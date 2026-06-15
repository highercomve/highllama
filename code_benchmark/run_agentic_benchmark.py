#!/usr/bin/env python3
"""
Agentic variant of the code benchmark: run the SAME private tasks through an
agent with tools enabled, instead of a single chat-completion. Select the agent
backend with --agent and pass just a model name with --model. The model can
write a file, run it, read errors, and fix its code before finishing — so this
measures model+tooling+iteration, not raw one-shot generation. Results are
graded by the exact same hidden harness, so they sit directly next to the
single-shot numbers in score_compare.py.

Supported agents:
  opencode   `opencode run -m <model> --dir <workdir> ...` (default)
  pi         `pi --print --model <model> --tools edit,write <prompt>` (cwd=workdir)

For each task we:
  1. make a throwaway working dir,
  2. run the selected agent command to write the solution into
     solution.<ext> (and let it test however it likes),
  3. read that file back and grade it with runners.run_solution.

Usage:
    python run_agentic_benchmark.py --model kimi-k2.7-code
    python run_agentic_benchmark.py --agent opencode --model kimi-k2.7-code
    python run_agentic_benchmark.py --agent opencode --model llamacpp/gemma-4-26B-A4B-it-QAT-Q4_0
    python run_agentic_benchmark.py --agent pi --model gemma-4-26B-A4B-it-QAT-Q4_0
    python run_agentic_benchmark.py --agent pi --model gemma-4-26B-A4B-it-QAT-Q4_0 --tasks py_,rs_ --task-timeout 240

WARNING: the opencode backend runs with --dangerously-skip-permissions, i.e. it
will execute commands the model chooses inside the per-task scratch dir. The pi
backend is run with edit/write tools enabled in the same scratch dir. Only run
models you trust on this machine.
"""
import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import run_code_benchmark as R
from runners import available_languages, run_solution

HERE = os.path.dirname(os.path.abspath(__file__))

# What file the agent must write its solution into, per language. The contents
# are fed to run_solution() exactly like the single-shot extracted code, so the
# same {{SOLUTION}}/separate-file harness logic applies (Go -> its own file).
SOLUTION_FILE = {
    "python": "solution.py", "javascript": "solution.mjs", "typescript": "solution.ts",
    "rust": "solution.rs", "go": "solution.go", "c": "solution.c",
    "cpp": "solution.cpp", "bash": "solution.sh",
}

AGENT_COMMANDS = {
    "opencode": "opencode run -m {model} --dir {workdir} --dangerously-skip-permissions {prompt}",
    "pi": "pi --print --model {model} --tools edit,write {prompt}",
}


def agent_prompt(task):
    lang = task["language"]
    sol = SOLUTION_FILE[lang]
    extra = ""
    if lang == "go":
        extra = (f" The file {sol} must begin with `package main`, include the imports "
                 f"you need, and define the function — but do NOT add a main function.")
    return (
        f"{task['prompt']}\n\n"
        f"Create a file named {sol} in the current working directory containing ONLY "
        f"the {lang} implementation described above: the requested function/definition "
        f"with the exact name and signature, plus any imports it needs.{extra} Do not "
        f"put a main function, tests, example usage, or stray print/debug statements in "
        f"{sol}. You MAY create other files and run commands to test and debug your "
        f"solution — please verify it actually works before finishing. When done, {sol} "
        f"must contain just the implementation."
    )


def run_agent(agent, model, workdir, prompt, timeout, logpath):
    """Drive the selected agent command headlessly in workdir; return (returncode, timed_out)."""
    agent_cmd = AGENT_COMMANDS[agent].format(
        model=shlex.quote(model),
        workdir=shlex.quote(workdir),
        prompt=shlex.quote(prompt),
    )
    cmd = shlex.split(agent_cmd)
    with open(logpath, "w") as log:
        p = subprocess.Popen(cmd, cwd=workdir, stdout=log, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)
        try:
            p.communicate(timeout=timeout)
            return p.returncode, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.communicate()
            return -signal.SIGKILL, True


def read_solution(workdir, sol):
    """Return the agent's solution file contents, searching subdirs as a fallback."""
    direct = os.path.join(workdir, sol)
    if os.path.isfile(direct):
        return open(direct).read()
    for root, _dirs, files in os.walk(workdir):
        if sol in files:
            return open(os.path.join(root, sol)).read()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=list(AGENT_COMMANDS), default="opencode",
                    help="agent backend to drive (default: opencode)")
    ap.add_argument("--model", default="",
                    help="model name/id passed to the agent (just the model name, e.g. kimi-k2.7-code)")
    ap.add_argument("--tasks-dir", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--langs", default="", help="comma-separated language filter")
    ap.add_argument("--difficulty", default="",
                    help="comma-separated difficulty filter (easy,medium,hard,expert)")
    ap.add_argument("--tasks", default="", help="comma-separated id-prefix filter")
    ap.add_argument("--task-timeout", type=int, default=300,
                    help="per-task wall-clock budget for the agent (seconds)")
    ap.add_argument("--out", default="", help="results json (default results/YYYY-MM-DD/HH-MM-SS/<model>__agent.json)")
    ap.add_argument("--keep-workdirs", action="store_true", help="don't delete the agent scratch dirs")
    args = ap.parse_args()

    if not args.model:
        sys.exit("!! choose a model: --model <name>")

    binary = {"opencode": "opencode", "pi": "pi"}[args.agent]
    if shutil.which(binary) is None:
        sys.exit(f"!! {binary} not found on PATH")

    # opencode needs a provider prefix; default to opencode-go if the user gave a bare name.
    if args.agent == "opencode" and "/" not in args.model:
        model = f"opencode-go/{args.model}"
    else:
        model = args.model

    avail = available_languages()
    lang_filter = {x for x in args.langs.split(",") if x} or None
    diff_filter = {x for x in args.difficulty.split(",") if x} or None
    id_filters = [x for x in args.tasks.split(",") if x]
    tasks, skipped = R.load_tasks(args.tasks_dir, lang_filter, id_filters, avail, diff_filter)
    if not tasks:
        sys.exit("!! no tasks matched (check --tasks-dir / filters / toolchains)")

    label = f"{model} ({args.agent}-agent)"
    print(f">> agent={args.agent} model={model}  tasks={len(tasks)}  timeout={args.task_timeout}s/task")
    scratch_root = os.path.join(HERE, ".scratch-agent")
    os.makedirs(scratch_root, exist_ok=True)

    records = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        sol = SOLUTION_FILE[task["language"]]
        print(f"   [{i}/{len(tasks)}] {task['id']} ...", end=" ", flush=True)
        workdir = tempfile.mkdtemp(prefix=f"{task['id']}_", dir=scratch_root)
        logpath = os.path.join(workdir, "_agent.log")
        ts = time.time()
        rc, timed_out = run_agent(args.agent, model, workdir, agent_prompt(task),
                                  args.task_timeout, logpath)
        gen_s = round(time.time() - ts, 2)

        code = read_solution(workdir, sol)
        if code is None:
            stage = "timeout" if timed_out else "no_solution_file"
            rec = {"id": task["id"], "language": task["language"],
                   "difficulty": task.get("difficulty", "?"), "category": task.get("category", "?"),
                   "num_checks": task["num_checks"], "passed_checks": 0, "pass_at_1": False,
                   "stage": stage, "gen_s": gen_s, "completion_tokens": None, "tok_s": None, "code": ""}
        else:
            res = run_solution(task["language"], code, task["harness_template"], task.get("timeout_s", 15))
            n = task["num_checks"]
            passed = min(len(res["passed"]), n)
            rec = {"id": task["id"], "language": task["language"],
                   "difficulty": task.get("difficulty", "?"), "category": task.get("category", "?"),
                   "num_checks": n, "passed_checks": passed,
                   "pass_at_1": passed == n and not res["failed"],
                   "stage": ("timeout_partial" if timed_out else res["stage"]),
                   "gen_s": gen_s, "completion_tokens": None, "tok_s": None, "code": code}
        records.append(rec)
        print(f"{rec['passed_checks']}/{rec['num_checks']}  {gen_s}s ({rec['stage']})")
        if not args.keep_workdirs:
            shutil.rmtree(workdir, ignore_errors=True)

    summ = R.aggregate(records)
    R.print_summary(label, summ, records, skipped)

    if args.out:
        out = args.out
    else:
        now = datetime.datetime.now()
        out = os.path.join(HERE, "results", now.strftime("%Y-%m-%d"),
                           now.strftime("%H-%M-%S"), re.sub(r"[^\w.-]", "_", label) + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": label, "mode": f"{args.agent}-agent", "agent_model": model,
                   "elapsed_s": round(time.time() - t0, 1),
                   "params": {"task_timeout_s": args.task_timeout},
                   "summary": summ, "tasks": records}, f, indent=2)
    print(f"\n>> wrote {out}")


if __name__ == "__main__":
    main()
