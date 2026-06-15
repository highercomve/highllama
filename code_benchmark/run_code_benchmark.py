#!/usr/bin/env python3
"""
Private code-quality benchmark for highllama-served models.

Hits a *live* OpenAI-compatible server (whatever highllama is currently
serving on :8089 — this script never starts/stops the server), sends each
private coding task, extracts the model's code, compiles+runs it against a
hidden harness, and reports two headline quality numbers:

  pass@1     : fraction of tasks where EVERY hidden check passed (strict)
  test-pass% : fraction of all individual hidden checks that passed (partial)

Tasks live as JSON files in tasks/. They are hand-authored and intended to be
kept private (gitignored) so they don't leak into public training sets — that
is the whole point versus HumanEval/MBPP, which are already contaminated.

Usage:
    python run_code_benchmark.py                      # score the live model
    python run_code_benchmark.py --langs python,rust  # subset of languages
    python run_code_benchmark.py --tasks py_,rs_      # id-prefix filter
    python run_code_benchmark.py --temperature 0 --max-tokens 4096
    python run_code_benchmark.py --out results/foo.json --save-raw
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict

import requests

from runners import available_languages, extract_code, run_solution

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = "http://localhost:8089"

# OpenCode Go — big open coding models behind one subscription, exposed as plain
# OpenAI- or Anthropic-compatible HTTP. See https://opencode.ai/docs/go/
# ("Endpoints" table). Same private tasks, just a different backend, so local
# highllama models can be compared head-to-head against the frontier ones.
OPENCODE_GO_BASE = "https://opencode.ai/zen/go"
OPENCODE_GO_PROVIDER = {  # model id -> wire protocol
    "glm-5.1": "openai", "glm-5": "openai",
    "kimi-k2.7": "openai", "kimi-k2.6": "openai",
    "deepseek-v4-pro": "openai", "deepseek-v4-flash": "openai",
    "mimo-v2.5": "openai", "mimo-v2.5-pro": "openai",
    "minimax-m3": "anthropic", "minimax-m2.7": "anthropic", "minimax-m2.5": "anthropic",
    "qwen3.7-max": "anthropic", "qwen3.7-plus": "anthropic", "qwen3.6-plus": "anthropic",
}

SYSTEM_PROMPT = (
    "You are an expert programmer. Implement exactly what is asked. "
    "Respond with a SINGLE fenced code block in the requested language and "
    "nothing else — no explanation, no example usage, no tests. Define the "
    "requested function/symbol at top level with the exact name and signature."
)


def load_providers(path):
    """Read the provider registry (api keys, base URLs, protocols). Missing -> {}."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        print(f">> warning: could not parse {path}: {e} (ignoring)")
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def resolve_backend(args, providers):
    """
    Merge a named provider from providers.json with CLI flags into one backend.

    Precedence (highest first): explicit CLI flag > named-provider config >
    built-in default. `--opencode-go <id>` is sugar for `--provider opencode-go
    --model <id>` and still works with no config file (built-in base + env key).
    Returns (backend_dict, provider_name).
    """
    name = args.provider
    model = args.model
    if args.opencode_go:
        name = name or "opencode-go"
        model = model or args.opencode_go

    cfg = providers.get(name, {}) if name else {}
    if name and name not in providers and name != "opencode-go":
        print(f">> warning: provider '{name}' not in {os.path.basename(args.provider_config)}; "
              f"using CLI flags / defaults only")

    base = args.base or cfg.get("base") or (OPENCODE_GO_BASE if name == "opencode-go" else DEFAULT_BASE)
    api_key = args.api_key or cfg.get("api_key") or os.environ.get("OPENCODE_API_KEY", "")
    protocol = (args.protocol
                or (cfg.get("models") or {}).get(model)
                or (OPENCODE_GO_PROVIDER.get(model) if name == "opencode-go" else None)
                or cfg.get("protocol")
                or "openai")

    if name == "opencode-go" and not api_key:
        sys.exit("!! OpenCode Go needs an API key: set it in providers.json, or pass "
                 "--api-key / $OPENCODE_API_KEY (get one at https://opencode.ai/auth)")

    if not model:
        model = detect_model(base, api_key or None)

    return {"base": base, "model": model, "provider": protocol, "api_key": api_key}, name


def detect_model(base, api_key=None):
    """Pick the chat model from /v1/models, skipping embedding models."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    data = requests.get(f"{base}/v1/models", headers=headers, timeout=10).json()["data"]
    ids = [m["id"] for m in data]
    embed_pat = ("embed", "bge", "nomic", "gte", "e5-", "minilm", "arctic")
    chat = [i for i in ids if not any(p in i.lower() for p in embed_pat)]
    if not chat:
        sys.exit(f"!! no chat model found in {ids}")
    return chat[0]


def load_tasks(tasks_dir, lang_filter, id_filters, langs_available, diff_filter=None):
    tasks, skipped = [], []
    for path in sorted(glob.glob(os.path.join(tasks_dir, "*.json"))):
        with open(path) as f:
            t = json.load(f)
        t["_path"] = path
        if lang_filter and t["language"] not in lang_filter:
            continue
        if diff_filter and t.get("difficulty") not in diff_filter:
            continue
        if id_filters and not any(t["id"].startswith(p) for p in id_filters):
            continue
        if t["language"] not in langs_available:
            skipped.append((t["id"], t["language"]))
            continue
        tasks.append(t)
    return tasks, skipped


def _post(url, payload, headers):
    """POST json; on a 400 that complains about temperature, retry once without it.

    Some reasoning models (e.g. Kimi K2.7 Code via Moonshot) only allow their own
    fixed temperature and reject temperature=0. Dropping the field lets them use
    their default while models that accept temperature still get our value.
    """
    r = requests.post(url, json=payload, headers=headers, timeout=600)
    if (r.status_code == 400 and "temperature" in payload
            and "temperature" in r.text.lower()):
        payload = {k: v for k, v in payload.items() if k != "temperature"}
        r = requests.post(url, json=payload, headers=headers, timeout=600)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
    return r.json()


def _call_openai(backend, user, temperature, max_tokens):
    """OpenAI /v1/chat/completions -> (text, completion_tokens)."""
    headers = {}
    if backend.get("api_key"):
        headers["Authorization"] = f"Bearer {backend['api_key']}"
    payload = {
        "model": backend["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    b = _post(backend["base"].rstrip("/") + "/v1/chat/completions", payload, headers)
    text = b["choices"][0]["message"].get("content") or ""
    ctok = (b.get("usage") or {}).get("completion_tokens")
    return text, ctok


def _call_anthropic(backend, user, temperature, max_tokens):
    """Anthropic /v1/messages -> (text, output_tokens)."""
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if backend.get("api_key"):
        headers["x-api-key"] = backend["api_key"]
    payload = {
        "model": backend["model"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user}],
    }
    b = _post(backend["base"].rstrip("/") + "/v1/messages", payload, headers)
    text = "".join(p.get("text", "") for p in b.get("content", []) if p.get("type") == "text")
    ctok = (b.get("usage") or {}).get("output_tokens")
    return text, ctok


def ask_model(backend, task, temperature, max_tokens):
    """Send the task prompt via the backend's protocol; return (text, err, meta)."""
    user = (
        f"Language: {task['language']}\n\n{task['prompt']}\n\n"
        f"Return only the {task['language']} code in one fenced block."
    )
    call = _call_anthropic if backend["provider"] == "anthropic" else _call_openai
    t0 = time.time()
    try:
        text, ctok = call(backend, user, temperature, max_tokens)
    except Exception as e:
        return "", f"{type(e).__name__}: {e}", {"gen_s": round(time.time() - t0, 2),
                                                 "completion_tokens": None, "tok_s": None}
    gen_s = round(time.time() - t0, 2)
    tok_s = round(ctok / gen_s, 1) if ctok and gen_s > 0 else None
    return text, None, {"gen_s": gen_s, "completion_tokens": ctok, "tok_s": tok_s}


def grade(task, raw, meta):
    """Extract code, run it, return a per-task result record."""
    code = extract_code(raw, task["language"])
    res = run_solution(task["language"], code, task["harness_template"],
                       task.get("timeout_s", 15))
    n = task["num_checks"]
    passed = min(len(res["passed"]), n)
    return {
        "id": task["id"],
        "language": task["language"],
        "difficulty": task.get("difficulty", "?"),
        "category": task.get("category", "?"),
        "num_checks": n,
        "passed_checks": passed,
        "pass_at_1": passed == n and not res["failed"],
        "stage": res["stage"],
        "gen_s": meta["gen_s"],
        "completion_tokens": meta["completion_tokens"],
        "tok_s": meta["tok_s"],
        "code": code,
    }


def aggregate(records):
    """Compute headline + breakdown stats from per-task records."""
    def pct(num, den):
        return round(100.0 * num / den, 1) if den else 0.0

    total_tasks = len(records)
    solved = sum(r["pass_at_1"] for r in records)
    checks = sum(r["passed_checks"] for r in records)
    checks_total = sum(r["num_checks"] for r in records)
    gens = [r["gen_s"] for r in records if r.get("gen_s") is not None]
    ctoks = sum(r["completion_tokens"] or 0 for r in records)
    total_gen = sum(gens)

    def group(key):
        g = defaultdict(lambda: {"tasks": 0, "solved": 0, "checks": 0,
                                 "checks_total": 0, "gen_s": 0.0})
        for r in records:
            b = g[r[key]]
            b["tasks"] += 1
            b["solved"] += r["pass_at_1"]
            b["checks"] += r["passed_checks"]
            b["checks_total"] += r["num_checks"]
            b["gen_s"] += r.get("gen_s") or 0.0
        return {k: {"tasks": v["tasks"],
                    "pass_at_1_pct": pct(v["solved"], v["tasks"]),
                    "test_pass_pct": pct(v["checks"], v["checks_total"]),
                    "avg_gen_s": round(v["gen_s"] / v["tasks"], 2) if v["tasks"] else 0.0}
                for k, v in sorted(g.items())}

    return {
        "tasks": total_tasks,
        "pass_at_1_pct": pct(solved, total_tasks),
        "test_pass_pct": pct(checks, checks_total),
        "total_gen_s": round(total_gen, 1),
        "avg_gen_s": round(total_gen / len(gens), 2) if gens else 0.0,
        "completion_tokens": ctoks,
        "avg_tok_s": round(ctoks / total_gen, 1) if total_gen > 0 and ctoks else None,
        "by_language": group("language"),
        "by_difficulty": group("difficulty"),
    }


def print_summary(model, summ, records, skipped):
    print(f"\n=== code benchmark: {model} ===")
    tps = f"   {summ['avg_tok_s']} tok/s" if summ.get("avg_tok_s") else ""
    print(f"tasks: {summ['tasks']}   "
          f"pass@1: {summ['pass_at_1_pct']}%   "
          f"test-pass: {summ['test_pass_pct']}%")
    print(f"model time: {summ['total_gen_s']}s total   "
          f"{summ['avg_gen_s']}s/task avg{tps}")
    if skipped:
        sk = ", ".join(f"{i}({l})" for i, l in skipped)
        print(f"skipped (toolchain missing): {sk}")

    print("\nby language:")
    for lang, s in summ["by_language"].items():
        print(f"  {lang:<12} {s['tasks']:>2} tasks   "
              f"pass@1 {s['pass_at_1_pct']:>5}%   test {s['test_pass_pct']:>5}%"
              f"   {s['avg_gen_s']:>5}s/task")
    print("by difficulty:")
    for d, s in summ["by_difficulty"].items():
        print(f"  {d:<12} {s['tasks']:>2} tasks   "
              f"pass@1 {s['pass_at_1_pct']:>5}%   test {s['test_pass_pct']:>5}%"
              f"   {s['avg_gen_s']:>5}s/task")

    print("\nper task:")
    for r in records:
        mark = "OK " if r["pass_at_1"] else "   "
        gs = f"{r['gen_s']:>6.2f}s" if r.get("gen_s") is not None else "     ?s"
        print(f"  [{mark}] {r['id']:<28} {r['passed_checks']}/{r['num_checks']}"
              f"  {gs}  ({r['stage']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="", help="endpoint base URL (overrides the provider config)")
    ap.add_argument("--tasks-dir", default=os.path.join(HERE, "tasks"))
    ap.add_argument("--langs", default="", help="comma-separated language filter")
    ap.add_argument("--difficulty", default="",
                    help="comma-separated difficulty filter (easy,medium,hard,expert)")
    ap.add_argument("--tasks", default="", help="comma-separated id-prefix filter")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="generation cap (reasoning models need headroom for thinking + code)")
    ap.add_argument("--out", default="", help="results json (default results/YYYY-MM-DD/HH-MM-SS/<model>.json)")
    ap.add_argument("--save-raw", action="store_true", help="also keep raw model output")
    ap.add_argument("--provider-config", default=os.path.join(HERE, "providers.json"),
                    help="provider registry json (default providers.json)")
    ap.add_argument("--provider", default="", metavar="NAME",
                    help="named provider from providers.json (e.g. opencode-go); use with --model")
    ap.add_argument("--opencode-go", default="", metavar="MODEL",
                    help="sugar for --provider opencode-go --model MODEL (e.g. kimi-k2.7)")
    ap.add_argument("--protocol", choices=["openai", "anthropic"], default="",
                    help="wire protocol override (default from config, else openai)")
    ap.add_argument("--api-key", default="", help="bearer/x-api-key override (or $OPENCODE_API_KEY)")
    ap.add_argument("--model", default="",
                    help="model id; skips /v1/models autodetect (required for remote endpoints)")
    args = ap.parse_args()

    providers = load_providers(args.provider_config)
    backend, _name = resolve_backend(args, providers)
    model = backend["model"]

    avail = available_languages()
    lang_filter = {x for x in args.langs.split(",") if x} or None
    diff_filter = {x for x in args.difficulty.split(",") if x} or None
    id_filters = [x for x in args.tasks.split(",") if x]

    tasks, skipped = load_tasks(args.tasks_dir, lang_filter, id_filters, avail, diff_filter)
    if not tasks:
        sys.exit("!! no tasks matched (check --tasks-dir / filters / toolchains)")

    print(f">> model={model}  provider={backend['provider']}  base={backend['base']}")
    print(f">> tasks={len(tasks)}  languages={sorted({t['language'] for t in tasks})}")
    records, raws = [], {}
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        print(f"   [{i}/{len(tasks)}] {task['id']} ...", end=" ", flush=True)
        raw, err, meta = ask_model(backend, task, args.temperature, args.max_tokens)
        if err:
            print(f"request failed: {err}")
            records.append({"id": task["id"], "language": task["language"],
                            "difficulty": task.get("difficulty", "?"),
                            "category": task.get("category", "?"),
                            "num_checks": task["num_checks"], "passed_checks": 0,
                            "pass_at_1": False, "stage": "request_error",
                            "gen_s": meta["gen_s"], "completion_tokens": None,
                            "tok_s": None, "code": ""})
            continue
        rec = grade(task, raw, meta)
        if args.save_raw:
            raws[task["id"]] = raw
        records.append(rec)
        print(f"{rec['passed_checks']}/{rec['num_checks']}  {rec['gen_s']}s ({rec['stage']})")

    summ = aggregate(records)
    print_summary(model, summ, records, skipped)

    if args.out:
        out = args.out
    else:
        now = datetime.datetime.now()
        out = os.path.join(HERE, "results", now.strftime("%Y-%m-%d"),
                           now.strftime("%H-%M-%S"), re.sub(r"[^\w.-]", "_", model) + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    blob = {
        "model": model,
        "base": backend["base"],
        "provider": backend["provider"],
        "elapsed_s": round(time.time() - t0, 1),
        "params": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "summary": summ,
        "tasks": records,
    }
    if args.save_raw:
        blob["raw"] = raws
    with open(out, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"\n>> wrote {out}")


if __name__ == "__main__":
    main()
