"""
Per-language execution backends for the highllama code benchmark.

Each task ships a `harness_template`: a *complete* program in the target
language containing the literal placeholder ``{{SOLUTION}}`` where the model's
extracted code is spliced in. The combined program is compiled (if needed) and
run in a temp dir under a wall-clock timeout. The harness prints one line per
assertion:

    @@CHECK@@ <name> PASS
    @@CHECK@@ <name> FAIL

The runner counts PASS lines. A compile error, crash, or timeout simply means
fewer (or zero) PASS lines were printed — that falls out as partial credit with
no special-casing.

Only languages whose toolchain is actually installed are exposed (see
`available_languages`), so the same task set degrades gracefully on a box that
lacks, say, a Go compiler.
"""
import os
import re
import shutil
import signal
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Sandboxes live under the benchmark dir, NOT /tmp: the Go toolchain refuses to
# honor a go.mod located in the system temp root, which breaks `go run .`.
SCRATCH = os.path.join(HERE, ".scratch")

PLACEHOLDER = "{{SOLUTION}}"
CHECK_RE = re.compile(r"^@@CHECK@@\s+(\S+)\s+(PASS|FAIL)\s*$", re.MULTILINE)


# --- language registry -------------------------------------------------------
# Each entry:
#   src      : filename the combined program is written to
#   probe    : a command whose presence (exit 0) proves the toolchain exists
#   compile  : optional argv list run first; non-zero exit = compile failure
#   run      : argv list executed to produce the @@CHECK@@ output
# {ext}/{bin} are not templated — keep argv concrete and simple.
LANGS = {
    "python": {
        "src": "main.py",
        "probe": ["python3", "--version"],
        "run": ["python3", "-I", "main.py"],
    },
    "javascript": {
        "src": "main.mjs",
        "probe": ["node", "--version"],
        "run": ["node", "main.mjs"],
    },
    "typescript": {
        "src": "main.ts",
        "probe": ["deno", "--version"],
        "run": ["deno", "run", "--quiet", "--no-check", "main.ts"],
    },
    "rust": {
        "src": "main.rs",
        "probe": ["rustc", "--version"],
        "compile": ["rustc", "-O", "--edition", "2021", "main.rs", "-o", "prog"],
        "run": ["./prog"],
    },
    "go": {
        # Go imports are file-scoped, so the model's code goes in its own file
        # (sol.go) next to the harness (main.go); `go run .` compiles the package.
        "src": "main.go",
        "sol": "sol.go",
        "extra": {"go.mod": "module hlbench\n\ngo 1.21\n"},
        "probe": ["go", "version"],
        "run": ["go", "run", "."],
    },
    "c": {
        "src": "main.c",
        "probe": ["gcc", "--version"],
        "compile": ["gcc", "-O2", "-std=c17", "main.c", "-o", "prog", "-lm"],
        "run": ["./prog"],
    },
    "cpp": {
        "src": "main.cpp",
        "probe": ["g++", "--version"],
        "compile": ["g++", "-O2", "-std=c++20", "main.cpp", "-o", "prog"],
        "run": ["./prog"],
    },
    "bash": {
        "src": "main.sh",
        "probe": ["bash", "--version"],
        "run": ["bash", "main.sh"],
    },
}

# Markdown fence aliases -> canonical language key.
FENCE_ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "mjs": "javascript", "node": "javascript",
    "ts": "typescript",
    "rs": "rust",
    "golang": "go",
    "c++": "cpp", "cc": "cpp", "cxx": "cpp",
    "sh": "bash", "shell": "bash",
}


def _which_ok(cmd):
    if shutil.which(cmd[0]) is None:
        return False
    try:
        subprocess.run(cmd, capture_output=True, timeout=20)
        return True
    except Exception:
        return False


_AVAIL_CACHE = None


def available_languages():
    """Set of language keys whose toolchain is installed (memoized)."""
    global _AVAIL_CACHE
    if _AVAIL_CACHE is None:
        _AVAIL_CACHE = {k for k, c in LANGS.items() if _which_ok(c["probe"])}
    return _AVAIL_CACHE


def _exec(cmd, cwd, timeout, env):
    """Run argv in its own process group; SIGKILL the whole group on timeout."""
    p = subprocess.Popen(
        cmd, cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            out, err = p.communicate(timeout=5)
        except Exception:
            out, err = "", ""
        return -signal.SIGKILL, out, (err or "") + "\n[killed: timeout]"


def run_solution(language, solution_code, harness_template, timeout):
    """
    Splice `solution_code` into `harness_template`, build+run it, and return a
    dict: {passed: [names], failed: [names], stage, returncode, stderr}.

    `stage` is "compile", "run", or "timeout" — useful for triage. PASS/FAIL
    names come straight from the program's @@CHECK@@ output.
    """
    if language not in LANGS:
        return {"passed": [], "failed": [], "stage": "unsupported",
                "returncode": None, "stderr": f"no runner for {language}"}

    cfg = LANGS[language]
    sol = solution_code or ""
    # Two write modes: splice the solution into the harness (placeholder present),
    # or keep it in a separate file the harness compiles alongside (cfg["sol"]).
    if PLACEHOLDER in harness_template:
        program, sep_solution = harness_template.replace(PLACEHOLDER, sol), None
    else:
        program, sep_solution = harness_template, sol

    env = dict(os.environ)
    os.makedirs(SCRATCH, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="hl_codebench_", dir=SCRATCH)
    # Keep every toolchain's scratch (go build cache, rustc temp, …) inside the
    # sandbox dir so nothing leaks into $HOME and cleanup is a single rmtree.
    env["HOME"] = workdir
    # NOT workdir itself: Go ignores a go.mod that sits in $TMPDIR (temp root).
    env["TMPDIR"] = os.path.join(workdir, ".tmp")
    os.makedirs(env["TMPDIR"], exist_ok=True)
    env["GOCACHE"] = os.path.join(workdir, ".gocache")
    env["GOFLAGS"] = "-mod=mod"
    env.setdefault("GOTOOLCHAIN", "local")

    try:
        with open(os.path.join(workdir, cfg["src"]), "w") as f:
            f.write(program)
        if sep_solution is not None:
            with open(os.path.join(workdir, cfg["sol"]), "w") as f:
                f.write(sep_solution)
        for fname, content in cfg.get("extra", {}).items():
            with open(os.path.join(workdir, fname), "w") as f:
                f.write(content)

        if cfg.get("compile"):
            rc, _out, err = _exec(cfg["compile"], workdir, timeout, env)
            if rc != 0:
                return {"passed": [], "failed": [], "stage": "compile",
                        "returncode": rc, "stderr": (err or "")[-4000:]}

        rc, out, err = _exec(cfg["run"], workdir, timeout, env)
        passed, failed = [], []
        for name, verdict in CHECK_RE.findall(out or ""):
            (passed if verdict == "PASS" else failed).append(name)
        stage = "timeout" if rc == -signal.SIGKILL else "run"
        return {"passed": passed, "failed": failed, "stage": stage,
                "returncode": rc, "stderr": (err or "")[-4000:]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# --- code extraction ---------------------------------------------------------
_FENCE_RE = re.compile(r"```([\w+#.-]*)\s*\n(.*?)```", re.DOTALL)
# gpt-oss "harmony" channel markers and similar control wrappers some local
# models emit around their real answer.
_HARMONY_RE = re.compile(r"<\|?/?(?:channel|message|start|end|return|assistant|final)[^>]*\|?>")


def extract_code(text, language):
    """
    Pull the most plausible code block for `language` out of a chat completion.

    Preference order: a fenced block tagged with the language (or an alias),
    then the longest untagged fenced block, then the raw text (last resort, for
    models that forget the fence). Harmony/channel control tokens are stripped.
    """
    if not text:
        return ""
    text = _HARMONY_RE.sub("", text)

    tagged, untagged = [], []
    for tag, body in _FENCE_RE.findall(text):
        key = FENCE_ALIASES.get(tag.lower(), tag.lower())
        (tagged if key == language else untagged).append(body)
    if tagged:
        return max(tagged, key=len).strip("\n")
    if untagged:
        return max(untagged, key=len).strip("\n")
    return text.strip()
