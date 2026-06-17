# code_benchmark — private code-quality benchmark for highllama models

A small, **private** coding benchmark that scores any model highllama serves and
gives you a single quality percentage per model. The tasks are hand-written and
original — *not* from HumanEval / MBPP / LeetCode / GitHub — so they should not
be present in any model's training data. That is the whole point versus public
benchmarks, which are already contaminated.

Each task asks the model to implement one function/class; a hidden harness then
compiles and runs the model's code against assertions and reports two numbers:

- **pass@1** — fraction of tasks where *every* hidden check passed (strict: does it actually work).
- **test-pass %** — fraction of all individual hidden checks passed (partial credit).

It also records **how long the model took to answer each task** (`gen_s`) plus
`tok/s` from the server's usage stats — per task, per language, and as run totals
— so you can weigh quality against speed across models.

Languages covered (toolchain auto-detected; missing ones are skipped):
**python, javascript, typescript, rust, go, c, cpp, bash**.

## Layout

| file | what |
|---|---|
| `build_tasks.py` | authoring source for the tasks (private; gitignored). Run it to emit `tasks/*.json`. |
| `tasks/*.json` | generated task files the runner consumes (private; gitignored). |
| `runners.py` | per-language compile/run sandbox + model-output code extraction. |
| `run_code_benchmark.py` | the runner: hits the live server, scores the current model, writes `results/YYYY-MM-DD/HH-MM-SS/<model>.json`. |
| `score_compare.py` | leaderboard across all `results/**/*.json`. |
| `selftest.py` | validates every harness against its private reference solution (no server needed). |
| `run_all_models.sh` | optional: swaps the highllama model between runs (the only piece that touches the server). |
| `run_opencode_go.sh` | optional: sweep several **OpenCode Go** frontier models (remote API — never touches the local server). |
| `bench.sh` | optional: run single-shot + agentic benchmarks for one model in a single pass. |
| `providers.example.json` | template for `providers.json` (gitignored) — endpoint base URLs, API keys, and per-model protocol. |

## Quick start

```bash
# 1. (first time / after editing tasks) generate task files and sanity-check them
python3 build_tasks.py
python3 selftest.py            # every harness must report ALL HARNESSES OK

# 2. with a model already served by highllama on :8089, score it
python3 run_code_benchmark.py  # -> results/YYYY-MM-DD/HH-MM-SS/<model>.json + a printed summary

# 3. score more models, then compare
#    (switch the served model yourself, re-run step 2, OR use the orchestrator)
python3 score_compare.py       # leaderboard across results/ (recursively, including dated subdirs)
```

The runner is **server-agnostic**: it reads whatever model `/v1/models` reports
and never starts or stops the server. To benchmark several models hands-free use
`./run_all_models.sh <model> <model> ...`, but **don't** run that while another
benchmark is using the server — it restarts `llama-server` between models.

## Difficulty tiers

Tasks are tagged `easy` / `medium` / `hard` / `expert`. Run a subset with
`--difficulty` (works on both the single-shot and agentic runners):

```bash
python3 run_code_benchmark.py --difficulty expert          # only the hardest
python3 run_agentic_benchmark.py --agent opencode --model kimi-k2.7-code --difficulty hard,expert
```

The **expert** tier is aimed at the agentic runs: a write-run-fix loop can pass
the worked example in a prompt yet still miss the *hidden* edge cases or a
performance gate. For example `rs_count_inversions` includes a 1,000,000-element
hidden check, so an O(n²) solution passes the small cases but times out and
misses pass@1 — only an O(n log n) answer scores 5/5. This keeps the benchmark
discriminating once tooling pushes the easy/medium tasks to 100%.

## Comparing against big OpenCode Go models

[OpenCode Go](https://opencode.ai/docs/go/) is a cheap subscription that exposes
strong open coding models (GLM-5.1, Kimi K2.7, DeepSeek V4, Qwen3.7, MiniMax,
MiMo) over plain OpenAI- or Anthropic-compatible HTTP. Pointing the same private
tasks at it lets you benchmark your local highllama models head-to-head against
the frontier ones — your private problems are sent to OpenCode's endpoint but
they stay out of any public training set.

First put your key in a provider config (kept out of git):

```bash
cp providers.example.json providers.json   # then edit providers.json:
#   "opencode-go": { ..., "api_key": "sk-..." }
```

Then pick a model at runtime — the base URL, key, and wire protocol come from
the config:

```bash
# one model ('opencode-go' provider implied by --opencode-go)
python3 run_code_benchmark.py --opencode-go kimi-k2.7
python3 run_code_benchmark.py --opencode-go qwen3.7-max   # anthropic-style, handled automatically

# a sweep, then a leaderboard mixing local + remote results
./run_opencode_go.sh kimi-k2.7 glm-5.1 deepseek-v4-pro qwen3.7-max
python3 score_compare.py
```

`--opencode-go <id>` is sugar for `--provider opencode-go --model <id>`; the
per-model protocol (chat/completions vs messages) comes from the `models` map in
the config, per the [Endpoints table](https://opencode.ai/docs/go/). This is
**remote** — it never starts/stops or touches your local llama-server, so it's
safe to run while a local benchmark is in progress. (No `providers.json`? It
still works from the built-in defaults if you `export OPENCODE_API_KEY=...`.)

### Provider config (`providers.json`)

Each top-level key is a provider name you select with `--provider <name>`. Add
any OpenAI- or Anthropic-compatible endpoint once and reuse it:

```json
{
  "opencode-go": { "base": "https://opencode.ai/zen/go", "api_key": "sk-...",
                   "protocol": "openai",
                   "models": { "kimi-k2.7": "openai", "qwen3.7-max": "anthropic" } },
  "my-vllm":     { "base": "http://10.0.0.5:8000", "api_key": "", "protocol": "openai" }
}
```

- `protocol` — default wire format for the provider (`openai` | `anthropic`).
- `models` — optional per-model protocol override (OpenCode Go needs this since
  it serves some models as openai and others as anthropic).

```bash
python3 run_code_benchmark.py --provider my-vllm --model Qwen3.5-9B
```

CLI flags (`--base --api-key --protocol --model`) always override the config, so
you can still hit a one-off endpoint without editing the file.

> Note: OpenCode Go bills per request and enforces usage limits — roughly one
> request per task. Cheap, but mind the per-window caps if you sweep many models.
> Use `--difficulty expert` to run only the few hardest tasks while iterating.
>
> Reasoning models (e.g. Kimi K2.7 Code) spend many tokens "thinking" before the
> code, so the default `--max-tokens` is 8192. Some also forbid `temperature=0`
> (Moonshot's Kimi only allows its own value); the runner detects that 400 and
> automatically retries the request without the temperature field, so runs just
> work — they're simply not at temp 0 for those models.

## Agentic mode (same tasks, with tooling)

`run_agentic_benchmark.py` runs the *same* private tasks through an agent with
tools enabled instead of a single completion, so the model can write
`solution.<ext>`, run it, read the error, and fix its code before finishing. The
file it leaves behind is graded by the same hidden harness, so the result lands
right next to the single-shot number in `score_compare.py`. Latency now includes
the whole tool loop (expect minutes per task, not seconds).

Pick the agent backend with `--agent {opencode,pi}` and pass a plain model name
with `--model`. For `opencode`, bare model names are automatically prefixed with
`opencode-go/`; if you need a different provider, pass the full
`provider/model` id. For `pi`, the model name is passed straight through.

```bash
# OpenCode Go frontier model (default agent)
python3 run_agentic_benchmark.py --model kimi-k2.7-code

# local highllama via opencode (exposed as the 'llamacpp' provider)
python3 run_agentic_benchmark.py --agent opencode --model llamacpp/gemma-4-26B-A4B-it-QAT-Q4_0

# pi coding assistant
python3 run_agentic_benchmark.py --agent pi --model gemma-4-26B-A4B-it-QAT-Q4_0

python3 score_compare.py                     # single-shot vs agent, side by side
python3 score_compare.py --serve --port 8080 # interactive HTML leaderboard
```

Notes / gotchas:
- `--agent` chooses the backend (`opencode` is the default). The model string is
  always just a name; no `opencode models` lookup or provider selection is
  required unless you want a non-default opencode provider.
- The opencode backend runs headless with `--dangerously-skip-permissions`, i.e.
  the model executes commands it chooses inside a per-task `.scratch-agent/`
  dir. The pi backend is run with edit/write tools enabled in the same dir.
  Only run models you trust.
- `--keep-workdirs` preserves those dirs (incl. `_agent.log`) for debugging;
  `--task-timeout` caps each task (default 300s).
- Weak local models may have limited tool-calling and can underperform their
  single-shot score here (they fumble the edit/run loop). That gap *is* the
  signal — it's what agentic tooling does or doesn't buy you per model.

## One-pass benchmark for a single model (`bench.sh`)

`bench.sh` runs the single-shot code benchmark plus any agentic backends you
want for a single model, then prints (and optionally serves) the leaderboard.
It manages the local highllama server for you and picks the endpoint
automatically: local models hit `http://localhost:8089`, and OpenCode Go
models hit `https://opencode.ai/zen/go`. There is no `--base` flag for
`bench.sh`; use `--remote opencode-go` or the `opencode-go/` model prefix.

```bash
# local model: single-shot + opencode + pi
./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent opencode,pi

# same, but only agentic with opencode
./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent opencode

# remote OpenCode Go model: single-shot + opencode agent
# (the opencode-go/ prefix is auto-detected, so --remote is optional)
./code_benchmark/bench.sh --model opencode-go/kimi-k2.7-code --agent opencode
./code_benchmark/bench.sh --model kimi-k2.7-code --remote opencode-go --agent opencode

# run only python/rust tasks and serve the HTML report when done
./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent all \
    --langs python,rust --tasks py_,rs_ --serve
```

Use `--agent all` as a shortcut for `--agent opencode,pi`. If `--agent` is
omitted, only the single-shot benchmark runs.

## Useful flags (run_code_benchmark.py)

```
--langs python,rust       only these languages
--difficulty hard,expert  only these difficulties (easy,medium,hard,expert)
--tasks py_,rs_           only tasks whose id starts with one of these prefixes
--temperature 0.0         sampling temp (default 0 = deterministic, closest to pass@1)
--max-tokens 8192         generation cap (reasoning models need headroom)
--thinking-effort none|low|medium|high  control thinking via chat_template_kwargs (OpenAI-protocol backends; ignored by Anthropic)
--save-raw                also store the raw model output in the results json
--out results/foo.json    custom output path (bypasses the dated default)
--opencode-go <model-id>  sugar for --provider opencode-go --model <id>
--provider <name>         named provider from providers.json
--model <id>              model id (skip /v1/models autodetect; required for remote)
--base http://host:port   endpoint base URL override
--protocol openai|anthropic   wire protocol override
--api-key <key>           bearer/x-api-key override (or $OPENCODE_API_KEY)
--provider-config <path>  provider registry json (default providers.json)
```

## Useful flags (run_agentic_benchmark.py)

```
--agent {opencode,pi}     agent backend to drive (default: opencode)
--model <name>            model name passed to the agent (e.g. kimi-k2.7-code)
--langs python,rust       only these languages
--difficulty hard,expert  only these difficulties
--tasks py_,rs_           only tasks whose id starts with one of these prefixes
--task-timeout 300        per-task wall-clock budget for the agent (seconds)
--keep-workdirs           don't delete the agent scratch dirs (useful for debugging)
--out results/foo.json    custom output path (bypasses the dated default)
```

## Useful flags (score_compare.py)

```
results/*.json ...        result files to compare (default: all results/**/*.json)
--serve                   start a local web server with charts
--port 8080               port for the web server (default 8080)
```

The web report loads Chart.js from a CDN, so it needs internet access on the
machine that opens the page.

## Useful flags (`bench.sh`)

```
--model <name>            model to benchmark (required). Prefix with
                          opencode-go/ to auto-detect remote mode.
--agent <list|all>        comma-separated agents: opencode, pi, or all
--remote opencode-go      use OpenCode Go instead of the local server
--ctx <size>              context size for the local server (default 32k)
--langs <list>            language filter passed to runners
--difficulty <list>       difficulty filter passed to runners
--tasks <prefixes>        task-prefix filter passed to runners
--serve                   launch score_compare.py --serve after benchmarking
```

## How scoring works

The model is asked for a single fenced code block. `extract_code` pulls it out
(handles ```lang fences, aliases like `py`/`rs`, gpt-oss harmony channel markers,
and a raw-text fallback). The code is spliced into the task's harness and run in
a throwaway sandbox under `.scratch/` with a per-task wall-clock timeout and its
own `$HOME`/caches. The harness prints `@@CHECK@@ <name> PASS|FAIL` per
assertion; the runner counts PASS lines. A compile error, crash, or timeout just
yields fewer PASS lines — partial credit falls out naturally, no special cases.

## Adding tasks

Append a `TASK(...)` call in `build_tasks.py`. Each task needs: a prompt (what
the model sees), a `harness_template` (a complete program in the target language
with the literal `{{SOLUTION}}` placeholder — except Go, which uses a separate
`sol.go` file, so its harness has no placeholder), `num_checks`, and a private
`reference` solution used only by `selftest.py`. Then:

```bash
python3 build_tasks.py && python3 selftest.py
```

`selftest.py` must stay green — if a harness fails on its own reference solution,
the harness is buggy, not the model. Keep new problems original and unusual so
they remain contamination-resistant.

## Privacy

`build_tasks.py`, `tasks/`, and `results/` (including dated subdirectories) are
gitignored on purpose. If you publish this repo, the framework ships but your
private problems and run artifacts do not. Keep the problems off the public
internet to preserve their value as an uncontaminated benchmark.

## Safety note

Generated code is compiled and executed locally. It runs in a temp sandbox with
a timeout and isolated env, but it is **not** a security sandbox. Only benchmark
models you're willing to run code from. For stronger isolation, wrap the run in
`firejail`/a container.
