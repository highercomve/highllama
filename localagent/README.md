# localagent — local-LLM sub-agents, validated by Claude

Run `claude` sub-agents whose **brain is your local `llama-server`** (not the Anthropic
API), so the orchestrating Claude (Opus) can **validate** the result before anything is
trusted or lands on a real branch.

## Two ways to run a local sub-agent

| | **CLI orchestrator** (`localagent` agent) | **Native gateway** (`local-llama` agent) |
|---|---|---|
| how | Opus calls the `localagent` CLI via Bash | a real subagent with `model: local-llama` |
| session | works in **any** session, unchanged | only in a session started with `claude-local` |
| blast radius | **none** — main session never routes through the proxy | proxy sits in front of ALL session traffic |
| isolation | git worktree per run | normal subagent (parent decides) |
| best for | the safe default; validate-then-apply | when you want a native subagent on a different model |

Both are powered by the same `anthropic_proxy.py`. The proxy is a **router**: requests
for a "local" model are translated to llama-server; everything else (Opus) is passed
through verbatim to `api.anthropic.com`.

### Native gateway mode (the `model:` field)

A Claude Code subagent's `model:` is a *name* override, not a network override — subagents
share the session's endpoint. So to run a subagent on the local model natively, the whole
session must point at the router. The `claude-local` launcher does this opt-in:

```bash
claude-local            # interactive Opus session; `local-llama` subagent available
                        #   Opus → relayed to Anthropic;  local-llama → llama-server
claude-local --model local-llama -p "..."   # run the MAIN loop on the local model
```

Sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8090` + `CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1`
for that session only. Your plain `claude` is untouched. (Auth: subscription OAuth is
forwarded to Anthropic by the proxy; if Opus passthrough ever 401s, export `ANTHROPIC_API_KEY`.)

---

## CLI orchestrator mode

Headless `claude` sub-agents, isolated in a git worktree, so Opus can validate the
**diff + transcript** before anything lands on the real branch.

```
  Opus (orchestrator)
     │  localagent run --task "..."
     ▼
  claude --bare -p   ← LOCAL brain, real agentic tools (Bash/Read/Write/Edit/Grep/Glob)
     │  Anthropic /v1/messages
     ▼
  anthropic_proxy.py ← stdlib translation proxy (Anthropic ⇄ OpenAI)
     │  OpenAI /v1/chat/completions
     ▼
  llama-server :8089 ← gemma / qwen / whatever highllama loaded

  Opus reviews .git/localagent/meta/<branch>/changes.diff  →  localagent apply | discard
```

The local model does the cheap grunt work (drafting, investigating, implementing,
committing on a scratch branch); **Opus is the gatekeeper** for anything that reaches
your real branches.

## Why a proxy?

`claude` speaks the **Anthropic Messages API**; `llama-server` speaks the **OpenAI API**.
`anthropic_proxy.py` translates between them, including:
- streaming SSE (chunked) ⇄ OpenAI streaming chunks
- tool definitions, tool_use / tool_result round-trips
- `stop_reason` normalization, `count_tokens` endpoint
- **disables the model's `reasoning_content`** by default (`chat_template_kwargs.enable_thinking=false`)
  so reasoning models don't burn the token budget thinking — we validate
  *actions*, not chain-of-thought. Set `LLAMA_DISABLE_THINKING=0` to keep thinking.
- **tool-call salvage** — some local models (Qwen3-Coder especially) emit tool calls as
  TEXT (`<function=Name><parameter=k>v</parameter></function>` or Hermes
  `<tool_call>{json}`) when llama.cpp's parser misses them; Claude Code then sees no tool
  call and the loop stalls. `salvage_tool_calls()` detects these and rebuilds real
  `tool_use` blocks. To do this the streaming path **buffers each local response**, salvages,
  then emits clean Anthropic events (only local responses use this path — Opus is
  passthrough, so the watched session still streams live).

Zero dependencies — Python stdlib only.

## Install

```bash
ln -sf ~/Code/llms/localagent/localagent.sh   ~/.local/bin/localagent     # CLI orchestrator
ln -sf ~/Code/llms/localagent/claude-local.sh ~/.local/bin/claude-local   # gateway launcher
```

Agents (Claude Code, loaded at startup — restart to pick up changes):
- `~/.claude/agents/localagent.md` — CLI-orchestrator agent (runs on Opus, any session).
- `~/.claude/agents/local-llama.md` — native worker (`model: local-llama`, needs a
  `claude-local` session).

Requires `llama-server` running (use `highllama`) and the `claude` CLI.

## Usage

```bash
# proxy lifecycle (run auto-starts it; this is for manual control)
localagent proxy start|status|stop|restart

# implement something in an ISOLATED worktree (default for git repos)
localagent run --repo /path/to/repo --name feat-x --timeout 600 \
  --task "Add a --json flag to the CLI that prints output as JSON. Add a test. Commit it."

# read-only investigation, IN PLACE (no worktree, never mutates the index)
localagent run --repo /path/to/repo --readonly \
  --task "Where is rate limiting enforced? List files and line numbers."

# after reviewing the diff:
localagent diff   la/feat-x-20260608-...     # re-print the diff
localagent apply  la/feat-x-20260608-...      # merge the branch into the current branch
localagent discard la/feat-x-20260608-...     # throw it away

localagent list                               # active worktrees awaiting validation

# export logged dataset calls:
localagent export --format sharegpt -o exported_dataset.json  # export all calls
localagent export --latest 10 --has-tools                    # view latest 10 tool calls on stdout
```

### Flags for `run`
| flag | meaning |
|------|---------|
| `--task "..."` / `--task-file F` / stdin | the task (required) |
| `--repo DIR` | target repo (default: cwd) |
| `--in-place` | skip worktree isolation; work directly in the repo |
| `--readonly` | read-only tool set (`Read,Grep,Glob,Bash`), implies in-place |
| `--tools "Bash,Read,..."` | override which tool *schemas* are sent (smaller = more reliable on weak models) |
| `--timeout SECS` | wall-clock cap (default 600; there is no `--max-turns` in the CLI) |
| `--name LABEL` | label baked into the branch/run name |

Artifacts per run (worktree mode): `<repo>/.git/localagent/meta/<branch>/`
(`changes.diff`, `transcript.json`, `stderr.log`). Worktree lives in
`<repo>/.git/localagent/wt/<branch>/`.

## How Opus orchestrates this

1. Decompose the task; for each cheap/parallelizable piece call `localagent run`.
2. Read the printed `result`, `changes.diff`, and (if needed) `transcript.json`.
3. **Validate**: correctness, scope, no stray edits. Re-run with a sharper task if wrong.
4. `localagent apply` the good ones; `localagent discard` the rest.

The machine-readable footer line `::localagent:: repo=… branch=… diff=… transcript=… files=… exit=…`
is emitted on stdout for the orchestrator to parse.

## Config (env)
| var | default | meaning |
|-----|---------|---------|
| `LLAMA_BASE` | `http://127.0.0.1:8089` | upstream llama-server |
| `LLAMA_PROXY_PORT` | `8090` | proxy listen port |
| `LLAMA_MODEL` | auto-detect | force a specific upstream model id |
| `LOCAL_MODEL_ALIAS` | `local-llama` | extra name that routes local (also: any name starting `local`) |
| `ANTHROPIC_PASSTHROUGH_BASE` | `https://api.anthropic.com` | where non-local models (Opus) are relayed |
| `LLAMA_DISABLE_THINKING` | `1` | `0` keeps the model's reasoning_content |
| `LLAMA_PROXY_LOG` | `~/.local/state/localagent/proxy.log` | proxy debug log |

## Choosing a local model — findings

Tested gemma-4-26B-A4B vs Qwen3-Coder-30B-A3B as the agent brain, same harness/task/context:

| | gemma-4-26B-A4B | Qwen3-Coder-30B-A3B |
|---|---|---|
| tool-call format | clean (llama.cpp parses it) | **leaks as text** (needs the salvage above) |
| open-ended discovery (grep→read→synthesize) | ✅ completes, accurate, exact citations | ❌ flails, gives up, hallucinates |
| best at | the agentic loop here | raw coding when *handed* exact files |

**Use gemma-4-26B as the local brain** (`highllama -m gemma`). Despite being the "coder"
model, Qwen3-Coder underperforms in this agentic harness. **Context size was a red herring**
— 64k vs 128k made no difference; the model was the deciding factor. 64k is faster on gemma
(~47 t/s) and plenty.

Lessons that shaped the agents' prompts:
- Weak models **loop** with no turn cap → the CLI mode bounds them with `--timeout`; native
  agents rely on a tool-call budget in the prompt.
- They **corrupt unfamiliar names** (it "corrected" the dir `pantacor`→`pantavisor` and failed)
  → give exact paths.
- When tools fail they **hallucinate** rather than stop → **always validate**: check cited
  files exist and contain the claim. A wrong run here literally admitted it had no evidence.

## opencode integration

opencode talks to llama-server **directly** via its `llamacpp` provider (OpenAI-native) — no
proxy, so the salvage/thinking-disable above do **not** apply there (another reason to use
gemma in opencode). Per-subagent models are native to opencode, so no gateway is needed.

- `~/.config/opencode/config.json` — the `llamacpp` provider has a **stable alias model**
  `local-llama` (llama-server serves whatever GGUF is loaded regardless of the id sent), so
  it survives model swaps.
- `~/.config/opencode/agents/local-llama.md` — `mode: subagent`, `model: llamacpp/local-llama`,
  same worker prompt. Your primary opencode agent delegates to it; it runs on the local model
  at zero cloud cost and you validate the result.

```bash
opencode run -m llamacpp/local-llama "..."     # run the primary loop on the local model
# inside opencode, ask the primary agent to "use the local-llama subagent" to delegate
```

## Caveats
- The local model is far weaker than Opus. Keep tasks **small and concrete**, constrain
  `--tools`, and **always validate**. Treat sub-agent commits as drafts.
- `--bare` forces `ANTHROPIC_API_KEY` auth (never reads your real OAuth/keychain) and
  skips hooks/plugins/CLAUDE.md — the nested instance is fully isolated.
- No `--max-turns`; runaway loops are bounded by `--timeout`.
