#!/usr/bin/env bash
#
# code_benchmark/bench.sh — run the code benchmark (single-shot + agentic) for one model.
#
# For local highllama models it starts/stops the server. For OpenCode Go models
# it skips server management and hits the remote endpoint directly.
#
# Usage:
#   ./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent opencode,pi
#   ./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent pi --tasks py_,go_
#   ./code_benchmark/bench.sh --model opencode-go/kimi-k2.7-code --agent opencode
#   ./code_benchmark/bench.sh --model gemma-4-26B-A4B-it-QAT-Q4_0 --agent all --serve
#
# Provider prefixes in --model are auto-detected: opencode-go/ forces remote mode,
# llamacpp/ is stripped for local serving.
#
# After it finishes:  python3 code_benchmark/score_compare.py
set -euo pipefail
cd "$(dirname "$0")"

HL="${HL:-../highllama}"          # path to the highllama launcher
BASE="${BASE:-http://localhost:8089}"
CTX="${CTX:-32k}"
PY="${PY:-python3}"

MODEL=""
AGENTS=""
REMOTE=""                         # empty = local; "opencode-go" = remote OpenCode Go
TASKS=""
LANGS=""
DIFFICULTY=""
SERVE=0

usage() {
    echo "usage: $0 --model <name> [--agent <opencode,pi|all>] [--remote opencode-go]"
    echo "       [--ctx <size>] [--tasks <prefixes>] [--langs <list>] [--difficulty <list>] [--serve]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL="$2"; shift 2 ;;
        --agent)      AGENTS="$2"; shift 2 ;;
        --remote)     REMOTE="$2"; shift 2 ;;
        --ctx)        CTX="$2"; shift 2 ;;
        --tasks)      TASKS="$2"; shift 2 ;;
        --langs)      LANGS="$2"; shift 2 ;;
        --difficulty) DIFFICULTY="$2"; shift 2 ;;
        --serve)      SERVE=1; shift ;;
        -h|--help)    usage ;;
        *) echo "!! unknown argument: $1"; usage ;;
    esac
done

[[ -n "$MODEL" ]] || usage

# Auto-detect provider prefixes in the model name.
if [[ -z "$REMOTE" ]]; then
    if [[ "$MODEL" == opencode-go/* ]]; then
        REMOTE="opencode-go"
        MODEL="${MODEL#opencode-go/}"
    elif [[ "$MODEL" == llamacpp/* ]]; then
        MODEL="${MODEL#llamacpp/}"
    fi
fi

# Sanitize model name for filenames (log path, etc.)
LOG_NAME="${MODEL//\//_}"

# Normalize agents list
if [[ "$AGENTS" == "all" ]]; then
    AGENTS="opencode,pi"
fi
IFS=',' read -ra AGENT_LIST <<<"$AGENTS"

# Build common filter flags
FILTERS=()
[[ -n "$TASKS" ]]      && FILTERS+=(--tasks "$TASKS")
[[ -n "$LANGS" ]]      && FILTERS+=(--langs "$LANGS")
[[ -n "$DIFFICULTY" ]] && FILTERS+=(--difficulty "$DIFFICULTY")

wait_ready() { # poll /v1/models until the chat model answers, or fail after ~5 min
    for _ in $(seq 1 300); do
        curl -s --max-time 3 "$BASE/v1/models" 2>/dev/null | grep -q '"id"' && return 0
        sleep 1
    done
    echo "!! server did not become ready" >&2
    return 1
}

start_server() {
    echo "================ serving $MODEL ================"
    "$HL" stop >/dev/null 2>&1 || true
    pkill -x llama-server 2>/dev/null || true
    sleep 1
    nohup "$HL" -m "$MODEL" -c "$CTX" --no-embeddings >"/tmp/codebench-$LOG_NAME.log" 2>&1 &
    wait_ready || { tail -n 20 "/tmp/codebench-$LOG_NAME.log"; exit 1; }
}

stop_server() {
    "$HL" stop >/dev/null 2>&1 || true
}

# Run single-shot benchmark
run_single_shot() {
    echo ""
    echo "================ single-shot code benchmark ================"
    if [[ "$REMOTE" == "opencode-go" ]]; then
        "$PY" run_code_benchmark.py --opencode-go "$MODEL" "${FILTERS[@]}"
    else
        "$PY" run_code_benchmark.py --base "$BASE" "${FILTERS[@]}"
    fi
}

# Run agentic benchmark for one agent
run_agentic() {
    local agent="$1"
    echo ""
    echo "================ agentic code benchmark ($agent) ================"

    local model_arg="$MODEL"
    if [[ "$REMOTE" == "opencode-go" && "$agent" == "pi" ]]; then
        # pi needs the provider prefix for remote OpenCode Go models
        model_arg="opencode-go/$MODEL"
    elif [[ -z "$REMOTE" && "$agent" == "opencode" ]]; then
        # opencode exposes local models as the 'llamacpp' provider
        model_arg="llamacpp/$MODEL"
    fi

    "$PY" run_agentic_benchmark.py --agent "$agent" --model "$model_arg" "${FILTERS[@]}"
}

# Main flow
if [[ -z "$REMOTE" ]]; then
    start_server
fi

run_single_shot

for agent in "${AGENT_LIST[@]}"; do
    [[ -n "$agent" ]] || continue
    run_agentic "$agent"
done

if [[ -z "$REMOTE" ]]; then
    stop_server
fi

echo ""
echo "================ leaderboard ================"
if [[ "$SERVE" -eq 1 ]]; then
    "$PY" score_compare.py --serve
else
    "$PY" score_compare.py
fi
