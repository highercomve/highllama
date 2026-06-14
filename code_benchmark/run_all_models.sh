#!/usr/bin/env bash
#
# run_all_models.sh — score several models on the private code benchmark by
# swapping the highllama-served model between runs.
#
# This is the ONLY part of the benchmark that starts/stops the server. Do NOT
# run it while another benchmark is using the server. The core runner
# (run_code_benchmark.py) is server-agnostic and just hits whatever is live.
#
# Usage:
#   ./run_all_models.sh gemma-4-12B-it-qat-UD-Q4_K_XL gpt-oss-20b Qwen3.5-9B
#   MODELS="gemma-4-12B-it-qat-UD-Q4_K_XL gpt-oss-20b" CTX=32k ./run_all_models.sh
#
# After it finishes:  python3 score_compare.py
set -euo pipefail
cd "$(dirname "$0")"

HL="${HL:-../highllama}"          # path to the highllama launcher
BASE="${BASE:-http://localhost:8089}"
CTX="${CTX:-32k}"
PY="${PY:-python3}"
MODELS="${MODELS:-$*}"

[ -n "$MODELS" ] || { echo "usage: $0 <model> [model ...]   (or MODELS=...)"; exit 1; }

wait_ready() { # poll /v1/models until the chat model answers, or fail after ~5 min
    for _ in $(seq 1 300); do
        curl -s --max-time 3 "$BASE/v1/models" 2>/dev/null | grep -q '"id"' && return 0
        sleep 1
    done
    echo "!! server did not become ready" >&2
    return 1
}

for model in $MODELS; do
    echo "================ serving $model ================"
    "$HL" stop >/dev/null 2>&1 || true
    pkill -x llama-server 2>/dev/null || true
    sleep 1
    nohup "$HL" -m "$model" -c "$CTX" --no-embeddings >"/tmp/codebench-$model.log" 2>&1 &
    wait_ready || { tail -n 20 "/tmp/codebench-$model.log"; exit 1; }
    "$PY" run_code_benchmark.py --base "$BASE"
done

"$HL" stop >/dev/null 2>&1 || true
echo ""
echo "================ leaderboard ================"
"$PY" score_compare.py
