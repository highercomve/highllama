#!/usr/bin/env bash
#
# run_opencode_go.sh — score several OpenCode Go models on the private benchmark.
#
# OpenCode Go is a remote API (https://opencode.ai/docs/go/), so this does NOT
# touch your local llama-server at all — safe to run alongside a local benchmark.
# It just calls run_code_benchmark.py once per model id.
#
# Needs an OpenCode Zen API key (https://opencode.ai/auth), provided EITHER via
# the "opencode-go" entry in providers.json OR $OPENCODE_API_KEY.
#
# Usage:
#   ./run_opencode_go.sh kimi-k2.7 glm-5.1 deepseek-v4-pro qwen3.7-max
#   ./run_opencode_go.sh           # default set if none given
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3}"
MODELS="${MODELS:-$*}"
: "${MODELS:=kimi-k2.7 glm-5.1 deepseek-v4-pro qwen3.7-max}"

if [ -z "${OPENCODE_API_KEY:-}" ] && ! grep -q '"opencode-go"' providers.json 2>/dev/null; then
    echo "!! no OpenCode Go key: add it to providers.json or set OPENCODE_API_KEY (https://opencode.ai/auth)"
    exit 1
fi

for model in $MODELS; do
    echo "================ OpenCode Go: $model ================"
    "$PY" run_code_benchmark.py --opencode-go "$model" || echo "!! $model failed, continuing"
done

echo ""
echo "================ leaderboard (local + OpenCode Go) ================"
"$PY" score_compare.py
