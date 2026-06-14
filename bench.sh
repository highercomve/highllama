#!/usr/bin/env bash
#
# bench.sh — measure gemma-4 token-gen speed vs context window, MTP on vs off.
#
# Launches highllama twice (baseline single-process, then with --mtp), runs the
# context-window sweep against each, and prints a comparison (speedup, draft
# acceptance, needle accuracy, and the temp-0 lossless-MTP output identity check).
#
# Both runs use --no-embeddings so each is a clean single-process server (no
# router), for an apples-to-apples MTP comparison.
#
# Usage:
#   ./bench.sh                                  # defaults below
#   MODEL=gemma-4-12B-it-qat-UD-Q4_K_XL CTX=131072 SIZES=512,2048,8192,32768,65536 ./bench.sh
#   GEN=160 REPEATS=3 ./bench.sh
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-gemma-4-12B-it-qat-UD-Q4_K_XL}"
CTX="${CTX:-131072}" # server context; must exceed the largest sweep size
SIZES="${SIZES:-512,2048,8192,32768,65536}"
GEN="${GEN:-160}"
REPEATS="${REPEATS:-3}"
BASE="${BASE:-http://localhost:8089}"
PY="${PY:-python3}"

wait_ready() { # poll until the chat model answers /v1/models, or fail after ~5 min
    for _ in $(seq 1 300); do
        curl -s --max-time 3 "$BASE/v1/models" 2>/dev/null | grep -q '"id"' && return 0
        sleep 1
    done
    echo "!! server did not become ready" >&2
    return 1
}

run_config() { # $1=label  $2...=extra highllama flags
    local label="$1"
    shift
    echo "================ launching highllama ($label): $* ================"
    ./highllama stop >/dev/null 2>&1 || true
    pkill -x llama-server 2>/dev/null || true
    sleep 1
    nohup ./highllama -m "$MODEL" -c "$CTX" --no-embeddings "$@" >"/tmp/bench-$label.log" 2>&1 &
    wait_ready || {
        tail -n 20 "/tmp/bench-$label.log"
        exit 1
    }
    "$PY" run_benchmark.py --base "$BASE" --label "$label" \
        --out "results_$label.json" --sizes "$SIZES" --gen "$GEN" --repeats "$REPEATS"
}

# ensure the needle datasets for the requested sizes exist
IFS=',' read -ra WANT <<<"$SIZES"
need_gen=0
for s in "${WANT[@]}"; do
    [ -f "benchmark_data/needle_${s}_tokens.txt" ] || need_gen=1
done
if [ "$need_gen" -eq 1 ]; then
    echo ">> generating needle datasets for sizes: $SIZES"
    SIZES="$SIZES" "$PY" generate_benchmark_data.py
fi

run_config nomtp
run_config mtp --mtp

echo ""
echo "================ comparison ================"
"$PY" bench_compare.py results_mtp.json results_nomtp.json

./highllama stop >/dev/null 2>&1 || true
echo ">> done. raw: results_mtp.json results_nomtp.json"
