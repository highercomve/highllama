# speed_benchmark — highllama token-generation speed benchmark

Measures llama.cpp/highllama decode and prefill speed across context windows,
plus needle-in-a-haystack retrieval accuracy and (when MTP is enabled) draft
acceptance. Results are organized by run date so you can compare multiple runs
over time.

## Layout

| file | what |
|---|---|
| `run_benchmark.py` | sweep context sizes and record speed/accuracy metrics |
| `bench_compare.py` | compare two runs (usually MTP-on vs MTP-off) |
| `bench.sh` | orchestrate the full MTP-on vs MTP-off comparison |
| `generate_benchmark_data.py` | create `benchmark_data/needle_<N>_tokens.txt` files |
| `benchmark_data/` | generated needle prompts (gitignored) |
| `results/YYYY-MM-DD/HH-MM-SS/` | one directory per benchmark run (gitignored) |

## Quick start

Make sure a highllama server is **not** already running (`bench.sh` manages its
own server).

```bash
# full MTP comparison: writes results/2026-06-15/14-32-01/results_{nomtp,mtp}.json
./speed_benchmark/bench.sh

# or run just one sweep manually
python3 speed_benchmark/run_benchmark.py --label nomtp
python3 speed_benchmark/run_benchmark.py --label mtp

# compare any two result files
python3 speed_benchmark/bench_compare.py \
    speed_benchmark/results/2026-06-15/14-32-01/results_mtp.json \
    speed_benchmark/results/2026-06-15/14-32-01/results_nomtp.json

# serve an interactive HTML report with charts
python3 speed_benchmark/bench_compare.py \
    speed_benchmark/results/2026-06-15/14-32-01/results_mtp.json \
    speed_benchmark/results/2026-06-15/14-32-01/results_nomtp.json \
    --serve --port 8080
```

`run_benchmark.py` writes to `results/YYYY-MM-DD/HH-MM-SS/results_<label>.json`
by default. Use `--out` to override.

## Useful flags (bench_compare.py)

```
result_a.json result_b.json   two run_benchmark result files
--serve                       start a local web server with charts
--port 8080                   port for the web server (default 8080)
```

The web report loads Chart.js from a CDN, so it needs internet access on the
machine that opens the page.

## Useful flags (run_benchmark.py)

```
--base http://localhost:8089   server endpoint
--model <id>                   model id (auto-detected from /v1/models if omitted)
--label <name>                 run label, e.g. mtp / nomtp / fp16 / q4
--out <path>                   custom output json
--sizes 512,2048,8192          token sizes to sweep
--gen 160                      tokens to generate per size
--repeats 3                    repeats per size (median reported)
```

## Notes

- `bench.sh` launches a fresh single-process server for each config (`--no-embeddings`),
  so MTP-on vs MTP-off numbers are comparable.
- Needle prompts are generated on demand via `generate_benchmark_data.py`. They are
  gitignored and can be regenerated at any time.
- Older result files that were previously at the repository root have been moved to
  `speed_benchmark/results/legacy/`.
