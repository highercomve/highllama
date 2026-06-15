#!/usr/bin/env python3
"""
Benchmark llama.cpp/highllama token-generation speed across context windows,
and (when run against an MTP server) capture draft-acceptance so MTP-on vs
MTP-off runs can be compared with bench_compare.py.

Per context size it sends a needle-in-a-haystack prompt, generates a fixed batch
of tokens at temperature 0, and records prefill/decode tok/s, draft acceptance,
needle-retrieval success, and the exact generated text (for the lossless-MTP
identity check). Runs are repeated and aggregated by median.

Usage:
    python run_benchmark.py --label mtp   --out results_mtp.json
    python run_benchmark.py --label nomtp --out results_nomtp.json
    python run_benchmark.py --sizes 512,2048,8192,32768,65536 --gen 160 --repeats 3
"""
import argparse
import datetime
import json
import os
import re
import statistics
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = "http://localhost:8089"
DATA_DIR = os.path.join(HERE, "benchmark_data")
NEEDLE_KEY = "key-256k-alpha"  # matched case-insensitively in the answer
KEY_RE = re.compile(r"key-256k-alpha", re.IGNORECASE)
QUESTION = "\n\nWhat was the unique verification key mentioned in the documentation? Answer with only the key."


def detect_model(base):
    """Pick the chat model from /v1/models (skip embedding models)."""
    ids = [m["id"] for m in requests.get(f"{base}/v1/models", timeout=10).json()["data"]]
    embed_pat = ("embed", "bge", "nomic", "gte", "e5-", "minilm", "arctic")
    chat = [i for i in ids if not any(p in i.lower() for p in embed_pat)]
    if not chat:
        sys.exit(f"!! no chat model found in {ids}")
    return chat[0]


def needle_prompts(sizes):
    """Map requested sizes -> available benchmark_data/needle_*.txt files.

    Returns sorted [(approx_size, path)]. If --sizes is given, only those files
    are used; otherwise every needle_*.txt found is swept.
    """
    if not os.path.isdir(DATA_DIR):
        sys.exit(f"!! {DATA_DIR}/ not found — run generate_benchmark_data.py first")
    found = {}
    for f in os.listdir(DATA_DIR):
        if f.startswith("needle_") and f.endswith("_tokens.txt"):
            try:
                found[int(f.split("_")[1])] = os.path.join(DATA_DIR, f)
            except ValueError:
                pass
    if sizes:
        missing = [s for s in sizes if s not in found]
        if missing:
            sys.exit(f"!! no needle file for sizes {missing}; generate them first "
                     f"(have {sorted(found)})")
        found = {s: found[s] for s in sizes}
    return sorted(found.items())


def one_request(base, model, prompt, gen):
    """Single completions call at temp 0; returns metrics + the generated text."""
    payload = {
        "model": model,
        "prompt": prompt + QUESTION,
        "n_predict": gen,
        "temperature": 0,
        "stream": False,
        "cache_prompt": False,  # measure real prefill, not a KV restore
    }
    t0 = time.time()
    r = requests.post(f"{base}/v1/completions", json=payload, timeout=900)
    r.raise_for_status()
    d = r.json()
    wall = time.time() - t0
    text = (d["choices"][0].get("text") if "choices" in d else d.get("text")) or ""
    t = d.get("timings", {}) or {}
    u = d.get("usage", {}) or {}
    draft_n = t.get("draft_n", 0) or 0
    gen_tokens = t.get("predicted_n", u.get("completion_tokens", 0)) or 0
    raw_dtps = t.get("predicted_per_second", 0.0) or 0.0
    # llama.cpp returns predicted_per_second = 1e6 when ~nothing was generated
    # (e.g. an immediate stop token). Treat a too-short generation or an absurd
    # rate as an invalid speed sample (None) so it can't pollute the medians.
    decode_tps = round(raw_dtps, 2) if (gen_tokens >= 2 and raw_dtps < 5000) else None
    m = KEY_RE.search(text)
    return {
        "prompt_tokens": u.get("prompt_tokens", t.get("prompt_n", 0)),
        "gen_tokens": gen_tokens,
        "prefill_tps": round(t.get("prompt_per_second", 0.0) or 0.0, 2),
        "decode_tps": decode_tps,
        "draft_n": draft_n,
        "draft_accepted": t.get("draft_n_accepted", 0) or 0,
        "accept_rate": round((t.get("draft_n_accepted", 0) or 0) / draft_n, 4) if draft_n else None,
        "needle_ok": m is not None,
        "answer_key": m.group(0).upper() if m else None,  # the meaningful answer
        "wall_s": round(wall, 2),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--model", default=None, help="model id (auto-detected if omitted)")
    ap.add_argument("--label", default="run", help="config label, e.g. mtp / nomtp")
    ap.add_argument("--out", default=None, help="output json (default results/YYYY-MM-DD/HH-MM-SS/results_<label>.json)")
    ap.add_argument("--sizes", default=None, help="comma-separated token sizes to sweep")
    ap.add_argument("--gen", type=int, default=160, help="tokens to generate per call")
    ap.add_argument("--repeats", type=int, default=3, help="runs per size (median reported)")
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")] if args.sizes else None
    if args.out:
        out = args.out
    else:
        now = datetime.datetime.now()
        out = os.path.join(HERE, "results", now.strftime("%Y-%m-%d"),
                           now.strftime("%H-%M-%S"), f"results_{args.label}.json")
    model = args.model or detect_model(args.base)
    cases = needle_prompts(sizes)

    print(f">> base={args.base}  model={model}  label={args.label}")
    print(f">> sweeping {len(cases)} sizes x {args.repeats} repeats, gen={args.gen} tok\n")

    per_size = []
    mtp_seen = False
    for approx, path in cases:
        with open(path) as f:
            prompt = f.read()
        runs = []
        for i in range(args.repeats):
            try:
                m = one_request(args.base, model, prompt, args.gen)
            except requests.HTTPError as e:
                print(f"  {approx:>7}: HTTP error {e} — {e.response.text[:160]}")
                break
            runs.append(m)
            if m["draft_n"]:
                mtp_seen = True
            acc = ("%.0f%%" % (m["accept_rate"] * 100)) if m["accept_rate"] is not None else "  -- "
            dtps = f"{m['decode_tps']:>6.1f}" if m["decode_tps"] is not None else "  none"
            print(f"  {approx:>7} tok  run{i+1}: prefill {m['prefill_tps']:>7.1f} t/s | "
                  f"decode {dtps} t/s | accept {acc} | "
                  f"needle {'OK' if m['needle_ok'] else 'MISS'}")
        if not runs:
            continue
        decs = [r["decode_tps"] for r in runs if r["decode_tps"] is not None]
        accs = [r["accept_rate"] for r in runs if r["accept_rate"] is not None]
        keys = [r["answer_key"] for r in runs if r["answer_key"]]
        per_size.append({
            "size": approx,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "prefill_tps_med": round(statistics.median(r["prefill_tps"] for r in runs), 2),
            "decode_tps_med": round(statistics.median(decs), 2) if decs else None,
            "accept_rate_med": round(statistics.median(accs), 4) if accs else None,
            "needle_success": sum(r["needle_ok"] for r in runs) / len(runs),
            "answer_key": keys[0] if keys else None,  # the meaningful answer (KEY-256K-ALPHA)
            "answer_text": runs[0]["text"],           # full text; temp 0 -> deterministic
            "runs": runs,
        })
        print()

    result = {
        "label": args.label,
        "model": model,
        "base": args.base,
        "gen_tokens": args.gen,
        "repeats": args.repeats,
        "mtp_active": mtp_seen,
        "sizes": per_size,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f">> wrote {out}  (mtp_active={mtp_seen})")


if __name__ == "__main__":
    main()
