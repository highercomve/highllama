#!/usr/bin/env python3
"""
Leaderboard across one or more code-benchmark result files.

Reads results/*.json (or the paths you pass) and prints a model ranking by
pass@1, with the partial-credit test-pass% alongside and a per-language pass@1
matrix so you can see where each model is strong/weak.

Usage:
    python score_compare.py                       # everything in results/
    python score_compare.py results/a.json results/b.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(json.load(f))
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    if not paths:
        sys.exit("!! no result files (run run_code_benchmark.py first)")
    blobs = load(paths)
    blobs.sort(key=lambda b: b["summary"]["pass_at_1_pct"], reverse=True)

    langs = sorted({l for b in blobs for l in b["summary"]["by_language"]})

    name_w = max(len(b["model"]) for b in blobs)
    name_w = max(name_w, 5)
    hdr = (f"{'model':<{name_w}} | {'pass@1':>7} | {'test%':>6} | {'tasks':>5} | "
           f"{'s/task':>6} | {'tok/s':>6}")
    for l in langs:
        hdr += f" | {l[:6]:>6}"
    print(hdr)
    print("-" * len(hdr))

    for b in blobs:
        s = b["summary"]
        tps = s.get("avg_tok_s")
        row = (f"{b['model']:<{name_w}} | {s['pass_at_1_pct']:>6}% | "
               f"{s['test_pass_pct']:>5}% | {s['tasks']:>5} | "
               f"{s.get('avg_gen_s', 0):>6} | {tps if tps else '-':>6}")
        for l in langs:
            cell = s["by_language"].get(l)
            row += f" | {str(cell['pass_at_1_pct'])+'%' if cell else '-':>6}"
        print(row)

    print("\n(pass@1 = task fully correct; test% = fraction of hidden checks passed; "
          "s/task = avg model latency)")


if __name__ == "__main__":
    main()
