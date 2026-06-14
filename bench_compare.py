#!/usr/bin/env python3
"""
Compare two run_benchmark.py result files (MTP-on vs MTP-off) and report, per
context window:

  - decode tok/s for each, and the MTP speedup factor
  - draft acceptance (MTP run)
  - needle-retrieval accuracy for each (the task-accuracy signal)
  - answer match: does the extracted key (KEY-256K-ALPHA) agree between configs
  - greedy token agreement %: how much of the raw generated text is byte-identical

On MTP accuracy: speculative decoding is distribution-preserving, so task
accuracy should be unchanged. Exact greedy token sequences can still diverge
slightly — batched draft-verification computes the target logits in parallel,
and floating-point non-associativity occasionally flips a near-tie argmax vs
sequential decoding. That is expected and *not* a quality regression, so this
tool only flags a problem (⚠) when the needle accuracy or the answer key
actually differ — not when the surrounding text diverges.

Usage:
    python bench_compare.py results_mtp.json results_nomtp.json
"""
import json
import math
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def pick(a, b):
    """Return (mtp_run, base_run) regardless of argument order."""
    if a.get("mtp_active") and not b.get("mtp_active"):
        return a, b
    if b.get("mtp_active") and not a.get("mtp_active"):
        return b, a
    return (a, b) if "mtp" in a.get("label", "").lower() else (b, a)


def token_agreement(a, b):
    """Fraction of the shorter string that matches byte-for-byte from the start."""
    if not a and not b:
        return 1.0
    n = min(len(a), len(b)) or 1
    same = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    return same / max(len(a), len(b), 1)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python bench_compare.py <results_A.json> <results_B.json>")
    mtp, base = pick(load(sys.argv[1]), load(sys.argv[2]))
    bm = {s["size"]: s for s in base["sizes"]}

    print(f"model: {mtp['model']}")
    print(f"MTP run='{mtp['label']}' (gen={mtp['gen_tokens']}, repeats={mtp['repeats']})  "
          f"vs base='{base['label']}'\n")

    hdr = (f"{'ctx':>8} | {'decode no-MTP':>13} | {'decode MTP':>11} | {'speedup':>7} | "
           f"{'accept':>6} | {'needle b/m':>11} | {'answer':>9} | {'tok-agree':>9}")
    print(hdr)
    print("-" * len(hdr))

    speedups, accuracy_ok = [], True
    for s in mtp["sizes"]:
        sz = s["size"]
        b = bm.get(sz)
        if not b:
            continue
        dm, db = s["decode_tps_med"], b["decode_tps_med"]
        if dm and db:
            spd = dm / db
            speedups.append(spd)
            spd_s = f"{spd:.2f}x"
        else:
            spd_s = "n/a"
        acc = f"{s['accept_rate_med']*100:.0f}%" if s.get("accept_rate_med") is not None else "--"
        nb, nm = b["needle_success"], s["needle_success"]
        kb, km = b.get("answer_key"), s.get("answer_key")
        # A genuine MTP regression = MTP retrieves the needle less often, or both
        # produce a key but they conflict. MTP doing *better* than the baseline
        # (base miss, MTP hit) is not a regression — at borderline depths the FP
        # divergence just flips a single greedy sample either way.
        if (nm < nb) or (kb and km and kb != km):
            accuracy_ok = False
        if kb == km:
            ans = "same"
        elif kb and km:
            ans = "CONFLICT"  # both answered, different keys -> real problem
        else:
            ans = "base-miss" if km else "mtp-miss"  # one found it, one didn't (noise)
        agree = token_agreement(b.get("answer_text", ""), s.get("answer_text", ""))
        dm_s = f"{dm:.1f}" if dm else "none"
        db_s = f"{db:.1f}" if db else "none"
        print(f"{sz:>8} | {db_s:>13} | {dm_s:>11} | {spd_s:>7} | {acc:>6} | "
              f"{f'{nb*100:.0f}%/{nm*100:.0f}%':>11} | {ans:>9} | "
              f"{agree*100:>7.0f}%")

    if speedups:
        gm = math.exp(sum(map(math.log, speedups)) / len(speedups))
        print(f"\nMTP decode speedup: {gm:.2f}x geometric mean over {len(speedups)} valid size(s)")
    print(f"MTP accuracy preserved (needle + answer key unchanged): "
          f"{'YES ✅' if accuracy_ok else 'NO — REGRESSION ⚠'}")
    print("note: <100% tok-agree is expected FP divergence in batched verification, "
          "not a quality loss (see header).")


if __name__ == "__main__":
    main()
