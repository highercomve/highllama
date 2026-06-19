#!/usr/bin/env python3
"""
Estimate a starting --n-cpu-moe for a GGUF model on a given free-VRAM budget.

Reads only the GGUF header (stops before the big tokenizer arrays, so it's
instant even on 20GB files). SWA-aware KV sizing so sliding-window models
(e.g. Gemma) aren't massively over-offloaded.

Usage:  gguf-estimate.py <model.gguf> <free_mib> <ctx_tokens> <parallel> <kv_bits>
Prints: "<mode> <value> <block_count>"
        mode=moe   value=n-cpu-moe (MoE expert layers to put on CPU; 0 = full GPU)
        mode=dense value=n-gpu-layers (layers to keep on GPU; offload knob is -ngl)
On any failure prints nothing and exits non-zero (caller falls back to buckets).
"""

import math
import os
import struct
import sys


def main():
    path = sys.argv[1]
    free_mib = float(sys.argv[2])
    ctx = int(sys.argv[3])
    parallel = int(sys.argv[4])
    kv_bits = float(sys.argv[5])

    want = {
        "block_count",
        "attention.head_count_kv",
        "attention.key_length",
        "attention.value_length",
        "attention.sliding_window",
        "attention.sliding_window_pattern",
        "attention.key_length_swa",
        "attention.value_length_swa",
        "expert_count",
    }
    md = {}
    with open(path, "rb") as f:

        def rd(n):
            return f.read(n)

        def u32():
            return struct.unpack("<I", rd(4))[0]

        def u64():
            return struct.unpack("<Q", rd(8))[0]

        def st():
            n = u64()
            return rd(n).decode("utf-8", "replace")

        def val(t):
            if t == 0:
                return rd(1)[0]
            if t == 1:
                return struct.unpack("<b", rd(1))[0]
            if t == 2:
                return struct.unpack("<H", rd(2))[0]
            if t == 3:
                return struct.unpack("<h", rd(2))[0]
            if t == 4:
                return u32()
            if t == 5:
                return struct.unpack("<i", rd(4))[0]
            if t == 6:
                return struct.unpack("<f", rd(4))[0]
            if t == 7:
                return rd(1)[0]
            if t == 8:
                return st()
            if t == 10:
                return u64()
            if t == 11:
                return struct.unpack("<q", rd(8))[0]
            if t == 12:
                return struct.unpack("<d", rd(8))[0]
            if t == 9:
                et = u32()
                n = u64()
                return [val(et) for _ in range(n)]
            raise ValueError("bad type %d" % t)

        if rd(4) != b"GGUF":
            raise ValueError("not a gguf")
        u32()
        u64()  # version, tensor_count
        nkv = u64()
        for _ in range(nkv):
            k = st()
            t = u32()
            v = val(t)
            for w in want:
                if k.endswith(w):
                    md[w] = v
            # all arch metadata precedes the big tokenizer arrays -> stop there
            if k.startswith("tokenizer."):
                break

    L = int(md["block_count"])
    dk = int(md["attention.key_length"])
    dv = int(md["attention.value_length"])
    hck = md["attention.head_count_kv"]
    heads = hck if isinstance(hck, list) else [int(hck)] * L
    if len(heads) < L:  # scalar-ish, pad
        heads = (heads + [heads[-1]] * L)[:L]
    win = int(md.get("attention.sliding_window", 0) or 0)
    dks = int(md.get("attention.key_length_swa", dk))
    dvs = int(md.get("attention.value_length_swa", dv))
    pattern = md.get("attention.sliding_window_pattern")  # per-layer: 1=SWA, 0=global

    # Per-layer KV: SWA layers cap at the window (and may use smaller dims);
    # global layers store the full context. Only trust SWA when we know exactly
    # which layers are windowed (pattern array); otherwise assume full attention
    # (safe over-estimate).
    def layer_swa(i):
        if not (win and win < ctx):
            return False
        if isinstance(pattern, list) and i < len(pattern):
            return bool(pattern[i])
        return False

    kv_elems = 0
    for i in range(L):
        h = heads[i]
        if layer_swa(i):
            kv_elems += h * (dks + dvs) * min(ctx, win)
        else:
            kv_elems += h * (dk + dv) * ctx
    kv_mib = kv_elems * (kv_bits / 8.0) / (1024 * 1024)

    weights_mib = os.path.getsize(path) / (1024 * 1024)
    overhead = 650 + 130 * parallel + (ctx / 1024.0) * 8.0  # cuda + compute buffers
    reserve = kv_mib + overhead + 350  # + small safety
    gpu_budget = free_mib - reserve

    if gpu_budget >= weights_mib:
        print(f"moe 0 {L}")  # fits fully; -ngl 99 works for dense or MoE
        return

    experts = int(md.get("expert_count", 0) or 0)
    if experts > 1:
        expert_frac = 0.90  # experts dominate MoE weight
        per_layer = expert_frac * weights_mib / L
        need = weights_mib - gpu_budget
        ncmoe = max(0, min(L, math.ceil(need / per_layer)))
        print(f"moe {ncmoe} {L}")
    else:
        # dense model: --n-cpu-moe is a no-op, offload via -ngl instead.
        # +1 layer accounts for the non-repeating output/embedding tensors.
        per_layer = weights_mib / (L + 1)
        ngl = max(0, min(L, int(gpu_budget / per_layer)))
        print(f"dense {ngl} {L}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(str(e), file=sys.stderr)
        sys.exit(1)
