#!/usr/bin/env python3
"""Print the ggml tensor types a GGUF uses, one per line. Exits 2 if it cannot be read.

Used to decide which llama.cpp build can load a model: PrismML's ternary types (142 PQ2_0,
143 PTQ1_0) are not upstream, and a filename is only a hint.
"""
import struct
import sys


def main() -> int:
    path = sys.argv[1]
    with open(path, "rb") as f:
        def rd(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError("short read")
            return b

        u32 = lambda: struct.unpack("<I", rd(4))[0]
        u64 = lambda: struct.unpack("<Q", rd(8))[0]

        def st():
            return rd(u64()).decode("utf-8", "replace")

        def skip_val(t):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if t == 8:
                st()
            elif t == 9:
                et = u32()
                n = u64()
                for _ in range(n):
                    skip_val(et)
            elif t in sizes:
                rd(sizes[t])
            else:
                raise ValueError(f"bad type {t}")

        if rd(4) != b"GGUF":
            raise ValueError("not a gguf")
        u32()
        n_tensors = u64()
        n_kv = u64()
        for _ in range(n_kv):
            st()
            skip_val(u32())
        types = set()
        for _ in range(n_tensors):
            st()                      # name
            n_dims = u32()
            rd(8 * n_dims)            # dimensions
            types.add(u32())          # ggml type
            rd(8)                     # offset
    for t in sorted(types):
        print(t)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"gguf: {e}", file=sys.stderr)
        sys.exit(2)
