#!/usr/bin/env python3
"""Compare two backend-benchmark transcript runs token-for-token.

Usage: i8_flip_diff.py REF.json CAND.json
Prints, per scenario: first divergence index, count of differing positions,
and the shared-prefix length. A position "flips" when the candidate's token
differs from the reference's; after a flip the streams resynchronize only if
the candidate returns to the reference token sequence.
"""
import json
import sys


def load(path):
    d = json.load(open(path))
    out = {}
    for m in d.get("measurements", []):
        out[m["scenario"]] = m["output_tokens"]
    return out


def main():
    ref = load(sys.argv[1])
    cand = load(sys.argv[2])
    total_flip = total = 0
    for name in sorted(ref):
        r, c = ref[name], cand.get(name, [])
        n = min(len(r), len(c))
        flips = [i for i in range(n) if r[i] != c[i]]
        first = flips[0] if flips else None
        print(f"{name}: len={n} first_diff={first} diffs={len(flips)}")
        if flips:
            print(f"   idx={first}: ref={r[first]} cand={c[first]}")
        total_flip += len(flips)
        total += n
    print(f"TOTAL: {total_flip}/{total} positions differ")


if __name__ == "__main__":
    main()
