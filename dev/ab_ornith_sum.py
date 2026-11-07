#!/usr/bin/env python3
"""Summarize build/ab-ornith/*.json A/B runs.

Decode runs: median aggregate_decode_wall_tokens_per_second per width,
median draft_acceptance_rate, token hashes (parity check vs base),
median per-step decode_gpu_ms.
Prefill (short) runs: median ttft_ms / prefill_gpu_ms per prompt_tokens.
"""
import json
import statistics
import sys
from pathlib import Path

OUT = Path("build/ab-ornith")


def load(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def decode_rows(path):
    d = load(path)
    if not d:
        return None
    samples = d.get("decode_throughput", {}).get("samples", [])
    if not samples:
        return None
    by_w = {}
    for s in samples:
        by_w.setdefault(s["width"], []).append(s)
    row = {"name": path.stem}
    for w in sorted(by_w):
        ss = by_w[w]
        row[f"B{w}_tok_s"] = statistics.median(
            x["aggregate_decode_wall_tokens_per_second"] for x in ss)
        row[f"B{w}_acc"] = statistics.median(
            x["draft_acceptance_rate"] for x in ss)
        row[f"B{w}_hash"] = ss[0]["output_token_hash"]
        row[f"B{w}_step_ms"] = statistics.median(
            x["decode_gpu_ms"] / x["decode_batches"] for x in ss)
    return row


def prefill_rows(path):
    d = load(path)
    if not d:
        return None
    ms = [m for m in d.get("measurements", []) if m.get("scenario") == "short"]
    if not ms:
        return None
    groups = {}
    for m in ms:
        groups.setdefault(m.get("prompt_tokens"), []).append(m)
    row = {"name": path.stem}
    for k in sorted(groups, key=lambda x: (x is None, x)):
        g = groups[k]
        row[f"p{k}_ttft"] = statistics.median(x["ttft_ms"] for x in g)
        row[f"p{k}_gpu"] = statistics.median(x["prefill_gpu_ms"] for x in g)
    return row


def fmt(v):
    if isinstance(v, float):
        return f"{v:>10.4g}"
    return f"{str(v):>10}"


def main():
    decode, prefill, bad = [], [], []
    for p in sorted(OUT.glob("*.json")):
        r = decode_rows(p)
        if r:
            decode.append(r)
            continue
        r = prefill_rows(p)
        if r:
            prefill.append(r)
        else:
            bad.append(p.name)
    if decode:
        keys = [k for k in decode[0] if k != "name"]
        print("=== DECODE: wall tok/s | acceptance | hash | step gpu ms ===")
        print(f"{'run':<17}" + "".join(f"{k:>22}" for k in keys))
        for r in decode:
            print(f"{r['name']:<17}" + "".join(fmt(r.get(k)) for k in keys))
    if prefill:
        keys = [k for k in prefill[0] if k != "name"]
        print("\n=== PREFILL short: ttft ms | prefill gpu ms ===")
        print(f"{'run':<17}" + "".join(f"{k:>22}" for k in keys))
        for r in prefill:
            print(f"{r['name']:<17}" + "".join(fmt(r.get(k)) for k in keys))
    if bad:
        print("\n# unparsed:", ", ".join(bad), file=sys.stderr)


if __name__ == "__main__":
    main()
