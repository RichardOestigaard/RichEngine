#!/usr/bin/env python3
"""Turn a duoattention_probe run into a per-head table + headMask.

Reads the probe's ``--out`` JSON (exact per-row region fractions at the
recorded ``--settings``) and ``--profiles-out`` .npz (normalized per-token
mass per call), averages each (full-attention layer, KV head)'s mass split
across prompts and calls, and prints the DuoAttention classification:

* ``old``  — fraction of softmax mass on tokens between the sink region
  and the recent window. Retrieval heads keep this high on NIAH prompts.
* ``recent``/``sink`` — the streaming head's mass.
* ``needle`` — fraction inside the passkey span (from the profiles), the
  direct retrieval signal.

Mask rules:
  --threshold F  heads whose mean old fraction >= F keep the full cache.
  --budget K     keep the K heads with the highest mean old fraction.

The emitted ``duoAttention`` object is the shape the family-json consumer
uses: per absolute layer id a list of ``kv_heads`` booleans (true = keep
full history; false = sink + window streaming). See duoattention.md.

usage: duoattention_mask.py RUN.json [PROFILES.npz]
        [--sink 4] [--window 1024] [--threshold 0.05 | --budget 24]
        [--json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def setting_key(sink, window):
    return f"sink{sink}_win{window}"


def load_run(path):
    with open(path) as f:
        return json.load(f)


def region_table(run, sink, window):
    """Mean per-(fa_index, kv_head) region fractions over recorded calls.

    Uses the wrapper's exact per-row fractions when the requested
    (sink, window) pair was recorded at probe time.
    """
    key = setting_key(sink, window)
    table = defaultdict(lambda: np.zeros(3))
    counts = defaultdict(int)
    for call in run["calls"]:
        if key not in call["region_fractions"]:
            continue
        fa, fracs = call["fa_index"], call["region_fractions"][key]
        for h, triple in enumerate(fracs):
            table[(fa, h)] += np.asarray(triple)
            counts[(fa, h)] += 1
    for head in table:
        table[head] /= max(counts[head], 1)
    return table


def needle_fractions(run, profiles_path):
    """Mass inside each prompt's needle span, per (fa_index, kv_head)."""
    if not profiles_path or not Path(profiles_path).exists():
        return {}
    profiles = np.load(profiles_path)
    spans = {p["id"]: p["needle_span"] for p in run["prompts"]}
    acc = defaultdict(list)
    for key in profiles.files:
        # p{prompt}_s{step}_l{layer}_h{head}_c{call}
        parts = key.split("_")
        prompt = int(parts[0][1:])
        layer = int(parts[2][1:])
        head = int(parts[3][1:])
        fa = run["fa_layers"].index(layer) if layer in run["fa_layers"] else layer
        start, end = spans[prompt]
        prof = profiles[key]
        acc[(fa, head)].append(float(prof[start:end].sum()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("profiles", nargs="?", default=None)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--window", type=int, default=1024)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, default=None)
    group.add_argument("--budget", type=int, default=None,
                       help="keep the K heads with most old mass")
    ap.add_argument("--json", action="store_true",
                    help="print only the duoAttention object")
    args = ap.parse_args()

    run = load_run(args.run)
    profiles_path = args.profiles or str(Path(args.run).with_suffix(".npz"))
    if not Path(profiles_path).exists():
        profiles_path = None

    recorded = {tuple(s) for s in run["settings"]}
    exact = (args.sink, args.window) in recorded
    table = region_table(run, args.sink, args.window)
    if not table:
        print(f"no calls recorded at sink={args.sink} window={args.window}; "
              f"recorded settings: {sorted(recorded)}", file=sys.stderr)
        return 64
    needles = needle_fractions(run, profiles_path)
    fa_layers = run["fa_layers"]
    hkv = run["kv_heads"]

    if not args.json:
        print(f"# {run['model']}  sink={args.sink} window={args.window}"
              f"{'  (exact)' if exact else ''}")
        print("# fa layer kv_head  sink    mid/old  recent  needle")
    old = {}
    for fa, layer in enumerate(fa_layers):
        for h in range(hkv):
            sink_f, mid_f, rec_f = table.get((fa, h), np.zeros(3))
            n_frac = needles.get((fa, h), float("nan"))
            old[(fa, h)] = mid_f
            if not args.json:
                print(f"  {fa:3d} {layer:5d} {h:7d}  "
                      f"{sink_f:.4f}  {mid_f:6.4f}  {rec_f:.4f}  "
                      f"{'' if np.isnan(n_frac) else f'{n_frac:.4f}'}")

    if args.budget is not None:
        keep = {h for h, _ in sorted(old.items(), key=lambda kv: -kv[1])[: args.budget]}
    else:
        threshold = 0.05 if args.threshold is None else args.threshold
        keep = {head for head, frac in old.items() if frac >= threshold}

    mask_by_layer = {
        str(layer): [bool((fa, h) in keep) for h in range(hkv)]
        for fa, layer in enumerate(fa_layers)
    }
    duo = {
        "duoAttention": {
            "sink": args.sink,
            "window": args.window,
            "retrievalHeads": [
                [fa_layers[fa], h] for (fa, h) in sorted(keep, key=lambda x: (x[0], x[1]))
            ],
            "headMask": mask_by_layer,
            "faLayerOrder": fa_layers,
            "kvHeads": hkv,
            "source": {"run": args.run, "model": run["model"]},
        }
    }
    if not args.json:
        kept = sum(sum(v) for v in mask_by_layer.values())
        print(f"# keep-full {kept}/{len(fa_layers) * hkv} heads")
        print(json.dumps(duo))
    else:
        print(json.dumps(duo, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
