"""Per-layer KV-quantization sensitivity for Qwen3.8-27B (hybrid GDN+attn).

Prefills ~8K tokens on mlx-community/Qwen3.8-27B-4bit, captures the KV
cache of the 16 full-attention layers, then for EACH attention layer
independently dequantize-in-place ONLY that layer's K and/or V through the
engine's int4 scheme (per-(head,token) absmax symmetric, scale=max/7,
codes [-7,7] — runtime/metal/kernels/common/paged_store_row.h) and runs a
256-token greedy verify pass. Reports argmax token-match vs the bf16
reference and vs the all-int8 reference.

Anchors: all-int4 and all-int8 across all 16 layers. Then a greedy sweep:
rank layers by int4 sensitivity, measure combined "int8 for top-k, int4
for the rest" assignments, and recommend the cheapest table that recovers
most of the all-int4 acceptance tax.

usage: kv_sensitivity_experiment.py [--prompt-tokens 8192]
        [--draft-tokens 256] [--skip-kv-split] [--layers 3,7,...]
        [--out dev/kv_sensitivity_results.json]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

MODEL = "mlx-community/Qwen3.8-27B-4bit"


def scalar_quant(x, bits):
    """Engine KV scheme: per-(token, head) absmax symmetric quant.

    Matches paged_store_row.h: scale = absmax / qmax, rint codes clipped
    to [-qmax-1, qmax] (the -qmax-1 end is unreachable), one fp32 scale
    per (head, token) per tensor. axis=-1 is the head dim; axes are
    [B, heads, T, D] so max over -1 gives [B, heads, T, 1].
    """
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=-1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-8)
    return np.clip(np.round(x / scale), -qmax - 1, qmax) * scale


def attention_layer_indices(model):
    """Model-layer indices of full-attention layers (skip is_linear)."""
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers")
    out = []
    for index, layer in enumerate(layers):
        if getattr(layer, "is_linear", False):
            continue
        if hasattr(layer, "self_attn") or hasattr(layer, "attention"):
            out.append(index)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--draft-tokens", type=int, default=256)
    ap.add_argument("--skip-kv-split", action="store_true",
                    help="skip per-layer int4-K-only / int4-V-only trials")
    ap.add_argument("--combo-ks", default="1,2,4,8",
                    help="top-k int8 layer counts for combined trials")
    ap.add_argument("--layers", default="",
                    help="comma list of model layer indices to sweep "
                         "(default: all full-attention layers)")
    ap.add_argument("--out", default="dev/kv_sensitivity_results.json")
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load(args.model)
    print(f"loaded {args.model} {time.time() - t0:.0f}s", flush=True)

    text = (
        "The quarterly infrastructure report details deployment topology, "
        "cache coherence protocols, and latency budgets for each region. "
    )
    ids = tok.encode(text)
    reps = args.prompt_tokens // len(ids) + 1
    prompt = (ids * reps)[: args.prompt_tokens]

    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    mx.eval(
        logits,
        *[
            s
            for c in cache
            for s in (c.state if isinstance(c.state, (list, tuple)) else [c.state])
            if s is not None
        ],
    )
    print(f"prefilled {args.prompt_tokens} tokens {time.time() - t0:.0f}s",
          flush=True)

    fa_model_layers = attention_layer_indices(model)
    saved = [c.state for c in cache]
    kv_idx = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    print(f"model attn layers {fa_model_layers}", flush=True)
    print(f"cache KVCache idx  {kv_idx}", flush=True)
    assert len(kv_idx) == len(fa_model_layers), (
        f"KVCache count {len(kv_idx)} != attn layers {len(fa_model_layers)}"
    )
    # cache list is layer-ordered, so cache index == model layer index.
    layer_of_cache = {ci: li for ci, li in zip(kv_idx, fa_model_layers)}

    k0 = np.array(
        cache[kv_idx[0]].keys[..., : cache[kv_idx[0]].offset, :].astype(mx.float32)
    )
    n_kv_heads, T, D = k0.shape[1], k0.shape[2], k0.shape[3]
    print(f"KV per layer: {n_kv_heads} heads x {T} x {D}", flush=True)

    # Greedy draft (proxy for a correct draft at temp 0), as in the reference.
    draft = []
    nxt = int(mx.argmax(logits[0, -1]))
    for _ in range(args.draft_tokens):
        draft.append(nxt)
        lg = model(mx.array([[nxt]]), cache=cache)
        nxt = int(mx.argmax(lg[0, -1]))
    print(f"draft {len(draft)} tokens {time.time() - t0:.0f}s", flush=True)

    def restore(spec=None):
        """spec: {cache_idx: (k_bits|None, v_bits|None)}; None -> bf16."""
        spec = spec or {}
        for i, c in enumerate(cache):
            if i in kv_idx:
                keys, values, offset = saved[i]
                k = np.array(keys[..., :offset, :].astype(mx.float32))
                v = np.array(values[..., :offset, :].astype(mx.float32))
                kb, vb = spec.get(i, (None, None))
                if kb:
                    k = scalar_quant(k, kb)
                if vb:
                    v = scalar_quant(v, vb)
                c.state = (
                    mx.array(k).astype(keys.dtype),
                    mx.array(v).astype(values.dtype),
                    offset,
                )
            else:
                c.state = saved[i]

    results = {}
    REFS = {}

    def trial(name, spec=None):
        t1 = time.time()
        restore(spec)
        logits = model(mx.array(draft)[None], cache=cache)
        pred = np.array(mx.argmax(logits[0], axis=-1))[:-1]
        entry = {"spec": spec or {}, "seconds": round(time.time() - t1, 2)}
        if "bf16" in name:
            REFS["bf16"] = pred
            REFS["bf16_logits"] = logits
            m = pred == np.array(draft[1:])
            entry["self_match_draft"] = float(m.mean())
            # continuous baselines vs itself
            entry["logit_l1"] = 0.0
            lsm = mx.log_softmax(logits[0, :-1], axis=-1)
            idx = mx.array(pred)
            logp = lsm[mx.arange(len(pred)), idx]
            entry["mean_p_refargmax"] = float(mx.mean(mx.exp(logp)))
            print(f"{name:34s} self-match {m.mean() * 100:.1f}%", flush=True)
        else:
            if name == "all-int8":
                REFS["int8"] = pred
            # continuous perturbation metrics vs the bf16 pass (cheap, on-GPU)
            ref_lg = REFS["bf16_logits"]
            entry["logit_l1"] = float(
                mx.mean(mx.abs(logits - ref_lg))
            )
            lsm = mx.log_softmax(logits[0, :-1], axis=-1)
            idx = mx.array(REFS["bf16"])
            logp = lsm[mx.arange(len(REFS["bf16"])), idx]
            entry["mean_p_refargmax"] = float(mx.mean(mx.exp(logp)))
            for ref_name, ref in REFS.items():
                if ref_name == "bf16_logits":
                    continue
                match = pred == ref
                entry[f"match_{ref_name}"] = float(match.mean())
                # clean 4-token verify windows fully matching
                w = match[: len(match) // 4 * 4].reshape(-1, 4).all(axis=1)
                entry[f"clean4_{ref_name}"] = float(w.mean())
            line = "  ".join(
                f"vs {r}: {entry[f'match_{r}'] * 100:.1f}%"
                for r in REFS
                if r != "bf16_logits"
            )
            print(
                f"{name:34s} {line}  l1={entry['logit_l1']:.4f} "
                f"p={entry['mean_p_refargmax']:.4f} ({entry['seconds']}s)",
                flush=True,
            )
        results[name] = entry
        return entry

    # ---- anchors ----
    trial("bf16 (ref)")
    trial("all-int8", {i: (8, 8) for i in kv_idx})
    trial("all-int4", {i: (4, 4) for i in kv_idx})

    # ---- per-layer sweep ----
    if args.layers:
        sweep_layers = [int(x) for x in args.layers.split(",")]
    else:
        sweep_layers = fa_model_layers
    cache_of_layer = {li: ci for ci, li in layer_of_cache.items()}
    per_layer = {}
    for li in sweep_layers:
        ci = cache_of_layer[li]
        e = trial(f"L{li} int4-KV", {ci: (4, 4)})
        per_layer[li] = {
            "int4_kv": e["match_bf16"],
            "l1_int4_kv": e["logit_l1"],
            "p_int4_kv": e["mean_p_refargmax"],
        }
        if not args.skip_kv_split:
            e = trial(f"L{li} int4-K-only", {ci: (4, None)})
            per_layer[li]["int4_k_only"] = e["match_bf16"]
            per_layer[li]["l1_k_only"] = e["logit_l1"]
            per_layer[li]["p_k_only"] = e["mean_p_refargmax"]
            e = trial(f"L{li} int4-V-only", {ci: (None, 4)})
            per_layer[li]["int4_v_only"] = e["match_bf16"]
            per_layer[li]["l1_v_only"] = e["logit_l1"]
            per_layer[li]["p_v_only"] = e["mean_p_refargmax"]
        print(f"  -- L{li} done {time.time() - t0:.0f}s", flush=True)

    # ---- greedy combined assignment ----
    # Rank by acceptance-proxy sensitivity: lowest mean_p_refargmax under
    # single-layer int4 first (argmax match is chaotic; p is smooth).
    bf16_p = results["bf16 (ref)"]["mean_p_refargmax"]
    ranked = sorted(per_layer, key=lambda li: per_layer[li]["p_int4_kv"])
    all4 = results["all-int4"]["match_bf16"]
    all8 = results["all-int8"]["match_bf16"]
    all4_p = results["all-int4"]["mean_p_refargmax"]
    all8_p = results["all-int8"]["mean_p_refargmax"]
    all4_l1 = results["all-int4"]["logit_l1"]
    all8_l1 = results["all-int8"]["logit_l1"]
    combo_ks = sorted({int(x) for x in args.combo_ks.split(",")})
    combos = {}
    for k in combo_ks:
        top = set(ranked[:k])
        spec = {
            cache_of_layer[li]: (8, 8) if li in top else (4, 4)
            for li in sweep_layers
        }
        name = f"combo int8-top{k}-int4-rest"
        e = trial(name, spec)
        combos[k] = e

    # Byte model: per token per KV head per tensor.
    # int4: D/2 data + 4B scale ; int8: D + 4B ; bf16: 2*D.
    def bytes_per_layer(kb, vb):
        def one(bits):
            if bits is None:
                return 2 * D
            return D * bits // 8 + 4
        return one(kb) + one(vb)

    byte_all4 = sum(bytes_per_layer(4, 4) for _ in sweep_layers)
    byte_all8 = sum(bytes_per_layer(8, 8) for _ in sweep_layers)
    byte_bf16 = sum(bytes_per_layer(None, None) for _ in sweep_layers)

    table = []
    for li in ranked:
        r = per_layer[li]
        table.append(
            {
                "layer": li,
                "match_int4_kv": r["int4_kv"],
                "drop_vs_bf16": 1.0 - r["int4_kv"],
                "match_int4_k_only": r.get("int4_k_only"),
                "match_int4_v_only": r.get("int4_v_only"),
                "drop_k_only": (1.0 - r["int4_k_only"])
                if "int4_k_only" in r else None,
                "drop_v_only": (1.0 - r["int4_v_only"])
                if "int4_v_only" in r else None,
            }
        )

    # Recommendation: smallest k whose measured combo match is within
    # 0.3pt of all-int8 (or the best measured if none qualifies).
    rec_k = None
    best = None
    for k in combo_ks:
        m = combos[k]["match_bf16"]
        if best is None or m > best[1]:
            best = (k, m)
        if m >= all8 - 0.003:
            rec_k = k
            break
    if rec_k is None:
        rec_k = best[0]
    rec_bytes = sum(
        bytes_per_layer(8, 8) if li in set(ranked[:rec_k]) else bytes_per_layer(4, 4)
        for li in sweep_layers
    )

    summary = {
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "draft_tokens": args.draft_tokens,
        "attn_layers": sweep_layers,
        "kv_shape": [n_kv_heads, T, D],
        "anchors": {
            "all_int4_match_bf16": all4,
            "all_int4_clean4": results["all-int4"]["clean4_bf16"],
            "all_int8_match_bf16": all8,
            "all_int8_clean4": results["all-int8"]["clean4_bf16"],
        },
        "per_layer": table,
        "combos": {
            str(k): {
                "int8_layers": sorted(ranked[:k]),
                "match_bf16": combos[k]["match_bf16"],
                "clean4_bf16": combos[k]["clean4_bf16"],
                "bytes": sum(
                    bytes_per_layer(8, 8)
                    if li in set(ranked[:k])
                    else bytes_per_layer(4, 4)
                    for li in sweep_layers
                ),
            }
            for k in combo_ks
        },
        "bytes": {
            "unit": "bytes per token per KV head, K+V",
            "all_int4": byte_all4,
            "all_int8": byte_all8,
            "all_bf16": byte_bf16,
        },
        "recommendation": {
            "int8_layers": sorted(ranked[:rec_k]),
            "int4_layers": sorted(set(sweep_layers) - set(ranked[:rec_k])),
            "k": rec_k,
            "measured_match_bf16": combos[rec_k]["match_bf16"],
            "measured_clean4": combos[rec_k]["clean4_bf16"],
            "bytes": rec_bytes,
            "bytes_vs_all_int4": rec_bytes / byte_all4,
            "bytes_vs_all_int8": rec_bytes / byte_all8,
        },
        "wall_seconds": round(time.time() - t0, 1),
    }
    for k in combo_ks:
        b = summary["combos"][str(k)]["bytes"]
        summary["combos"][str(k)]["bytes_vs_all4"] = b / byte_all4
        summary["combos"][str(k)]["bytes_vs_all8"] = b / byte_all8

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=1)

    # ---- printed report ----
    print("\n=== per-layer int4 sensitivity (sorted worst-first) ===")
    print(f"{'layer':>5} {'int4KV%':>8} {'drop':>7} {'Konly%':>8} {'Vonly%':>8}")
    for row in table:
        print(
            f"{row['layer']:>5} {row['match_int4_kv'] * 100:>7.1f}% "
            f"{row['drop_vs_bf16'] * 100:>6.2f} "
            f"{(row['match_int4_k_only'] or 0) * 100:>7.1f}% "
            f"{(row['match_int4_v_only'] or 0) * 100:>7.1f}%"
        )
    print(f"\nall-int4 match {all4 * 100:.1f}%   all-int8 match {all8 * 100:.1f}%")
    print("\n=== combined int8-top-k + int4-rest ===")
    for k in combo_ks:
        c = summary["combos"][str(k)]
        print(
            f"k={k:>2} int8={c['int8_layers']} match {c['match_bf16'] * 100:.1f}% "
            f"clean4 {c['clean4_bf16'] * 100:.0f}% "
            f"bytes {c['bytes'] / byte_all4:.2f}x all-int4"
        )
    r = summary["recommendation"]
    print(
        f"\nRECOMMEND int8 layers {r['int8_layers']} int4 {r['int4_layers']} "
        f"-> match {r['measured_match_bf16'] * 100:.1f}% "
        f"at {r['bytes_vs_all_int4']:.2f}x int4 bytes "
        f"({r['bytes_vs_all_int8']:.2f}x int8 bytes)"
    )
    print(f"wrote {args.out}  total {summary['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
