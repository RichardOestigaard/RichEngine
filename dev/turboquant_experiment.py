"""Offline TurboQuant vs Splash int4/int8 KV experiment.

Prefills a long prompt on mlx-community/Qwen3.8-27B-4bit, captures the KV
cache, then for each quantization scheme: dequantize-in-place into the
cache and run a 4-token greedy draft through the verify pass. Reports
argmax flips and spec-decode prefix acceptance.
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

MODEL = "mlx-community/Qwen3.8-27B-4bit"
PROMPT_TOKENS = 8192
DRAFT_TOKENS = 256


def scalar_quant(x, bits):
    """Splash-style symmetric quant, one fp32 scale per (token, head) vector."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=-1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-8)
    return np.clip(np.round(x / scale), -qmax - 1, qmax) * scale


def recency(fn, window):
    """Apply fn to all but the last `window` tokens (axis 2 = T)."""
    def wrapped(x):
        out = x.copy()
        if window < x.shape[2]:
            out[:, :, :-window, :] = fn(x[:, :, :-window, :])
        return out
    return wrapped


def mlx_quant(x, mode, group_size):
    """Round-trip through MLX mxfp4/nvfp4 quantize->dequantize."""
    a = mx.array(x)
    q = mx.quantize(a, group_size=group_size, mode=mode)
    return np.array(mx.dequantize(*q, group_size=group_size, mode=mode).astype(mx.float32))


def tq_quant(x, bit_width, mode, seed):
    from turboquant import TurboQuant

    d = x.shape[-1]
    tq = TurboQuant(dim=d, bit_width=bit_width, mode=mode, seed=seed)
    flat = x.reshape(-1, d).astype(np.float64)
    out = np.empty_like(flat)
    bs = 4096
    for i in range(0, len(flat), bs):
        out[i : i + bs] = tq.dequantize(tq.quantize(flat[i : i + bs]))
    return out.reshape(x.shape).astype(np.float32)


def main():
    t0 = time.time()
    model, tok = load(MODEL)
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    text = (
        "The quarterly infrastructure report details deployment topology, "
        "cache coherence protocols, and latency budgets for each region. "
    )
    ids = tok.encode(text)
    reps = PROMPT_TOKENS // len(ids) + 1
    prompt = (ids * reps)[:PROMPT_TOKENS]

    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    mx.eval(logits, *[s for c in cache for s in (c.state if isinstance(c.state, (list, tuple)) else [c.state]) if s is not None])
    print(f"prefilled {PROMPT_TOKENS} tokens {time.time() - t0:.0f}s", flush=True)

    saved = [c.state for c in cache]
    kv_idx = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    k0 = np.array(cache[kv_idx[0]].keys[..., : cache[kv_idx[0]].offset, :].astype(mx.float32))
    n_layers, n_heads, T, D = len(kv_idx), k0.shape[1], k0.shape[2], k0.shape[3]
    print(f"KV: {n_layers} attn layers x {n_heads} heads x {T} x {D}", flush=True)

    # reference greedy draft (proxy for a correct draft model at temp 0)
    draft = []
    nxt = int(mx.argmax(logits[0, -1]))
    for _ in range(DRAFT_TOKENS):
        draft.append(nxt)
        lg = model(mx.array([[nxt]]), cache=cache)
        nxt = int(mx.argmax(lg[0, -1]))
    print("draft:", draft, repr(tok.decode(draft)), flush=True)

    def restore(kq=None, vq=None):
        for i, c in enumerate(cache):
            if i in kv_idx:
                keys, values, offset = saved[i]
                k = np.array(keys[..., :offset, :].astype(mx.float32))
                v = np.array(values[..., :offset, :].astype(mx.float32))
                c.state = (mx.array(kq(k) if kq else k).astype(keys.dtype), mx.array(vq(v) if vq else v).astype(values.dtype), offset)
            else:
                c.state = saved[i]

    REF = {}

    def trial(name, kq=None, vq=None):
        restore(kq, vq)
        logits = model(mx.array(draft)[None], cache=cache)
        pred = np.array(mx.argmax(logits[0], axis=-1))[:-1]
        if name == "bf16 (ref)":
            REF["pred"] = pred
            m = pred == np.array(draft[1:])
            print(f"{name:26s} self-match {m.mean()*100:.1f}%", flush=True)
            return
        ref = REF["pred"]
        match = pred == ref
        # 4-token verify windows fully matching = spec-decode block accepted
        w = match[: len(match) // 4 * 4].reshape(-1, 4).all(axis=1)
        print(f"{name:26s} token-match {match.mean()*100:.1f}%  clean-4win {w.mean()*100:.0f}% ({int(w.sum())}/{len(w)})", flush=True)

    trial("bf16 (ref)")
    for b in (8, 4):
        trial(f"splash int{b} K+V", lambda x, b=b: scalar_quant(x, b), lambda x, b=b: scalar_quant(x, b))
    for kb, vb in ((4, 8), (6, 8), (4, 6), (8, 4), (8, 2), (4, 2)):
        trial(
            f"scalar K{kb} V{vb}",
            lambda x, b=kb: scalar_quant(x, b),
            lambda x, b=vb: scalar_quant(x, b),
        )
    for mode, g in (("mxfp4", 32), ("nvfp4", 16), ("mxfp8", 32)):
        trial(f"{mode} K+V g{g}",
              lambda x, m=mode, g=g: mlx_quant(x, m, g),
              lambda x, m=mode, g=g: mlx_quant(x, m, g))
    trial("K int8 / V mxfp4",
          lambda x: scalar_quant(x, 8),
          lambda x: mlx_quant(x, "mxfp4", 32))
    trial("K mxfp4 / V int8",
          lambda x: mlx_quant(x, "mxfp4", 32),
          lambda x: scalar_quant(x, 8))
    for w in (128, 512, 2048):
        trial(
            f"int4 + bf16 last {w}",
            recency(lambda x: scalar_quant(x, 4), w),
            recency(lambda x: scalar_quant(x, 4), w),
        )
    trial("int2 + bf16 last 512",
          recency(lambda x: scalar_quant(x, 2), 512),
          recency(lambda x: scalar_quant(x, 2), 512))
    trial("K4V4 old + bf16 last 512 / V2",
          recency(lambda x: scalar_quant(x, 4), 512),
          recency(lambda x: scalar_quant(x, 2), 512))
    for kb, vb in ((3, 3), (3, 2), (2, 2), (4, 3)):
        trial(
            f"tq K{kb}(prod) V{vb}(mse)",
            lambda x, b=kb: tq_quant(x, b, "inner_product", 7),
            lambda x, b=vb: tq_quant(x, b, "mse", 7),
        )
    restore()


if __name__ == "__main__":
    main()
