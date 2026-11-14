"""Offline RaBitQ-style KV quantization experiment.

Same harness as turboquant_experiment.py: prefill 8K tokens on
mlx-community/Qwen3.8-27B-4bit, capture the KV cache, dequantize-in-place
per scheme, run a 256-token verify pass, compare per-position argmax vs a
bf16 reference (paired).

RaBitQ-style K codec: randomized Hadamard rotation -> 1-bit sign + fp16
per-vector scale -> inverse rotation (quantize-dequantize, then normal
attention). This measures CODEC quality, not the asymmetric sign-dot
scoring path — true RaBitQ attention (XOR+popcount on sign bits) would
need a custom attention kernel and is out of scope for this harness.
V codec: 4-bit per-group (64) affine min/max.

Run with .venv-tq/bin/python (has mlx_lm).
"""

import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

MODEL = "mlx-community/Qwen3.8-27B-4bit"
PROMPT_TOKENS = 8192
DRAFT_TOKENS = 256
SEED = 7


# ---------- scalar / group quantizers (same conventions as reference) ----------


def scalar_quant(x, bits):
    """Symmetric quant, one fp32 scale per (token, head) vector."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(x).max(axis=-1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-8)
    return np.clip(np.round(x / scale), -qmax - 1, qmax) * scale


def affine_quant(x, bits, group=64):
    """Asymmetric min/max quant, `group` elems along last axis, fp16 lo/hi."""
    d = x.shape[-1]
    assert d % group == 0, f"head_dim {d} not divisible by group {group}"
    xg = x.reshape(x.shape[:-1] + (d // group, group))
    lo = xg.min(axis=-1, keepdims=True).astype(np.float16).astype(np.float32)
    hi = xg.max(axis=-1, keepdims=True).astype(np.float16).astype(np.float32)
    qmax = 2**bits - 1
    scale = np.maximum((hi - lo) / qmax, 1e-8)
    q = np.clip(np.round((xg - lo) / scale), 0, qmax)
    return (q * scale + lo).reshape(x.shape)


# ---------- randomized Hadamard rotation (RaBitQ-style) ----------


def hadamard_matrix(d):
    """Sylvester Hadamard, d must be a power of two. Returns H/sqrt(d)."""
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


class Rotation:
    """Fixed random orthogonal map: permute -> sign-flip -> Hadamard."""

    def __init__(self, d, seed=SEED):
        rng = np.random.default_rng(seed)
        self.perm = rng.permutation(d)
        self.inv_perm = np.empty(d, dtype=np.int64)
        self.inv_perm[self.perm] = np.arange(d)
        self.flips = rng.choice(np.array([-1.0, 1.0]), size=d)
        self.h = hadamard_matrix(d).astype(np.float32)

    def fwd(self, x):
        """x (..., d) -> rotated coordinates (row-vector convention)."""
        return (np.take(x, self.perm, axis=-1) * self.flips) @ self.h.T

    def inv(self, z):
        """Inverse of fwd for orthogonal H (H symmetric, H@H=I)."""
        y = z @ self.h
        return np.take(y * self.flips, self.inv_perm, axis=-1)


_ROT = None


def get_rot(d):
    global _ROT
    if _ROT is None or _ROT.h.shape[0] != d:
        _ROT = Rotation(d)
    return _ROT


# ---------- RaBitQ-style K codecs ----------


def k_rabitq1(x, scale_mode="ls"):
    """1-bit sign after randomized Hadamard + fp16 scale, codec round-trip.

    scale_mode 'ls': least-squares optimal c = mean(|z|) for z ~ c*sign(z).
    scale_mode 'rabitq': paper's alpha = ||x||/sqrt(d) (rms), kept for
    comparison — reconstruction is biased low-MSE-wise but matches the
    published estimator's magnitude convention.
    """
    r = get_rot(x.shape[-1])
    z = r.fwd(x)
    bits = np.sign(z)
    bits[bits == 0] = 1.0
    if scale_mode == "ls":
        scale = np.mean(np.abs(z), axis=-1, keepdims=True)
    else:  # rabitq rms scale
        scale = np.sqrt(np.mean(z * z, axis=-1, keepdims=True))
    scale = scale.astype(np.float16).astype(np.float32)  # stored fp16
    return r.inv(bits * scale)


def k_rabitq1_norot(x):
    """Ablation: 1-bit sign + scale WITHOUT rotation (shows rotation is
    load-bearing — raw K has outlier dims)."""
    bits = np.sign(x)
    bits[bits == 0] = 1.0
    scale = np.mean(np.abs(x), axis=-1, keepdims=True)
    scale = scale.astype(np.float16).astype(np.float32)
    return bits * scale


def k_rabitq2(x):
    """Ablation/context: 2-bit symmetric scalar AFTER rotation."""
    r = get_rot(x.shape[-1])
    return r.inv(scalar_quant(r.fwd(x), 2))


# ---------- harness (same as turboquant_experiment.py) ----------


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
    mx.eval(
        logits,
        *[
            s
            for c in cache
            for s in (c.state if isinstance(c.state, (list, tuple)) else [c.state])
            if s is not None
        ],
    )
    print(f"prefilled {PROMPT_TOKENS} tokens {time.time() - t0:.0f}s", flush=True)

    saved = [c.state for c in cache]
    kv_idx = [i for i, c in enumerate(cache) if isinstance(c, KVCache)]
    k0 = np.array(
        cache[kv_idx[0]].keys[..., : cache[kv_idx[0]].offset, :].astype(mx.float32)
    )
    n_layers, n_heads, T, D = len(kv_idx), k0.shape[1], k0.shape[2], k0.shape[3]
    get_rot(D)
    print(
        f"KV: {n_layers} attn layers x {n_heads} heads x {T} x {D} "
        f"(head_dim {D}, rot ready)",
        flush=True,
    )

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
                c.state = (
                    mx.array(kq(k) if kq else k).astype(keys.dtype),
                    mx.array(vq(v) if vq else v).astype(values.dtype),
                    offset,
                )
            else:
                c.state = saved[i]

    REF = {}

    def trial(name, kq=None, vq=None, bits=""):
        restore(kq, vq)
        logits = model(mx.array(draft)[None], cache=cache)
        pred = np.array(mx.argmax(logits[0], axis=-1))[:-1]
        if name == "bf16 (ref)":
            REF["pred"] = pred
            m = pred == np.array(draft[1:])
            print(f"{name:30s} self-match {m.mean() * 100:.1f}%", flush=True)
            return
        ref = REF["pred"]
        match = pred == ref
        # 4-token verify windows fully matching = spec-decode block accepted
        w = match[: len(match) // 4 * 4].reshape(-1, 4).all(axis=1)
        b = f"  [{bits}]" if bits else ""
        print(
            f"{name:30s} token-match {match.mean() * 100:.1f}%  "
            f"clean-4win {w.mean() * 100:.0f}% ({int(w.sum())}/{len(w)}){b}",
            flush=True,
        )

    # effective bits/elem: intN scalar = N + 32/256 (fp32 scale per token-head)
    # K1bit = 1 + 16/256 (fp16 scale per 256-vec); V4g64 = 4 + 32/64 (fp16 lo+hi)
    # K2bit = 2 + 32/256
    BITS = {
        "int8": "8.1/8.1",
        "int4": "4.1/4.1",
        "K1+V4": "1.1/4.5",
        "K1+V8": "1.1/8.1",
        "K1norot+V4": "1.1/4.5",
        "K1rms+V4": "1.1/4.5",
        "K2+V4": "2.1/4.5",
    }

    trial("bf16 (ref)")
    trial("richengine int8 K+V", lambda x: scalar_quant(x, 8), lambda x: scalar_quant(x, 8), BITS["int8"])
    trial("richengine int4 K+V", lambda x: scalar_quant(x, 4), lambda x: scalar_quant(x, 4), BITS["int4"])
    # RaBitQ-style: K 1-bit sign after randomized Hadamard, codec round-trip
    trial(
        "rabitq K1bit(LS) + V4g64",
        lambda x: k_rabitq1(x, "ls"),
        lambda x: affine_quant(x, 4, 64),
        BITS["K1+V4"],
    )
    trial(
        "rabitq K1bit(LS) + V8",
        lambda x: k_rabitq1(x, "ls"),
        lambda x: scalar_quant(x, 8),
        BITS["K1+V8"],
    )
    trial(
        "rabitq K1bit(rms) + V4g64",
        lambda x: k_rabitq1(x, "rabitq"),
        lambda x: affine_quant(x, 4, 64),
        BITS["K1rms+V4"],
    )
    trial(
        "ablate K1bit NOROT + V4g64",
        k_rabitq1_norot,
        lambda x: affine_quant(x, 4, 64),
        BITS["K1norot+V4"],
    )
    trial(
        "ablate K2bit(rot) + V4g64",
        k_rabitq2,
        lambda x: affine_quant(x, 4, 64),
        BITS["K2+V4"],
    )
    restore()


if __name__ == "__main__":
    main()
