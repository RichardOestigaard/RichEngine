#!/usr/bin/env python3
"""DuoAttention head-classification probe for RichEngine targets.

Runs NIAH-style passkey prompts through the real model weights with
``mlx_lm`` and records, per (full-attention layer, KV head), where the
softmax mass of the trailing query rows lands: the sink region ``[0, S)``,
the recent window ``[end - W, end)``, and the old ("mid") tokens between
them. That split is the signal DuoAttention uses to decide which heads
keep the whole cache (retrieval heads) and which stream to sink+recent.

Mechanics: ``mlx_lm`` attention modules call a module-global
``scaled_dot_product_attention(queries, keys, values, cache=..., scale=...,
mask=...)`` with *post-RoPE* queries and the *full-history* keys returned
by ``cache.update_and_fetch`` — see e.g.
``mlx_lm/models/qwen3_next.py`` ``Qwen3NextAttention.__call__``. This tool
wraps that function in every loaded ``mlx_lm.models.*`` module, maps the
bound ``cache`` object back to its layer via ``model.make_cache()``, and
accumulates:

* a normalized per-token mass profile ``[T]`` per KV head (mean over the
  KV head's query group and the last ``--probe-rows`` query rows), saved to
  ``--profiles-out`` (.npz) for offline re-bucketing, and
* exact per-row region fractions for each ``--settings`` ``S:W`` pair,
  evaluated inside the wrapper with each row's true causal end, saved to
  ``--out`` (.json).

The companion ``duoattention_mask.py`` turns a run into the per-head table
and the recommended ``duoAttention`` family-json mask.

usage: duoattention_probe.py [--model mlx-community/Qwen3.8-27B-4bit]
        [--lengths 1024,4096,16384,32768] [--depths 0.1,0.5,0.9]
        [--probe-rows 8] [--decode-steps 2] [--prefill-chunk 0]
        [--settings 4:512,4:1024,8:2048] [--seed N]
        [--out run.json] [--profiles-out profiles.npz]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load as load_model
from mlx_lm.models.cache import make_prompt_cache

# Filler is domain-neutral repeated prose; the needle carries the passkey.
FILLER = (
    "The archive holds quarterly ledgers, harvest notes, tide tables, and "
    "shipping manifests from a dozen coastal towns. Curators index every "
    "entry by hand and cross-reference the marginalia with port records. "
)
NEEDLE = "The magic passkey is {word}-{number}. Remember it exactly."
QUESTION = (
    "\n\nQuestion: What is the magic passkey mentioned earlier "
    "in this document? Answer with the passkey only.\nAnswer:"
)
NEEDLE_WORDS = [
    "orion",
    "banyan",
    "cobalt",
    "falcon",
    "harbor",
    "lumen",
    "marmot",
    "quartz",
]


def encode_fill(tokenizer, target):
    """Filler token list of exactly `target` tokens."""
    base = tokenizer.encode(FILLER)
    reps = target // len(base) + 1
    return (base * reps)[:target]


def make_niah_prompt(tokenizer, length, depth, needle_word, needle_number):
    """Passkey at fractional `depth` of `length` tokens; question at the end.

    Returns (token_ids, needle_span, needle_text): needle_span is the
    [start, end) token range the needle occupies.
    """
    needle_text = NEEDLE.format(word=needle_word, number=needle_number)
    needle = tokenizer.encode(needle_text)
    question = tokenizer.encode(QUESTION)
    body = length - len(question)
    if body <= len(needle):
        raise ValueError(f"length {length} too small for needle + question")
    needle_start = int((body - len(needle)) * depth)
    ids = (
        encode_fill(tokenizer, needle_start)
        + needle
        + encode_fill(tokenizer, body - needle_start - len(needle))
        + question
    )
    assert len(ids) == length, (len(ids), length)
    return ids, (needle_start, needle_start + len(needle)), needle_text


def attention_layers(model):
    """[(absolute_index, DecoderLayer)] for full-attention layers.

    Hybrid models (qwen3_5 / qwen3_next) give non-attention layers an
    ``is_linear`` flag and no ``self_attn``; dense models report every
    layer. The cache list from ``model.make_cache()`` is in the same order
    as ``model.layers``.
    """
    layers = getattr(model, "layers", None)
    if layers is None:
        inner = getattr(model, "model", model)
        layers = getattr(inner, "layers")
    result = []
    for index, layer in enumerate(layers):
        if getattr(layer, "is_linear", False):
            continue
        if hasattr(layer, "self_attn") or hasattr(layer, "attention"):
            result.append((index, layer))
    return layers, result


class MassRecorder:
    """sdpa tap: region fractions + per-token mass profile per KV head."""

    def __init__(self, layers, prompt_cache, settings, probe_rows):
        self.probe_rows = probe_rows
        self.settings = settings  # list of (sink, window)
        self.enabled = False
        self.calls = []
        self.profiles = {}
        self.prompt = -1
        self.step = 0
        self.layer_calls = {}
        self.errors = 0
        # The cache list is layer-ordered; record only full-attention
        # layers, whose caches are KVCache-shaped (have .offset).
        self.cache_layer = {
            id(cache): index for index, cache in enumerate(prompt_cache)
        }
        fa = [i for i, layer in enumerate(layers)
              if not getattr(layer, "is_linear", False)
              and (hasattr(layer, "self_attn") or hasattr(layer, "attention"))]
        self.fa_index = {layer: k for k, layer in enumerate(fa)}
        self.fa_layers = fa

    def begin(self, prompt_id):
        self.prompt = prompt_id
        self.step = 0
        self.layer_calls = {}

    def next_step(self):
        """Advance the forward-pass counter; call after each model()."""
        self.step += 1

    def record(self, queries, keys, cache, scale, mask):
        if not self.enabled or cache is None or id(cache) not in self.cache_layer:
            return
        layer = self.cache_layer[id(cache)]
        if layer not in self.fa_index:
            return
        B, hq, L, _ = queries.shape
        hkv = keys.shape[1]
        T = keys.shape[2]
        rows = min(self.probe_rows, L)
        group = hq // hkv
        if hq % hkv:
            return
        q = np.asarray(
            queries[:, :, L - rows :, :].astype(mx.float32), dtype=np.float32
        )
        k = np.asarray(keys.astype(mx.float32), dtype=np.float32)
        ends = self._causal_ends(mask, rows, T)
        # Slice the mask to the probed rows before any host conversion.
        mask_arr = None
        if mask is not None and not isinstance(mask, str):
            mask_arr = np.asarray(mask[..., -rows:, -T:])
        profile = np.zeros((hkv, T), np.float32)
        fracs = np.zeros((hkv, len(self.settings), 3), np.float32)
        t = np.arange(T, dtype=np.int64)[None, :]
        end_col = ends[:, None]
        admitted = t < end_col  # [rows, T]
        for h in range(hkv):
            qh = q[:, h * group : (h + 1) * group]  # [B,g,rows,D]
            scores = np.einsum(
                "bgrd,btd->bgrt", qh, k[:, h], optimize=True
            ) * float(scale)
            if mask_arr is not None:
                m = np.broadcast_to(mask_arr, scores.shape)
                if mask_arr.dtype == bool:
                    scores = np.where(m, scores, -np.inf)
                else:
                    scores += m
            scores = np.where(admitted[None, None], scores, -np.inf)
            top = scores.max(axis=-1, keepdims=True)
            exp = np.where(
                admitted[None, None], np.exp(scores - np.maximum(top, -1e30)), 0.0
            )
            exp /= np.maximum(exp.sum(axis=-1, keepdims=True), 1e-30)
            profile[h] = exp.mean(axis=(0, 1, 2))
            for s, (snk, win) in enumerate(self.settings):
                is_sink = t < snk
                is_recent = (
                    t >= np.maximum(end_col - win, snk)
                ) & admitted
                is_mid = admitted & ~is_sink & ~is_recent
                reg = np.stack(
                    [
                        (exp * is_sink[None, None]).sum(-1),
                        (exp * is_mid[None, None]).sum(-1),
                        (exp * is_recent[None, None]).sum(-1),
                    ],
                    axis=-1,
                )
                fracs[h, s] = reg.mean(axis=(0, 1, 2))
        entry = {
            "prompt": self.prompt,
            "step": self.step,
            "layer": layer,
            "fa_index": self.fa_index[layer],
            "kv_heads": int(hkv),
            "T": int(T),
            "rows": int(rows),
            "region_fractions": {
                f"sink{sink}_win{window}": np.round(fracs[:, s], 6).tolist()
                for s, (sink, window) in enumerate(self.settings)
            },
        }
        self.calls.append(entry)
        call_no = self.layer_calls.get(layer, 0)
        self.layer_calls[layer] = call_no + 1
        for h in range(hkv):
            key = f"p{self.prompt}_s{self.step}_l{layer}_h{h}_c{call_no}"
            self.profiles[key] = profile[h].astype(np.float32)

    def _causal_ends(self, mask, rows, T):
        """Causal end (first excluded token) of each probed query row."""
        if mask is None:
            return np.full(rows, T, dtype=np.int64)
        if isinstance(mask, str):
            # "causal": rows are the last `rows` of the call's L; row j's
            # absolute index is T - rows + j, so its end is +1.
            return (T - rows + 1 + np.arange(rows)).astype(np.int64)
        m = np.asarray(mask[..., -rows:, -T:]).reshape(-1, rows, T)
        if m.dtype == bool:
            admitted = m[0]
        else:
            admitted = m[0] > -1e29
        # End is one past the last admitted key of each row.
        rev = admitted[:, ::-1]
        ends = np.where(
            admitted.any(-1), T - rev.argmax(-1), 0
        ).astype(np.int64)
        return ends


def patch_sdpa(recorder):
    """Wrap scaled_dot_product_attention in already-imported model modules.

    Call after ``load_model`` so the target's module (and its attention
    helper imports, e.g. qwen3_5 -> qwen3_next) are in ``sys.modules``.
    Importing every module is unsafe: some raise on missing optional deps
    (mlx_lm.models.olmo exits when ai2-olmo is absent).
    """
    wrapped = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("mlx_lm.models."):
            continue
        fn = getattr(module, "scaled_dot_product_attention", None)
        if callable(fn) and not getattr(fn, "_duo_tap", False):

            def make(fn):
                def tapped(queries, keys, values, *args, **kwargs):
                    cache = kwargs.get("cache")
                    if cache is None and len(args) >= 1:
                        cache = args[0]
                    scale = kwargs.get("scale")
                    if scale is None:
                        scale = args[1] if len(args) >= 2 else 1.0
                    mask = kwargs.get("mask")
                    if mask is None and len(args) >= 3:
                        mask = args[2]
                    try:
                        recorder.record(queries, keys, cache, scale, mask)
                    except Exception as error:
                        # A tap failure must not break the forward pass.
                        if recorder.errors < 3:
                            print(f"tap record failed: {error}",
                                  file=sys.stderr)
                        recorder.errors += 1
                    return fn(queries, keys, values, *args, **kwargs)

                tapped._duo_tap = True
                return tapped

            setattr(module, "scaled_dot_product_attention", make(fn))
            wrapped.append(name)
    return wrapped


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--lengths", default="1024,4096,16384,32768")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--probe-rows", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=2)
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="chunk prefill at N tokens (0: one shot)")
    ap.add_argument("--settings", default="4:512,4:1024,8:2048",
                    help="sink:window pairs scored exactly in the wrapper")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="dev/tools/duoattention_run.json")
    ap.add_argument("--profiles-out", default="dev/tools/duoattention_profiles.npz")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    settings = []
    for pair in args.settings.split(","):
        sink, window = pair.split(":")
        settings.append((int(sink), int(window)))

    model, tokenizer = load_model(args.model)
    layers, fa = attention_layers(model)
    print(f"model {args.model}: {len(layers)} layers, "
          f"{len(fa)} full-attention {[i for i, _ in fa]}",
          file=sys.stderr)

    recorder_cache = make_prompt_cache(model)
    recorder = MassRecorder(layers, recorder_cache, settings, args.probe_rows)
    wrapped = patch_sdpa(recorder)
    print(f"tapped sdpa in {len(wrapped)} modules", file=sys.stderr)

    rng = np.random.default_rng(args.seed)
    prompts_meta = []
    pid = 0
    for length in lengths:
        for depth in depths:
            word = NEEDLE_WORDS[pid % len(NEEDLE_WORDS)]
            number = int(rng.integers(1000, 9999))
            ids, span, needle_text = make_niah_prompt(
                tokenizer, length, depth, word, number
            )
            prompts_meta.append(
                {
                    "id": pid,
                    "length": length,
                    "depth": depth,
                    "needle_span": list(span),
                    "needle_text": needle_text,
                    "answer": f"{word}-{number}",
                }
            )
            t0 = time.time()
            prompt_cache = make_prompt_cache(model)
            # Re-key the recorder to this prompt's fresh cache objects.
            recorder.cache_layer = {
                id(cache): index for index, cache in enumerate(prompt_cache)
            }
            recorder.begin(pid)
            tokens = np.asarray(ids, dtype=np.int32)
            if args.prefill_chunk > 0:
                # Only the last chunk's trailing rows are probed.
                logits = None
                for off in range(0, len(ids), args.prefill_chunk):
                    recorder.enabled = off + args.prefill_chunk >= len(ids)
                    logits = model(
                        mx.array(tokens[None, off : off + args.prefill_chunk]),
                        cache=prompt_cache,
                    )
                if recorder.enabled:
                    recorder.next_step()
            else:
                recorder.enabled = True
                logits = model(mx.array(tokens[None]), cache=prompt_cache)
                recorder.next_step()
            # Prefill's last row already predicts the first generated token.
            y = mx.argmax(logits[0, -1])
            mx.eval(y)
            generated = [int(y)]
            for _ in range(args.decode_steps - 1):
                logits = model(mx.array(y[None, None]), cache=prompt_cache)
                recorder.next_step()
                y = mx.argmax(logits[0, -1])
                mx.eval(y)
                generated.append(int(y))
            recorder.enabled = False
            text = tokenizer.decode(generated)
            prompts_meta[-1]["generated_tokens"] = generated
            prompts_meta[-1]["generated_text"] = text
            prompts_meta[-1]["hit"] = prompts_meta[-1]["answer"] in text
            print(
                f"prompt {pid}: len={length} depth={depth} "
                f"needle@{span[0]} gen={text!r} "
                f"hit={prompts_meta[-1]['hit']} "
                f"({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )
            pid += 1

    results = {
        "model": args.model,
        "probe_rows": args.probe_rows,
        "decode_steps": args.decode_steps,
        "prefill_chunk": args.prefill_chunk,
        "settings": [list(s) for s in settings],
        "layers": len(layers),
        "fa_layers": recorder.fa_layers,
        "kv_heads": max((c["kv_heads"] for c in recorder.calls), default=0),
        "prompts": prompts_meta,
        "calls": recorder.calls,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(args.profiles_out, **recorder.profiles)
    print(f"wrote {args.out} and {args.profiles_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
