# DuoAttention head classification

Tools for ranking each (full-attention layer, KV head) of a target model by
how much of its softmax mass lands on old context versus the sink + recent
window — the signal DuoAttention uses to pick which heads keep the whole
cache and which stream to `sink` + `window` tokens only.

For Qwen3.8-27B this is the 16 full-attention layers
(`full_attention_interval: 4` of the 64-layer `qwen3_5` hybrid) x 4 KV
heads = 64 mask entries. GDN/linear layers do not attend a KV cache and
are skipped.

## Why the probe is MLX, not a Metal kernel

A debug Metal kernel reusing `paged_attention_tile.h` (option a) was
considered and dropped: the mass split needs the model's *real* queries,
which only exist transiently inside the engine's graph, so a kernel alone
cannot produce them without runtime changes (which this work package must
not make). The probe instead taps
`scaled_dot_product_attention(queries, keys, values, cache=…, scale, mask)`
inside `mlx_lm.models.*` — at that call site queries are post-RoPE and
keys are the full committed history, so the fractions it measures are the
exact softmax the production kernels compute. `.venv` already carries
`mlx` + `mlx_lm` (used by `dev/distill/dflash_distill_mlx.py`).

## Files

- `duoattention_probe.py` — builds the NIAH set, runs the model, records.
- `duoattention_mask.py` — aggregates a run into the mass table and emits
  the `duoAttention` JSON.
- Output: `duoattention_run.json` (exact per-call region fractions +
  prompt metadata) and `duoattention_profiles.npz` (normalized per-token
  mass `[T]` per call/layer/head for offline re-bucketing).

## Run

```sh
.venv/bin/python dev/tools/duoattention_probe.py \
    --model mlx-community/Qwen3.8-27B-4bit \
    --lengths 1024,4096,16384,32768 --depths 0.1,0.5,0.9 \
    --probe-rows 8 --decode-steps 2 \
    --settings 4:512,4:1024,8:2048 \
    --out dev/tools/duoattention_run.json \
    --profiles-out dev/tools/duoattention_profiles.npz
```

Each prompt is filler prose with `The magic passkey is <word>-<number>.`
at fractional `depth`, then a question; the last `--probe-rows` query rows
of prefill plus the first `--decode-steps` generated tokens are the probed
query rows. `hit` in the output says the greedy answer reproduced the
passkey — heads must retrieve the needle for hits at long lengths.

`--prefill-chunk N` chunks prefill for memory; only the last chunk's rows
are probed (non-final chunks still populate the cache normally).

```sh
.venv/bin/python dev/tools/duoattention_mask.py \
    dev/tools/duoattention_run.json dev/tools/duoattention_profiles.npz \
    --sink 4 --window 1024 --threshold 0.05
# or keep a fixed head count:
#   --budget 24
```

The table prints `sink` / `mid` ("old") / `recent` fractions and the
`needle`-span mass per head. A `--sink/--window` pair not in the probe's
`--settings` is rejected — re-run the probe with the setting you want
bucketted exactly (the profiles can be re-bucketed by hand for sink-side
questions; the recent edge is evaluated at each row's true causal end
only inside the wrapper).

## Feeding the family json

`duoattention_mask.py --json` emits:

```json
{"duoAttention": {
  "sink": 4, "window": 1024,
  "retrievalHeads": [[layer, kvHead], ...],
  "headMask": {"3": [true, true, false, true], ...},
  "faLayerOrder": [3, 7, 11, ..., 63],
  "kvHeads": 4
}}
```

`headMask` is keyed by *absolute* layer id; `faLayerOrder` maps the mask's
implicit full-attention ordering (`fa_index` in the run's `calls`) back to
layer ids — the runtime consumer indexes layers 0..15 or absolutely per
its own convention. `retrievalHeads` is the same mask as `[layer, kvHead]`
pairs. The other agent owns the `duoAttention` field name/shape in the
family schema and the runtime-side head mask plumbing (paged-attention
kernel gating); this output is the calibration input they consume — copy
`sink`/`window`/`headMask` into the family json, e.g.
`install/families/qwen3-8-27b.json`.

## Tuning notes

- `--threshold` (default 0.05) marks a head retrieval if >= 5% of its
  mass sits on old tokens; DuoAttention's paper budgets ~10-25% of heads.
  `--budget` keeps the top-K heads by old mass directly.
- Check `needle` mass: retrieval heads show a clear spike on the needle
  span at long lengths; streaming heads do not.
- Lengths beyond ~32K at 4-bit on laptop memory should use
  `--prefill-chunk 8192`.
- The probe adds no runtime dependencies: anything importable by
  `mlx_lm.load` works, including non-hybrid models (every layer is then a
  full-attention layer).
