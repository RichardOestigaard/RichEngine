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

## Measured A/B — Qwen3.8-27B, Apple M5 Pro (2026-11)

**Promoted**: `qwen38TargetLayout()` now declares the calibrated policy
(`{128, 1024, 0x4000000000bfa317}`) as the default; it applies on int8
and int4 KV and drops to dense on bf16/fp8
(`RuntimeGeometry::from` gates it by format). `RICHENGINE_DUO_OVERRIDE`
still replaces the policy; `0,0,0` restores dense attention.

`backend-benchmark --scenario longdecode`, sink 128 / window 1024, int8
KV. Baseline is the same build without the override. Single sample per
length; prompts are synthetic filler ending in the chat decode prompt.

Two masks were measured:

- `0x1111111111111111` — arbitrary: KV head 0 of every layer streams
  (16 heads), including heads the probe later showed carry heavy
  old-context mass. Upper-bound datapoint, not a candidate.
- `0x4000000000bfa317` — calibrated: the probe's `--budget 48` mask
  (16 lowest-old-mass heads stream, the rest keep full history).

| context | prefill tok/s base → arb → cal | decode tok/s base → arb → cal | output |
|--------:|-------------------------------:|------------------------------:|--------|
| 32,768  | 451 → 561 → 496                | 25.7 → 27.9 → 27.5            | arb identical, cal identical |
| 65,536  | 388 → 468 → 414                | 16.3 → 17.8 → 15.7            | arb identical, cal identical |
| 131,072 | 286 → 344 → 321                | 7.1 → 7.2 → 7.1               | arb identical, cal **diverged** |

Follow-up split-policy change (same calibrated mask, `PagedAttention.cpp`):
duo verify lanes now size splits by full history instead of the
sink+window bound, and duo prefill takes the history-scaled
`prefillSplits` instead of `sharedPrefillSplits`. Measured vs the dense
baseline:

| context | prefill tok/s | decode tok/s | output |
|--------:|--------------:|-------------:|--------|
| 32,768  | 451 → **531 (+18%)** | 25.7 → **28.1 (+9%)**  | identical |
| 65,536  | 388 → **449 (+16%)** | 16.3 → **18.3 (+12%)** | identical |
| 131,072 | 286 → **353 (+24%)** | 7.1 → **7.6 (+8%)**    | identical |

Decode-scenario throughput (39-token prompt) is unchanged — the whole
history fits inside sink+window, so DuoAttention is a no-op there.

### `_duo` q8q and `_a16` kernel variants (2026-11)

The `_duo` splits now have the same optional inner-loop variants as the
dense kernels: `RICHENGINE_Q8Q_ATTENTION=1` selects the INT8-query
`_duo_q8q` splits (prefill `prefill_attention_q8q_split_duo`, verify
`verify_attention_q8q_split_duo{,_m8,_r2,_r4}`) and
`RICHENGINE_ATTENTION_ACC16=1` selects the bfloat-accumulator
`_duo_a16` splits (`..._duo{,_m8,_r2,_r4}_a16`, int8/int4 pages). The two
stay mutually exclusive, as on the dense path. Both remain opt-in env
flags; the shipped default is still the fp32 `_duo` split.

`longdecode`, same calibrated mask and build, vs the dense baseline:

| context | prefill tok/s dense → duo+a16 | decode tok/s dense → duo+a16 |
|--------:|------------------------------:|-----------------------------:|
| 32,768  | 451 → **576 (+28%)** | 25.7 → 27.0 (+5%)  |
| 65,536  | 388 → **519 (+34%)** | 16.3 → **19.1 (+17%)** |
| 131,072 | 286 → **412 (+44%)** | 7.1 → **8.6 (+21%)** |

| context | prefill tok/s dense → duo+q8q | decode tok/s dense → duo+q8q |
|--------:|------------------------------:|-----------------------------:|
| 32,768  | 451 → **562 (+25%)** | 25.7 → **29.1 (+13%)** |
| 65,536  | 388 → **494 (+27%)** | 16.3 → **18.2 (+12%)** |
| 131,072 | 286 → **384 (+34%)** | 7.1 → **8.0 (+12%)** |

Against the fp32 `_duo` splits alone, `_a16` adds roughly +6–17% prefill
and up to +13% decode at 131K; `q8q` adds roughly +6–10% prefill and
~+4% decode at 131K. `_a16` is the larger of the two — the PV
accumulator's bandwidth matters more at long context than query
quantization does. Both runs completed 64 tokens/lane with identical
output hashes to each other and `performance_pass: true`; single-sample
runs, so treat sub-10% deltas as soft.

### Head classification (the probe run above)

At sink 128 / window 1024, 61 of 64 (layer, KV head) pairs carry ≥5%
old-context softmax mass — this model is retrieval-heavy and the
DuoAttention paper's ~10–25% streaming budget is the ceiling, not the
default. `--budget 48` streams the 16 lightest heads; `--budget 56`
streams 8 (mask `0x000000000012a116`) for a more conservative point.
The default `--threshold 0.05` keeps 61/64 heads — only ~5% streaming,
not worth the plumbing.

### Quality check — NIAH through `richengine serve`

Passkey prompts (32,768 and 131,072 tokens, needle at mid-depth)
greedy-decoded with and without the calibrated override:

- 32K: baseline and duo both answered `orion-4242` — byte-identical.
- 131K: duo answered `harbor-7719` correctly (needle at ~65.5K tokens,
  far outside sink+window). The 131K benchmark hash divergence was in
  degenerate synthetic-prompt output, not retrieval.

Caveats:

- Single-sample benchmark runs; ±5% run-to-run noise is plausible, so
  treat sub-10% deltas as soft.
- Decode gain shrinks with depth because the step is weight-bandwidth
  bound; prefill is where the win concentrates.
- The override path needed `RuntimeGeometry::from` to carry the resolved
  policy into `kvLayout.duo` — before that fix bootstrap threw
  "prefill attention duo policy does not match plan".

## Ornith 1.5 9B — kv4/g4 `_duo` kernels (2026-11)

Ornith's full-attention layers run the `Kv4Group4` core, so the `_duo`
set was instantiated a second time at Group 4: verify splits
`verify_attention_{q8,int4,q8q}_split_kv4_g4_duo{,_m8,_r2,_r4}` plus the
`_a16` twins, prefill `prefill_attention_{q8,int4,q8q}_split_kv4_g4_duo`
plus `_a16`, and the matching reduces
(`verify_attention_reduce{,_gate}_kv4_g4_duo{,_m8}`,
`prefill_attention_reduce_kv4_g4_duo`). `requireDuoLayout` admits both
kv4 cores; the mask packing (`4*ordinal+head`, ≤16 ordinals) is
unchanged — Ornith's 8 FA layers use 32 bits.

Probe (`ornith_duoattention_run.json`/`.npz`, Ornith-1.5-9B-MLX-4bit):
32 (layer, KV head) pairs, all carrying ≥10% mid/old mass at every
recorded setting — even more retrieval-heavy than the 27B. `--budget
24` keeps the 24 heaviest heads full and streams the 8 lightest:
`{128, 1024, 0xE0002017}` declared in `ornith9TargetLayout()`.

`longdecode`, int8 KV, same-build A/B (`RICHENGINE_DUO_OVERRIDE=0,0,0`
vs the declared policy), `--samples 1`, M5 Pro:

| context | prefill tok/s dense → duo | decode tok/s dense → duo |
|--------:|--------------------------:|-------------------------:|
| 32,768  | 1812 → **1836 (+1%)**  | 69.5 → **81.5 (+17%)** |
| 65,536  | 1420 → **1551 (+9%)**  | 46.4 → **48.1 (+4%)**  |
| 131,072 | 1022 → **1154 (+13%)** | 17.9 → **20.8 (+16%)** |

`performance_pass: true`, 64 tokens/lane everywhere. Decode deltas are
noisy — wall tok/s tracks draft acceptance and batch count (23 vs 25
batches at 32K), so read them as directional; prefill is the cleaner
signal and improves monotonically with depth, matching the KV-read
bound the policy targets. Output hashes differ from the dense baseline
at 32K and 131K (expected under approximation); NIAH below is the
quality check.

Quality: NIAH passkey probes through `richengine serve` — 32K answered
`orion-4242` identically with and without the policy; 131K answered
`harbor-7719` correctly under duo. Caveat: the mask was calibrated on
the MLX-4bit install; the ornith9 arch also covers the GGUF and
Distilled variants, which were not probed.

## qwen3_6moe — kv2/g8 `_duo` kernels, Ornith-1.5-35B-A3B (2026-11)

The MoE family's full-attention layers run the `Kv2Group8` core (16
query heads, 2 KV heads, head-dim 256, FA every 4th of 40 layers → 10
FA layers), so the `_duo` set was instantiated a third time. kv2/g8's
fused verify tile is M = 8 × VERIFY_ROWS = 128 rows — the same size
that forces the dense kernels' internal two-pass `_m2` halving — so its
plain and `_m8` `_duo` splits run `PAGED_VERIFY_SPLIT_DUO_M2` (MPasses=2
inside the tile, DuoAttention policy trailing), while the chain's `_r4`
splits keep the grid pass split like the kv4 cores'. q8q and `_a16`
twins included; the duo reduce macros took a KVHeads parameter (they
hardcoded 4). `requireDuoLayout` admits all three kv cores; the mask
packing is unchanged — 10 ordinals × 2 heads uses 40 bits.

Probe (`ornith35_duoattention_run.json`/`.npz`,
Ornith-1.5-35B-A3B-MLX-4bit): all 20 (layer, KV head) pairs carry
0.10–0.32 mid/old mass — the most retrieval-dense model profiled so
far. `--budget 16` keeps the 16 heaviest heads full and streams the
four lightest (ordinals 0h1, 1h1, 2h0, 2h1): `{128, 1024, 0x322}`,
declared in `ornith36Descriptor` only — Qwen3.6-35B-A3B shares
`qwen36MoeTargetLayout` but was not probed, so it stays dense.

`longdecode`, int8 KV, same-build A/B (`RICHENGINE_DUO_OVERRIDE=0,0,0`
vs the declared policy), `--samples 1`, M5 Pro:

| context | prefill tok/s dense → duo | decode tok/s dense → duo |
|--------:|--------------------------:|-------------------------:|
| 32,768  | 2278 → **2305 (+1%)**  | 106.9 → **109.6 (+2%)** |
| 65,536  | 1560 → **1602 (+3%)**  | 58.0 → **58.7 (+1%)**   |
| 131,072 | 972 → **988 (+2%)**    | 22.9 → **23.0 (flat)**  |

`performance_pass: true`, and all three output hashes are identical to
the dense baseline — the four streamed heads did not flip a token on
the synthetic prompts. Gains are small by construction: 4 of 20 FA KV
heads stream, FA layers are 10 of 40, and the MoE step is
expert-bandwidth-bound so attention is a minor share — prefill is
still the cleaner signal.

Quality: NIAH passkey probes through `richengine serve` — 32K
answered `orion-4242` and 131K answered `harbor-7719` (needle ~65K
tokens deep, far outside sink+window) under the duo policy, both with
clean `stop` finishes. `RICHENGINE_DUO_OVERRIDE=0,0,0` remains the
dense rollback; Qwen3.6-35B-A3B and other `qwen3_6moe` variants stay
dense until separately probed.
