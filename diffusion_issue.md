# DiffusionGemma 26B-A4B — runtime bring-up status

Status: the model **loads, warms up, serves, and generates finite (NaN-free)
canvas bursts**. Output is still not coherent text — the trunk's numerics are
close but wrong somewhere; the remaining gap needs a reference bisect
(HF/torch layer-0 comparison). All NaN paths are fixed and explained.

## What works

- Install/pack: `packed-diffusiongemma` format, `model.decoder.*` names,
  `self_conditioning.bin`, `encoder_scalars.bin`, no `draft/` requirement.
- Bootstrap through `decode_warmup` and `composite_state_restore`.
- API: `/v1/chat/completions` (plain + streaming), 256-token bursts emit
  correctly (`maxTokenBatch` raised via `ModelCapabilities.maximumStepTokens`).
- Kernel microbenchmarks all pass (`canvas-kernels` incl. bf16 logits,
  `gelu-saturation` new).
- HD512 attention parity (Q8/Int4/BF16, canvas + causal, scalar oracle).
- Q4 prefill projection parity incl. the shared-expert shape {256, 2816, 2304}.
- TTFT ≈ 6-7 s for prompt+first canvas (48 steps).

## Root causes fixed across sessions

| Failure | Fix |
|---|---|
| Canvas attention not bidirectional (tile offset joined the prefix) | `canvas_tile.h` + `prefill/canvas.metal` take total canvas rows; visibility = committed + rows |
| Canvas RoPE used row-local positions | Denoising positions = `prefixTokens + row` (RuntimeDiffusion.mm) |
| Missing `draft/` dir iterated | `ModelFactory.cpp` skips `DraftKind::Null` |
| Moved `projectionSums` used after move | `headSums` saved pre-move |
| FP32 canvas logits at 256 rows | Canvas logits → bf16 end to end |
| `rope_build_tables` empty draft buffers | `addRopeTables` borrows target buffers |
| Empty final command | no-op `canvas_uniform_noise` into `specTokens` |
| Warmup checks vs canvas commits | diffusion warmup accepts canvas-shaped output |
| `maxTokenBatch=9` vs 256-token burst | `maximumStepTokens` = canvasLength |
| **NaN trunk: `tanh` overflow** | **Metal's exp-based tanh NaNs for |x| ≳ 44 (exp(2x) → inf/inf). Real gate rows reach ±30. `richengine_tanh` clamps to ±10; used by `richengine_gelu_tanh` (geglu_multiply, the routed expert gelu) and every softcap kernel.** This was THE NaN root cause — probes showed sparse NaNs entering at `denseInter` (shared-expert geglu), one per row at a fixed column in the commit prefill, then whole rows through the down projection's dot products |
| **Missing final post-FFN norm** | Upstream: `out = R + post_ffn(post_1(shared) + post_2(routed))`. The runtime added `R + post_1 + post_2` directly. `post_feedforward_layernorm` now packed (new section `post-ffn-norm`) and applied in `addPrefillGemmaLayer` + `addVerifyGemmaLayer` |
| **Routed experts' input norm** | Upstream norms the routed experts with `pre_feedforward_layernorm_2`, not the shared expert's `pre_feedforward_layernorm`. Now packed (`pre-ffn-norm-routed`) and applied (prefill + verify) |
| Affine oracle test didn't compile | `DiffusionGemmaLayout` added to the packed-only constexpr guard |

**The packed format changed** (two extra norm sections per layer): repack
required. The installed package was rebuilt (`install/models/.packed/96a6f04…`).

## Debug-harness findings (for the next session)

- `RICHENGINE_CANVAS_DEBUG=2/3` short-circuits the step; every probe after
  the short-circuit reads never-written buffers (zeros) — the earlier
  "all-zero trunk" observation was this artifact, not model behavior.
- The trunk ping-pongs `hidden[0]/hidden[1]`: after a full trunk, the
  `hidden0` probe holds layer 29's output, not the embedding. Use
  `RICHENGINE_PREFILL_LAYERS=1` for layer-0 probes.
- `RICHENGINE_CANVAS_STEPS_PER_CMD=1` isolates step 48 (self-conditioning
  zero) — cleanest first-step probes.
- `RICHENGINE_GEGLU_OFF` leaves raw shared-up output in `denseInter`: with
  it, the whole trunk runs NaN-free (confirms the geglu/tanh localization).
- `canvas_soft_embed_exact` (full-softmax reference, 192 GMAC/step) exceeds
  the engine's 30 s status cadence across 48 steps — the watchdog kills the
  engine. Do not A/B it through the live server without raising the timeout.

## Open bug: output quality (no NaN, wrong text)

Predictions collapse to single characters/punctuation (',' '-' '+' '.' '9')
with high confidence (raw logits 39-48 pre-softcap). Outputs differ per
prompt (the prompt KV reaches the model) but stay character-level mush.
Verified against upstream (`transformers` v5.11 `modeling_diffusion_gemma.py`):

- trunk structure, norms, router (scale-free RMS × scale × h^-0.5), MoE
  combine, per-expert scale, gelu_pytorch_tanh — all match
- attention: scaling = 1.0 (not 1/sqrt(d)); q/k norm before rope; globals
  p-RoPE 64 pairs of 512 at theta 1e6; k_eq_v V = scaleless RMS of the
  pre-norm K; sliding V = scaleless RMS of V — all match
- encoder pass = decoder weights (the checkpoint's encoder LM layers carry
  only `layer_scalar`) + encoder scalars — prompt prefill and commit prefill
  both pass `encoderLayerScalars()`
- decoder positions = `arange(prefix, prefix+256)`; canvas bidirectional;
  sliding window bounds the prefix only
- schedule: t = tMin + (tMax-tMin)·step/48, steps descend 48→1; entropy
  accept = exclusive prefix sum of sorted entropies ≤ bound; commit = argmax

### Next steps

1. Layer-0 numerical bisect against a torch reference (CPU, per-tensor
   safetensors reads — the full model does not fit in 20 GB): run the
   runtime with `RICHENGINE_PREFILL_LAYERS=1` on a fixed canvas, dump
   `hidden1`, compare with a numpy/torch reimplementation of layer 0 from
   the checkpoint (Q4 error budget ~1e-2).
2. Suspects the bisect will separate: QKV prepare layout (packed QKV row
   order), the int8 KV quantization of the canvas scratch, the soft-embed
   top-64 approximation (A/B via `RICHENGINE_CANVAS_EMBED_HIST=0` gave the
   same mush — both are top-k; try exact offline), the head's tied-Q4
   rounding.
3. `execution-plans` test fails pre-existing ("GGUF MoE prefill plan left
   the device's tile") — GGUF path, unrelated to the packed Gemma target;
   triage separately.

## Perf notes (once quality works)

- ~6 s/canvas ≈ ~43 committed tok/s; kernel micro-wins landed (histogram
  soft-embed ~10-13×, fused row stats 1.7×, hd512 M-split 18-45%,
  softcap/temp fusion, 2 steps/command, prefix-exit, commit-tail skip).
- Bigger levers pending: fused head→stats, whole-canvas single command,
  cross-request canvas batching, speculative prefill.

## Debug harness notes

- `RICHENGINE_CANVAS_DEBUG=1` prints committed ids + logits stats; `=2`/`=3`
  short-circuit the step (probes then read unwritten buffers).
- `RICHENGINE_PREFILL_LAYERS=N` caps trunk layers in prefill.
- `RICHENGINE_GEGLU_OFF`, `RICHENGINE_CANVAS_EMBED_HIST=0`,
  `RICHENGINE_CANVAS_STATS_FUSED=0`, `RICHENGINE_CANVAS_STEPS_PER_CMD`,
  `RICHENGINE_CANVAS_PREFIX_EXIT`, `RICHENGINE_CANVAS_COMMIT_TAIL`.
- Debug swaps make several arena buffers `Shared` so the host can scan for
  NaN (RuntimeDiffusion.mm, env-gated). Revert to `device` before shipping.

## Still left

- Bisect the remaining numerics gap (layer-0 torch reference).
- Verify real text output + EOS/stop behavior.
- Measure real tok/s end-to-end; target ≥300 committed tok/s on M5 Pro.
- Remove debug probes/envs, restore private buffer storage.
- Full regression suite (the GGUF execution-plans failure predates this
  work).
