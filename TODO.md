# TODO

## Explore

1. GPU-driven dispatch — repeatedly surfaced, still the biggest untapped lever (compute commands encoding the next layer's dispatches on-GPU). Architecture project.
2. Q4 packed-tile epilogue loads — `q4_mpp_tiles.h` reads `scales[]`/`biases[]` per element per group straight from device memory. Staging the group's scale+bias vectors into threadgroup once per group (~128B) batches those scattered loads. Small, measurable.
3. Activation-quantized prefill (fp8e4m3 A) — the only remaining NA-throughput play; gated on activation-accuracy, not kernel work.
4. fp8-KV fp16 Q stage — quality fix for the landed tier (~0.97 cosine at long history), needs ~2KB threadgroup budget or a device-side stage.
5. Planner tier sweep — `kMxfp4Tiers`/`stagedTiers` were tuned on one shape; after the uint4b/int2b formats land, re-sweep N×K space. Mechanical.
6. `tensor_offset`/`replaceSliceOrigin` for KV page chunk rebasing — modest.
7. `atomic_wait`/`notify` — only if a real poll exists; needs a grep.
8. mxfp4p for prefill chunks ≤32 rows + packed norm for prefill — `packsPrefill` is env-gated (`SPLASH_GGUF_PACKED_ON`, Linear.cpp:535-548); decode tiles measured ~2-2.5× the staged tile at every height. Pair with emitting `LinearInput::Packed` from prefill norms (`norm_rms_packed_decode` is already row-generic; `packedInput_` gating at Linear.cpp:589) to kill the `gguf_pack_half` dispatch per chunk. Risk: bitwise identity vs the 128-row tile test unverified; if not bitwise, relax the test or keep gated.
9. `q4_prefill_write_output_sums` extension → drop `prefill_linear_q4_sums32` — the up kernel already emits consumer input sums post-store (prefill/linear_q4.metal:141-160); extend to Plain/Residual prefill kernels and mixer-output producers so `addPrefillSums` (QwenTarget.cpp:300) goes away. ~21 MB/layer saved at 2048×5120. Needs the existing `mem_device` barrier pattern.
10. KV store → `verify_attention_qkv` merge — `verify_attention_*_store` is a state commit (can't vanish), but its threadgroups could write the paged slot inside the qkv kernel, which already has the chunk rows + per-head-dim reduction in flight. ~1 dispatch/attention layer. Per-format variants (bf16/q8/int4/fp8) needed. Moderate complexity.
11. B-operand static slicing on non-blockwise tensors — A operands now sliced in the mxfp4 tiles; packed-format B (uint4b/int8/int2b in `gguf_decode_packed_tile`) could follow for its bounds-check elimination. Do NOT slice blockwise (E8M0 scale-plane) tensors — `slice<E0,E1>` mis-slices the scale plane (verified: gguf_projection fails, values off by ~2^9). Verify the non-blockwise case with the same test before keeping.
