#pragma once

// Gemma 4 26B-A4B kernel contract, for the integration code that binds these
// kernels. Everything below is a kernel name plus its buffer layout; the
// parameter structs are the existing ABI ones (metal/abi/*.h).
//
// Two per-layer shapes:
//   SLIDING ("h256"): QHeads 16, KVHeads 8 (group 2), HeadDim 256, full
//     rotary of 128 pairs, theta 1e4, SWA 1024, scaleless-RMS V.
//   GLOBAL ("hd512", layers 5, 11, 17, 23, 29): QHeads 16, KVHeads 2
//     (group 8), HeadDim 512, p-RoPE of 64 pairs (rotates dims 0..127),
//     theta 1e6, k_eq_v (no V projection; the V page slot holds the
//     scale-free RMS norm of the pre-norm K). Both carry learned-scale QK
//     norms (bfloat weights) and no query gate; attention score_scale = 1.0.
//
// RoPE tables: rope_build_tables is unchanged and generic. Build one table
// set per layer type — target_dims = 128 (sliding) or 64 (global),
// target_axes = 1 — and bind each layer's set at buffers 3/4 of the QKV
// kernels. The same buffers may hold both tables concatenated per dispatch
// if the host prefers; each kernel reads only its row stride.
//
// ---------------------------------------------------------------------------
// QKV prepare (norms + rope + KV staging). Both take:
//   0 packed QKV rows (bfloat). Row stride:
//       sliding: 16*256 q + 8*256 k + 8*256 v
//       global:  16*512 q + 2*512 k  (no v region)
//   1 q_norm (bfloat[HeadDim])   2 k_norm (bfloat[HeadDim])
//   3 rope_cos  4 rope_sin  (float[row][RotaryPairs])
//   5 queries   6 chunk_keys   7 chunk_values
//   8 FullPrefillParams (prefill) / FullDecodeBatchParams (verify)
// Threadgroup = HeadDim threads; grid = tokens*(QHeads+KHeads) (prefill) or
// (rows*(QHeads+KHeads), lanes) (verify).
//   prefill_attention_qkv_gemma_h256   verify_attention_qkv_gemma_h256
//   prefill_attention_qkv_gemma_hd512  verify_attention_qkv_gemma_hd512
//
// Output gather (no gate) into the out-projection input:
//   prefill_attention_gather_gemma_h256 / _gemma_hd512
//   verify_attention_gather_gemma_h256 / _gemma_hd512
//   verify_attention_gather_table64_gemma_{h256,hd512} and _table16_ variants
//     write the affine/GGUF operand tables alongside.
//   verify_attention_reduce_gather_gemma_{h256,hd512} fuse the split reduce
//     and the gather in one dispatch (reduce threadgroup = HeadDim).
//
// ---------------------------------------------------------------------------
// KV stores, one per cache format (q8/int4/bf16), both shapes:
//   prefill_attention_{q8,int4,bf16}_store_gemma_{h256,hd512}
//   verify_attention_{q8,int4,bf16}_store_gemma_{h256,hd512}
//
// Paged attention splits (threadgroup 256; grid (kv_heads, tiles, splits) for
// prefill, (kv_heads, splits, lanes) for verify):
//   prefill_attention_{q8,int4,bf16}_split_hd512          global, no window
//   prefill_attention_{q8,int4,bf16}_split_swa_h256       sliding, window
//   prefill_attention_{q8,int4,bf16}_split_swa_hd512      global, window
//   verify_attention_{q8,int4,bf16}_split_gemma_{h256,hd512}   no window
//   verify_attention_{q8,int4,bf16}_split_swa_{h256,hd512}     window
// The _swa kernels take one extra trailing buffer — constant uint
// window_tokens (prefill buffer 5, verify buffer 8): 0 = full causal, else
// each row admits only the last window_tokens tokens up to its causal end
// (1024 for the sliding layers). Reduces are window-agnostic:
//   prefill_attention_reduce_gemma_{h256,hd512}
//   verify_attention_reduce_gemma_{h256,hd512}
//   verify_attention_reduce_gather_gemma_{h256,hd512}
// RichPrefillAttentionParams / RichVerifyAttentionParams: set score_scale =
// 1.0f (attention scale 1.0; zero would select 1/sqrt(d)).
//
// ---------------------------------------------------------------------------
// MoE (128 experts, top-8, expert intermediate 704, GeGLU; dense shared
// expert width 2112 is NOT a routed slot — run it as its own GeGLU triple):
//   norm_rms_scaleless(input, output, width)  — scaleless RMS norm of the
//       post-attention residual; then multiply by learned scale *
//       hidden^-0.5 with layer_scalar_scale (or fold into router weights).
//   moe_route_scores_gemma(input, router_weights, scores, MoeRouteParams):
//       bf16 dense router [experts][input_size] -> fp32 scores, row stride
//       256. Grid (rows, experts/8), threadgroup 256.
//   moe_route_select_gemma(scores, per_expert_scale, selected,
//       routing_weights, MoeRouteParams): fp32 softmax over all experts,
//       top_k by score, winners renormalized, times per_expert_scale[e].
//       Routes per row = top_k (no shared slot); dispatch moe_group_routes
//       with MoeGroupParams.shared = 0.
//   geglu_multiply(gate, up, output, count): gelu_pytorch_tanh(gate) * up.
//   Decode experts: moe_expert_gate_up_q4_m8_gelu (or _n128_sg4) then the
//       unchanged moe_expert_down_q4_m8*.
//   Prefill experts: unchanged gate/down N256 passes; replace the up pass
//       with prefill_moe_expert_q4_n256_up_gelu_indirect_m32.
//   moe_combine applies routing_weights unchanged.
//
// ---------------------------------------------------------------------------
// Misc:
//   embedding_q4_h2816          Q4 gather, hidden 2816.
//   embedding_q4_scaled_h2816   same plus constant float embedding_scale at
//                               buffer 6 (bind sqrt(2816)).
//   layer_scalar_scale(values, scale, count): values[i] *= scale in place —
//   the per-layer learned scalar on the layer output.
//   decode_logit_softcap(logits, cap, count): logits[i] = cap *
//   tanh(logits[i]/cap), in place (cap 30), before sampling. Strictly
//   monotonic — the fused decode_head_argmax path needs no change.

// ===========================================================================
// DiffusionGemma CANVAS mode (256 canvas positions, vocabulary 262144,
// hidden 2816). One denoising step adds a 256-row bidirectional block after
// the read-only encoder prefix; the block is re-randomized each step and its
// K/V never enters the request's paged cache — it lives in a scratch extent
// of the SAME pool format as the prefix (q8/int4/bf16), whose 8 page entries
// the host appends to the dispatch's page table. RichKvPage entries are
// extent GPU addresses, so the canvas extent needs no kernel-side distinction.
//
// Attention (per decoder layer):
//   1. prefill_attention_qkv_gemma_{h256,hd512} — unchanged; FullPrefillParams
//      .tokens = 256, .stride = the chunk staging stride. Rope rows are the
//      canvas positions' rope rows (host picks the row offset).
//   2. prefill_attention_{q8,int4,bf16}_store_gemma_{h256,hd512} — unchanged;
//      RichChunkedPrefillParams.committed_tokens = prefix_tokens, so rows
//      land in the canvas pages the combined page table names. Nothing is
//      committed: the next step rewrites the same scratch.
//   3. prefill_attention_{q8,int4,bf16}_split_canvas[_swa]_{h256,hd512} —
//      the canvas splits (common/canvas_tile.h). Same signature and grid
//      (kv_heads, tiles, splits; tg 256) as the _swa AR splits; buffer 5 is
//      window_tokens on every variant (non-_swa_ ignore it). Params:
//      committed_tokens = prefix length, rows = 256, split_count partitions
//      prefix+canvas pages. Every canvas row admits [window_begin, visible)
//      where window_begin = max(0, prefix - window_tokens): sliding layers
//      see the last window_tokens prefix tokens plus the whole canvas
//      (bidirectional), globals see everything.
//   4. prefill_attention_reduce_gemma_{h256,hd512} + the existing gather —
//      unchanged; the reduce's written-split accounting already covers the
//      appended canvas pages.
//
// Logit tail (per step), on the fp32 [256 x 262144] lm-head output:
//   decode_logit_softcap(logits, cap=30, count=256*262144) — count is
//     arbitrary; reuse unchanged.
//   canvas_logits_scale(logits, scale=1/temperature, count) — in place.
//   canvas_row_stats(logits, sampled, argmax, entropy,
//       CanvasRowStatsParams{vocabulary, seed}) — grid (256) x tg 256.
//     Per row: multinomial draw of softmax(logits), argmax (lowest index on
//     ties), entropy. seed decorrelates steps (hash RNG, no device state).
//   canvas_entropy_accept(entropy, sampled, argmax_prev, canvas_out,
//       argmax_out, stats[2], CanvasAcceptParams{entropy_bound, vocabulary,
//       seed}, argmax_cur) — one tg of 256. Bitonic-sorts entropies
//     ascending; position i is accepted iff the exclusive prefix sum of the
//     sorted order at i's rank <= entropy_bound (cheapest-first budget).
//     canvas_out[i] = sampled[i] if accepted else uniform-renoise.
//     stats = {mean_entropy, all_argmax_equal_to_previous}; argmax_out is
//     this step's argmax — feed it back as argmax_prev next step (early
//     exit when the flag stays 1).
//   canvas_uniform_noise(tokens, vocabulary, seed) — init/reset fill.
//
// Soft embeddings (the step's self-conditioning signal):
//   canvas_soft_embed_topk(logits, weights, scales, biases, out[256x2816],
//       CanvasSoftEmbedParams{vocabulary, top_k}, embedding_scale) —
//     production path; bisects a threshold keeping ~top_k vocab rows and
//     renormalizes the kept mass.
//   canvas_soft_embed_exact — same buffers (top_k ignored); the reference:
//     softmax(logits) @ dequant(Q4 embedding) * sqrt(2816), 192 G MACs.
//   Embedding operands are the packed Q4 planes of embedding_q4_h2816
//   (uchar codes [vocab][1408], bf16 scales/biases [vocab][44]).
//   NOTE: upstream mlx-vlm is EXACT (softmax @ embed_tokens, or
//   quantized_matmul over the quantized embedding) — it does NOT top-k.
//   topk is a speed approximation; keep `exact` for A/B checks. An
//   integrator may instead pass precomputed self_conditioning_embeddings
//   [256,2816] and skip both.
//
// Self-conditioning MLP, hidden 2816, inter 2112:
//   normed = norm_rms(soft_embeds, pre_norm_weight)          (existing)
//   g, u   = gate_proj(normed), up_proj(normed)              (existing GEMMs)
//   act    = geglu_multiply(g, u)                            (existing)
//   sc     = down_proj(act)                                  (existing GEMM)
//   inputs_embeds = canvas_self_condition(inputs_embeds, sc, out, 2816)
//     — fused residual add + scaleless RMS norm.
// inputs_embeds for the step = embedding_q4_scaled_h2816(canvas_tokens)
// before self-conditioning adds in.
