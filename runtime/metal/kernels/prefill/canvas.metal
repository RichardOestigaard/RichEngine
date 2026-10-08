#include "metal/kernels/common/canvas_tile.h"

// DiffusionGemma canvas-mode attention splits. One canvas step is 256 query
// rows attending bidirectionally over [encoder prefix pages | canvas scratch
// pages]: the host builds the request page table with the prefix entries
// followed by the entries of the canvas scratch extent (8 pages for a
// 256-token canvas), sets RichPrefillAttentionParams.committed_tokens to the
// prefix length, rows = canvas rows, chunk_stride to the queries staging
// stride, and split_count as usual. The canvas rows' K/V reach those scratch
// pages through the unchanged prefill_attention_*_store_gemma_* kernels —
// the canvas extent holds the same pool format (q8/int4/bf16) as the rest of
// the table, so one CacheElement serves both sources.
//
// Grid is (kv_heads, tiles, splits), threadgroup 256, as the AR prefill
// splits; canvas uses its own reducer because every tile sees the full canvas.
//
// _canvas_swa_ variants take constant uint window_tokens at buffer 5 (1024
// for the sliding layers): the window bounds only the paged prefix — every
// canvas position is always admitted. The non-swa variants take no window.

#define PREFILL_ATTENTION_SPLIT_CANVAS(Name, Heads, Group, CacheElement, HeadDim, Windowed) \
  kernel void Name(                                                         \
      device bfloat *queries [[buffer(0)]],                                 \
      device float *partials [[buffer(1)]],                                 \
      device float *statistics [[buffer(2)]],                               \
      device const RichKvPage *page_table [[buffer(3)]],                  \
      constant RichPrefillAttentionParams &params [[buffer(4)]],          \
      constant uint &window_tokens [[buffer(5)]],                           \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]]) {                  \
    constexpr uint M = Group * RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;          \
    constexpr uint N = RichKvPageTokens;                                  \
    alignas(16) threadgroup float scores[M * N];                            \
    alignas(16) threadgroup bfloat probabilities[2 * M * N];                \
    threadgroup float row_max[M];                                           \
    threadgroup float row_sum[M];                                           \
    threadgroup float previous_scale[M];                                    \
    threadgroup atomic_uint rescale;                                        \
    constexpr uint D = HeadDim;                                             \
    uint kv_head = group.x;                                                 \
    uint split = group.z;                                                   \
    uint tile = group.y;                                                    \
    uint tile_start = tile * RichCanvasTileRows;                            \
    if (!richengine_prefill_attention_contract_valid(params) ||             \
        kv_head >= Heads || split >= params.split_count ||                  \
        tile_start >= params.rows)                                          \
      return;                                                               \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) \
                        * Group * D;                                        \
    ulong slot = (ulong(tile) * Heads + kv_head) * params.split_count +     \
                 split;                                                     \
    richengine_canvas_attention_tile<Heads, Group,                          \
        RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, CacheElement, HeadDim>(         \
        queries + tile_offset, page_table, params.kv, kv_head,              \
        params.committed_tokens, params.rows,                               \
        params.split_count, split, partials, statistics, slot,              \
        params.score_scale, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, thread_index,                             \
        Windowed ? window_tokens : 0u);                                     \
  }

// Buffer 5 (window_tokens) is part of every canvas signature, windowed or
// not, so the host always binds the same constant.

// Sliding layers: KV8 group-2, head dim 256, windowed.
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_q8_split_canvas_swa_h256, 8, 2, int8_t, 256, true)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_int4_split_canvas_swa_h256, 8, 2, RichKvPacked4, 256, true)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_bf16_split_canvas_swa_h256, 8, 2, bfloat, 256, true)
// Global layers: KV2 group-8, head dim 512, full visibility (window ignored).
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_q8_split_canvas_hd512, 2, 8, int8_t, 512, false)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_int4_split_canvas_hd512, 2, 8, RichKvPacked4, 512, false)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_bf16_split_canvas_hd512, 2, 8, bfloat, 512, false)
// Windowed hd512 canvas variants for completeness.
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_q8_split_canvas_swa_hd512, 2, 8, int8_t, 512, true)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_int4_split_canvas_swa_hd512, 2, 8, RichKvPacked4, 512, true)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_bf16_split_canvas_swa_hd512, 2, 8, bfloat, 512, true)
// Non-windowed h256 (full prefix) canvas variants.
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_q8_split_canvas_h256, 8, 2, int8_t, 256, false)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_int4_split_canvas_h256, 8, 2, RichKvPacked4, 256, false)
PREFILL_ATTENTION_SPLIT_CANVAS(prefill_attention_bf16_split_canvas_h256, 8, 2, bfloat, 256, false)
#undef PREFILL_ATTENTION_SPLIT_CANVAS

// The _m2 hd512 canvas variants run the canvas tile in two fused-row passes
// (MP = M/2 = 32), halving the PV accumulator's ~128 fp32/thread — the M5
// per-core register/SRAM pool — plus the score/probability scratch. Buffer
// layout, grid, threadgroup count and the gemma_hd512 reduce are unchanged.
#define PREFILL_ATTENTION_SPLIT_CANVAS_M2(Name, Heads, Group, CacheElement, HeadDim, Windowed) \
  kernel void Name(                                                         \
      device bfloat *queries [[buffer(0)]],                                 \
      device float *partials [[buffer(1)]],                                 \
      device float *statistics [[buffer(2)]],                               \
      device const RichKvPage *page_table [[buffer(3)]],                  \
      constant RichPrefillAttentionParams &params [[buffer(4)]],          \
      constant uint &window_tokens [[buffer(5)]],                           \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]]) {                  \
    constexpr uint M = Group * RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;          \
    constexpr uint MP = M / 2;                                              \
    constexpr uint N = RichKvPageTokens;                                  \
    alignas(16) threadgroup float scores[MP * N];                           \
    alignas(16) threadgroup bfloat probabilities[2 * MP * N];               \
    threadgroup float row_max[MP];                                          \
    threadgroup float row_sum[MP];                                          \
    threadgroup float previous_scale[MP];                                   \
    threadgroup atomic_uint rescale;                                        \
    constexpr uint D = HeadDim;                                             \
    uint kv_head = group.x;                                                 \
    uint split = group.z;                                                   \
    uint tile = group.y;                                                    \
    uint tile_start = tile * RichCanvasTileRows;                            \
    if (!richengine_prefill_attention_contract_valid(params) ||             \
        kv_head >= Heads || split >= params.split_count ||                  \
        tile_start >= params.rows)                                          \
      return;                                                               \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) \
                        * Group * D;                                        \
    ulong slot = (ulong(tile) * Heads + kv_head) * params.split_count +     \
                 split;                                                     \
    richengine_canvas_attention_tile<Heads, Group,                          \
        RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, CacheElement, HeadDim, 2>(    \
        queries + tile_offset, page_table, params.kv, kv_head,              \
        params.committed_tokens, params.rows,                               \
        params.split_count, split, partials, statistics, slot,              \
        params.score_scale, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, thread_index,                             \
        Windowed ? window_tokens : 0u);                                     \
  }
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_q8_split_canvas_hd512_m2, 2, 8, int8_t, 512, false)
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_int4_split_canvas_hd512_m2, 2, 8, RichKvPacked4, 512, false)
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_bf16_split_canvas_hd512_m2, 2, 8, bfloat, 512, false)
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_q8_split_canvas_swa_hd512_m2, 2, 8, int8_t, 512, true)
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_int4_split_canvas_swa_hd512_m2, 2, 8, RichKvPacked4, 512, true)
PREFILL_ATTENTION_SPLIT_CANVAS_M2(prefill_attention_bf16_split_canvas_swa_hd512_m2, 2, 8, bfloat, 512, true)
#undef PREFILL_ATTENTION_SPLIT_CANVAS_M2

#define PREFILL_ATTENTION_REDUCE_CANVAS(Name, KVHeads, Group, HeadDim, Groups) \
  kernel void Name(                                                         \
      device const float *partials [[buffer(0)]],                           \
      device const float *statistics [[buffer(1)]],                         \
      device bfloat *output [[buffer(2)]],                                  \
      constant RichPrefillAttentionParams &params [[buffer(3)]],            \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]]) {                  \
    threadgroup float weights[RichCanvasMaximumSplits];                     \
    threadgroup float group_values[Groups];                                 \
    richengine_canvas_attention_reduce_phase<KVHeads, Group, HeadDim>(      \
        partials, statistics, output, params, group, thread_index, weights,  \
        group_values);                                                      \
  }
PREFILL_ATTENTION_REDUCE_CANVAS(prefill_attention_reduce_canvas_gemma_h256,
                                8, 2, 256, 8)
PREFILL_ATTENTION_REDUCE_CANVAS(prefill_attention_reduce_canvas_gemma_hd512,
                                2, 8, 512, 16)
#undef PREFILL_ATTENTION_REDUCE_CANVAS
