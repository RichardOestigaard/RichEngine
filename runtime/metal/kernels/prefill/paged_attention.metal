#include "metal/kernels/common/paged_attention_tile.h"

// Prefill runs the shared device-operand page loop
// (richengine_paged_attention_tile) over eight query rows of one KV head.

// Prefill split: KV heads vary first, then query tiles, then history splits.
template <uint KVHeads, uint QueryHeadsPerKVHead, typename CacheElement,
          uint HeadDim = RichKvHeadDimension, uint MPasses = 1>
inline void richengine_prefill_attention_split_phase(
    device bfloat *queries, device float *partials,
    device float *statistics, device const RichKvPage *page_table,
    constant RichPrefillAttentionParams &params,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint3 group, uint thread_index) {
  constexpr uint D = HeadDim;
  uint kv_head = group.x;
  uint split = group.z;
  uint tile = group.y;
  uint tile_start = tile * RichPrefillTileRows;
  if (!richengine_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || split >= params.split_count ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(RichPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  ulong slot = (ulong(tile) * KVHeads + kv_head) * params.split_count + split;
  richengine_paged_attention_tile<KVHeads, QueryHeadsPerKVHead,
                              RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, CacheElement,
                              HeadDim, MPasses>(
      queries + tile_offset, page_table, params.kv, kv_head,
      params.committed_tokens + tile_start, active_rows, params.split_count, split,
      partials, statistics, slot, nullptr, params.score_scale, scores,
      probabilities, row_max, row_sum, previous_scale, rescale,
      thread_index);
}

template <uint KVHeads, uint QueryHeadsPerKVHead,
          uint HeadDim = RichKvHeadDimension>
inline void richengine_prefill_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant RichPrefillAttentionParams &params, uint3 group,
    uint thread_index, threadgroup float *weights,
    threadgroup float *group_values) {
  constexpr ushort M = RICHENGINE_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * RichPrefillTileRows;
  if (!richengine_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  uint active_rows = min(RichPrefillTileRows, params.rows - tile_start);
  ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *
                      QueryHeadsPerKVHead * D;
  richengine_attention_reduce_row<QueryHeadsPerKVHead,
                              RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, HeadDim>(
      partials, statistics, output + tile_offset,
      params.committed_tokens + tile_start, active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index, weights, group_values);
}

#define PAGED_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table [[buffer(3)]],                     \
      constant RichPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = RichKvPageTokens;                                     \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[2 * M * N];                   \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    richengine_prefill_attention_split_phase<Heads, Group, CacheElement>(          \
        queries, partials, statistics, page_table, params, scores,             \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        group, thread_index);                                                  \
  }

// Packed-INT4 pages feed matmul2d as native int4b operands; no staging.
PAGED_PREFILL_SPLIT(prefill_attention_q8_split, 4, 6, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv4_g4, 4, 4, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_q8_split_kv2_g8, 2, 8, int8_t)
PAGED_PREFILL_SPLIT(prefill_attention_int4_split, 4, 6, RichKvPacked4)
PAGED_PREFILL_SPLIT(prefill_attention_int4_split_kv4_g4, 4, 4, RichKvPacked4)
PAGED_PREFILL_SPLIT(prefill_attention_int4_split_kv2_g8, 2, 8, RichKvPacked4)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split, 4, 6, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv4_g4, 4, 4, bfloat)
PAGED_PREFILL_SPLIT(prefill_attention_bf16_split_kv2_g8, 2, 8, bfloat)
#undef PAGED_PREFILL_SPLIT

kernel void prefill_attention_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichPrefillMaximumSplits];
  threadgroup float group_values[8];
  richengine_prefill_attention_reduce_phase<4, 6>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

kernel void prefill_attention_reduce_kv4_g4(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichPrefillMaximumSplits];
  threadgroup float group_values[8];
  richengine_prefill_attention_reduce_phase<4, 4>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

kernel void prefill_attention_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichPrefillMaximumSplits];
  threadgroup float group_values[8];
  richengine_prefill_attention_reduce_phase<2, 8>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

// Head-dimension variants: the dense target's KV2/Group8 pages of 128
// dimensions (_hd128) and LFM2's KV8/Group4 pages of 64 (_hd64). The reduce
// threadgroup is the head dimension.
#define PREFILL_ATTENTION_SPLIT_HD(Name, Heads, Group, CacheElement, HeadDim) \
  kernel void Name(                                                         \
      device bfloat *queries [[buffer(0)]],                                 \
      device float *partials [[buffer(1)]],                                 \
      device float *statistics [[buffer(2)]],                               \
      device const RichKvPage *page_table [[buffer(3)]],                  \
      constant RichPrefillAttentionParams &params [[buffer(4)]],          \
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
    richengine_prefill_attention_split_phase<Heads, Group, CacheElement,        \
                                         HeadDim>(                          \
        queries, partials, statistics, page_table, params, scores,          \
        probabilities, row_max, row_sum, previous_scale, &rescale,          \
        group, thread_index);                                               \
  }
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_q8_split_hd128, 2, 8, int8_t, 128)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_int4_split_hd128, 2, 8, RichKvPacked4, 128)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_bf16_split_hd128, 2, 8, bfloat, 128)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_q8_split_hd64, 8, 4, int8_t, 64)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_int4_split_hd64, 8, 4, RichKvPacked4, 64)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_bf16_split_hd64, 8, 4, bfloat, 64)
// Granite: 3B's KV8/Group5 of 64 and 8B's KV8/Group4 of 128.
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_q8_split_k8q5d64, 8, 5, int8_t, 64)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_int4_split_k8q5d64, 8, 5, RichKvPacked4, 64)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_bf16_split_k8q5d64, 8, 5, bfloat, 64)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_q8_split_k8q4d128, 8, 4, int8_t, 128)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_int4_split_k8q4d128, 8, 4, RichKvPacked4, 128)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_bf16_split_k8q4d128, 8, 4, bfloat, 128)
// Gemma 4's global layers: KV2 group-8 pages of 512 dimensions.
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_q8_split_hd512, 2, 8, int8_t, 512)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_int4_split_hd512, 2, 8, RichKvPacked4, 512)
PREFILL_ATTENTION_SPLIT_HD(prefill_attention_bf16_split_hd512, 2, 8, bfloat, 512)
#undef PREFILL_ATTENTION_SPLIT_HD

// The _m2 hd512 variants run the tile in two fused-row passes (MP = M/2 =
// 32), halving the PV accumulator's ~128 fp32/thread — the M5 per-core
// register/SRAM pool — plus halving score/probability scratch. Partials,
// statistics, grid (kv heads, tiles, splits), threadgroup count (256) and
// the shared gemma_hd512 reduce are the plain variants'; the host only swaps
// the pipeline name.
#define PREFILL_ATTENTION_SPLIT_HD_M2(Name, Heads, Group, CacheElement, HeadDim) \
  kernel void Name(                                                         \
      device bfloat *queries [[buffer(0)]],                                 \
      device float *partials [[buffer(1)]],                                 \
      device float *statistics [[buffer(2)]],                               \
      device const RichKvPage *page_table [[buffer(3)]],                  \
      constant RichPrefillAttentionParams &params [[buffer(4)]],          \
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
    richengine_prefill_attention_split_phase<Heads, Group, CacheElement,        \
                                         HeadDim, 2>(                       \
        queries, partials, statistics, page_table, params, scores,          \
        probabilities, row_max, row_sum, previous_scale, &rescale,          \
        group, thread_index);                                               \
  }
PREFILL_ATTENTION_SPLIT_HD_M2(prefill_attention_q8_split_hd512_m2, 2, 8, int8_t, 512)
PREFILL_ATTENTION_SPLIT_HD_M2(prefill_attention_int4_split_hd512_m2, 2, 8, RichKvPacked4, 512)
PREFILL_ATTENTION_SPLIT_HD_M2(prefill_attention_bf16_split_hd512_m2, 2, 8, bfloat, 512)
#undef PREFILL_ATTENTION_SPLIT_HD_M2

#define PREFILL_ATTENTION_REDUCE_HD(Name, KVHeads, Group, HeadDim)            \
  kernel void Name(                                                         \
      device const float *partials [[buffer(0)]],                           \
      device const float *statistics [[buffer(1)]],                         \
      device bfloat *output [[buffer(2)]],                                  \
      constant RichPrefillAttentionParams &params [[buffer(3)]],          \
      uint3 group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]]) {                  \
    threadgroup float weights[RichPrefillMaximumSplits];                  \
    threadgroup float group_values[8];                                      \
    richengine_prefill_attention_reduce_phase<KVHeads, Group, HeadDim>(         \
        partials, statistics, output, params, group, thread_index, weights, \
        group_values);                                                      \
  }
PREFILL_ATTENTION_REDUCE_HD(prefill_attention_reduce_hd128, 2, 8, 128)
PREFILL_ATTENTION_REDUCE_HD(prefill_attention_reduce_hd64, 8, 4, 64)
PREFILL_ATTENTION_REDUCE_HD(prefill_attention_reduce_k8q5d64, 8, 5, 64)
PREFILL_ATTENTION_REDUCE_HD(prefill_attention_reduce_k8q4d128, 8, 4, 128)
#undef PREFILL_ATTENTION_REDUCE_HD

// ---------------------------------------------------------------------------
// Gemma 4 26B-A4B: the global layers' KV2 group-8 pages of 512 dimensions,
// and sliding-window (_swa) variants taking a per-dispatch window in tokens
// (buffer 5, 0 = full causal). The _swa entries also cover the sliding
// layers' KV8 group-2 pages of 256; the reduce pass needs no window — the
// splits' statistics already carry the masked pages as zero weight.
#define PREFILL_ATTENTION_SPLIT_SWA(Name, Heads, Group, CacheElement, HeadDim) \
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
    uint tile_start = tile * RichPrefillTileRows;                           \
    if (!richengine_prefill_attention_contract_valid(params) ||             \
        kv_head >= Heads || split >= params.split_count ||                  \
        tile_start >= params.rows)                                          \
      return;                                                               \
    uint active_rows = min(RichPrefillTileRows, params.rows - tile_start);  \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) \
                        * Group * D;                                        \
    ulong slot = (ulong(tile) * Heads + kv_head) * params.split_count +     \
                 split;                                                     \
    richengine_paged_attention_tile<Heads, Group,                             \
        RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, CacheElement, HeadDim>(         \
        queries + tile_offset, page_table, params.kv, kv_head,              \
        params.committed_tokens + tile_start, active_rows,                  \
        params.split_count, split, partials, statistics, slot, nullptr,     \
        params.score_scale, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, thread_index, window_tokens);             \
  }
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_q8_split_swa_h256, 8, 2, int8_t, 256)
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_int4_split_swa_h256, 8, 2, RichKvPacked4, 256)
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_bf16_split_swa_h256, 8, 2, bfloat, 256)
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_q8_split_swa_hd512, 2, 8, int8_t, 512)
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_int4_split_swa_hd512, 2, 8, RichKvPacked4, 512)
PREFILL_ATTENTION_SPLIT_SWA(prefill_attention_bf16_split_swa_hd512, 2, 8, bfloat, 512)
#undef PREFILL_ATTENTION_SPLIT_SWA

// Windowed _m2 hd512 variants: the same two-pass tile as _hd512_m2 plus the
// buffer(5) window. Same dispatch contract as the _swa_ variants.
#define PREFILL_ATTENTION_SPLIT_SWA_M2(Name, Heads, Group, CacheElement, HeadDim) \
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
    uint tile_start = tile * RichPrefillTileRows;                           \
    if (!richengine_prefill_attention_contract_valid(params) ||             \
        kv_head >= Heads || split >= params.split_count ||                  \
        tile_start >= params.rows)                                          \
      return;                                                               \
    uint active_rows = min(RichPrefillTileRows, params.rows - tile_start);  \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) \
                        * Group * D;                                        \
    ulong slot = (ulong(tile) * Heads + kv_head) * params.split_count +     \
                 split;                                                     \
    richengine_paged_attention_tile<Heads, Group,                             \
        RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, CacheElement, HeadDim, 2>(    \
        queries + tile_offset, page_table, params.kv, kv_head,              \
        params.committed_tokens + tile_start, active_rows,                  \
        params.split_count, split, partials, statistics, slot, nullptr,     \
        params.score_scale, scores, probabilities, row_max, row_sum,        \
        previous_scale, &rescale, thread_index, window_tokens);             \
  }
PREFILL_ATTENTION_SPLIT_SWA_M2(prefill_attention_q8_split_swa_hd512_m2, 2, 8, int8_t, 512)
PREFILL_ATTENTION_SPLIT_SWA_M2(prefill_attention_int4_split_swa_hd512_m2, 2, 8, RichKvPacked4, 512)
PREFILL_ATTENTION_SPLIT_SWA_M2(prefill_attention_bf16_split_swa_hd512_m2, 2, 8, bfloat, 512)
#undef PREFILL_ATTENTION_SPLIT_SWA_M2

// Gemma reduces: h256's fused rows are 8 * 2 = 16, hd512's 8 * 8 = 64, with a
// 512-thread reduce threadgroup whose simdgroup maxima span 16 groups.
kernel void prefill_attention_reduce_gemma_h256(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichPrefillMaximumSplits];
  threadgroup float group_values[8];
  richengine_prefill_attention_reduce_phase<8, 2, 256>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

kernel void prefill_attention_reduce_gemma_hd512(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichPrefillAttentionParams &params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichPrefillMaximumSplits];
  threadgroup float group_values[16];
  richengine_prefill_attention_reduce_phase<2, 8, 512>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}
