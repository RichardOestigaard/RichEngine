#include "metal/kernels/common/paged_attention_fp8_tile.h"

// FP8 E4M3 prefill splits over eight query rows of one KV head: the same
// page loop and staged fp16 operands as the verify entries, sharing the
// production reduce.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__)

#define PAGED_PREFILL_FP8_SPLIT(Name, Heads, Group, HeadDim)                   \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table [[buffer(3)]],                     \
      constant RichPrefillAttentionParams &params [[buffer(4)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint MP = M / 2;                                                 \
    constexpr uint N = RichKvPageTokens;                                     \
    constexpr uint D = HeadDim;                                                \
    alignas(16) threadgroup float scores[MP * N];                              \
    alignas(16) threadgroup half probabilities[MP * N];                        \
    alignas(16) threadgroup half staged_queries[MP * D];                        \
    threadgroup float row_max[MP];                                             \
    threadgroup float row_sum[MP];                                             \
    threadgroup float previous_scale[MP];                                      \
    threadgroup atomic_uint rescale;                                           \
    uint kv_head = group.x;                                                    \
    uint split = group.z;                                                      \
    uint tile = group.y;                                                       \
    uint tile_start = tile * RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;               \
    if (!richengine_prefill_attention_contract_valid(params) ||                    \
        kv_head >= Heads || split >= params.split_count ||                     \
        tile_start >= params.rows)                                             \
      return;                                                                  \
    uint active_rows =                                                         \
        min(RICHENGINE_PREFILL_ATTENTION_TILE_ROWS, params.rows - tile_start);     \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *  \
                        Group * D;                                             \
    ulong slot =                                                               \
        (ulong(tile) * Heads + kv_head) * params.split_count + split;          \
    richengine_paged_attention_tile_fp8<Heads, Group,                              \
                                    RICHENGINE_PREFILL_ATTENTION_TILE_ROWS,        \
                                    HeadDim>(                                  \
        queries + tile_offset, page_table, params.kv, kv_head,                 \
        params.committed_tokens + tile_start, active_rows,                     \
        params.split_count, split, partials, statistics, slot, scores,         \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        staged_queries, thread_index, params.score_scale);                                         \
  }

PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split, 4, 6, 256)
PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split_kv4_g4, 4, 4, 256)
PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split_kv2_g8, 2, 8, 256)
PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split_hd128, 2, 8, 128)
PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split_hd64, 8, 4, 64)
// Granite 3B's group-5 tile is not a valid fp8 matmul shape.
PAGED_PREFILL_FP8_SPLIT(prefill_attention_fp8_split_k8q4d128, 8, 4, 128)
#undef PAGED_PREFILL_FP8_SPLIT

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__
