#include "metal/kernels/common/paged_attention_fp8_tile.h"

// FP8 E4M3 verify splits: the shared page loop over native fp8 pages, with
// fp16-staged queries and probabilities (matmul2d admits no bf16 operand
// against fp8). The row merge is the shared reduce the other tiers use.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__)

#define PAGED_VERIFY_FP8_SPLIT(Name, Heads, Group, HeadDim)                    \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table0 [[buffer(3)]],                    \
      device const RichKvPage *page_table1 [[buffer(4)]],                    \
      device const RichKvPage *page_table2 [[buffer(5)]],                    \
      device const RichKvPage *page_table3 [[buffer(6)]],                    \
      constant RichVerifyAttentionParams *params [[buffer(7)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint M = Group * RICHENGINE_TARGET_VERIFY_ROWS;                      \
    constexpr uint MP = M / 2;                                                 \
    constexpr uint N = RichKvPageTokens;                                     \
    alignas(16) threadgroup float scores[MP * N];                              \
    alignas(16) threadgroup half probabilities[MP * N];                        \
    alignas(16) threadgroup half staged_queries[MP * HeadDim];                  \
    threadgroup float row_max[MP];                                             \
    threadgroup float row_sum[MP];                                             \
    threadgroup float previous_scale[MP];                                      \
    threadgroup atomic_uint rescale;                                           \
    constexpr uint D = HeadDim;                                                \
    constexpr ulong group_stride =                                             \
        ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * Group * D;                         \
    uint kv_head = group.x;                                                    \
    uint split = group.y;                                                      \
    uint batch = group.z;                                                      \
    constant RichVerifyAttentionParams &lane_params = params[batch];         \
    if (!richengine_verify_attention_contract_valid(lane_params) ||                \
        kv_head >= Heads || split >= lane_params.split_count)                  \
      return;                                                                  \
    device const RichKvPage *page_table =                                    \
        batch == 0 ? page_table0                                               \
                   : (batch == 1 ? page_table1                                 \
                                 : (batch == 2 ? page_table2 : page_table3));  \
    richengine_paged_attention_tile_fp8<Heads, Group, RICHENGINE_TARGET_VERIFY_ROWS,   \
                                    HeadDim>(                                  \
        queries + (ulong(batch) * Heads + kv_head) * group_stride,             \
        page_table, lane_params.kv, kv_head, lane_params.committed_tokens,     \
        RICHENGINE_TARGET_VERIFY_ROWS, lane_params.split_count, split, partials,   \
        statistics,                                                            \
        (ulong(batch) * Heads + kv_head) * lane_params.slot_splits + split,    \
        scores, probabilities, row_max, row_sum, previous_scale, &rescale,     \
        staged_queries, thread_index, lane_params.score_scale);                                         \
  }

PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split, 4, 6, 256)
PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split_kv4_g4, 4, 4, 256)
PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split_kv2_g8, 2, 8, 256)
PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split_hd128, 2, 8, 128)
PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split_hd64, 8, 4, 64)
// Granite 3B's group-5 tile is not a valid fp8 matmul shape (M=20).
PAGED_VERIFY_FP8_SPLIT(verify_attention_fp8_split_k8q4d128, 8, 4, 128)
#undef PAGED_VERIFY_FP8_SPLIT

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__
