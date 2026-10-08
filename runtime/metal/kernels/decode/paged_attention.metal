#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/activation.h"

// Verify tiles process one lane's eight rows per KV head and history split.
// Verify and prefill share the device-operand page loop in paged_attention_tile.h.

// The tile one verify threadgroup owns: group.x is the KV head, group.y the
// history split and group.z the lane, whose parameters select the page table
// and the query tile. A threadgroup past its lane's split count, or whose
// lane fails the contract, is inactive and does nothing.
struct RichVerifyTile {
  device bfloat *queries;
  device const RichKvPage *page_table;
  ulong slot;
  uint kv_head;
  uint split;
  uint splits;
  uint committed_tokens;
  uint active_rows;
  bool active;
};

template <uint KVHeads, uint QueryHeadsPerKVHead,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline RichVerifyTile richengine_verify_attention_tile_at(
    device bfloat *queries, device const RichKvPage *page_table0,
    device const RichKvPage *page_table1, device const RichKvPage *page_table2,
    device const RichKvPage *page_table3,
    constant RichVerifyAttentionParams *params, uint3 group) {
  constexpr uint D = HeadDim;
  RichVerifyTile tile{};
  uint kv_head = group.x;
  uint split = group.y;
  uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || split >= lane_params.split_count)
    return tile;
  constexpr ulong group_stride =
      ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  tile.queries = queries + (ulong(batch) * KVHeads + kv_head) * group_stride;
  tile.page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  tile.slot =
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits + split;
  tile.kv_head = kv_head;
  tile.split = split;
  tile.splits = lane_params.split_count;
  tile.committed_tokens = lane_params.committed_tokens;
  tile.active_rows = lane_params.active_rows;
  tile.active = true;
  return tile;
}

// Verify entries: one lane per group.z, eight rows, one configured history
// partition, with scratch for scores, probabilities and row statistics.
// INT8 and BF16 entries share one signature: pages are reached through the
// tables, so no entry binds KV storage.
#define PAGED_VERIFY_SPLIT_SIGNATURE(Name)                                     \
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
      uint thread_index [[thread_index_in_threadgroup]])

#define PAGED_VERIFY_SCRATCH(Group)                                            \
  constexpr uint M = Group * RICHENGINE_TARGET_VERIFY_ROWS;                        \
  constexpr uint N = RichKvPageTokens;                                       \
  alignas(16) threadgroup float scores[M * N];                                 \
  alignas(16) threadgroup bfloat probabilities[2 * M * N];                     \
  threadgroup float row_max[M];                                                \
  threadgroup float row_sum[M];                                                \
  threadgroup float previous_scale[M];                                         \
  threadgroup atomic_uint rescale;

#define PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                            \
  const RichVerifyTile tile =                                                \
      richengine_verify_attention_tile_at<Heads, Group, HeadDim>(                  \
          queries, page_table0, page_table1, page_table2, page_table3, params, \
          group);                                                              \
  if (!tile.active)                                                            \
    return;

#define PAGED_VERIFY_SPLIT(Name, Heads, Group, CacheElement, HeadDim)          \
  PAGED_VERIFY_SPLIT_SIGNATURE(Name) {                                         \
    PAGED_VERIFY_SCRATCH(Group)                                                \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                                \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TARGET_VERIFY_ROWS,       \
                                CacheElement, HeadDim>(                        \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, nullptr,                \
        params[group.z].score_scale, scores,                                 \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }

// Packed-INT4 pages feed matmul2d as native int4b operands; no staging.
PAGED_VERIFY_SPLIT(verify_attention_q8_split, 4, 6, int8_t, 256)
PAGED_VERIFY_SPLIT(verify_attention_q8_split_kv4_g4, 4, 4, int8_t, 256)
PAGED_VERIFY_SPLIT(verify_attention_int4_split, 4, 6, RichKvPacked4, 256)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_kv4_g4, 4, 4, RichKvPacked4, 256)
// BF16 shares the page loop and reduction, without quantization scales.
PAGED_VERIFY_SPLIT(verify_attention_bf16_split, 4, 6, bfloat, 256)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_kv4_g4, 4, 4, bfloat, 256)
// Head-dimension variants: the dense target's KV2/Group8 of 128 (_hd128)
// and LFM2's KV8/Group4 of 64 (_hd64).

PAGED_VERIFY_SPLIT(verify_attention_q8_split_hd64, 8, 4, int8_t, 64)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_hd64, 8, 4, RichKvPacked4, 64)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_hd64, 8, 4, bfloat, 64)
// Granite: 3B's KV8/Group5 of 64 and 8B's KV8/Group4 of 128.
PAGED_VERIFY_SPLIT(verify_attention_q8_split_k8q5d64, 8, 5, int8_t, 64)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_k8q5d64, 8, 5, RichKvPacked4, 64)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_k8q5d64, 8, 5, bfloat, 64)
PAGED_VERIFY_SPLIT(verify_attention_q8_split_k8q4d128, 8, 4, int8_t, 128)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_k8q4d128, 8, 4, RichKvPacked4, 128)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_k8q4d128, 8, 4, bfloat, 128)
// Gemma 4: the sliding layers' KV8 group-2 pages of 256 and the global
// layers' KV2 group-8 pages of 512.
PAGED_VERIFY_SPLIT(verify_attention_q8_split_gemma_h256, 8, 2, int8_t, 256)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_gemma_h256, 8, 2, RichKvPacked4, 256)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_gemma_h256, 8, 2, bfloat, 256)
PAGED_VERIFY_SPLIT(verify_attention_q8_split_gemma_hd512, 2, 8, int8_t, 512)
PAGED_VERIFY_SPLIT(verify_attention_int4_split_gemma_hd512, 2, 8, RichKvPacked4, 512)
PAGED_VERIFY_SPLIT(verify_attention_bf16_split_gemma_hd512, 2, 8, bfloat, 512)
#undef PAGED_VERIFY_SPLIT

// The _m2 hd512 verify variants run the tile in two fused-row passes
// (MP = M/2 = 32), halving the PV accumulator's ~128 fp32/thread — the M5
// per-core register/SRAM pool — plus the score/probability scratch. The
// signature, grid (kv heads, splits, lanes), threadgroup count (256), slot
// layout and the shared gemma_hd512 reduces are the plain variants'; the
// host only swaps the pipeline name.
#define PAGED_VERIFY_SCRATCH_M2(Group)                                       \
  constexpr uint M = Group * RICHENGINE_TARGET_VERIFY_ROWS;                       \
  constexpr uint MP = M / 2;                                                 \
  constexpr uint N = RichKvPageTokens;                                       \
  alignas(16) threadgroup float scores[MP * N];                              \
  alignas(16) threadgroup bfloat probabilities[2 * MP * N];                  \
  threadgroup float row_max[MP];                                             \
  threadgroup float row_sum[MP];                                             \
  threadgroup float previous_scale[MP];                                      \
  threadgroup atomic_uint rescale;

#define PAGED_VERIFY_SPLIT_M2(Name, Heads, Group, CacheElement, HeadDim)     \
  PAGED_VERIFY_SPLIT_SIGNATURE(Name) {                                       \
    PAGED_VERIFY_SCRATCH_M2(Group)                                           \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                              \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TARGET_VERIFY_ROWS,       \
                                CacheElement, HeadDim, 2>(                   \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, nullptr,                \
        params[group.z].score_scale, scores,                                 \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index);                                                         \
  }
PAGED_VERIFY_SPLIT_M2(verify_attention_q8_split_gemma_hd512_m2, 2, 8, int8_t, 512)
PAGED_VERIFY_SPLIT_M2(verify_attention_int4_split_gemma_hd512_m2, 2, 8, RichKvPacked4, 512)
PAGED_VERIFY_SPLIT_M2(verify_attention_bf16_split_gemma_hd512_m2, 2, 8, bfloat, 512)
// The kv2/g8 head-dimension-128 variants fuse M = 8 x VERIFY_ROWS rows: at
// 16 verify rows the full-M scratch overflows the 32 KB threadgroup budget,
// so they run the same two-pass halving the hd512 shapes use. The group-8
// head-dimension-256 layout (_kv2_g8) overflows the same way.
PAGED_VERIFY_SPLIT_M2(verify_attention_q8_split_hd128, 2, 8, int8_t, 128)
PAGED_VERIFY_SPLIT_M2(verify_attention_int4_split_hd128, 2, 8, RichKvPacked4, 128)
PAGED_VERIFY_SPLIT_M2(verify_attention_bf16_split_hd128, 2, 8, bfloat, 128)
PAGED_VERIFY_SPLIT_M2(verify_attention_q8_split_kv2_g8, 2, 8, int8_t, 256)
PAGED_VERIFY_SPLIT_M2(verify_attention_int4_split_kv2_g8, 2, 8, RichKvPacked4, 256)
PAGED_VERIFY_SPLIT_M2(verify_attention_bf16_split_kv2_g8, 2, 8, bfloat, 256)
#undef PAGED_VERIFY_SPLIT_M2

// Sliding-window variants of the two Gemma shapes: the same verify split
// plus buffer(8), the per-dispatch window in tokens (0 = full causal). The
// reduce pass needs no window; a fully masked page's statistics carry zero
// weight.
#define PAGED_VERIFY_SPLIT_SWA(Name, Heads, Group, CacheElement, HeadDim)    \
  kernel void Name(                                                          \
      device bfloat *queries [[buffer(0)]],                                  \
      device float *partials [[buffer(1)]],                                  \
      device float *statistics [[buffer(2)]],                                \
      device const RichKvPage *page_table0 [[buffer(3)]],                    \
      device const RichKvPage *page_table1 [[buffer(4)]],                    \
      device const RichKvPage *page_table2 [[buffer(5)]],                    \
      device const RichKvPage *page_table3 [[buffer(6)]],                    \
      constant RichVerifyAttentionParams *params [[buffer(7)]],              \
      constant uint &window_tokens [[buffer(8)]],                            \
      uint3 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]]) {                   \
    PAGED_VERIFY_SCRATCH(Group)                                              \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                              \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TARGET_VERIFY_ROWS,       \
                                CacheElement, HeadDim>(                        \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, nullptr,                \
        params[group.z].score_scale, scores,                                 \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index, window_tokens);                                        \
  }
PAGED_VERIFY_SPLIT_SWA(verify_attention_q8_split_swa_h256, 8, 2, int8_t, 256)
PAGED_VERIFY_SPLIT_SWA(verify_attention_int4_split_swa_h256, 8, 2, RichKvPacked4, 256)
PAGED_VERIFY_SPLIT_SWA(verify_attention_bf16_split_swa_h256, 8, 2, bfloat, 256)
PAGED_VERIFY_SPLIT_SWA(verify_attention_q8_split_swa_hd512, 2, 8, int8_t, 512)
PAGED_VERIFY_SPLIT_SWA(verify_attention_int4_split_swa_hd512, 2, 8, RichKvPacked4, 512)
PAGED_VERIFY_SPLIT_SWA(verify_attention_bf16_split_swa_hd512, 2, 8, bfloat, 512)
#undef PAGED_VERIFY_SPLIT_SWA

// Windowed _m2 hd512 verify variants: the two-pass tile plus buffer(8), the
// per-dispatch window in tokens (0 = full causal).
#define PAGED_VERIFY_SPLIT_SWA_M2(Name, Heads, Group, CacheElement, HeadDim) \
  kernel void Name(                                                          \
      device bfloat *queries [[buffer(0)]],                                  \
      device float *partials [[buffer(1)]],                                  \
      device float *statistics [[buffer(2)]],                                \
      device const RichKvPage *page_table0 [[buffer(3)]],                    \
      device const RichKvPage *page_table1 [[buffer(4)]],                    \
      device const RichKvPage *page_table2 [[buffer(5)]],                    \
      device const RichKvPage *page_table3 [[buffer(6)]],                    \
      constant RichVerifyAttentionParams *params [[buffer(7)]],              \
      constant uint &window_tokens [[buffer(8)]],                            \
      uint3 group [[threadgroup_position_in_grid]],                          \
      uint thread_index [[thread_index_in_threadgroup]]) {                   \
    PAGED_VERIFY_SCRATCH_M2(Group)                                           \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                              \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TARGET_VERIFY_ROWS,       \
                                CacheElement, HeadDim, 2>(                   \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS, tile.splits,         \
        tile.split, partials, statistics, tile.slot, nullptr,                \
        params[group.z].score_scale, scores,                                 \
        probabilities, row_max, row_sum, previous_scale, &rescale,             \
        thread_index, window_tokens);                                        \
  }
PAGED_VERIFY_SPLIT_SWA_M2(verify_attention_q8_split_swa_hd512_m2, 2, 8, int8_t, 512)
PAGED_VERIFY_SPLIT_SWA_M2(verify_attention_int4_split_swa_hd512_m2, 2, 8, RichKvPacked4, 512)
PAGED_VERIFY_SPLIT_SWA_M2(verify_attention_bf16_split_swa_hd512_m2, 2, 8, bfloat, 512)
#undef PAGED_VERIFY_SPLIT_SWA_M2
#undef PAGED_VERIFY_SCRATCH_M2

// The tree verify splits: RICHENGINE_TREE_VERIFY_NODES rows of a lane's comb.
// The emitted nodes occupy scratch slots committed..committed+rows-1 in row
// order; row_masks[lane*row_capacity+row] holds each row's ancestor bitmask
// over those scratch slots (history is always admitted). A row whose mask is
// zero — the spare row, or a leaf the selector did not emit — attends only
// committed history, matching the causal rows' behavior.
#define PAGED_VERIFY_TREE_SCRATCH(Group)                                       \
  constexpr uint M = Group * RICHENGINE_TREE_VERIFY_NODES;                         \
  constexpr uint N = RichKvPageTokens;                                       \
  alignas(16) threadgroup float scores[M * N];                                 \
  alignas(16) threadgroup bfloat probabilities[2 * M * N];                     \
  threadgroup float row_max[M];                                                \
  threadgroup float row_sum[M];                                                \
  threadgroup float previous_scale[M];                                         \
  threadgroup atomic_uint rescale;

#define PAGED_VERIFY_TREE_SPLIT(Name, Heads, Group, CacheElement, HeadDim)     \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table0 [[buffer(3)]],                    \
      device const RichKvPage *page_table1 [[buffer(4)]],                    \
      device const RichKvPage *page_table2 [[buffer(5)]],                    \
      device const RichKvPage *page_table3 [[buffer(6)]],                    \
      device const uint *row_masks [[buffer(7)]],                              \
      constant RichVerifyAttentionParams *params [[buffer(8)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    PAGED_VERIFY_TREE_SCRATCH(Group)                                           \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                                \
    device const uint *lane_masks =                                            \
        row_masks + ulong(group.z) * params[group.z].row_capacity;             \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TREE_VERIFY_NODES,        \
                                CacheElement, HeadDim>(                        \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, tile.active_rows, tile.splits,                  \
        tile.split, partials, statistics, tile.slot, lane_masks,             \
        params[group.z].score_scale, scores,                                   \
        probabilities,                                                         \
        row_max, row_sum, previous_scale, &rescale, thread_index);             \
  }

PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_q8_split, 4, 6, int8_t, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_q8_split_kv4_g4, 4, 4, int8_t, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_int4_split, 4, 6, RichKvPacked4, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_int4_split_kv4_g4, 4, 4, RichKvPacked4, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_bf16_split, 4, 6, bfloat, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_bf16_split_kv4_g4, 4, 4, bfloat, 256)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_q8_split_hd64, 8, 4, int8_t, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_int4_split_hd64, 8, 4, RichKvPacked4, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_bf16_split_hd64, 8, 4, bfloat, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_q8_split_k8q5d64, 8, 5, int8_t, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_int4_split_k8q5d64, 8, 5, RichKvPacked4, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_bf16_split_k8q5d64, 8, 5, bfloat, 64)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_q8_split_k8q4d128, 8, 4, int8_t, 128)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_int4_split_k8q4d128, 8, 4, RichKvPacked4, 128)
PAGED_VERIFY_TREE_SPLIT(verify_tree_attention_bf16_split_k8q4d128, 8, 4, bfloat, 128)
#undef PAGED_VERIFY_TREE_SPLIT

// The Group-8 tree layouts fuse 128 rows (8 query heads x 16 tree nodes) —
// the score/probability scratch alone is 34 KB, over the 32 KB threadgroup
// limit — so they run the tile in two fused-row passes, the gemma_hd512
// verify splits' halving. The shared reduces are unchanged.
#define PAGED_VERIFY_TREE_SCRATCH_M2(Group)                                    \
  constexpr uint M = Group * RICHENGINE_TREE_VERIFY_NODES;                         \
  constexpr uint MP = M / 2;                                                 \
  constexpr uint N = RichKvPageTokens;                                       \
  alignas(16) threadgroup float scores[MP * N];                              \
  alignas(16) threadgroup bfloat probabilities[2 * MP * N];                  \
  threadgroup float row_max[MP];                                             \
  threadgroup float row_sum[MP];                                             \
  threadgroup float previous_scale[MP];                                      \
  threadgroup atomic_uint rescale;

#define PAGED_VERIFY_TREE_SPLIT_M2(Name, Heads, Group, CacheElement, HeadDim)  \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table0 [[buffer(3)]],                    \
      device const RichKvPage *page_table1 [[buffer(4)]],                    \
      device const RichKvPage *page_table2 [[buffer(5)]],                    \
      device const RichKvPage *page_table3 [[buffer(6)]],                    \
      device const uint *row_masks [[buffer(7)]],                              \
      constant RichVerifyAttentionParams *params [[buffer(8)]],              \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    PAGED_VERIFY_TREE_SCRATCH_M2(Group)                                        \
    PAGED_VERIFY_TILE_AT(Heads, Group, HeadDim)                                \
    device const uint *lane_masks =                                            \
        row_masks + ulong(group.z) * params[group.z].row_capacity;             \
    richengine_paged_attention_tile<Heads, Group, RICHENGINE_TREE_VERIFY_NODES,        \
                                CacheElement, HeadDim, 2>(                     \
        tile.queries, tile.page_table, params[group.z].kv, tile.kv_head,       \
        tile.committed_tokens, tile.active_rows, tile.splits,                  \
        tile.split, partials, statistics, tile.slot, lane_masks,             \
        params[group.z].score_scale, scores,                                   \
        probabilities,                                                         \
        row_max, row_sum, previous_scale, &rescale, thread_index);             \
  }

PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_q8_split_hd128_m2, 2, 8, int8_t, 128)
PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_int4_split_hd128_m2, 2, 8, RichKvPacked4, 128)
PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_bf16_split_hd128_m2, 2, 8, bfloat, 128)
PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_q8_split_kv2_g8_m2, 2, 8, int8_t, 256)
PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_int4_split_kv2_g8_m2, 2, 8, RichKvPacked4, 256)
PAGED_VERIFY_TREE_SPLIT_M2(verify_tree_attention_bf16_split_kv2_g8_m2, 2, 8, bfloat, 256)
#undef PAGED_VERIFY_TREE_SPLIT_M2
#undef PAGED_VERIFY_TREE_SCRATCH_M2
#undef PAGED_VERIFY_TILE_AT
#undef PAGED_VERIFY_TREE_SCRATCH
#undef PAGED_VERIFY_SCRATCH
#undef PAGED_VERIFY_SPLIT_SIGNATURE

template <uint KVHeads, uint QueryHeadsPerKVHead,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_verify_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant RichVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr ushort M = RICHENGINE_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  constexpr ulong group_stride =
      ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  richengine_attention_reduce_row<QueryHeadsPerKVHead,
                                   RICHENGINE_TARGET_VERIFY_ROWS, HeadDim>(
      partials, statistics,
      output + (ulong(batch) * KVHeads + kv_head) * group_stride,
      lane_params.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
}

kernel void verify_attention_reduce(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[8];
  richengine_verify_attention_reduce_phase<4, 6>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}

kernel void verify_attention_reduce_kv4_g4(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[8];
  richengine_verify_attention_reduce_phase<4, 4>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}

kernel void verify_attention_reduce_kv2_g8(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[8];
  richengine_verify_attention_reduce_phase<2, 8>(
      partials, statistics, output, params, group, thread_index, weights, group_values);
}

// The reduce/gate fusion for a Plain-input out-projection
// (ops::PagedAttention::addVerify with gate buffers): each reduce thread owns
// one output element, which the gate's elementwise dispatch would read back,
// so the fused kernel scales it by sigmoid of its packed QKV gate value and
// writes the hidden row itself. The bits match verify_attention_reduce plus
// verify_attention_gate: the same bf16 attention value is the product's
// factor. `packed_qkv` is the lane's [q|gate]xQHeads + [k|v]xKVHeads rows and
// `hidden` its [row][query head][dimension] gate output.
template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void richengine_verify_attention_reduce_gate_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output, device const bfloat *packed_qkv,
    device bfloat *hidden,
    constant RichVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr uint QHeads = KVHeads * QueryHeadsPerKVHead;
  constexpr uint D = RichKvHeadDimension;
  constexpr uint PackedStride = 2 * QHeads * D + 2 * KVHeads * D;
  constexpr ushort M = RICHENGINE_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  const uint kv_head = group.x;
  const uint fused_row = group.y;
  const uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  constexpr ulong group_stride =
      ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  const bfloat value = richengine_attention_reduce_value<QueryHeadsPerKVHead,
                                                     RICHENGINE_TARGET_VERIFY_ROWS>(
      partials, statistics, lane_params.committed_tokens, RICHENGINE_TARGET_VERIFY_ROWS,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
  output[(ulong(batch) * KVHeads + kv_head) * group_stride + fused_row * D +
         thread_index] = value;
  const uint row = fused_row / QueryHeadsPerKVHead;
  const uint query_head = kv_head * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
  const float gate = float(
      packed_qkv[(ulong(batch) * RICHENGINE_TARGET_VERIFY_ROWS + row) * PackedStride +
                 query_head * 2 * D + D + thread_index]);
  hidden[((ulong(batch) * RICHENGINE_TARGET_VERIFY_ROWS + row) * QHeads + query_head) *
             D +
         thread_index] = bfloat(float(value) * richengine_sigmoid(gate));
}

#define PAGED_VERIFY_REDUCE_GATE(Name, Heads, Group)                            \
  kernel void Name(                                                             \
      device const float *partials [[buffer(0)]],                               \
      device const float *statistics [[buffer(1)]],                             \
      device bfloat *output [[buffer(2)]],                                      \
      device const bfloat *packed_qkv [[buffer(3)]],                            \
      device bfloat *hidden [[buffer(4)]],                                      \
      constant RichVerifyAttentionParams *params [[buffer(5)]],               \
      uint3 group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]]) {                      \
    threadgroup float weights[RichVerifyMaximumSplits];                       \
    threadgroup float group_values[8];                                          \
    richengine_verify_attention_reduce_gate_phase<Heads, Group>(                    \
        partials, statistics, output, packed_qkv, hidden, params, group,        \
        thread_index, weights, group_values);                                   \
  }

PAGED_VERIFY_REDUCE_GATE(verify_attention_reduce_gate, 4, 6)
PAGED_VERIFY_REDUCE_GATE(verify_attention_reduce_gate_kv4_g4, 4, 4)
PAGED_VERIFY_REDUCE_GATE(verify_attention_reduce_gate_kv2_g8, 2, 8)
#undef PAGED_VERIFY_REDUCE_GATE

// The reduce entries of the head-dimension variants: their threadgroups are
// the head dimension (128 and 64), the same grids as the 256-dimension
// entries.
#define PAGED_VERIFY_REDUCE_HD(Name, Heads, Group, HeadDim)                  \
  kernel void Name(                                                        \
      device const float *partials [[buffer(0)]],                          \
      device const float *statistics [[buffer(1)]],                        \
      device bfloat *output [[buffer(2)]],                                 \
      constant RichVerifyAttentionParams *params [[buffer(3)]],          \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint thread_index [[thread_index_in_threadgroup]]) {                 \
    threadgroup float weights[RichVerifyMaximumSplits];                  \
    threadgroup float group_values[8];                                     \
    richengine_verify_attention_reduce_phase<Heads, Group, HeadDim>(           \
        partials, statistics, output, params, group, thread_index,         \
        weights, group_values);                                            \
  }
PAGED_VERIFY_REDUCE_HD(verify_attention_reduce_hd128, 2, 8, 128)
PAGED_VERIFY_REDUCE_HD(verify_attention_reduce_hd64, 8, 4, 64)
PAGED_VERIFY_REDUCE_HD(verify_attention_reduce_k8q5d64, 8, 5, 64)
PAGED_VERIFY_REDUCE_HD(verify_attention_reduce_k8q4d128, 8, 4, 128)
#undef PAGED_VERIFY_REDUCE_HD

// Gemma 4's reduces: the sliding layers' KV8 group-2 of 256 (threadgroup
// 256, eight simdgroup maxima) and the global layers' KV2 group-8 of 512
// (threadgroup 512, sixteen).
kernel void verify_attention_reduce_gemma_h256(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[8];
  richengine_verify_attention_reduce_phase<8, 2, 256>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

kernel void verify_attention_reduce_gemma_hd512(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[16];
  richengine_verify_attention_reduce_phase<2, 8, 512>(
      partials, statistics, output, params, group, thread_index, weights,
      group_values);
}

// The reduce/gather fusion of the no-gate targets (dense hd128, LFM2 hd64):
// the reduce/gate kernel's structure minus the gate multiply — the same
// per-row split reduce as verify_attention_reduce_hd*, writing the bf16
// value straight into the out-projection's [row][query head][dimension]
// hidden layout. One dispatch replaces reduce plus verify_attention_gather.
// The reduced bits equal the reduce kernel's, and the gather's bf16 -> float
// -> bf16 copy is exact, so the hidden rows are bit-identical. The attention
// staging rows are not written: with no gate nothing reads them afterward.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline void richengine_verify_attention_reduce_gather_phase(
    device const float *partials, device const float *statistics,
    device bfloat *hidden,
    constant RichVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr uint QHeads = KVHeads * QueryHeadsPerKVHead;
  constexpr ushort M = RICHENGINE_TARGET_VERIFY_ROWS * QueryHeadsPerKVHead;
  const uint kv_head = group.x;
  const uint fused_row = group.y;
  const uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= HeadDim)
    return;
  const bfloat value =
      richengine_attention_reduce_value<QueryHeadsPerKVHead,
                                    RICHENGINE_TARGET_VERIFY_ROWS, HeadDim>(
          partials, statistics, lane_params.committed_tokens,
          RICHENGINE_TARGET_VERIFY_ROWS, lane_params.split_count,
          (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits,
          fused_row, thread_index, weights, group_values);
  const uint row = fused_row / QueryHeadsPerKVHead;
  const uint query_head =
      kv_head * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
  hidden[((ulong(batch) * RICHENGINE_TARGET_VERIFY_ROWS + row) * QHeads +
          query_head) *
             HeadDim +
         thread_index] = value;
}

#define PAGED_VERIFY_REDUCE_GATHER(Name, Heads, Group, HeadDim)            \
  kernel void Name(                                                        \
      device const float *partials [[buffer(0)]],                          \
      device const float *statistics [[buffer(1)]],                        \
      device bfloat *hidden [[buffer(2)]],                                 \
      constant RichVerifyAttentionParams *params [[buffer(3)]],          \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint thread_index [[thread_index_in_threadgroup]]) {                 \
    threadgroup float weights[RichVerifyMaximumSplits];                  \
    threadgroup float group_values[8];                                     \
    richengine_verify_attention_reduce_gather_phase<Heads, Group, HeadDim>(    \
        partials, statistics, hidden, params, group, thread_index,         \
        weights, group_values);                                            \
  }
PAGED_VERIFY_REDUCE_GATHER(verify_attention_reduce_gather_hd128, 2, 8, 128)
PAGED_VERIFY_REDUCE_GATHER(verify_attention_reduce_gather_hd64, 8, 4, 64)
PAGED_VERIFY_REDUCE_GATHER(verify_attention_reduce_gather_k8q5d64, 8, 5, 64)
PAGED_VERIFY_REDUCE_GATHER(verify_attention_reduce_gather_k8q4d128, 8, 4, 128)
#undef PAGED_VERIFY_REDUCE_GATHER

// Gemma's fused reduce/gathers, with group_values sized to the head
// dimension's simdgroup count (hd512's sixteen).
kernel void verify_attention_reduce_gather_gemma_h256(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[8];
  richengine_verify_attention_reduce_gather_phase<8, 2, 256>(
      partials, statistics, hidden, params, group, thread_index, weights,
      group_values);
}

kernel void verify_attention_reduce_gather_gemma_hd512(
    device const float *partials [[buffer(0)]],
    device const float *statistics [[buffer(1)]],
    device bfloat *hidden [[buffer(2)]],
    constant RichVerifyAttentionParams *params [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float weights[RichVerifyMaximumSplits];
  threadgroup float group_values[16];
  richengine_verify_attention_reduce_gather_phase<2, 8, 512>(
      partials, statistics, hidden, params, group, thread_index, weights,
      group_values);
}

// The tree reduces: a RICHENGINE_TREE_VERIFY_NODES-row tile per lane, with the
// runtime active row count deciding which rows carry splits. Rows past a
// lane's emitted nodes reduce to zeros, exactly as inactive chain rows do.
template <uint KVHeads, uint QueryHeadsPerKVHead,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_verify_tree_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output,
    constant RichVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr ushort M = RICHENGINE_TREE_VERIFY_NODES * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  constexpr ulong group_stride =
      ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  richengine_attention_reduce_row<QueryHeadsPerKVHead,
                              RICHENGINE_TREE_VERIFY_NODES, HeadDim>(
      partials, statistics,
      output + (ulong(batch) * KVHeads + kv_head) * group_stride,
      lane_params.committed_tokens, lane_params.active_rows,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
}

#define PAGED_VERIFY_TREE_REDUCE(Name, Heads, Group, HeadDim)              \
  kernel void Name(                                                        \
      device const float *partials [[buffer(0)]],                          \
      device const float *statistics [[buffer(1)]],                        \
      device bfloat *output [[buffer(2)]],                                 \
      constant RichVerifyAttentionParams *params [[buffer(3)]],          \
      uint3 group [[threadgroup_position_in_grid]],                        \
      uint thread_index [[thread_index_in_threadgroup]]) {                 \
    threadgroup float weights[RichVerifyMaximumSplits];                  \
    threadgroup float group_values[8];                                     \
    richengine_verify_tree_attention_reduce_phase<Heads, Group, HeadDim>(      \
        partials, statistics, output, params, group, thread_index,         \
        weights, group_values);                                            \
  }
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce, 4, 6, 256)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_kv4_g4, 4, 4, 256)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_kv2_g8, 2, 8, 256)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_hd128, 2, 8, 128)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_hd64, 8, 4, 64)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_k8q5d64, 8, 5, 64)
PAGED_VERIFY_TREE_REDUCE(verify_tree_attention_reduce_k8q4d128, 8, 4, 128)
#undef PAGED_VERIFY_TREE_REDUCE

// The tree reduce/gate fusion: the same per-element gate as the chain's,
// with the packed QKV and hidden row strides taken from the lane's runtime
// row count.
template <uint KVHeads, uint QueryHeadsPerKVHead>
inline void richengine_verify_tree_attention_reduce_gate_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output, device const bfloat *packed_qkv,
    device bfloat *hidden,
    constant RichVerifyAttentionParams *params, uint3 group,
    uint thread_index, threadgroup float *weights, threadgroup float *group_values) {
  constexpr uint QHeads = KVHeads * QueryHeadsPerKVHead;
  constexpr uint D = RichKvHeadDimension;
  constexpr uint PackedStride = 2 * QHeads * D + 2 * KVHeads * D;
  constexpr ushort M = RICHENGINE_TREE_VERIFY_NODES * QueryHeadsPerKVHead;
  const uint kv_head = group.x;
  const uint fused_row = group.y;
  const uint batch = group.z;
  constant RichVerifyAttentionParams &lane_params = params[batch];
  if (!richengine_verify_attention_contract_valid(lane_params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D)
    return;
  constexpr ulong group_stride =
      ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * QueryHeadsPerKVHead * D;
  const bfloat value = richengine_attention_reduce_value<QueryHeadsPerKVHead,
                                                     RICHENGINE_TREE_VERIFY_NODES>(
      partials, statistics, lane_params.committed_tokens, lane_params.active_rows,
      lane_params.split_count,
      (ulong(batch) * KVHeads + kv_head) * lane_params.slot_splits, fused_row,
      thread_index, weights, group_values);
  output[(ulong(batch) * KVHeads + kv_head) * group_stride + fused_row * D +
         thread_index] = value;
  const uint row = fused_row / QueryHeadsPerKVHead;
  if (row >= lane_params.active_rows)
    return;
  const uint query_head = kv_head * QueryHeadsPerKVHead + fused_row % QueryHeadsPerKVHead;
  const float gate = float(
      packed_qkv[(ulong(batch) * RICHENGINE_TREE_VERIFY_NODES + row) * PackedStride +
                 query_head * 2 * D + D + thread_index]);
  hidden[((ulong(batch) * RICHENGINE_TREE_VERIFY_NODES + row) * QHeads +
          query_head) *
             D +
         thread_index] = bfloat(float(value) * richengine_sigmoid(gate));
}

#define PAGED_VERIFY_TREE_REDUCE_GATE(Name, Heads, Group)                     \
  kernel void Name(                                                             \
      device const float *partials [[buffer(0)]],                               \
      device const float *statistics [[buffer(1)]],                             \
      device bfloat *output [[buffer(2)]],                                      \
      device const bfloat *packed_qkv [[buffer(3)]],                            \
      device bfloat *hidden [[buffer(4)]],                                      \
      constant RichVerifyAttentionParams *params [[buffer(5)]],               \
      uint3 group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]]) {                      \
    threadgroup float weights[RichVerifyMaximumSplits];                       \
    threadgroup float group_values[8];                                          \
    richengine_verify_tree_attention_reduce_gate_phase<Heads, Group>(               \
        partials, statistics, output, packed_qkv, hidden, params, group,        \
        thread_index, weights, group_values);                                   \
  }

PAGED_VERIFY_TREE_REDUCE_GATE(verify_tree_attention_reduce_gate, 4, 6)
PAGED_VERIFY_TREE_REDUCE_GATE(verify_tree_attention_reduce_gate_kv4_g4, 4, 4)
PAGED_VERIFY_TREE_REDUCE_GATE(verify_tree_attention_reduce_gate_kv2_g8, 2, 8)
#undef PAGED_VERIFY_TREE_REDUCE_GATE
