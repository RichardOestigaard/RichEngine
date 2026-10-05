#include "metal/kernels/common/paged_store_row.h"

template <uint KVHeads, typename CacheElement,
          uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
inline void splash_store_verify_phase(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table0, device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2, device const SplashKvPage *page_table3,
    constant SplashChunkedPrefillParams *params,
    threadgroup float *maxima, uint group, uint thread_index, uint simd_lane,
    uint simd_group) {
  // The row count is the lane's runtime chunk: eight chain rows, or the
  // comb's emitted nodes (SPLASH_TREE_VERIFY_NODES - 1) under tree verify.
  // Both store their rows at committed + row in scratch page order.
  constant SplashChunkedPrefillParams &first = params[0];
  const uint groups_per_lane = 2 * first.chunk_tokens * KVHeads;
  if (groups_per_lane == 0)
    return;
  uint batch = group / groups_per_lane;
  uint local_group = group % groups_per_lane;
  constant SplashChunkedPrefillParams &lane_params = params[batch];
  const uint rows = lane_params.chunk_tokens;
  if (!splash_chunk_contract_valid(lane_params) ||
      rows > SPLASH_TREE_VERIFY_NODES - 1 ||
      lane_params.chunk_stride != SPLASH_VERIFY_CHUNK_STRIDE ||
      thread_index >= HeadDim)
    return;
  device const SplashKvPage *page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  const ulong lane_tensor_stride =
      ulong(KVHeads) * SPLASH_VERIFY_CHUNK_STRIDE * HeadDim;
  chunk_keys += batch * lane_tensor_stride;
  chunk_values += batch * lane_tensor_stride;

  const uint head_rows = rows * KVHeads;
  bool value_tensor = local_group >= head_rows;
  uint row = value_tensor ? local_group - head_rows : local_group;
  uint chunk_token = row % rows;
  uint head = row / rows;
  splash_store_kv_row<KVHeads, CacheElement, HeadDim>(
      chunk_keys, chunk_values, page_table, lane_params, maxima, value_tensor,
      head, chunk_token, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_q8_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_store_verify_phase<4, int8_t>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, maxima, group, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_q8_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_store_verify_phase<2, int8_t>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, maxima, group, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_int4_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_store_verify_phase<4, SplashKvPacked4>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, maxima, group, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_int4_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float maxima[8];
  splash_store_verify_phase<2, SplashKvPacked4>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, maxima, group, thread_index, simd_lane, simd_group);
}

// BF16 entries copy the source bits and need no scale reduction.

kernel void verify_attention_bf16_store(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  splash_store_verify_phase<4, bfloat>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, nullptr, group, thread_index, simd_lane, simd_group);
}

kernel void verify_attention_bf16_store_kv2_g8(
    device const bfloat *chunk_keys [[buffer(0)]],
    device const bfloat *chunk_values [[buffer(1)]],
    device const SplashKvPage *page_table0 [[buffer(2)]],
    device const SplashKvPage *page_table1 [[buffer(3)]],
    device const SplashKvPage *page_table2 [[buffer(4)]],
    device const SplashKvPage *page_table3 [[buffer(5)]],
    constant SplashChunkedPrefillParams *params [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  splash_store_verify_phase<2, bfloat>(
      chunk_keys, chunk_values, page_table0, page_table1, page_table2,
      page_table3, params, nullptr, group, thread_index, simd_lane, simd_group);
}

// Head-dimension variants: the dense target's KV2 pages of 128 (_hd128)
// and LFM2's KV8 pages of 64 (_hd64); the reduce over scale maxima covers
// headDim/32 simdgroups.
#define VERIFY_ATTENTION_STORE_HD(Name, Heads, CacheElement, HeadDim)       \
  kernel void Name(                                                       \
      device const bfloat *chunk_keys [[buffer(0)]],                      \
      device const bfloat *chunk_values [[buffer(1)]],                    \
      device const SplashKvPage *page_table0 [[buffer(2)]],               \
      device const SplashKvPage *page_table1 [[buffer(3)]],               \
      device const SplashKvPage *page_table2 [[buffer(4)]],               \
      device const SplashKvPage *page_table3 [[buffer(5)]],               \
      constant SplashChunkedPrefillParams *params [[buffer(6)]],          \
      uint group [[threadgroup_position_in_grid]],                        \
      uint thread_index [[thread_index_in_threadgroup]],                  \
      uint simd_lane [[thread_index_in_simdgroup]],                       \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {               \
    threadgroup float maxima[HeadDim / 32];                               \
    splash_store_verify_phase<Heads, CacheElement, HeadDim>(              \
        chunk_keys, chunk_values, page_table0, page_table1, page_table2,  \
        page_table3, params, maxima, group, thread_index, simd_lane,      \
        simd_group);                                                      \
  }
VERIFY_ATTENTION_STORE_HD(verify_attention_q8_store_hd128, 2, int8_t, 128)
VERIFY_ATTENTION_STORE_HD(verify_attention_int4_store_hd128, 2, SplashKvPacked4, 128)
VERIFY_ATTENTION_STORE_HD(verify_attention_bf16_store_hd128, 2, bfloat, 128)
VERIFY_ATTENTION_STORE_HD(verify_attention_q8_store_hd64, 8, int8_t, 64)
VERIFY_ATTENTION_STORE_HD(verify_attention_int4_store_hd64, 8, SplashKvPacked4, 64)
VERIFY_ATTENTION_STORE_HD(verify_attention_bf16_store_hd64, 8, bfloat, 64)
#undef VERIFY_ATTENTION_STORE_HD

// Tree verify compaction. A tree lane's retained path names the scratch rows
// (in emitted order) that survived acceptance, depth-ordered; chain rows
// already sit at committed + depth, but a retained leaf's slab must move to
// its path position. The copy runs sequentially over the path inside one
// threadgroup per (lane, head): a destination slot is always a smaller path
// index than the source row that may occupy it, so the forward order never
// clobbers a row still to be read. All loads of a row finish before its
// stores. Packed-INT4 pages copy bytes; quantized tensors move their
// per-token scale alongside.
template <uint KVHeads, typename CacheElement,
          uint HeadDim = SPLASH_KV_HEAD_DIMENSION>
inline void splash_compact_verify_tree_phase(
    device const SplashKvPage *page_table0,
    device const SplashKvPage *page_table1,
    device const SplashKvPage *page_table2,
    device const SplashKvPage *page_table3,
    constant SplashChunkedPrefillParams *params,
    device const uint *retained_path, device const uint *retained_count,
    uint group, uint thread_index) {
  const uint batch = group / KVHeads;
  const uint head = group % KVHeads;
  constant SplashChunkedPrefillParams &lane_params = params[batch];
  const uint count = retained_count[batch];
  if (count == 0)
    return;
  device const SplashKvPage *page_table =
      batch == 0 ? page_table0
                 : (batch == 1 ? page_table1
                               : (batch == 2 ? page_table2 : page_table3));
  const SplashKvAddressing<KVHeads, CacheElement, HeadDim> addressing(
      lane_params.kv, head);
  constexpr bool packed = is_same<CacheElement, SplashKvPacked4>::value;
  // Key slabs are token-major: a token's HeadDim elements are contiguous —
  // HeadDim / 2 bytes for INT4, whose nibbles are (dim, dim + 1) pairs and so
  // copy byte-exact. Packed values are token-major too — the same [token]
  // [dim pair] bytes — while INT8 and BF16 value slabs are dimension-major,
  // a token's elements one per dimension, one thread each.
  constexpr uint key_width = packed ? HeadDim / 2 : HeadDim;
  constexpr bool quantized =
      SplashKvPageBytes<KVHeads, CacheElement, HeadDim>::Quantized;
  const uint path_base = batch * SPLASH_TARGET_VERIFY_ROWS;
  for (uint path_index = 0; path_index < count; ++path_index) {
    const uint source_row = retained_path[path_base + path_index];
    if (source_row == path_index)
      continue;
    const uint source_token = lane_params.committed_tokens + source_row;
    const uint destination_token =
        lane_params.committed_tokens + path_index;
    const auto source =
        addressing.page(page_table[source_token / SplashKvPageTokens]);
    const auto destination =
        addressing.page(page_table[destination_token / SplashKvPageTokens]);
    const uint source_offset = source_token % SplashKvPageTokens;
    const uint destination_offset = destination_token % SplashKvPageTokens;
    const bool active = thread_index < key_width;
    const float key_scale =
        quantized && thread_index == 0 ? source.key_scales[source_offset] : 0.0f;
    const float value_scale =
        quantized && thread_index == 0 ? source.value_scales[source_offset] : 0.0f;
    if constexpr (packed) {
      // Keys and values are both token-major pairs of (dim, dim + 1):
      // byte-exact copies of the token's rows.
      const auto key = active
          ? source.keys[source_offset * key_width + thread_index]
          : uchar(0);
      const auto value = active
          ? source.values[source_offset * key_width + thread_index]
          : uchar(0);
      threadgroup_barrier(mem_flags::mem_device);
      if (active) {
        destination.keys[destination_offset * key_width + thread_index] = key;
        destination.values[destination_offset * key_width + thread_index] =
            value;
      }
    } else {
      const auto key = source.keys[source_offset * key_width + thread_index];
      const auto value = source.values[thread_index * SplashKvPageTokens +
                                     source_offset];
      threadgroup_barrier(mem_flags::mem_device);
      destination.keys[destination_offset * key_width + thread_index] = key;
      destination.values[thread_index * SplashKvPageTokens +
                         destination_offset] = value;
    }
    if (quantized && thread_index == 0) {
      destination.key_scales[destination_offset] = key_scale;
      destination.value_scales[destination_offset] = value_scale;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

#define VERIFY_TREE_COMPACT(Name, Heads, CacheElement, HeadDim)            \
  kernel void Name(                                                        \
      device const SplashKvPage *page_table0 [[buffer(0)]],                \
      device const SplashKvPage *page_table1 [[buffer(1)]],                \
      device const SplashKvPage *page_table2 [[buffer(2)]],                \
      device const SplashKvPage *page_table3 [[buffer(3)]],                \
      device const uint *retained_path [[buffer(4)]],                      \
      device const uint *retained_count [[buffer(5)]],                     \
      constant SplashChunkedPrefillParams *params [[buffer(6)]],           \
      uint group [[threadgroup_position_in_grid]],                         \
      uint thread_index [[thread_index_in_threadgroup]]) {                 \
    splash_compact_verify_tree_phase<Heads, CacheElement, HeadDim>(        \
        page_table0, page_table1, page_table2, page_table3, params,        \
        retained_path, retained_count, group, thread_index);               \
  }
VERIFY_TREE_COMPACT(verify_tree_attention_q8_compact, 4, int8_t, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_q8_compact_kv4_g4, 4, int8_t, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_q8_compact_kv2_g8, 2, int8_t, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_int4_compact, 4, SplashKvPacked4, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_int4_compact_kv4_g4, 4, SplashKvPacked4, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_int4_compact_kv2_g8, 2, SplashKvPacked4, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_bf16_compact, 4, bfloat, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_bf16_compact_kv4_g4, 4, bfloat, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_bf16_compact_kv2_g8, 2, bfloat, 256)
VERIFY_TREE_COMPACT(verify_tree_attention_q8_compact_hd128, 2, int8_t, 128)
VERIFY_TREE_COMPACT(verify_tree_attention_int4_compact_hd128, 2, SplashKvPacked4, 128)
VERIFY_TREE_COMPACT(verify_tree_attention_bf16_compact_hd128, 2, bfloat, 128)
VERIFY_TREE_COMPACT(verify_tree_attention_q8_compact_hd64, 8, int8_t, 64)
VERIFY_TREE_COMPACT(verify_tree_attention_int4_compact_hd64, 8, SplashKvPacked4, 64)
VERIFY_TREE_COMPACT(verify_tree_attention_bf16_compact_hd64, 8, bfloat, 64)
#undef VERIFY_TREE_COMPACT
