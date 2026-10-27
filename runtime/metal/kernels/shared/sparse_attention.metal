#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/sparse_attention_tile.h"

// Sparse attention (metal/abi/PagedAttention.h): the block lists a tile's rows'
// block masks make, and the split and reduce entries that attend through
// them, for the GQA group of twelve over two KV heads on a KV layout that
// keeps values token-major (Qwen3.8-Flash-Next's, for one). The verify lanes'
// rows run in eight-row tiles (a list entry holds a bit per tile row), a
// lane's tile t at group.z = lane * row_tiles + t.

// Threadgroup per tile, group.x the tile and group.y the sequence
// (RichAttentionBlockListParams): only the blocks of the tile's history are
// listed (the tail bit of a row whose token completes its block names the
// next block), and every thread takes a contiguous run of their words, so the
// entries keep block order. A list keeps to its stride whatever the masks
// hold.
constant constexpr uint kBlockListThreads = RICHENGINE_ATTENTION_BLOCK_LIST_THREADS;

// The bits of mask word `w` that name one of the first `blocks` blocks.
inline uint richengine_history_word_bits(uint blocks, uint w) {
  const uint remaining = blocks - w * 32;
  return remaining >= 32 ? ~0u : (1u << remaining) - 1u;
}

kernel void attention_block_lists(device const uint *mask [[buffer(0)]],
                                  device uint *lists [[buffer(1)]],
                                  device uint *counts [[buffer(2)]],
                                  constant RichAttentionBlockListParams &p [[buffer(3)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint simd_counts[kBlockListThreads / 32];
  if (group.y >= min(p.sequences, uint(RICHENGINE_MAXIMUM_BATCH_WIDTH)))
    return;
  const RichAttentionBlockListSequence s = p.sequence[group.y];
  const uint tile = group.x, first_row = tile * p.tile_rows;
  if (first_row >= s.rows)
    return;
  const uint rows = min(p.tile_rows, s.rows - first_row);
  const uint blocks = richengine_sparse_history_blocks(s.first_position + first_row, rows);
  const uint words = min((blocks + 31) / 32, p.mask_words);
  const uint per = (words + kBlockListThreads - 1) / kBlockListThreads;
  const uint w0 = min(tid * per, words), w1 = min(w0 + per, words);
  device const uint *bits = mask + ulong(s.mask_row + first_row) * p.mask_words;
  // A tile's rows, unrolled over an entry's row bits, so that each row's word
  // stays in a register; a short last tile's missing rows read as zero.
  constexpr uint RowBits = RICHENGINE_ATTENTION_BLOCK_ROW_BITS;
  uint attended = 0;
  for (uint w = w0; w < w1; ++w) {
    uint any = 0;
#pragma unroll
    for (uint r = 0; r < RowBits; ++r)
      any |= r < rows ? bits[ulong(r) * p.mask_words + w] : 0u;
    attended += popcount(any & richengine_history_word_bits(blocks, w));
  }
  const uint inclusive = simd_prefix_inclusive_sum(attended);
  if (lane == 31)
    simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint before = inclusive - attended;
  for (uint g = 0; g < sg; ++g)
    before += simd_counts[g];
  device uint *list = lists + ulong(s.list + tile) * p.stride;
  for (uint w = w0; w < w1; ++w) {
    uint row_bits[RowBits];
    uint any = 0;
#pragma unroll
    for (uint r = 0; r < RowBits; ++r) {
      row_bits[r] = r < rows ? bits[ulong(r) * p.mask_words + w] : 0u;
      any |= row_bits[r];
    }
    any &= richengine_history_word_bits(blocks, w);
    while (any) {
      const uint b = ctz(any);
      any &= any - 1;
      uint entry = (w * 32 + b) << RICHENGINE_ATTENTION_BLOCK_ROW_BITS;
#pragma unroll
      for (uint r = 0; r < RowBits; ++r)
        entry |= (row_bits[r] >> b & 1u) << r;
      if (before < p.stride)
        list[before] = entry;
      ++before;
    }
  }
  if (tid == kBlockListThreads - 1)
    counts[s.list + tile] = min(before, p.stride);
}

// The entries of list `index` of the bound lists, at most its stride, or
// without a stride (no lists bound) the blocks of the tile's history.
inline uint richengine_sparse_list_count(device const uint *counts, uint stride, uint index,
                                         uint first_position, uint active_rows) {
  return stride ? min(counts[index], stride)
                : richengine_sparse_history_blocks(first_position, active_rows);
}

// A tile's list, null without a stride, and its entries' count.
struct RichSparseList {
  device const uint *entries;
  uint count;
};

inline RichSparseList richengine_sparse_list(device const uint *lists, device const uint *counts,
                                             uint stride, uint index, uint first_position,
                                             uint active_rows) {
  return {stride ? lists + ulong(index) * stride : nullptr,
          richengine_sparse_list_count(counts, stride, index, first_position, active_rows)};
}

// Verify: group.x is the KV head, group.y the split and group.z the lane's
// tile, lane * row_tiles + t of the row_capacity/8 eight-row tiles of the
// lane's rows, which attend through list block_list + t. The reduce takes
// the KV head in group.x. A tile past the lane's active rows is inactive.
#define SPARSE_VERIFY_SPLIT(Name, Heads, Group, CacheElement)                    \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *partials [[buffer(1)]],                                   \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table0 [[buffer(3)]],                      \
      device const RichKvPage *page_table1 [[buffer(4)]],                      \
      device const RichKvPage *page_table2 [[buffer(5)]],                      \
      device const RichKvPage *page_table3 [[buffer(6)]],                      \
      device const uint *lists [[buffer(7)]],                                  \
      device const uint *counts [[buffer(8)]],                                 \
      constant RichVerifyAttentionParams *params [[buffer(9)]],                \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = RichSparseTileRows;                                  \
    threadgroup RichSparseScratch<CacheElement, Rows * Group> scratch;         \
    const uint row_tiles = (params[0].row_capacity + Rows - 1) / Rows;         \
    const uint batch = group.z / row_tiles, tile = group.z % row_tiles;        \
    constant RichVerifyAttentionParams &lane = params[batch];                  \
    const uint kv_head = group.x, split = group.y;                             \
    const uint first_row = tile * Rows;                                        \
    if (!richengine_verify_attention_contract_valid(lane) ||                   \
        kv_head >= Heads || split >= lane.split_count ||                       \
        first_row >= lane.active_rows)                                         \
      return;                                                                  \
    const uint active_rows = min(Rows, lane.active_rows - first_row);          \
    const uint first_position = lane.committed_tokens + first_row;             \
    constexpr ulong group_stride =                                             \
        ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * Group * RichKvHeadDimension;   \
    const ulong offset =                                                       \
        (ulong(batch) * Heads + kv_head) * group_stride +                      \
        ulong(first_row) * Group * RichKvHeadDimension;                        \
    const ulong slot =                                                         \
        ((ulong(batch) * Heads + kv_head) * row_tiles + tile) *                \
            lane.slot_splits +                                                 \
        split;                                                                 \
    const RichSparseList list = richengine_sparse_list(                        \
        lists, counts, lane.block_stride, lane.block_list + tile,              \
        first_position, active_rows);                                          \
    richengine_sparse_attention_tile<Heads, Group, Rows, CacheElement>(        \
        queries + offset,                                                    \
        RICHENGINE_LANE_BINDING(batch, page_table0, page_table1, page_table2,  \
                                page_table3),                                  \
        lane.kv, kv_head, first_position, active_rows, list.entries,           \
        list.count, lane.split_count, split, partials, statistics, slot,       \
        lane.score_scale, scratch, thread_index);                              \
  }
SPARSE_VERIFY_SPLIT(verify_attention_q8_sparse_kv2_g12, 2, 12, int8_t)
SPARSE_VERIFY_SPLIT(verify_attention_bf16_sparse_kv2_g12, 2, 12, bfloat)
#undef SPARSE_VERIFY_SPLIT

// Combines the splits that took entries of the lane tile's list: the same
// lane-tile decode as the split, one threadgroup per fused row.
#define SPARSE_VERIFY_REDUCE(Name, Heads, Group)                                 \
  kernel void Name(                                                            \
      device const bfloat *partials [[buffer(0)]],                             \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const uint *counts [[buffer(3)]],                                 \
      constant RichVerifyAttentionParams *params [[buffer(4)]],                \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = RichSparseTileRows;                                  \
    threadgroup float weights[RichVerifyMaximumSplits];                        \
    threadgroup float group_values[8];                                         \
    const uint row_tiles = (params[0].row_capacity + Rows - 1) / Rows;         \
    const uint batch = group.z / row_tiles, tile = group.z % row_tiles;        \
    constant RichVerifyAttentionParams &lane = params[batch];                  \
    const uint kv_head = group.x, fused_row = group.y;                         \
    const uint first_row = tile * Rows;                                        \
    if (!richengine_verify_attention_contract_valid(lane) ||                   \
        kv_head >= Heads || fused_row >= Rows * Group ||                       \
        thread_index >= RichKvHeadDimension)                                   \
      return;                                                                  \
    const uint active_rows =                                                   \
        first_row < lane.active_rows                                           \
            ? min(Rows, lane.active_rows - first_row)                          \
            : 0u;                                                              \
    const uint first_position = lane.committed_tokens + first_row;             \
    const uint count = active_rows                                             \
                           ? richengine_sparse_list_count(                     \
                                 counts, lane.block_stride,                    \
                                 lane.block_list + tile, first_position,       \
                                 active_rows)                                  \
                           : 0u;                                               \
    constexpr ulong group_stride =                                             \
        ulong(RICHENGINE_VERIFY_CHUNK_STRIDE) * Group * RichKvHeadDimension;   \
    const ulong offset =                                                       \
        (ulong(batch) * Heads + kv_head) * group_stride +                      \
        ulong(first_row) * Group * RichKvHeadDimension;                        \
    const ulong head_slot =                                                    \
        ((ulong(batch) * Heads + kv_head) * row_tiles + tile) *                \
        lane.slot_splits;                                                      \
    richengine_sparse_combine_splits<Group, Rows>(                             \
        partials, statistics, output + offset, active_rows,                    \
        richengine_sparse_written_splits(count, lane.split_count), head_slot,  \
        fused_row, thread_index, weights, group_values);                       \
  }
SPARSE_VERIFY_REDUCE(verify_attention_sparse_reduce_kv2_g12, 2, 12)
#undef SPARSE_VERIFY_REDUCE

// Prefill: group.x is the KV head, group.y the query tile and group.z the
// split; tile t of the chunk attends through list block_list + t.
#define SPARSE_PREFILL_SPLIT(Name, Heads, Group, CacheElement)                   \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device bfloat *partials [[buffer(1)]],                                   \
      device float *statistics [[buffer(2)]],                                  \
      device const RichKvPage *page_table [[buffer(3)]],                       \
      device const uint *lists [[buffer(4)]],                                  \
      device const uint *counts [[buffer(5)]],                                 \
      constant RichPrefillAttentionParams &params [[buffer(6)]],               \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = RichSparseTileRows;                                  \
    threadgroup RichSparseScratch<CacheElement, Rows * Group> scratch;         \
    const uint kv_head = group.x, tile = group.y, split = group.z;             \
    const uint start = tile * Rows;                                            \
    if (!richengine_prefill_attention_contract_valid(params) ||                \
        kv_head >= Heads || split >= params.split_count ||                     \
        start >= params.rows)                                                  \
      return;                                                                  \
    const uint active_rows = min(Rows, params.rows - start);                   \
    const uint first_position = params.committed_tokens + start;               \
    const ulong offset =                                                       \
        (ulong(kv_head) * params.chunk_stride + start) * Group *               \
        RichKvHeadDimension;                                                   \
    const ulong slot =                                                         \
        (ulong(tile) * Heads + kv_head) * params.split_count + split;          \
    const RichSparseList list = richengine_sparse_list(                        \
        lists, counts, params.block_stride, params.block_list + tile,          \
        first_position, active_rows);                                          \
    richengine_sparse_attention_tile<Heads, Group, Rows, CacheElement>(        \
        queries + offset, page_table, params.kv, kv_head, first_position,      \
        active_rows, list.entries, list.count, params.split_count, split,      \
        partials, statistics, slot, params.score_scale, scratch,               \
        thread_index);                                                         \
  }
SPARSE_PREFILL_SPLIT(prefill_attention_q8_sparse_kv2_g12, 2, 12, int8_t)
SPARSE_PREFILL_SPLIT(prefill_attention_bf16_sparse_kv2_g12, 2, 12, bfloat)
#undef SPARSE_PREFILL_SPLIT

#define SPARSE_PREFILL_REDUCE(Name, Heads, Group)                                \
  kernel void Name(                                                            \
      device const bfloat *partials [[buffer(0)]],                             \
      device const float *statistics [[buffer(1)]],                            \
      device bfloat *output [[buffer(2)]],                                     \
      device const uint *counts [[buffer(3)]],                                 \
      constant RichPrefillAttentionParams &params [[buffer(4)]],               \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]]) {                     \
    constexpr uint Rows = RichSparseTileRows;                                  \
    threadgroup float weights[RichPrefillMaximumSplits];                       \
    threadgroup float group_values[8];                                         \
    const uint kv_head = group.x, fused_row = group.y, tile = group.z;         \
    const uint start = tile * Rows;                                            \
    if (!richengine_prefill_attention_contract_valid(params) ||                \
        kv_head >= Heads || fused_row >= Rows * Group ||                       \
        thread_index >= RichKvHeadDimension || start >= params.rows)           \
      return;                                                                  \
    const uint active_rows = min(Rows, params.rows - start);                   \
    const uint first_position = params.committed_tokens + start;               \
    const uint count = richengine_sparse_list_count(                           \
        counts, params.block_stride, params.block_list + tile,                 \
        first_position, active_rows);                                          \
    const ulong offset =                                                       \
        (ulong(kv_head) * params.chunk_stride + start) * Group *               \
        RichKvHeadDimension;                                                   \
    richengine_sparse_combine_splits<Group, Rows>(                             \
        partials, statistics, output + offset, active_rows,                    \
        richengine_sparse_written_splits(count, params.split_count),           \
        (ulong(tile) * Heads + kv_head) * params.split_count, fused_row,       \
        thread_index, weights, group_values);                                  \
  }
SPARSE_PREFILL_REDUCE(prefill_attention_sparse_reduce_kv2_g12, 2, 12)
#undef SPARSE_PREFILL_REDUCE
