#include "metal/abi/Mbsa.h"
#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/lane_bindings.h"
#include "metal/kernels/common/rms_inverse.h"

#include <metal_stdlib>

using namespace metal;

// The Maxwell MBSA indexer (metal/abi/Mbsa.h). Unlike the QSA block indexer
// there is no pooled-key pool or pending state: every token's index key
// lands in its page's index plane with its K/V row, normalized and rotated
// at store time, and a query row's route reduces each 128-token block at
// score time. mbsa_index_project fuses the project's two halves — the four
// group queries (a head each, RMS-normed and rotated as the main branch's
// are, written bf16 to the queries scratch) and the shared index key head
// (same treatment, written to the row's token slot of the index plane).
// mbsa_route turns the routes into the sparse attention's block lists: one
// threadgroup per (lane, group, tile) scores each tile row's remote-
// eligible blocks, keeps the top MBSA_REMOTE_BLOCKS, and emits the union of
// every row's prefix, remote and local blocks as ascending 4-token list
// entries (subblock << RICHENGINE_ATTENTION_BLOCK_ROW_BITS | one bit per tile
// row attending), the format the _k4q8d128 sparse splits and reduce read.

constant constexpr uint kDimension = MBSA_INDEX_DIMENSION;
constant constexpr uint kHeads = MBSA_INDEX_HEADS;
constant constexpr uint kPairs = MBSA_ROPE_PAIRS; // rotated pairs (i, i + kPairs)
constant constexpr uint kPageTokens = RICHENGINE_TARGET_KV_BLOCK_TOKENS;
constant constexpr uint kRouteBlock = MBSA_ROUTE_BLOCK;
constant constexpr uint kPrefix = MBSA_PREFIX_BLOCKS;
constant constexpr uint kLocal = MBSA_LOCAL_BLOCKS;
constant constexpr uint kRemote = MBSA_REMOTE_BLOCKS;
constant constexpr uint kRows = MBSA_ROUTE_ROWS;
constant constexpr uint kThreads = MBSA_ROUTE_THREADS;
constant constexpr uint kSub = MBSA_ROUTE_SUBBLOCKS;
constant constexpr uint kMaxBlocks = MBSA_ROUTE_MAX_BLOCKS;
constant constexpr uint kRowBits = RICHENGINE_ATTENTION_BLOCK_ROW_BITS;
constant constexpr uint kBlockTokens = RICHENGINE_ATTENTION_BLOCK_TOKENS;
// index_softmax_scale of the modeling code.
constant constexpr float kScoreScale = 0.08838834764831845f; // 128^-0.5

// Lane `lane` of a dispatch, none past its lanes.
inline bool mbsa_lane(uint lane, uint lanes) {
  return lane < min(lanes, RICHENGINE_MAXIMUM_BATCH_WIDTH);
}

// The index keys of a page's tokens, through its index-table entry: the
// extent's index plane plus the layer's index region, then the page's own
// kPageTokens rows of kDimension (the entry's low bits hold the page's index
// in its extent, as every KV entry).
inline device bfloat *mbsa_index_keys(RichKvPage entry, uint index_offset) {
  const uint index = richengine_kv_page_index(entry);
  return reinterpret_cast<device bfloat *>(richengine_kv_extent(entry, index) + index_offset) +
         ulong(index) * kPageTokens * kDimension;
}

// The inverse RMS of the threadgroup's kDimension values, one per thread.
inline float mbsa_rms_scale(float value, threadgroup float *scratch, uint lane, uint sg) {
  const float sum = simd_sum(value * value);
  if (lane == 0) scratch[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  float total = 0.0f;
  for (uint g = 0; g < kDimension / 32; ++g) total += scratch[g];
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  return rsqrt(total / kDimension + kRmsEpsilon);
}

// Rotates the whole 128-wide head as the attention does: pairs (i, i +
// kPairs) by the table's frequency i, i < kPairs = kDimension / 2. `rotated`
// stages the normed row for the cross-half read.
inline float mbsa_rotate(float value, threadgroup float *rotated, device const float *cosine,
                         device const float *sine, uint tid) {
  rotated[tid] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  float result = value;
  if (tid < kPairs) result = value * cosine[tid] - rotated[tid + kPairs] * sine[tid];
  else result = value * cosine[tid - kPairs] + rotated[tid - kPairs] * sine[tid - kPairs];
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  return result;
}

// Threadgroup (row, head, lane) of kDimension threads: heads 0..kHeads-1
// write the row's normalized, rotated group queries bf16; head kHeads writes
// the row's shared index key, normalized and rotated the same way, to the
// token's slot of the index plane. The threadgroup first dots the row's
// fp32 normed input against its head's weight rows — the projections the
// split path ran as two gguf_float dispatches into fp32 scratch — then
// applies the norm, rotation and store. The query heads' weight rows are
// query_weights[head * kDimension + output], the key head's
// key_weights[output]; the projected rows follow the lanes' (lane * rows +
// row), as do the RoPE tables'.
kernel void mbsa_index_project(device const bfloat *input [[buffer(0)]],
                               device const float *query_weights [[buffer(1)]],
                               device const float *key_weights [[buffer(2)]],
                               device const float *query_norm [[buffer(3)]],
                               device const float *key_norm [[buffer(4)]],
                               device const float *rope_cos [[buffer(5)]],
                               device const float *rope_sin [[buffer(6)]],
                               device bfloat *queries [[buffer(7)]],
                               device const RichKvPage *index_table0 [[buffer(8)]],
                               device const RichKvPage *index_table1 [[buffer(9)]],
                               device const RichKvPage *index_table2 [[buffer(10)]],
                               device const RichKvPage *index_table3 [[buffer(11)]],
                               constant MbsaIndexParams &p [[buffer(12)]],
                               uint3 group [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float scratch[kDimension / 32];
  threadgroup float rotated[kDimension];
  threadgroup float x[MBSA_INDEX_INPUT_WIDTH];
  const uint head = group.y, batch = group.z;
  if (group.x >= p.rows || !mbsa_lane(batch, p.lanes)) return;
  const ulong row = ulong(batch) * p.rows + group.x;
  const ulong table = row * kPairs;
  for (uint i = tid; i < p.input_width; i += kDimension)
    x[i] = float(input[row * p.input_width + i]);
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  // The projection: thread `tid` dots its output row of the head's weight
  // plane against the staged input — the same weight bytes the split
  // path's gguf_float read, one threadgroup-resident input row for all of
  // them.
  const uint weight_row = head == kHeads ? tid : head * kDimension + tid;
  device const float *weights =
      head == kHeads ? key_weights : query_weights;
  const float4 *x4 = reinterpret_cast<threadgroup const float4 *>(x);
  const float4 *w4 =
      reinterpret_cast<device const float4 *>(weights + ulong(weight_row) * p.input_width);
  float value = 0.0f;
  for (uint k = 0; k < p.input_width / 4; ++k) {
    const float4 xv = x4[k], wv = w4[k];
    value += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
  }
  if (head == kHeads) {
    const float scale = mbsa_rms_scale(value, scratch, lane, sg);
    const float result =
        mbsa_rotate(value * scale * key_norm[tid], rotated, rope_cos + table, rope_sin + table, tid);
    const uint position = p.first_position[batch] + group.x;
    device const RichKvPage *index_table =
        RICHENGINE_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
    device bfloat *slot = mbsa_index_keys(index_table[position / kPageTokens], p.index_offset) +
                          (position % kPageTokens) * kDimension;
    slot[tid] = bfloat(result);
    return;
  }
  const float scale = mbsa_rms_scale(value, scratch, lane, sg);
  const float result =
      mbsa_rotate(value * scale * query_norm[tid], rotated, rope_cos + table, rope_sin + table, tid);
  queries[(row * kHeads + head) * kDimension + tid] = bfloat(result);
}

// A block b of the route of the row at block index `bp` is in the prefix
// ([0, kPrefix) reaching the row's own block), the local band
// ([max(bp - kLocal + 1, kPrefix), bp]), or the remote band the threadgroup
// selected into remote[r]: the three disjoint bands the modeling code
// unions.
inline bool mbsa_route_has(uint b, uint bp, threadgroup const uint *remote_row) {
  if (b <= bp) {
    if (b < kPrefix || b + kLocal > bp) return true;
    for (uint j = 0; j < kRemote; ++j)
      if (remote_row[j] == b) return true;
  }
  return false;
}

// Threadgroup (tile, group, lane): the group's route of each of the tile's
// up-to-kRows rows, unioned into the (lane, group, tile) block list the
// _k4q8d128 splits read. A row at position p = first + r routes to the
// prefix and local blocks (arithmetic, no scoring) plus the top-kRemote of
// the eligible band [kPrefix, bp - kLocal], whose score is the largest
// q_g . k_t * kScoreScale of its kRouteBlock index keys — every eligible
// token precedes p, so the scan needs no causal mask (the sparse tile's
// causal mask covers the row's own open block's tail). Selection is sixteen
// rounds of a threadgroup argmax of a sortable (score, ~block) key, ties
// toward the earlier block; a band of at most kRemote blocks is taken
// whole. The emit keeps each block's subblocks in ascending order and none
// past the tile's history, which the splits would discard anyway.
kernel void mbsa_route(device const bfloat *queries [[buffer(0)]],
                       device const RichKvPage *index_table0 [[buffer(1)]],
                       device const RichKvPage *index_table1 [[buffer(2)]],
                       device const RichKvPage *index_table2 [[buffer(3)]],
                       device const RichKvPage *index_table3 [[buffer(4)]],
                       device uint *lists [[buffer(5)]],
                       device uint *counts [[buffer(6)]],
                       constant MbsaRouteParams &p [[buffer(7)]],
                       uint3 group [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float q[kRows * kDimension];
  threadgroup float block_score[kMaxBlocks];
  threadgroup uchar emit[kMaxBlocks];
  threadgroup uint remote[kRows][kRemote];
  threadgroup uint sg_best[kThreads / 32];
  threadgroup uint sg_score[kThreads / 32];
  threadgroup uint simd_counts[kThreads / 32];
  const uint tile = group.x, head = group.y, batch = group.z;
  if (!mbsa_lane(batch, p.lanes)) return;
  const uint list_index = (p.list + batch * p.row_tiles + tile) * kHeads + head;
  const uint first_row = tile * kRows;
  const uint active = p.active_rows[batch];
  if (first_row >= active) {
    // A tile past the lane's live rows lists nothing — the splits skip it,
    // but the reduce still reads its count.
    if (tid == 0) counts[list_index] = 0;
    return;
  }
  const uint rows = min(kRows, active - first_row);
  const uint first = p.first_position[batch] + first_row;
  device const RichKvPage *index_table =
      RICHENGINE_LANE_BINDING(batch, index_table0, index_table1, index_table2, index_table3);
  // Stage the group's queries for the tile's rows (rows are kHeads heads
  // apart in the queries scratch).
  device const bfloat *qrow =
      queries + ((ulong(batch) * p.rows + first_row) * kHeads + head) * kDimension;
  for (uint i = tid; i < kRows * kDimension; i += kThreads)
    q[i] = i / kDimension < rows
               ? float(qrow[(i / kDimension) * kHeads * kDimension + i % kDimension])
               : 0.0f;
  for (uint j = tid; j < kRows * kRemote; j += kThreads) remote[j / kRemote][j % kRemote] = ~0u;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  // Each row's remote band: score every eligible block, then the argmax
  // rounds. block_score[b] holds the current row's score of block b.
  for (uint r = 0; r < rows; ++r) {
    const uint bp = (first + r) / kRouteBlock;
    // Eligible: kPrefix <= b and b + 1 <= max(bp - kLocal + 1, 0) — nonempty
    // from bp >= kPrefix + kLocal.
    if (bp < kPrefix + kLocal) continue;
    const uint lo = kPrefix, hi = bp - kLocal; // inclusive
    const uint band = hi - lo + 1;
    for (uint b = lo + tid; b <= hi; b += kThreads) {
      const threadgroup float *qr = q + r * kDimension;
      float best = -INFINITY;
      for (uint token = 0; token < kRouteBlock; ++token) {
        const uint t = b * kRouteBlock + token;
        device const bfloat *k = mbsa_index_keys(index_table[t / kPageTokens], p.index_offset) +
                                 (t % kPageTokens) * kDimension;
        float dot = 0.0f;
        for (uint d = 0; d < kDimension; ++d) dot += qr[d] * float(k[d]);
        best = max(best, dot);
      }
      block_score[b] = best * kScoreScale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    if (band <= kRemote) {
      for (uint j = tid; j < band; j += kThreads) remote[r][j] = lo + j;
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
      continue;
    }
    // Sortable keys: the score's bits sign-flipped into order, ties toward
    // the earlier block. Each thread argmaxes its strided run, the
    // simdgroup its lanes', and thread 0 the simdgroups' (two 32-bit
    // reductions; simd_max has no 64-bit form).
    const auto sortable = [](float score) -> uint {
      uint u = as_type<uint>(score);
      return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
    };
    for (uint k = 0; k < kRemote; ++k) {
      uint u_best = 0, b_best = ~0u;
      for (uint b = lo + tid; b <= hi; b += kThreads) {
        const uint u = sortable(block_score[b]);
        if (u > u_best || (u == u_best && b < b_best)) {
          u_best = u;
          b_best = b;
        }
      }
      const uint mine = u_best;
      u_best = simd_max(u_best);
      b_best = simd_min(mine == u_best ? b_best : ~0u);
      if (lane == 0) {
        sg_score[sg] = u_best;
        sg_best[sg] = b_best;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
      if (tid == 0) {
        uint score = sg_score[0], block = sg_best[0];
        for (uint g = 1; g < kThreads / 32; ++g) {
          if (sg_score[g] > score || (sg_score[g] == score && sg_best[g] < block)) {
            score = sg_score[g];
            block = sg_best[g];
          }
        }
        if (block < kMaxBlocks) {
          remote[r][k] = block;
          block_score[block] = -INFINITY;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    }
  }
  // The union over the tile's rows: emit[b] is the row bits of 128-block b.
  // Only blocks of the row's route up to its own count; candidates stop at
  // the last row's block.
  const uint last = first + rows - 1;
  const uint candidates = last / kRouteBlock + 1;
  for (uint b = tid; b < candidates; b += kThreads) {
    uint bits = 0;
    for (uint r = 0; r < rows; ++r) {
      const uint bp = (first + r) / kRouteBlock;
      bits |= mbsa_route_has(b, bp, remote[r]) ? 1u << r : 0u;
    }
    emit[b] = uchar(bits);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  // The block-list history the splits compute for this tile: entries past
  // it would be discarded, so they are never emitted.
  const uint history = (first + rows + kBlockTokens - 1) / kBlockTokens;
  const auto subblocks = [&](uint b) -> uint {
    if (!emit[b] || b * kSub >= history) return 0u;
    return min(kSub, history - b * kSub);
  };
  // Contiguous runs per thread, a group-wide exclusive scan for the write
  // offsets, then each run's entries in ascending order (the
  // attention_block_lists pattern).
  const uint per = (candidates + kThreads - 1) / kThreads;
  const uint b0 = min(tid * per, candidates), b1 = min(b0 + per, candidates);
  uint mine = 0;
  for (uint b = b0; b < b1; ++b) mine += subblocks(b);
  const uint inclusive = simd_prefix_inclusive_sum(mine);
  if (lane == 31) simd_counts[sg] = inclusive;
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  uint before = inclusive - mine;
  for (uint g = 0; g < sg; ++g) before += simd_counts[g];
  device uint *list = lists + ulong(list_index) * p.list_stride;
  for (uint b = b0; b < b1; ++b) {
    const uint subs = subblocks(b);
    for (uint s = 0; s < subs; ++s) {
      if (before < p.list_stride)
        list[before] = (b * kSub + s) << kRowBits | uint(emit[b]);
      ++before;
    }
  }
  if (tid == kThreads - 1) counts[list_index] = min(before, p.list_stride);
}
