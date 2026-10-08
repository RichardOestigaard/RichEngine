#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Paged attention over INT8 or BF16 KV. INT8 has one fp32 scale per
// (token, KV head); BF16 reads the stored values directly. The preceding
// store writes current rows to their final slots; committed_tokens controls
// visibility, and the next command overwrites rejected verify rows.
//
// A threadgroup owns one KV head, query tile and history split. It processes
// the tile's GQA group as M = rows x query heads per KV head. Each split writes
// fp32 partials and statistics for a fixed-order reduce.
//
// Prefill and verify share this device-operand page loop.
constant uint RichVerifyMaximumSplits =
    RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS;
constant uint RichPrefillTileRows = RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;
constant uint RichPrefillMaximumSplits =
    RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS;

inline bool richengine_prefill_attention_contract_valid(
    constant RichPrefillAttentionParams &params) {
  return params.rows > 0 && params.rows <= RICHENGINE_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride >= params.rows &&
         params.chunk_stride <= RICHENGINE_PREFILL_TOKEN_BUDGET &&
         params.chunk_stride % RICHENGINE_TARGET_KV_BLOCK_TOKENS == 0 &&
         params.page_table_entries >=
             (params.committed_tokens + params.rows +
              RichKvPageTokens - 1) /
                 RichKvPageTokens &&
         params.kv.extent_pages > 0 && params.split_count > 0 &&
         params.split_count <= RichPrefillMaximumSplits &&
         ulong(params.committed_tokens) + params.rows <=
             ulong(RICHENGINE_MAXIMUM_PHYSICAL_KV_TOKENS);
}

inline bool richengine_verify_attention_contract_valid(
    constant RichVerifyAttentionParams &params) {
  return params.active_rows > 0 &&
         params.active_rows <= params.row_capacity &&
         params.page_table_entries >=
             (params.committed_tokens + params.active_rows +
              RichKvPageTokens - 1) /
                 RichKvPageTokens &&
         params.kv.extent_pages > 0 &&
         ulong(params.committed_tokens) + params.active_rows <=
             ulong(RICHENGINE_MAXIMUM_PHYSICAL_KV_TOKENS) &&
         params.split_count > 0 &&
         params.split_count <= RichVerifyMaximumSplits &&
         params.slot_splits >= params.split_count &&
         params.slot_splits <= RichVerifyMaximumSplits;
}

// Split and reduce derive the same balanced partition of each query tile's
// visible pages. Causal masking remains per query row inside each split.
inline uint richengine_attention_pages(uint visible_tokens) {
  return (visible_tokens + RichKvPageTokens - 1) / RichKvPageTokens;
}

inline uint richengine_attention_pages_per_split(uint pages, uint splits) {
  return (pages + splits - 1) / splits;
}

// One Page32 block of one tile: key-scaled scores become value-scaled bf16
// probabilities and the row statistics advance. Four lanes own one fused row,
// eight consecutive tokens each, so the row maximum and sum are two xor
// shuffles instead of a 32-lane reduction per row and the eight bf16
// probabilities leave as one 16-byte store. Threads past 4 x FusedRows
// (kv4_g6: simdgroups 6 and 7) only take part in the caller's barriers.
// The float4 loads need 16-byte aligned slabs (alignas in the entries) and
// the page's 32 key and value scales as float4 vectors, in device memory
// (richengine_q8_scale_index is a multiple of 32 floats and every region of an
// extent starts 64 KiB-aligned).
// Rows whose running maximum grew atomically set the shared boolean rescale
// flag. Existing threadgroup barriers separate reset, concurrent set, and
// read; relaxed atomics make the same-value writes safe without changing
// arithmetic.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, bool Quantized,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION, uint TokensPerLane = 8>
inline void richengine_attention_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start, uint fused_row_offset,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    device const uint *row_masks, uint thread_index, float qk_scale = 0.0f,
    uint window_tokens = 0) {
  constexpr uint N = RichKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint LanesPerRow = N / TokensPerLane;
  static_assert(N % TokensPerLane == 0 && TokensPerLane % 8 == 0,
                "each softmax lane covers whole float4 vector groups");
  static_assert(N == 32, "four lanes of eight tokens span one page");
  static_assert(LanesPerRow * FusedRows <= 256,
                "one softmax lane per thread of the 256-thread tile");
  if (thread_index >= LanesPerRow * FusedRows)
    return;
  const uint fused_row = thread_index / LanesPerRow;
  const uint column = thread_index % LanesPerRow * TokensPerLane;
  const uint query_row =
      (fused_row_offset + fused_row) / QueryHeadsPerKVHead;
  const uint causal_end =
      committed_tokens + min(query_row, active_rows - 1) + 1;
  const uint limit = min(visible_tokens, causal_end);
  // A nonzero window_tokens admits only the last window_tokens tokens ending
  // at the row's causal end (Gemma 4's sliding-window layers); zero is the
  // full causal history.
  const uint window_begin =
      window_tokens != 0 && causal_end > window_tokens
          ? causal_end - window_tokens
          : 0;
  // A tree lane's scratch tokens are not a prefix of its query row: the row's
  // ancestor bitmask decides which of the visible rows past the committed
  // history it attends. A null mask is the chain's causal limit.
  const uint row_mask = row_masks ? row_masks[query_row] : 0u;
  const uint token = token_start + column;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                   column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    const float score_scale =
        qk_scale != 0.0f ? qk_scale
                         : (HeadDim == 64    ? 0.125f
                            : HeadDim == 128 ? 0.08838834764831845f
                                             : 0.0625f);
#pragma unroll
    for (uint i = 0; i < TokensPerLane / 4; ++i) {
      float4 s = scores4[i];
      if constexpr (Quantized)
        s *= key_scales[vector + i];
      s *= score_scale;
      score[4 * i] = s.x, score[4 * i + 1] = s.y, score[4 * i + 2] = s.z,
      score[4 * i + 3] = s.w;
    }
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    const uint t = token + j;
    const bool admitted = t >= window_begin &&
        (!row_masks ? t < limit
        : (t < committed_tokens) ||
              (t < visible_tokens && (row_mask >> (t - committed_tokens)) & 1u));
    score[j] = admitted ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  // The butterfly spans a row's lanes only: with sixteen tokens per lane
  // (fused rows above 64) a row covers two lanes, so the second leg would
  // leak the adjacent row's maxima and sums into this one's softmax.
#pragma unroll
  for (uint lane = 1; lane < LanesPerRow; lane <<= 1)
    local_max = max(local_max, simd_shuffle_xor(local_max, lane));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    const uint t = token + j;
    const bool admitted = t >= window_begin &&
        (!row_masks ? t < limit
        : (t < committed_tokens) ||
              (t < visible_tokens && (row_mask >> (t - committed_tokens)) & 1u));
    probability[j] = admitted ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
#pragma unroll
  for (uint lane = 1; lane < LanesPerRow; lane <<= 1)
    local_sum += simd_shuffle_xor(local_sum, lane);
  if (column == 0) {
    const float scale = next_max == -INFINITY || next_max == previous_max
                            ? 1.0f
                            : fast::exp(previous_max - next_max);
    previous_scale[fused_row] = scale;
    row_sum[fused_row] = row_sum[fused_row] * scale + local_sum;
    row_max[fused_row] = next_max;
    if (scale != 1.0f)
      atomic_store_explicit(rescale, 1u, memory_order_relaxed);
  }
  // Masked tokens stay exactly zero whatever their stored value scale holds.
  threadgroup bfloat4 *probabilities4 = reinterpret_cast<threadgroup bfloat4 *>(
      probabilities + fused_row * N + column);
#pragma unroll
  for (uint i = 0; i < TokensPerLane / 4; ++i) {
    float4 vs(1.0f);
    if constexpr (Quantized)
      vs = value_scales[vector + i];
    const float4 out(
        score[4 * i] > -INFINITY ? probability[4 * i] * vs.x : 0.0f,
        score[4 * i + 1] > -INFINITY ? probability[4 * i + 1] * vs.y : 0.0f,
        score[4 * i + 2] > -INFINITY ? probability[4 * i + 2] * vs.z : 0.0f,
        score[4 * i + 3] > -INFINITY ? probability[4 * i + 3] * vs.w : 0.0f);
    probabilities4[i] = bfloat4(out);
  }
}

// One tile over its split's pages with the INT8 K/V and the scales consumed
// in place as device tensor operands: the prefill tile, and the verify tile
// in its direct placement, which differs only in RowsPerTile. Queries are
// KV-head-major [kv head][row][query head in group][dimension], so the tile's
// fused rows form one contiguous M x D tensor. Each page is reached through
// its table entry (kv_extent.h).
// Three barriers per page order the score store, the softmax and the
// probability reads of PV. Packed-INT4 pages take the same path: both packed
// tensors are token-major (the even element in the low nibble), so each is a
// {D, N} strides {1, D} int4b view of its slab. QK reads keys that way
// directly (NT); an int4b operand must be contiguous along its first extent,
// so PV reads the values as the NN right operand {N, K} = {D, N}. The scales
// are still consumed by the softmax epilogue, unchanged from INT8. The
// probability buffer ping-pongs between pages so a tensorop's operand reads
// — whose ordering with threadgroup writes through threadgroup_barrier is
// not documented — can never overlap the next softmax's stores (the
// probabilities parameter points at a two-page allocation).
// MPasses > 1 runs the fused rows in that many passes of MP = M / MPasses
// rows each, the fp8 tile's MP = M/2 pattern: every pass' QK/PV/softmax
// scratch and, what motivates it at hd512, PV accumulator registers shrink
// by the pass factor (M=64 x D=512 is 128 fp32 per thread — the whole M5
// per-core pool; MP=32 halves that). Row-halving is exact — every fused row
// attends independently, so each pass accumulates and stores its own rows of
// the shared [slot][M][D] partials and [slot][M]{max,sum} statistics layout.
// The reduce kernels and the workspace sizes are unchanged.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          typename CacheElement,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION, uint MPasses = 1>
inline void richengine_paged_attention_tile(
    device bfloat *tile_queries, device const RichKvPage *page_table,
    RichKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    device const uint *row_masks, float qk_scale,
    threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    uint thread_index, uint window_tokens = 0) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort MP = M / MPasses;
  static_assert(M % MPasses == 0, "row passes split the fused tile evenly");
  constexpr bool Packed = is_same<CacheElement, RichKvPacked4>::value;
  constexpr bool Quantized = !is_same<CacheElement, bfloat>::value;
  constexpr ushort N = RichKvPageTokens;
  constexpr ushort D = HeadDim;
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = richengine_attention_pages(visible_tokens);
  uint per_split = richengine_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);

  auto st = tensor(scores, dextents<int, 2>{N, MP}, array<int, 2>{1, N});
  // The probability buffer alternates by page parity: whether a tensorop's
  // operand reads are fenced by threadgroup_barrier the way plain loads are
  // is unproven, so a lingering PV read must never collide with the next
  // page's softmax writes (the callers allocate the buffer twice over).
  typedef decltype(tensor(static_cast<threadgroup bfloat *>(nullptr),
                          dextents<int, 2>{N, MP}, array<int, 2>{1, N})
                       .slice<N, MP>(0, 0)) PSliced;
  const RichKvAddressing<KVHeads, CacheElement, HeadDim> addressing(kv, kv_head);
  // QK writes a complete page score tile, the softmax applies an INT8 page's
  // key scales and PV accumulates the running output.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(MP, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  // Packed PV is NN: the token-major values are the {D, N} strides {1, D}
  // right operand, contiguous along D as int4b requires.
  constexpr auto pv_descriptor =
      matmul2d_descriptor(MP, D, N, false, !Packed, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;
  typedef tensor<device int4b_format, dextents<int, 2>, tensor_inline>
      Packed4Tensor;

  for (ushort pass = 0; pass < MPasses; ++pass) {
    const uint row_offset = uint(pass) * MP;
    if (thread_index < MP) {
      row_max[thread_index] = -INFINITY;
      row_sum[thread_index] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto qt = tensor(tile_queries + row_offset * D, dextents<int, 2>{D, MP},
                     array<int, 2>{1, D});
    auto q0 = qt.template slice<D, MP>(0, 0);
    auto running = [&] {
      if constexpr (Packed)
        return pv.template get_destination_cooperative_tensor<
            PSliced,
            decltype(Packed4Tensor(static_cast<device uchar *>(nullptr),
                                   dextents<int, 2>{D, N}, array<int, 2>{1, D})
                         .template slice<D, N>(0, 0)),
            float>();
      else
        return pv.template get_destination_cooperative_tensor<
            PSliced,
            decltype(tensor(static_cast<device CacheElement *>(nullptr),
                            dextents<int, 2>{N, D}, array<int, 2>{1, N})
                         .template slice<N, D>(0, 0)),
            float>();
    }();
    // Uniform capacity partitions the logical tile across eight 32-lane
    // groups. Equality proves no padding; otherwise only valid entries may
    // be accessed.
    const bool running_full =
        uint(running.get_capacity()) * (8u * 32u) == uint(MP) * D;
#pragma unroll
    for (ushort index = 0; index < running.get_capacity(); ++index) {
      if (running_full || running.is_valid_element(index))
        running[index] = 0.0f;
    }

    for (uint page = page_begin; page < page_end; ++page) {
      const auto tensors = addressing.page(page_table[page]);
      uint token_start = page * N;
      device const float *key_scales = tensors.key_scales;
      device const float *value_scales = tensors.value_scales;

      auto run_page = [&](auto ks, auto vs) {
        threadgroup bfloat *pbuf = probabilities + (page & 1) * (N * MP);
        auto pt = tensor(pbuf, dextents<int, 2>{N, MP}, array<int, 2>{1, N});
        auto p0 = pt.template slice<N, MP>(0, 0);
        auto page_scores = qk.template get_destination_cooperative_tensor<
            decltype(q0), decltype(ks), float>();
        // One full-dimension product initializes the score CT through MPP.
        qk.run(q0, ks, page_scores);
        page_scores.store(st.template slice<N, MP>(0, 0));
        if (thread_index == 0)
          atomic_store_explicit(rescale, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        richengine_attention_page_softmax<QueryHeadsPerKVHead,
            RowsPerTile / MPasses, Quantized, HeadDim,
            (MP > 64 ? 16 : 8)>(
            scores, pbuf, row_max, row_sum, previous_scale, rescale,
            reinterpret_cast<device const float4 *>(key_scales),
            reinterpret_cast<device const float4 *>(value_scales), token_start,
            row_offset, visible_tokens, committed_tokens, active_rows,
            row_masks, thread_index, qk_scale, window_tokens);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
          for (ushort index = 0; index < running.get_capacity(); ++index) {
            if (!running_full && !running.is_valid_element(index))
              continue;
            auto coordinates = running.get_multidimensional_index(index);
            running[index] *= previous_scale[coordinates[1]];
          }
        }
        pv.run(p0, vs, running);
        threadgroup_barrier(mem_flags::mem_threadgroup);
      };

      if constexpr (Packed) {
        run_page(Packed4Tensor(tensors.keys, dextents<int, 2>{D, N},
                               array<int, 2>{1, D})
                     .template slice<D, N>(0, 0),
                 Packed4Tensor(tensors.values, dextents<int, 2>{D, N},
                               array<int, 2>{1, D})
                     .template slice<D, N>(0, 0));
      } else {
        run_page(
            tensor(tensors.keys, dextents<int, 2>{D, N}, array<int, 2>{1, D})
                .template slice<D, N>(0, 0),
            tensor(tensors.values, dextents<int, 2>{N, D}, array<int, 2>{1, N})
                .template slice<N, D>(0, 0));
      }
    }
    auto target = tensor(partials + slot * M * D + row_offset * D,
                         dextents<int, 2>{D, MP}, array<int, 2>{1, D});
    running.store(target.template slice<D, MP>(0, 0));
    if (thread_index < MP) {
      statistics[(slot * M + row_offset + thread_index) * 2] =
          row_max[thread_index];
      statistics[(slot * M + row_offset + thread_index) * 2 + 1] =
          row_sum[thread_index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// Partials are [slot][fused row][dimension]; statistics are
// [slot][fused row]{max, sum}. The callers lay slots out as
// [tile][KV head][split] (prefill) and [lane][KV head][split] (verify).
// Combines the splits of one fused row in split order. Only splits that own
// at least one page were written; the partition is recomputed here. The
// statistics are shared by all 256 output dimensions: their weights are
// computed once per row, then each lane streams one dimension of the
// partials. A row past the tile's active rows is written as zeros; a
// threadgroup owns one fused row, so it returns uniformly. The value form
// feeds richengine_attention_reduce_row and the fused verify reduce/gate, which
// scales the same bf16 result by its query gate.
template <uint QueryHeadsPerKVHead, uint RowsPerTile,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline bfloat richengine_attention_reduce_value(
    device const float *partials, device const float *statistics,
    uint committed_tokens, uint active_rows,
    uint splits, ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values,
    uint visible_tokens = 0) {
  constexpr uint M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint D = HeadDim;
  constexpr uint Groups = D / 32;
  if (fused_row / QueryHeadsPerKVHead >= active_rows)
    return bfloat(0.0f);
  const uint pages = richengine_attention_pages(
      visible_tokens ? visible_tokens : committed_tokens + active_rows);
  const uint per_split = richengine_attention_pages_per_split(pages, splits);
  const uint written = (pages + per_split - 1) / per_split;
  const uint lane = thread_index % 32, sg = thread_index / 32;
  // Threads cover the written splits strided, so a head dimension below the
  // maximum split count still reduces every split.
  float maximum = -INFINITY;
  for (uint split = thread_index; split < written; split += D)
    maximum = max(maximum,
                  statistics[((head_slot + split) * M + fused_row) * 2]);
  const float group_maximum = simd_max(maximum);
  if (lane == 0) group_values[sg] = group_maximum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float row_maximum =
      simd_max(lane < Groups ? group_values[lane] : -INFINITY);
  for (uint split = thread_index; split < written; split += D)
    weights[split] = fast::exp(
        statistics[((head_slot + split) * M + fused_row) * 2] - row_maximum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float numerator = 0.0f, denominator = 0.0f;
  // Keep both accumulations in the original split order. A parallel sum of
  // the denominator changes speculative acceptance on real model prompts.
  for (uint split = 0; split < written; ++split) {
    const float weight = weights[split];
    const ulong stat = ((head_slot + split) * M + fused_row) * 2;
    numerator += weight *
        partials[((head_slot + split) * M + fused_row) * D + thread_index];
    denominator += weight * statistics[stat + 1];
  }
  return bfloat(denominator > 0.0f ? numerator / denominator : 0.0f);
}

template <uint QueryHeadsPerKVHead, uint RowsPerTile,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_attention_reduce_row(
    device const float *partials, device const float *statistics,
    device bfloat *tile_output, uint committed_tokens, uint active_rows,
    uint splits, ulong head_slot, uint fused_row, uint thread_index,
    threadgroup float *weights, threadgroup float *group_values,
    uint visible_tokens = 0) {
  tile_output[fused_row * HeadDim + thread_index] =
      richengine_attention_reduce_value<QueryHeadsPerKVHead, RowsPerTile, HeadDim>(
          partials, statistics, committed_tokens, active_rows, splits,
          head_slot, fused_row, thread_index, weights, group_values,
          visible_tokens);
}
