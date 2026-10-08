#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// DiffusionGemma canvas attention: a copy of the paged-attention tile loop
// (paged_attention_tile.h) with canvas mask semantics. A canvas query row
// attends bidirectionally to every visible token — all paged encoder-prefix
// tokens plus all canvas positions — so the per-row causal limit of the AR
// path becomes the whole visible range. The canvas tokens' K/V are not
// special-cased: the host appends the canvas scratch pages' entries to the
// request's page table (RichKvPage entries are extent GPU addresses; the
// scratch buffer is an ordinary pool-format extent), so prefix and canvas
// pages form one contiguous token space [0, committed + rows).
//
// The sliding-window (_swa) variants window only the prefix: window_begin =
// committed_tokens - window_tokens, so canvas positions are always admitted.
//
// No canvas reduce or store kernels exist: prefill_attention_reduce_gemma_*
// and the prefill_attention_*_store_gemma_* kernels are reused unchanged —
// their page accounting already covers committed + rows.

constant uint RichCanvasVerifyMaximumSplits =
    RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS;
constant uint RichCanvasTileRows = RICHENGINE_PREFILL_ATTENTION_TILE_ROWS;
constant uint RichCanvasMaximumSplits =
    RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS;

// One Page32 block of one canvas tile: identical to
// richengine_attention_page_softmax except the admission test. Every fused
// row admits t in [window_begin, visible_tokens); window_begin bounds the
// paged prefix only. Score scaling and the rescale machinery are unchanged.
template <uint QueryHeadsPerKVHead, uint RowsPerTile, bool Quantized,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION, uint TokensPerLane = 8>
inline void richengine_canvas_page_softmax(
    threadgroup const float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start,
    uint visible_tokens, uint committed_tokens,
    uint thread_index, float qk_scale = 0.0f,
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
  // Canvas rows see the whole visible range; a nonzero window bounds only
  // the paged prefix, so every canvas position (t >= committed_tokens) is
  // admitted regardless of the window.
  const uint limit = visible_tokens;
  const uint window_begin =
      window_tokens != 0 && committed_tokens > window_tokens
          ? committed_tokens - window_tokens
          : 0;
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
    const bool admitted = t >= window_begin && t < limit;
    score[j] = admitted ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
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
    const bool admitted = t >= window_begin && t < limit;
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

// The canvas tile: richengine_paged_attention_tile with
// richengine_canvas_page_softmax in place of the causal page softmax. All
// layout, split and partial/statistics contracts are the AR tile's,
// including MPasses: each pass covers M/MPasses fused rows of the shared
// [slot][M][D] partials, halving the hd512 PV accumulator's registers.
// Canvas admission is row-independent, so the page softmax needs no row
// offset — only its template row count shrinks.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          typename CacheElement,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION, uint MPasses = 1>
inline void richengine_canvas_attention_tile(
    device bfloat *tile_queries, device const RichKvPage *page_table,
    RichKvLayer kv, uint kv_head, uint committed_tokens, uint canvas_rows,
    uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    float qk_scale,
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
  uint visible_tokens = committed_tokens + canvas_rows;
  uint pages = richengine_attention_pages(visible_tokens);
  uint per_split = richengine_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);

  auto st = tensor(scores, dextents<int, 2>{N, MP}, array<int, 2>{1, N});
  typedef decltype(tensor(static_cast<threadgroup bfloat *>(nullptr),
                          dextents<int, 2>{N, MP}, array<int, 2>{1, N})
                       .slice<N, MP>(0, 0)) PSliced;
  const RichKvAddressing<KVHeads, CacheElement, HeadDim> addressing(kv, kv_head);
  constexpr auto qk_descriptor =
      matmul2d_descriptor(MP, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
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
        qk.run(q0, ks, page_scores);
        page_scores.store(st.template slice<N, MP>(0, 0));
        if (thread_index == 0)
          atomic_store_explicit(rescale, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        richengine_canvas_page_softmax<QueryHeadsPerKVHead,
            RowsPerTile / MPasses, Quantized, HeadDim, (MP > 64 ? 16 : 8)>(
            scores, pbuf, row_max, row_sum, previous_scale, rescale,
            reinterpret_cast<device const float4 *>(key_scales),
            reinterpret_cast<device const float4 *>(value_scales), token_start,
            visible_tokens, committed_tokens, thread_index, qk_scale,
            window_tokens);
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

template <uint KVHeads, uint QueryHeadsPerKVHead, uint HeadDim>
inline void richengine_canvas_attention_reduce_phase(
    device const float *partials, device const float *statistics,
    device bfloat *output, constant RichPrefillAttentionParams &params,
    uint3 group, uint thread_index, threadgroup float *weights,
    threadgroup float *group_values) {
  constexpr ushort M =
      RICHENGINE_PREFILL_ATTENTION_TILE_ROWS * QueryHeadsPerKVHead;
  constexpr ushort D = HeadDim;
  uint kv_head = group.x;
  uint fused_row = group.y;
  uint tile = group.z;
  uint tile_start = tile * RichCanvasTileRows;
  if (!richengine_prefill_attention_contract_valid(params) ||
      kv_head >= KVHeads || fused_row >= M || thread_index >= D ||
      tile_start >= params.rows)
    return;
  const uint active_rows = min(RichCanvasTileRows, params.rows - tile_start);
  const uint visible_tokens = params.committed_tokens + params.rows;
  const ulong tile_offset =
      (ulong(kv_head) * params.chunk_stride + tile_start) *
      QueryHeadsPerKVHead * D;
  richengine_attention_reduce_row<QueryHeadsPerKVHead,
                                  RICHENGINE_PREFILL_ATTENTION_TILE_ROWS,
                                  HeadDim>(
      partials, statistics, output + tile_offset, params.committed_tokens,
      active_rows, params.split_count,
      (ulong(tile) * KVHeads + kv_head) * params.split_count, fused_row,
      thread_index, weights, group_values, visible_tokens);
}
