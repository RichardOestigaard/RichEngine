#pragma once

#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

// Paged attention over FP8 E4M3 KV. The pages share the INT8 tier's layout:
// one byte per element and one fp32 scale per (token, KV head), so the
// RichKvAddressing<int8_t> slabs address them — only the byte's meaning
// differs. matmul2d admits no bf16 left operand against an fp8 right
// operand, so queries stage as fp16 (half x fp8 is a native operand pair);
// a full-width fp16 query tile would overflow the 32 KB threadgroup budget
// alongside the score and probability tiles, so the tile runs the fused
// rows in two passes of M/2 rows each, halving every threadgroup buffer.
// E4M3 is chosen over E5M2: the per-(head, token) scale already bounds the
// exponent range a row needs, and three mantissa bits halve the element
// error two give.
#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__) && \
    defined(__HAVE_PACKED_NUMERIC_TYPE_PACK_UNPACK__)

// Marker for FP8 KV pages: INT8 byte geometry and scales, e4m3 elements.
struct RichKvFp8E4m3 {};

// The fp32 per-(token, head) scales keep the stored e4m3 codes in a range
// where the format's 448 maximum is never reached and small elements keep
// their mantissa bits. QK scores take the key scales exactly as the INT8
// tile does; PV takes probability x value scale in fp16.
template <uint QueryHeadsPerKVHead, uint RowsPerTile,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_attention_page_softmax_fp8(
    threadgroup const float *scores, threadgroup half *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    device const float4 *key_scales, device const float4 *value_scales,
    uint token_start, uint fused_row_offset,
    uint visible_tokens, uint committed_tokens, uint active_rows,
    uint thread_index, float qk_scale = 0.0f) {
  constexpr uint N = RichKvPageTokens;
  constexpr uint FusedRows = RowsPerTile * QueryHeadsPerKVHead;
  constexpr uint TokensPerLane = 8;
  constexpr uint LanesPerRow = N / TokensPerLane;
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
  const uint token = token_start + column;
  threadgroup const float4 *scores4 =
      reinterpret_cast<threadgroup const float4 *>(scores + fused_row * N +
                                                   column);
  const uint vector = column / 4;
  float score[TokensPerLane];
  {
    float4 low = scores4[0], high = scores4[1];
    low *= key_scales[vector];
    high *= key_scales[vector + 1];
    const float score_scale =
        qk_scale != 0.0f ? qk_scale
                         : (HeadDim == 64    ? 0.125f
                            : HeadDim == 128 ? 0.08838834764831845f
                                             : 0.0625f);
    low *= score_scale;
    high *= score_scale;
    score[0] = low.x, score[1] = low.y, score[2] = low.z, score[3] = low.w;
    score[4] = high.x, score[5] = high.y, score[6] = high.z, score[7] = high.w;
  }
  float local_max = -INFINITY;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    score[j] = token + j < limit ? score[j] : -INFINITY;
    local_max = max(local_max, score[j]);
  }
  local_max = max(local_max, simd_shuffle_xor(local_max, 1));
  local_max = max(local_max, simd_shuffle_xor(local_max, 2));
  const float previous_max = row_max[fused_row];
  const float next_max = max(previous_max, local_max);
  float probability[TokensPerLane];
  float local_sum = 0.0f;
#pragma unroll
  for (uint j = 0; j < TokensPerLane; ++j) {
    probability[j] = token + j < limit ? fast::exp(score[j] - next_max) : 0.0f;
    local_sum += probability[j];
  }
  local_sum += simd_shuffle_xor(local_sum, 1);
  local_sum += simd_shuffle_xor(local_sum, 2);
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
  const float4 low_scales = value_scales[vector];
  const float4 high_scales = value_scales[vector + 1];
  const float4 low(token + 0 < limit ? probability[0] * low_scales.x : 0.0f,
                   token + 1 < limit ? probability[1] * low_scales.y : 0.0f,
                   token + 2 < limit ? probability[2] * low_scales.z : 0.0f,
                   token + 3 < limit ? probability[3] * low_scales.w : 0.0f);
  const float4 high(token + 4 < limit ? probability[4] * high_scales.x : 0.0f,
                    token + 5 < limit ? probability[5] * high_scales.y : 0.0f,
                    token + 6 < limit ? probability[6] * high_scales.z : 0.0f,
                    token + 7 < limit ? probability[7] * high_scales.w : 0.0f);
  threadgroup half4 *probabilities4 = reinterpret_cast<threadgroup half4 *>(
      probabilities + fused_row * N + column);
  probabilities4[0] = half4(low);
  probabilities4[1] = half4(high);
}

// One tile over its split's pages: fp16-staged queries against the page's
// e4m3 key and value bytes, with the page's fp32 scales consumed in the
// softmax exactly as the INT8 tile does. The fused M rows run in two passes
// of MP = M/2: the fp16 query stage, the score tile, and the probability
// tile all halve, keeping the tile inside the 32 KB threadgroup budget.
// Row-halving is exact — every fused row attends independently, so each
// pass accumulates and stores its own partials and statistics. The queries
// stage once per pass before the page loop: every page reads the same
// MP x D tile, whose bf16 elements each convert once.
template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile,
          uint HeadDim = RICHENGINE_KV_HEAD_DIMENSION>
inline void richengine_paged_attention_tile_fp8(
    device bfloat *tile_queries, device const RichKvPage *page_table,
    RichKvLayer kv, uint kv_head, uint committed_tokens, uint active_rows, uint splits,
    uint split, device float *partials, device float *statistics, ulong slot,
    threadgroup float *scores, threadgroup half *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum,
    threadgroup float *previous_scale, threadgroup atomic_uint *rescale,
    threadgroup half *staged_queries,
    uint thread_index, float qk_scale = 0.0f) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort MP = M / 2;
  constexpr ushort N = RichKvPageTokens;
  constexpr ushort D = HeadDim;
  static_assert(M % 2 == 0, "two row passes split the fused tile evenly");
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = richengine_attention_pages(visible_tokens);
  uint per_split = richengine_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  typedef tensor<device metal_fp8_e4m3_format, dextents<int, 2>,
                 tensor_inline>
      Fp8Tensor;
  typedef tensor<threadgroup half, dextents<int, 2>, tensor_inline>
      HalfTensorTg;
  auto st = tensor(scores, dextents<int, 2>{N, MP}, array<int, 2>{1, N});
  auto pt =
      tensor(probabilities, dextents<int, 2>{N, MP}, array<int, 2>{1, N});
  auto p0 = pt.template slice<N, MP>(0, 0);
  const RichKvAddressing<KVHeads, int8_t, HeadDim> addressing(kv, kv_head);
  // QK writes a complete page score tile, the softmax applies an FP8 page's
  // key scales and PV accumulates the running output.
  constexpr auto qk_descriptor =
      matmul2d_descriptor(MP, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(MP, D, N, false, true, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroups<8>> pv;

  for (ushort pass = 0; pass < 2; ++pass) {
    const uint row_offset = uint(pass) * MP;
    for (uint index = thread_index; index < uint(MP) * D; index += 256)
      staged_queries[index] = half(tile_queries[row_offset * D + index]);
    if (thread_index < MP) {
      row_max[thread_index] = -INFINITY;
      row_sum[thread_index] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    auto qt = HalfTensorTg(staged_queries, dextents<int, 2>{D, MP},
                           array<int, 2>{1, D});
    auto q0 = qt.template slice<D, MP>(0, 0);
    auto running =
        pv.template get_destination_cooperative_tensor<
            decltype(p0),
            decltype(Fp8Tensor(static_cast<device uchar *>(nullptr),
                               dextents<int, 2>{N, D}, array<int, 2>{1, N})
                         .template slice<N, D>(0, 0)),
            float>();
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

      // Format element tensors take their storage's uchar pointer; at eight
      // bits per element the element index functions already address bytes.
      device uchar *key_bytes =
          reinterpret_cast<device uchar *>(tensors.keys);
      device uchar *value_bytes =
          reinterpret_cast<device uchar *>(tensors.values);
      Fp8Tensor kt(key_bytes, dextents<int, 2>{D, N}, array<int, 2>{1, D});
      Fp8Tensor vt(value_bytes, dextents<int, 2>{N, D}, array<int, 2>{1, N});

      auto page_scores = qk.template get_destination_cooperative_tensor<
          decltype(q0), decltype(kt.template slice<D, N>(0, 0)), float>();
      // One full-dimension product initializes the score CT through MPP.
      auto ks = kt.template slice<D, N>(0, 0);
      qk.run(q0, ks, page_scores);
      page_scores.store(st.template slice<N, MP>(0, 0));
      if (thread_index == 0)
        atomic_store_explicit(rescale, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      richengine_attention_page_softmax_fp8<QueryHeadsPerKVHead, RowsPerTile / 2, HeadDim>(
          scores, probabilities, row_max, row_sum, previous_scale, rescale,
          reinterpret_cast<device const float4 *>(key_scales),
          reinterpret_cast<device const float4 *>(value_scales), token_start,
          row_offset, visible_tokens, committed_tokens, active_rows,
          thread_index, qk_scale);
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
      auto vs = vt.template slice<N, D>(0, 0);
      pv.run(p0, vs, running);
      threadgroup_barrier(mem_flags::mem_threadgroup);
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

#endif // __HAVE_METAL_FP8_E4M3_FORMAT_TYPE__
