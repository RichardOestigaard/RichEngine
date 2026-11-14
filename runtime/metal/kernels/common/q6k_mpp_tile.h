#pragma once
#include "metal/kernels/common/gguf_mxfp4_tile.h"

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
// Q6_K on the Metal 4.1 packed operand (the `q6km` decode kernels of
// kernels/decode/linear_gguf_q6k_sgmatrix.metal, Apple GPU family 10 and
// up): plane0's low 4 bits feed matmul2d directly as a uint4b B operand,
// and plane1's high 2 bits are expanded into a threadgroup half B holding
// 16 * qh — the group's 6-bit code is q_lo + 16 qh, so s * (code - 32)
// reads s * P_lo + s * P_hi - 32 s * S per 16-element half. The high bits
// could feed matmul2d as uint2b directly, but the 2-bit fields sit in the
// image's slot order while the nibble operand's B positions take the 4-bit
// planes' order — a second A permutation the stage cannot serve beside the
// first — so the expansion keeps one A layout (the uint4b one). Q6_K's
// int8 scales give each 32-group two coefficients, one per 16-half
// (FmtQ6K::coef's .x and .y, with -d * 32 as the halves' mins since Zero
// is 32), so the tile keeps q2k_mpp_tile.h's split: A_lo and A_hi hold the
// halves' activations, staged as +0 in the other's slots, and the epilogue
// is s_lo * (P4_lo + P_hi_lo) + s_hi * (P4_hi + P_hi_hi) + m_lo * S_lo +
// m_hi * S_hi with m_h = -32 s_h. The halves are the 4-bit operand's
// element order: stage slot s holds element
// 16 * ((s >> 1) & 1) + 4 * (s >> 3) + 2 * (s & 1) + ((s >> 2) & 1), the
// first 16 elements at slots with (s >> 1) & 1 == 0. The high-bit gather
// indexes image slot 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1) of B
// position k, whose 2-bit field is bits 2 * slot of the column's 8 plane1
// bytes. `stage` is 4 * Rows * 32 bf16 (two double-buffered halves), `hi`
// 2 * 2 * 32 * 32 half, `coefs` [2][2][32] float4 (s_lo, s_hi, m_lo,
// m_hi) and `sums` [2][2][Rows], as in the q2k tile.
template <ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_q6k_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                                 device Out *output, device coherent(device) float *partials,
                                 device atomic_uint *counter, device bfloat *aux, uint input_size, uint splits,
                                 uint out_stride, uint origin, uint column0, uint split_index, uint simd_lane,
                                 uint simd_group, threadgroup bfloat *stage, threadgroup half *hi,
                                 threadgroup float4 *coefs, threadgroup float *sums,
                                 threadgroup uint *arrival) {
  typedef FmtQ6K F;
  const uint groups = input_size / 32, per = groups / splits, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  typedef tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> A;
  typedef tensor<device uint4b_format, dextents<int, 2>, tensor_inline> B;
  typedef tensor<threadgroup half, dextents<int, 2>, tensor_inline> H;
  // The A stage's halves and buffers are the q2k tile's: buffer `buf` holds
  // A_lo at my + buf * 2 * Rows * 32 and A_hi a block of Rows * 32 later.
  threadgroup bfloat *my = stage;
  A a0lo(my, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a0hi(my + Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1lo(my + 2 * Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1hi(my + 3 * Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  const auto as0lo = a0lo.template slice<32, Rows>(0, 0);
  const auto as0hi = a0hi.template slice<32, Rows>(0, 0);
  const auto as1lo = a1lo.template slice<32, Rows>(0, 0);
  const auto as1hi = a1hi.template slice<32, Rows>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_lo = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_hi = operation.template get_destination_cooperative_tensor<A, H, float>();
  gguf_zero(acc);
  const auto stage32 = [&](uint g, uint buf) __attribute__((always_inline)) {
    threadgroup bfloat *lo = my + buf * 2 * (Rows * 32);
    threadgroup bfloat *hi_stage = lo + Rows * 32;
    threadgroup half *hb = hi + ((buf * 2 + simd_group) * GGUF_STAGED_COLUMNS) * 32;
    const uint s = simd_lane;
    const ulong b0 = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row;
    // The staged group's coefficients, per column of the simdgroup's 32:
    // lane s writes column s's (s_lo, s_hi, m_lo, m_hi) — the halves'
    // scales and -32 times them, the Zero point's contribution to the
    // activation sums — as it stages slot s of each row.
    const QuantCoef k = F::coef(F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS +
                                                    plane_row + s) * F::MetaBytes),
                              ushort(g % F::MetaGroups));
    coefs[buf * 64 + simd_group * 32 + s] =
        float4(k.s.x, k.s.y, k.m.x - k.s.x * float(F::Zero), k.m.y - k.s.y * float(F::Zero));
    // Column s's eight plane1 bytes: 32 two-bit fields, field j at bits
    // 2j..2j+1 of the little-endian bit string.
    const uint2 qh = *((device uint2 *)(w1 + (b0 + s) * F::P1));
#pragma unroll
    for (uint k = 0; k < 32; ++k) {
      // B position k takes the high bits of image slot
      // 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1).
      const uint slot = 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1);
      const uint field = slot < 16 ? (qh.x >> (2 * slot)) & 3u : (qh.y >> (2 * slot - 32)) & 3u;
      hb[s * 32 + k] = half(16.0f * float(field));
    }
    // Slot s holds element 16 * h + 4 * (s >> 3) + 2 * (s & 1) + ((s >> 2) & 1)
    // of the group, h = (s >> 1) & 1 its 16-half under the nibble order:
    // A_lo keeps the low half's activations and zeros the high half's
    // slots, A_hi the mirror.
    const bool h = (s >> 1) & 1;
    const uint e = 16 * uint(h) + 4 * (s >> 3) + 2 * (s & 1) + ((s >> 2) & 1);
#pragma unroll
    for (uint row = simd_group; row < Rows; row += 2) {
      const bfloat a = input[ulong(row) * input_size + g * 32 + e];
      lo[row * 32 + s] = h ? bfloat(0) : a;
      hi_stage[row * 32 + s] = h ? a : bfloat(0);
      // The halves' activation sums: the matmul sees the permuted block,
      // whose per-half sums are the same either way.
      const float sum_lo = simd_sum(h ? 0.0f : float(a));
      const float sum_hi = simd_sum(h ? float(a) : 0.0f);
      if (s == 0) {
        sums[buf * 2 * Rows + row] = sum_lo;
        sums[buf * 2 * Rows + Rows + row] = sum_hi;
      }
    }
  };
  const uint g0 = split_index * per, g1 = (split_index + 1) * per;
  stage32(g0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) stage32(g + 1, ((g + 1 - g0) & 1));
    const ulong b0 = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row;
    B b(w0 + b0 * F::P0, dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32});
    const uint buf = (g - g0) & 1;
    H h(hi + ((buf * 2 + simd_group) * GGUF_STAGED_COLUMNS) * 32, dextents<int, 2>{32, GGUF_STAGED_COLUMNS},
        array<int, 2>{1, 32});
    const auto &alo = buf ? as1lo : as0lo;
    const auto &ahi = buf ? as1hi : as0hi;
    // Each half's packed-low and expanded-high products land in their own
    // partials; the epilogue adds them under the half's scale before the
    // next pair overwrites them.
    operation.run(alo, b, partial_lo);
    operation.run(alo, h, partial_hi);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const float4 c = coefs[buf * 64 + simd_group * 32 + index[0]];
      acc[i] = fma(partial_lo[i] + partial_hi[i], c.x, acc[i]);
    }
    operation.run(ahi, b, partial_lo);
    operation.run(ahi, h, partial_hi);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const float4 c = coefs[buf * 64 + simd_group * 32 + index[0]];
      const uint row = index[1];
      acc[i] = fma(partial_lo[i] + partial_hi[i], c.y,
                   fma(sums[buf * 2 * Rows + row], c.z,
                       fma(sums[buf * 2 * Rows + Rows + row], c.w, acc[i])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
  }
  gguf_store_sums<Rows>(acc, splits, split_index, partials, counter,
                        out_stride, column0, simd_group * 32 + simd_lane, arrival,
                        [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
#endif
