#pragma once
#include "metal/kernels/common/gguf_mxfp4_tile.h"

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
// Q2_K on the Metal 4.1 packed operand (the `q2km` decode kernels of
// kernels/decode/linear_gguf_q2k_sgmatrix.metal, Apple GPU family 10 and
// up): plane0's 2-bit codes feed matmul2d directly as a uint2b B operand,
// as PQ2_0's do in gguf_decode_packed_tile. Where that tile applies one
// (s, m - s * Zero) coefficient per group of 32, a Q2_K group has two —
// scale byte 2j's low nibble times d over elements 0..15 and byte 2j + 1's
// over 16..31, with the bytes' high nibbles times -dmin as the halves' mins
// (FmtQ2K::coef's .x and .y) — and its B interleaves the halves: element
// perm(k) = 16 * ((k >> 2) & 1) + 4 * (k >> 3) + (k & 3) sits at slot k, so
// the first 16-group's elements are the slots with (k >> 2) & 1 == 0. No
// K-slice of B separates them, so the group runs two matmuls against two
// staged operands: A_lo holds the low half's activations in their slots
// with the high half's staged as +0 (whose products are exact zeros), A_hi
// its mirror. The epilogue is the staged path's s * q + m distributed over
// each half's sums: acc += s_lo * P_lo + s_hi * P_hi + m_lo * S_lo +
// m_hi * S_hi, where P_h is the half's raw-code product and S_h its
// activation sum (lane s stages slot s, so the half sums are masked
// simd_sums of the lane values). Products of a bf16 and a 2-bit code are
// exact in fp32, so a half's P_h is the staged half-sum up to association —
// the same contract the sibling packed tiles keep.
// `stage` is 4 * Rows * 32 bf16 (two double-buffered halves), `coefs`
// [2][2][32] float4 (s_lo, s_hi, m_lo, m_hi) per column and `sums`
// [2][2][Rows] the halves' activation sums, as in that tile.
template <ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_q2k_tile(device bfloat *input, device uchar *w0, device uchar *meta, device Out *output,
                                 device coherent(device) float *partials, device atomic_uint *counter,
                                 device bfloat *aux, uint input_size, uint splits, uint out_stride, uint origin,
                                 uint column0, uint split_index, uint simd_lane, uint simd_group,
                                 threadgroup bfloat *stage, threadgroup float4 *coefs,
                                 threadgroup float *sums, threadgroup uint *arrival) {
  typedef FmtQ2K F;
  const uint groups = input_size / 32, per = groups / splits, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  typedef tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> A;
  typedef tensor<device uint2b_format, dextents<int, 2>, tensor_inline> B;
  // Both simdgroups matmul the same staged activations, so the threadgroup
  // shares one stage: simdgroup s stages rows s, s + 2, ... of each block.
  // A half's buffer alternates with the other group's as in the sibling
  // tiles: buffer `buf` holds A_lo at my + buf * 2 * Rows * 32 and A_hi a
  // block of Rows * 32 later.
  threadgroup bfloat *my = stage;
  A a0lo(my, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a0hi(my + Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1lo(my + 2 * Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1hi(my + 3 * Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  // Static-extent slices skip the bounds checks a dynamic A would emit per
  // run(); B stays dynamic as in gguf_decode_packed_tile.
  const auto as0lo = a0lo.template slice<32, Rows>(0, 0);
  const auto as0hi = a0hi.template slice<32, Rows>(0, 0);
  const auto as1lo = a1lo.template slice<32, Rows>(0, 0);
  const auto as1hi = a1hi.template slice<32, Rows>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_lo = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_hi = operation.template get_destination_cooperative_tensor<A, B, float>();
  gguf_zero(acc);
  const auto stage32 = [&](uint g, uint buf) __attribute__((always_inline)) {
    threadgroup bfloat *lo = my + buf * 2 * (Rows * 32);
    threadgroup bfloat *hi = lo + Rows * 32;
    const uint s = simd_lane;
    // The staged group's coefficients, per column of the simdgroup's 32:
    // lane s writes column s's (s_lo, s_hi, m_lo, m_hi) — FmtQ2K::coef's
    // (s.x, s.y, m.x, m.y) — as it stages slot s of each row.
    const QuantCoef k = F::coef(F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS +
                                                    plane_row + s) * F::MetaBytes),
                              ushort(g % F::MetaGroups));
    coefs[buf * 64 + simd_group * 32 + s] = float4(k.s.x, k.s.y, k.m.x, k.m.y);
    // Slot s holds element 16 * h + 4 * (s >> 3) + (s & 3), h = (s >> 2) & 1
    // its 16-group: A_lo keeps the low half's activations and zeros the
    // high half's slots, A_hi the mirror.
    const bool h = (s >> 2) & 1;
    const uint e = 16 * uint(h) + 4 * (s >> 3) + (s & 3);
#pragma unroll
    for (uint row = simd_group; row < Rows; row += 2) {
      const bfloat a = input[ulong(row) * input_size + g * 32 + e];
      lo[row * 32 + s] = h ? bfloat(0) : a;
      hi[row * 32 + s] = h ? a : bfloat(0);
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
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) stage32(g + 1, ((g + 1 - g0) & 1));
    const ulong b0 = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row;
    B b(w0 + b0 * F::P0, dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32});
    const uint buf = (g - g0) & 1;
    if (buf) {
      operation.run(as1lo, b, partial_lo);
      operation.run(as1hi, b, partial_hi);
    } else {
      operation.run(as0lo, b, partial_lo);
      operation.run(as0hi, b, partial_hi);
    }
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const float4 c = coefs[buf * 64 + simd_group * 32 + index[0]];
      const uint row = index[1];
      acc[i] = fma(partial_lo[i], c.x,
                   fma(partial_hi[i], c.y,
                       fma(sums[buf * 2 * Rows + row], c.z,
                           fma(sums[buf * 2 * Rows + Rows + row], c.w, acc[i]))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  gguf_store_sums<Rows>(acc, splits, split_index, partials, counter,
                        out_stride, column0, simd_group * 32 + simd_lane, arrival,
                        [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
#endif
