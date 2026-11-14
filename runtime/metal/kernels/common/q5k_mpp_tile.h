#pragma once
#include "metal/kernels/common/gguf_mxfp4_tile.h"

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
// Q5_K on the Metal 4.1 packed operand (the `q5km` decode kernels of
// kernels/decode/linear_gguf_q5k_sgmatrix.metal, Apple GPU family 10 and
// up): plane0's low 4 bits feed matmul2d directly as a uint4b B operand, as
// Q4_K's do in gguf_decode_packed_tile. The fifth bits have no packed
// operand — no 1-bit numeric format exists — so plane1's one bit per slot
// is expanded into a threadgroup half B holding 16 * bit: the group's
// contribution s * (q_lo + 16 q_hi) + m then reads s * P_lo + s * P_hi +
// m * S with P_hi the expanded matmul's product (16 folded into the staged
// operand, exact in half) and S the row's activation sum, the packed
// tile's own epilogue. Q5_K's coefficient covers the whole 32-group
// (quant_k4_coef's .x and .y are equal), so the A stage needs no half
// split. The high-bit expansion runs in stage32 beside the A stage: lane s
// stages column s of the simdgroup's 32, reading the column's 4 plane1
// bytes once; B position k holds the bit of image slot
// 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1) — the nibble operand's slot
// order (gguf_mxfp4_tile.h's packed-B comment) — so the bit gather indexes
// that slot's bit of the column's bit string. `stage` is 2 * Rows * 32
// bf16, `hi` 2 * 2 * 32 * 32 half (buf, simdgroup, column, K), `coefs`
// [2][2][32] float2 (s, m) and `sums` [2][Rows], as in that tile.
template <ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_q5k_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                                 device Out *output, device coherent(device) float *partials,
                                 device atomic_uint *counter, device bfloat *aux, uint input_size, uint splits,
                                 uint out_stride, uint origin, uint column0, uint split_index, uint simd_lane,
                                 uint simd_group, threadgroup bfloat *stage, threadgroup half *hi,
                                 threadgroup float2 *coefs, threadgroup float *sums,
                                 threadgroup uint *arrival) {
  typedef FmtQ5K F;
  const uint groups = input_size / 32, per = groups / splits, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  typedef tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> A;
  typedef tensor<device uint4b_format, dextents<int, 2>, tensor_inline> B;
  typedef tensor<threadgroup half, dextents<int, 2>, tensor_inline> H;
  threadgroup bfloat *my = stage;
  A a0(my, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1(my + Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  // Static-extent slices skip the bounds checks a dynamic A would emit per
  // run(); B stays dynamic as in gguf_decode_packed_tile.
  const auto as0 = a0.template slice<32, Rows>(0, 0);
  const auto as1 = a1.template slice<32, Rows>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_lo = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial_hi = operation.template get_destination_cooperative_tensor<A, H, float>();
  gguf_zero(acc);
  const auto stage32 = [&](uint g, uint buf) __attribute__((always_inline)) {
    threadgroup bfloat *dst = my + buf * (Rows * 32);
    threadgroup half *hb = hi + ((buf * 2 + simd_group) * GGUF_STAGED_COLUMNS) * 32;
    const uint s = simd_lane;
    const ulong b0 = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row;
    // The staged group's coefficient and the column's fifth-bit word, per
    // column of the simdgroup's 32: lane s writes column s's (s, m) as it
    // stages slot s of each row, and expands column s's plane1 bits.
    const QuantCoef k = F::coef(F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS +
                                                    plane_row + s) * F::MetaBytes),
                              ushort(g % F::MetaGroups));
    coefs[buf * 64 + simd_group * 32 + s] = float2(k.s.x, k.m.x);
    const uint bits = *((device uint *)(w1 + (b0 + s) * F::P1));
#pragma unroll
    for (uint k = 0; k < 32; ++k) {
      // B position k takes the fifth bit of image slot
      // 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1).
      const uint slot = 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1);
      hb[s * 32 + k] = half(16.0f * float((bits >> slot) & 1));
    }
#pragma unroll
    for (uint row = simd_group; row < Rows; row += 2) {
      // The B element at index s of a column, and so of the stage's slot s
      // (the 4-bit planes' nibble order).
      const uint e = 16 * ((s >> 1) & 1) + 4 * (s >> 3) + 2 * (s & 1) + ((s >> 2) & 1);
      const bfloat a = input[ulong(row) * input_size + g * 32 + e];
      dst[row * 32 + s] = a;
      // The block sum of the elements: the matmul sees the permuted block,
      // whose sum is the same either way.
      const float sum = simd_sum(float(a));
      if (s == 0) sums[buf * Rows + row] = sum;
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
    if (buf) {
      operation.run(as1, b, partial_lo);
      operation.run(as1, h, partial_hi);
    } else {
      operation.run(as0, b, partial_lo);
      operation.run(as0, h, partial_hi);
    }
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const float2 c = coefs[buf * 64 + simd_group * 32 + index[0]];
      // s * (P_lo + P_hi) + m * S: the expanded operand already carries the
      // fifth bit's factor 16.
      acc[i] += fma(partial_lo[i], c.x,
                    fma(partial_hi[i], c.x, c.y * sums[buf * Rows + index[1]]));
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
