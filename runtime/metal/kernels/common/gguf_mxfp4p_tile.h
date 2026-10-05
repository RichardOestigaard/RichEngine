#pragma once
#include "metal/kernels/common/gguf_mxfp4_tile.h"

// The pre-packed MXFP4 multiplane decode tile (ops::Linear's `mxfp4p` decode
// kernels, Apple GPU family 10 and up), a variant of
// kernels/common/gguf_mxfp4_tile.h's gguf_decode_mxfp4_tile that reads its A
// operand straight from device memory: a per-dispatch pack kernel
// (gguf_pack_half, kernels/shared/gguf_mxfp4p.metal) converts the bf16
// activations once into the same slot-permuted fp16 the staged path builds
// per group, so the tile never touches threadgroup memory or barriers. The
// staged tile's restage cost is linear in the tile's rows, so this tile wins
// where that one loses: the 32-row tile beats the staged decode (mxfp4
// 5120 x 8192 on a 16-core M5 Pro: ~0.14 ms vs 0.34 staged).
//
// Scratch layout (gguf_pack_half writes it; the host allocates both planes):
//   packed    [rows][K] half: slot s of group g of row r, at
//             packed[r * K + g * 32 + s], holds input element
//             16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3) of the group —
//             the image's chunk-slot order, the same permutation the staged
//             tile applies — scaled by 2^-e of its (row, group).
//   exponents [rows][K / 32] uchar: exponents[r * (K / 32) + g] is the
//             127-biased exponent of (row, group), 127 when the block fit
//             fp16's range unscaled (the common case).
// A's tensor strides pick a group's 32 slots at [g * 32, g * 32 + 32) of
// every row, so the matmul sees exactly the staged tile's operand.

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
// One lane of each simdgroup covers one (row, group) exponent: lane l reads
// row l's byte of the group and simd_any reduces the group's scaled flag.
// A group with no scaled row accumulates straight into acc; a scaled one
// goes through a fresh partial and the 2^e epilogue, as in the staged tile.
template <ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_mxfp4p_tile(device half *packed, device uchar *exponents, device uchar *w0,
                                    device uchar *meta, device Out *output,
                                    device coherent(device) float *partials, device atomic_uint *counter,
                                    device bfloat *aux, uint input_size, uint splits, uint out_stride, uint origin,
                                    uint column0, uint split_index, uint simd_lane, uint simd_group,
                                    threadgroup uint *arrival) {
  const uint groups = input_size / 32, per = groups / splits;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  constexpr auto acc_descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  matmul2d<acc_descriptor, execution_simdgroups<1>> accumulate;
  typedef tensor<device half, dextents<int, 2>, tensor_inline> A;
  typedef tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1> Scales;
  typedef tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, Scales> B;
  auto acc = accumulate.template get_destination_cooperative_tensor<A, B, float>();
  auto partial = operation.template get_destination_cooperative_tensor<A, B, float>();
  gguf_zero(acc);
  const uint g0 = split_index * per, g1 = (split_index + 1) * per;
  for (uint g = g0; g < g1; ++g) {
    A a(packed + g * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, int(input_size)});
    Scales scales(meta + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row);
    // Static-extent slice: see gguf_decode_mxfp4_tile (B stays dynamic).
    const auto as = a.template slice<32, Rows>(0, 0);
    B b(w0 + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS * 16 + plane_row * 16,
        dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32}, scales);
    const uchar e = simd_lane < Rows ? exponents[simd_lane * groups + g] : uchar(127);
    if (!simd_any(e != 127)) {
      accumulate.run(as, b, acc);
    } else {
      operation.run(as, b, partial);
#pragma unroll
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const uint row = acc.get_multidimensional_index(i)[1];
        acc[i] = fma(partial[i], as_type<float>(uint(exponents[row * groups + g]) << 23), acc[i]);
      }
    }
  }
  gguf_store_sums<Rows>(acc, splits, split_index, partials, counter,
                        out_stride, column0, simd_group * 32 + simd_lane, arrival,
                        [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}

// The same tile for prefill (ops::Linear's `mxfp4p` prefill kernels): the
// staged prefill tile's geometry — GGUF_PREFILL_SIMDGROUPS simdgroups of Rows
// rows sharing one GGUF_TILE_COLUMNS-column tile of a GGUF_PREFILL_ROWS-row
// chunk — but A is the packed plane gguf_pack_half wrote for the whole chunk
// and B the fp4 planes, so there is no threadgroup stage and no barrier: a
// simdgroup whose rows start past `rows` (the staged tile's own rule) simply
// leaves. A simdgroup inside the chunk computes and stores all of its Rows
// rows, padding rows included, exactly as the staged tile does; the pack
// dispatch covers the chunk's rows rounded up to Rows.
//
// Kept for benchmarks (RICHENGINE_GGUF_PACKED_ON, ops/Linear.cpp's
// packedPrefillEnabled): unlike the memory-bound decode this tile loses to
// the staged prefill tile — the compute-bound GEMM pays the fp4 matmul's
// halved rate without winning back the stage dequantization (mxfp4
// 5120 x 8192 on a 20-core M5 Pro, ms at rows 128/512/2048: 0.52/2.05/7.88
// here vs 0.51/1.86/6.96 staged; a single N=64 matmul per group instead of
// the two N=32 ones was worse still at 0.83/2.85/10.96).
template <ushort Rows, GgufEpilogue Ep>
inline void gguf_prefill_mxfp4p_tile(device half *packed, device uchar *exponents, device uchar *w0,
                                     device uchar *meta, device bfloat *output, uint input_size,
                                     uint output_origin, uint rows, uint simd_lane, uint simd_group,
                                     uint out_stride, uint out_offset, device bfloat *aux = nullptr) {
  if (simd_group * Rows >= rows) return;
  const uint groups = input_size / 32;
  device half *row_packed = packed + ulong(simd_group) * Rows * input_size;
  device uchar *row_exponents = exponents + ulong(simd_group) * Rows * groups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  // Two N=32 matmuls per group — the decode tile's descriptor shape — over
  // the column tile's two 32-row halves of the plane tile.
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  constexpr auto acc_descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  matmul2d<acc_descriptor, execution_simdgroups<1>> accumulate;
  typedef tensor<device half, dextents<int, 2>, tensor_inline> A;
  typedef tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1> Scales;
  typedef tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, Scales> B;
  auto acc0 = accumulate.template get_destination_cooperative_tensor<A, B, float>();
  auto acc1 = accumulate.template get_destination_cooperative_tensor<A, B, float>();
  auto partial = operation.template get_destination_cooperative_tensor<A, B, float>();
  gguf_zero(acc0);
  gguf_zero(acc1);
  for (uint g = 0; g < groups; ++g) {
    A a(row_packed + g * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, int(input_size)});
    Scales scales0(meta + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row);
    Scales scales1(meta + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row + 32);
    B b0(w0 + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS * 16 + plane_row * 16,
         dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32}, scales0);
    B b1(w0 + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS * 16 + (plane_row + 32) * 16,
         dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32}, scales1);
    // Static-extent slice: see gguf_decode_mxfp4_tile (B stays dynamic).
    const auto as = a.template slice<32, Rows>(0, 0);
    const uchar e = simd_lane < Rows ? row_exponents[simd_lane * groups + g] : uchar(127);
    if (!simd_any(e != 127)) {
      accumulate.run(as, b0, acc0);
      accumulate.run(as, b1, acc1);
    } else {
      operation.run(as, b0, partial);
#pragma unroll
      for (ushort i = 0; i < acc0.get_capacity(); ++i) {
        if (!acc0.is_valid_element(i)) continue;
        const uint row = acc0.get_multidimensional_index(i)[1];
        acc0[i] = fma(partial[i], as_type<float>(uint(row_exponents[row * groups + g]) << 23), acc0[i]);
      }
      operation.run(as, b1, partial);
#pragma unroll
      for (ushort i = 0; i < acc1.get_capacity(); ++i) {
        if (!acc1.is_valid_element(i)) continue;
        const uint row = acc1.get_multidimensional_index(i)[1];
        acc1[i] = fma(partial[i], as_type<float>(uint(row_exponents[row * groups + g]) << 23), acc1[i]);
      }
    }
  }
  // The staged tile's store: every element of the simdgroup's Rows rows.
#pragma unroll
  for (ushort half_ = 0; half_ < 2; ++half_)
#pragma unroll
    for (ushort i = 0; i < acc0.get_capacity(); ++i) {
      if (!acc0.is_valid_element(i)) continue;
      const auto index = acc0.get_multidimensional_index(i);
      const ulong o = (ulong(simd_group) * Rows + index[1]) * out_stride +
                      out_offset + output_origin + half_ * GGUF_STAGED_COLUMNS + index[0];
      output[o] = gguf_epilogue<Ep>(half_ ? acc1[i] : acc0[i], aux, o);
    }
}
#endif
