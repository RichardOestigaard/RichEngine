#pragma once
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/split_reduce.h"

// The decode tile's split-K store and the MXFP4 multiplane decode tile,
// shared by kernels/shared/gguf_linear.metal and kernels/shared/
// moe_gguf.metal. Includers set `#pragma clang fp reassociate(off)` first.

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
// MXFP4 on the Metal 4.1 multiplane tensor (ops::Linear's `mxfp4m` decode
// kernels, Apple GPU family 10 and up): the image's E2M1 nibbles are B of a
// K = 32 matmul2d as tensor<device metal_fp4_e2m1_format> and their E8M0
// exponents its tensor_blockwise scale plane, so the weights enter the MMA
// still packed — no dequantization stage, no coefficient epilogue. A group
// of 32 is one scale block, so the GGUF_STAGED_STEP loop runs one matmul per
// group. A is the staged kernels' bf16 input staged as fp16 (the fp4
// operand admits no bf16 left operand), permuted into the image's
// chunk-slot order (metal/abi/QuantFormat.h): slot s of the stage holds
// element 16 * (s >> 2 & 1) + 4 * (s >> 3) + (s & 3) of the group. Activations
// past fp16's range (sparse inputs reach +-1e5) stay exact through a per
// (row, group) exponent: a block that does not fit fp16's range is staged
// times 2^-e and its matmul writes a fresh partial the epilogue scales back
// by 2^e; a block that fits (the common case) accumulates straight into acc.
// The two stage buffers alternate so the next group's conversion overlaps
// the current one's matmul; the threadgroup's two simdgroups share the
// stage since their matmuls read the same activations.
// `origin` is the simdgroup's first column of the segment's image and
// `column0` its first destination column; `counter` is the pre-indexed
// arrival counter of the threadgroup's column tile and `split_index` its K
// partition (the callers' group.y).
template <ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_mxfp4_tile(device bfloat *input, device uchar *w0, device uchar *meta, device Out *output,
                                   device coherent(device) float *partials, device atomic_uint *counter,
                                   device bfloat *aux, uint input_size, uint splits, uint out_stride, uint origin,
                                   uint column0, uint split_index, uint simd_lane, uint simd_group,
                                   threadgroup half *stage, threadgroup uchar *scales_plane,
                                   threadgroup uint *arrival) {
  const uint groups = input_size / 32, per = groups / splits;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  constexpr auto acc_descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                    matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  // A group whose staged block needed no exponent (the common case: no
  // activation overflows fp16's range) accumulates straight into acc; one
  // that did goes through a fresh partial and the 2^e epilogue.
  matmul2d<acc_descriptor, execution_simdgroups<1>> accumulate;
  typedef tensor<threadgroup half, dextents<int, 2>, tensor_inline> A;
  typedef tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1> Scales;
  typedef tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline, Scales> B;
  // Both simdgroups matmul the same A, so the threadgroup shares one stage:
  // simdgroup s stages rows s, s + 2, ... of each block.
  threadgroup half *my = stage;
  threadgroup uchar *sc = scales_plane;
  A a0(my, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1(my + Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  // Static-extent slices skip the bounds checks a dynamic A would emit per
  // run(); B stays dynamic — slicing a blockwise operand would slice its
  // E8M0 scale plane wrongly (gguf_projection_test catches it).
  const auto as0 = a0.template slice<32, Rows>(0, 0);
  const auto as1 = a1.template slice<32, Rows>(0, 0);
  auto acc = accumulate.template get_destination_cooperative_tensor<A, B, float>();
  auto partial = operation.template get_destination_cooperative_tensor<A, B, float>();
  gguf_zero(acc);
  // Lane s stages element s of each row's block; the block's 2^-e is exact
  // at any magnitude since bf16's eight mantissa bits fit fp16's eleven once
  // the exponent is scaled in.
  const auto stage32 = [&](uint g, uint buf) __attribute__((always_inline)) {
    threadgroup half *dst = my + buf * (Rows * 32);
    const uint s = simd_lane;
    bool scaled = false;
#pragma unroll
    for (uint row = simd_group; row < Rows; row += 2) {
      const float a = float(input[ulong(row) * input_size + g * 32 + 16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3)]);
      const float m = simd_max(fabs(a));
      // e = 0 stages the block unscaled when every element converts exactly
      // (half's normals reach 65504, bf16's mantissa fits half's), which is
      // the common case and lets the group accumulate without a partial;
      // only blocks at half's edges take a nonzero exponent.
      const int e = m > 30720.0f || (m > 0.0f && m < 6.1e-5f)
                        ? clamp(int(floor(log2(m))) - 14, -100, 100)
                        : 0;
      dst[row * 32 + s] = half(a * as_type<float>(uint(127 - e) << 23));
      scaled |= e != 0;
      if (s == 0) sc[buf * Rows + row] = uchar(127 + e);
    }
    // The group's rows all staged unscaled accumulate into acc in one
    // multiply_accumulate; a scaled row takes the partial-and-2^e path.
    if (s == 0) sc[2 * Rows + 2 * buf + simd_group] = uchar(simd_any(scaled));
  };
  const uint g0 = split_index * per, g1 = (split_index + 1) * per;
  stage32(g0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) stage32(g + 1, ((g + 1 - g0) & 1));
    Scales scales(meta + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row);
    B b(w0 + (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS * 16 + plane_row * 16,
        dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32}, scales);
    const uint buf = (g - g0) & 1;
    if (!(sc[2 * Rows + 2 * buf] | sc[2 * Rows + 2 * buf + 1])) {
      if (buf) accumulate.run(as1, b, acc); else accumulate.run(as0, b, acc);
    } else {
      if (buf) operation.run(as1, b, partial); else operation.run(as0, b, partial);
#pragma unroll
      for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) continue;
        const uint row = acc.get_multidimensional_index(i)[1];
        acc[i] = fma(partial[i], as_type<float>(uint(sc[buf * Rows + row]) << 23), acc[i]);
      }
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

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
// The packed-operand decode tile (the `<f>m` kernels of
// kernels/shared/gguf_linear.metal, Apple GPU family 10 and up) for the
// formats whose plane0 codes feed matmul2d directly as a packed B operand —
// uint4b (Q4_0, Q4_1, Q4_K), uint2b (PQ2_0) and int8 (Q8_0) — and whose
// scales are not powers of two, so they apply in an epilogue instead of an
// E8M0 scale plane. The matmul's group product is the sum of a * q over raw
// codes; a column's value s * (q - Zero) + m makes the group's contribution
// s * product + (m - s * Zero) * sum, sum the row's activation sum over the
// group, which the A stage computes alongside (lane s stages element s, so a
// simd_sum over the lane values is the block's sum). Unlike the MXFP4 tile
// the stage keeps bf16: these B operands take a bf16 left operand, which
// holds the sparse +-1e5 inputs natively — no exponent, no rescale epilogue.
// A product of a bf16 (eight mantissa bits) and a code of at most eight is
// exact in fp32, so the sum is the staged path's up to association. The two
// buffers alternate as in that tile.
//   B's element order is its column's byte stream in little-endian bit
// order (metal/abi/QuantFormat.h): for the 2- and 8-bit planes the field of
// slot s starts at bit s * width, so element k is slot k and stage slot k
// takes element 16 * ((k >> 2) & 1) + 4 * (k >> 3) + (k & 3) as in MXFP4. The
// 4-bit planes pack word c's pair p e0 at bits 4p and e1 at 16 + 4p, so
// nibble k is the code of slot 8 * (k >> 3) + 2 * (k & 3) + ((k >> 2) & 1)
// and stage slot k takes its element, 16 * ((k >> 1) & 1) + 4 * (k >> 3) +
// 2 * (k & 1) + ((k >> 2) & 1).
// `coefs` is [2][2][32] float2: buf's per-column (s, m - s * Zero), lane c of
// a simdgroup staging column c's coefficient of the next group. `sums` is
// [2][Rows], the staged row sums, only read when the format has a Zero or m.
//
// Measured verdict (ops::LinearGguf keeps these for RICHENGINE_GGUF_PACKED_ON
// benchmarks; dispatch is staged): unlike MXFP4, whose staged decode pays a
// codebook unpack per element, these formats' staged dequant is a cheap
// linear scale, so the restaged-A-plus-epilogue tile reads the same weight
// bytes and loses on the activation restage — 8192 x 5120 on a 20-core M5
// Pro, best split, ms: q4k 0.27 vs 0.108 staged, q40 0.30 vs 0.12, q80 0.30
// vs 0.166, pq20 0.31 vs 0.082; the gap holds at every row count.
template <class F, class BT, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_packed_tile(device bfloat *input, device uchar *w0, device uchar *meta, device Out *output,
                                    device coherent(device) float *partials, device atomic_uint *counter,
                                    device bfloat *aux, uint input_size, uint splits, uint out_stride, uint origin,
                                    uint column0, uint split_index, uint simd_lane, uint simd_group,
                                    threadgroup bfloat *stage, threadgroup float2 *coefs,
                                    threadgroup float *sums, threadgroup uint *arrival) {
  const uint groups = input_size / 32, per = groups / splits, units = groups / F::MetaGroups;
  const uint plane_tile = origin / QUANT_TILE_ROWS, plane_row = origin % QUANT_TILE_ROWS;
  constexpr auto descriptor = matmul2d_descriptor(Rows, GGUF_STAGED_COLUMNS, 32, false, true, false,
                                                  matmul2d_descriptor::mode::multiply);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  typedef tensor<threadgroup bfloat, dextents<int, 2>, tensor_inline> A;
  typedef tensor<device BT, dextents<int, 2>, tensor_inline> B;
  // The coefficient epilogue needs the row's sum only for a Zero or m
  // (Q4_0, Q4_1, Q4_K, PQ2_0; Q8_0 has neither).
  constexpr bool SumTerm = F::Zero != 0 || F::Id == GGUF_FMT_Q41 || F::Id == GGUF_FMT_Q4K;
  constexpr bool Nibble = is_same_v<BT, uint4b_format>;
  threadgroup bfloat *my = stage;
  A a0(my, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  A a1(my + Rows * 32, dextents<int, 2>{32, Rows}, array<int, 2>{1, 32});
  // Static-extent slices skip the bounds checks a dynamic A would emit per
  // run(); B stays dynamic until a sliced packed operand is separately
  // verified (the blockwise variant slices its scale plane wrongly).
  const auto as0 = a0.template slice<32, Rows>(0, 0);
  const auto as1 = a1.template slice<32, Rows>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<A, B, float>();
  auto partial = operation.template get_destination_cooperative_tensor<A, B, float>();
  gguf_zero(acc);
  const auto stage32 = [&](uint g, uint buf) __attribute__((always_inline)) {
    threadgroup bfloat *dst = my + buf * (Rows * 32);
    const uint s = simd_lane;
    // The staged group's scale, per column of the simdgroup's 32: lane s
    // writes column s's (s, m - s * Zero) as it stages element s of each row.
    const QuantCoef k = F::coef(F::loadMeta(meta + ((ulong(plane_tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS +
                                                    plane_row + s) * F::MetaBytes),
                              ushort(g % F::MetaGroups));
    coefs[buf * 64 + simd_group * 32 + s] = float2(k.s.x, k.m.x - k.s.x * float(F::Zero));
#pragma unroll
    for (uint row = simd_group; row < Rows; row += 2) {
      // The B element at index s of a column, and so of the stage's slot s.
      const uint e = Nibble ? 16 * ((s >> 1) & 1) + 4 * (s >> 3) + 2 * (s & 1) + ((s >> 2) & 1)
                            : 16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3);
      const bfloat a = input[ulong(row) * input_size + g * 32 + e];
      dst[row * 32 + s] = a;
      if constexpr (SumTerm) {
        // The block sum of the elements: the matmul sees the permuted block,
        // whose sum is the same either way.
        const float sum = simd_sum(float(a));
        if (s == 0) sums[buf * Rows + row] = sum;
      }
    }
  };
  const uint g0 = split_index * per, g1 = (split_index + 1) * per;
  stage32(g0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint g = g0; g < g1; ++g) {
    if (g + 1 < g1) stage32(g + 1, ((g + 1 - g0) & 1));
    const ulong b0 = (ulong(plane_tile) * groups + g) * QUANT_TILE_ROWS + plane_row;
    B b;
    if constexpr (is_same_v<BT, int8_t>)
      b = B((device int8_t *)(w0 + b0 * F::P0), dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32});
    else
      b = B(w0 + b0 * F::P0, dextents<int, 2>{32, GGUF_STAGED_COLUMNS}, array<int, 2>{1, 32});
    const uint buf = (g - g0) & 1;
    if (buf) operation.run(as1, b, partial); else operation.run(as0, b, partial);
#pragma unroll
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
      if (!acc.is_valid_element(i)) continue;
      const auto index = acc.get_multidimensional_index(i);
      const float2 c = coefs[buf * 64 + simd_group * 32 + index[0]];
      if constexpr (SumTerm)
        acc[i] += fma(partial[i], c.x, c.y * sums[buf * Rows + index[1]]);
      else
        acc[i] = fma(partial[i], c.x, acc[i]);
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

// The fused `_n` kernels dispatch a packed-operand segment's tile to the
// path above, as gguf_decode_fused_mxfp4 does for MXFP4, at every tile
// height: a format's decode must equal itself bitwise at every row count and
// fused or alone, so the rows>16 and fused fallbacks of the mxfp4 tile are
// not its pattern (the packed formats' epilogue rounding differs from the
// staged decode's, so the two paths may not share a format). `coefs` and
// `sums` are arrays of the kernel's, `stage` its half stage read as bfloat.
// Disabled while ops::LinearGguf's dispatch defaults the packed formats to
// the staged tile (the packed path measured 2.5-4x slower): a fused segment
// must equal its standalone projection bitwise, so the fused path matches
// the dispatch default, not the benchmark override.
constant constexpr bool kGgufFusedPacked = false;
template <ushort R>
inline bool gguf_decode_fused_packed(uint fmt, device bfloat *input, device uchar *w0, device uchar *meta,
                                     device bfloat *output, device coherent(device) float *partials,
                                     device atomic_uint *counter, uint input_size, uint splits, uint out_stride,
                                     uint origin, uint column0, uint split_index, uint simd_lane, uint simd_group,
                                     threadgroup bfloat *stage, threadgroup float2 *coefs,
                                     threadgroup float *sums, threadgroup uint *arrival) {
  if constexpr (!kGgufFusedPacked) return false;
#define GGUF_FUSED_PACKED(F, BT)                                                                          \
  gguf_decode_packed_tile<F, BT, R, EpNone>(input, w0, meta, output, partials, counter, nullptr,          \
                                            input_size, splits, out_stride, origin, column0, split_index, \
                                            simd_lane, simd_group, stage, coefs, sums, arrival)
  switch (fmt) {
  case GGUF_FMT_Q40: GGUF_FUSED_PACKED(FmtQ40, uint4b_format); return true;
  case GGUF_FMT_Q41: GGUF_FUSED_PACKED(FmtQ41, uint4b_format); return true;
  case GGUF_FMT_Q4K: GGUF_FUSED_PACKED(FmtQ4K, uint4b_format); return true;
  case GGUF_FMT_Q80: GGUF_FUSED_PACKED(FmtQ80, int8_t); return true;
  case GGUF_FMT_PQ20: GGUF_FUSED_PACKED(FmtPQ20, uint2b_format); return true;
  default: return false;
  }
#undef GGUF_FUSED_PACKED
}
#else
template <ushort R>
inline bool gguf_decode_fused_packed(uint, device bfloat *, device uchar *, device uchar *, device bfloat *,
                                     device coherent(device) float *, device atomic_uint *, uint, uint, uint, uint,
                                     uint, uint, uint, uint, threadgroup bfloat *,
                                     threadgroup float2 *, threadgroup float *, threadgroup uint *) {
  return false;
}
#endif

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
// The fused `_n` kernels dispatch an MXFP4 segment's tile to the multiplane
// path above: its destination cooperative tensor has different operand
// types than the staged loop's, so the segment runs the whole tile —
// accumulator, split-K store and all — and the kernel returns. Other
// formats keep the staged loop. `scales_plane` rides on the pair table's
// memory: a segment on this path never fills one. Only 8-row tiles take
// it — the fused kernel's staging outweighs the multiplane win at 16 and
// 32 rows (ops::LinearGguf's decodeFormat reaches the same split of
// workload). The fallback below (a toolchain without multiplane tensors)
// declines every segment, which then takes the staged path as in the
// plain fused kernels.
template <ushort R>
inline bool gguf_decode_fused_mxfp4(uint fmt, device bfloat *input, device uchar *w0, device uchar *meta,
                                    device bfloat *output, device coherent(device) float *partials,
                                    device atomic_uint *counter, uint input_size, uint splits, uint out_stride,
                                    uint origin, uint column0, uint split_index, uint simd_lane, uint simd_group,
                                    threadgroup half *stage, threadgroup uchar *scales_plane,
                                    threadgroup uint *arrival) {
  if (fmt != GGUF_FMT_MXFP4 || R > 8) return false;
  gguf_decode_mxfp4_tile<R, EpNone>(input, w0, meta, output, partials, counter, nullptr, input_size, splits,
                                    out_stride, origin, column0, split_index, simd_lane, simd_group, stage,
                                    scales_plane, arrival);
  return true;
}
#else
template <ushort R>
inline bool gguf_decode_fused_mxfp4(uint fmt, device bfloat *, device uchar *, device uchar *, device bfloat *,
                                    device coherent(device) float *, device atomic_uint *, uint, uint, uint, uint,
                                    uint, uint, uint, uint, threadgroup half *, threadgroup uchar *,
                                    threadgroup uint *) {
  static_cast<void>(fmt);
  return false;
}
#endif
