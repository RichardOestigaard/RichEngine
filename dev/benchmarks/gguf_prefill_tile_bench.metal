// Tile-shape variants of the GGUF prefill tile (kernels/shared/gguf_linear.metal's gguf_prefill_tile) for
// gguf_prefill_tile_bench.mm: the production shape beside simdgroup-count, K-step, column-tile and
// execution_simdgroups<2> pair variants, instantiated for the formats the sweep measures. Bench-only; production
// dispatch never names them.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"

// gguf_prefill_tile's body (kernels/shared/gguf_linear.metal), templated here so the variants pick their own
// geometry: a grid.x row tile of RowsPerSG * Simdgroups rows, a grid.y column tile of TileN columns.
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS>
inline void bench_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                               device bfloat *output, uint input_size, uint output_origin, uint rows,
                               threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint simd_group,
                               uint out_stride, uint out_offset) {
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto acc = staged_accumulator<RowsPerSG, TileN, KS>(rows_input, input_size, stage);
  gguf_zero(acc);
  gguf_staged_steps<F, RowsPerSG, TileN, KS, Simdgroups * 32>(rows_input, w0, w1, meta, input_size, output_origin,
                                                              stage, tl, simd_group * 32 + simd_lane, 0,
                                                              input_size / KS, 0, owns_rows, acc);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = bfloat(acc[i]);
  }
}

// The execution_simdgroups<2> variant: simdgroup pairs (simd_group / 2) cooperate on one 64-row matmul per pass
// instead of each simdgroup running its own 32-row matmul. The staging walk is gguf_staged_steps' with the pair's
// input rows and the wider destination; every thread still stages and meets every barrier.
template <class F, ushort TileN, ushort KS>
inline void es2_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                             device bfloat *output, uint input_size, uint output_origin, uint rows,
                             threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint simd_group,
                             uint out_stride, uint out_offset) {
  constexpr ushort Rows = 64, Threads = 128;
  const uint pair = simd_group / 2;
  const bool owns_rows = pair * Rows < rows;
  device bfloat *rows_input = input + ulong(pair) * Rows * input_size;
  constexpr ushort GPS = KS / 32, Items = TileN * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(rows_input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(Rows, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<2>> operation;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * TileN, dextents<int, 2>{KS, TileN},
                                                                array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, TileN>(0, 0), b1 = bt1.slice<KS, TileN>(0, 0);
  auto a0 = a.template slice<KS, Rows>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  const uint thread_index = simd_group * 32 + simd_lane;
  const uint step_end = input_size / KS;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  const auto code_at = [&](ulong g, uint col) { return gguf_code_at<F>(w0, plane_tile, plane_row + col, g, groups); };
  const auto meta_at = [&](ulong unit, uint col) {
    return gguf_meta_at<F>(meta, w1, plane_tile, plane_row + col, unit, units);
  };
  const uint count = step_end, first_run = step_end;
  const auto step_at = [&](uint w) __attribute__((always_inline)) {
    return w < first_run ? w : w - first_run;
  };
  typename F::Payload packed0[IPT], packed1[IPT];
  typename F::Meta hdr0[IPT], hdr1[IPT];
  const auto preload = [&](typename F::Payload (&packed)[IPT], typename F::Meta (&hdr)[IPT], uint w,
                           bool payload) __attribute__((always_inline)) {
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads;
      const bool live = item < Items;
      const uint col = live ? item % TileN : 0, gi = live ? item / TileN : 0;
      const ulong g = ulong(step_at(w)) * GPS + gi;
      if (live && payload) packed[it] = F::load(code_at(g, col), tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
      hdr[it] = meta_at(g / F::MetaGroups, col);
    }
  };
  preload(packed0, hdr0, 0, count > 0);
  if (count > 1) preload(packed1, hdr1, 1, true);
  const auto run_step = [&](typename F::Payload (&packed)[IPT], typename F::Meta (&hdr)[IPT],
                            uint w) __attribute__((always_inline)) {
    const uint step = step_at(w);
    threadgroup half *buf = stage + (w & 1) * (KS * TileN);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads;
      if (item >= Items) break;
      const uint col = item % TileN, gi = item / TileN, g = step * GPS + gi;
      dequant32<F>(packed[it], hdr[it], g % F::MetaGroups, tl, buf + col * KS + gi * 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    if (w + 2 < count) {
      const uint next = step_at(w + 2);
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads;
        if (item >= Items) break;
        const uint col = item % TileN, gi = item / TileN;
        const ulong g = ulong(next) * GPS + gi;
        packed[it] = F::load(code_at(g, col), tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
        const uint unit = uint(g / F::MetaGroups);
        if (unit != (step * GPS + gi) / F::MetaGroups) hdr[it] = meta_at(unit, col);
      }
    }
    if (owns_rows) {
      auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
      if (w & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
    }
  };
  uint w = 0;
  for (; w + 2 <= count; w += 2) {
    run_step(packed0, hdr0, w);
    run_step(packed1, hdr1, w + 1);
  }
  if (w < count) run_step(packed0, hdr0, w);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(pair) * Rows + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = bfloat(acc[i]);
  }
}


// Ceiling probes, not tile candidates: nodeq skips the dequantize (the stage holds whatever the threadgroup memory
// powers up with; the matmul and the payload loads still run), nomatmul skips the matmul. Their sum against the
// base tile's time says how the step time splits between staging and tensor work. Outputs are garbage; the harness
// check line for them is meaningless.
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, bool Dequant, bool Matmul>
inline void probe_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta,
                               device bfloat *output, uint input_size, uint output_origin, uint rows,
                               threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint simd_group,
                               uint out_stride, uint out_offset) {
  constexpr ushort Threads = Simdgroups * 32, GPS = KS / 32, Items = TileN * GPS, IPT = (Items + Threads - 1) / Threads;
  const bool owns_rows = simd_group * RowsPerSG < rows;
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto a = tensor(rows_input, dextents<int, 2>{int(input_size), RowsPerSG}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor =
      matmul2d_descriptor(RowsPerSG, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * TileN, dextents<int, 2>{KS, TileN},
                                                                array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, TileN>(0, 0), b1 = bt1.slice<KS, TileN>(0, 0);
  auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  const uint thread_index = simd_group * 32 + simd_lane;
  const uint count = input_size / KS;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint plane_tile = output_origin / QUANT_TILE_ROWS, plane_row = output_origin % QUANT_TILE_ROWS;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  const auto code_at = [&](ulong g, uint col) { return gguf_code_at<F>(w0, plane_tile, plane_row + col, g, groups); };
  const auto meta_at = [&](ulong unit, uint col) {
    return gguf_meta_at<F>(meta, w1, plane_tile, plane_row + col, unit, units);
  };
  for (uint step = 0; step < count; ++step) {
    threadgroup half *buf = stage + (step & 1) * (KS * TileN);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = thread_index + it * Threads;
      if (item >= Items) break;
      const uint col = item % TileN, gi = item / TileN, g = step * GPS + gi;
      typename F::Payload packed = F::load(code_at(g, col), tw1 + (ulong(g) * QUANT_TILE_ROWS + col) * F::P1);
      typename F::Meta hdr = meta_at(g / F::MetaGroups, col);
      if (Dequant) dequant32<F>(packed, hdr, g % F::MetaGroups, tl, buf + col * KS + gi * 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup, memory_order_acq_rel);
    if (Matmul && owns_rows) {
      auto a_slice = a.template slice<KS, RowsPerSG>(step * KS, 0);
      if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
    }
  }
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = bfloat(acc[i]);
  }
}

#define BENCH_ARGS                                                                                                  \
  device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],              \
      device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]],                                       \
      constant GgufPrefillParams &p [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],                   \
      uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define BENCH_FIRST RowsPerSG *Simdgroups
#define BENCH_BODY(F, RowsPerSG, Simdgroups, TileN, KS)                                                              \
  threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];                         \
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, Simdgroups * 32);                                             \
  threadgroup half stage[2 * TileN * KS];                                                                            \
  const uint first = group.x * (RowsPerSG * Simdgroups), rows = p.rows > first ? p.rows - first : 0;                 \
  bench_prefill_tile<F, RowsPerSG, Simdgroups, TileN, KS>(input + ulong(first) * p.input_size, w0, w1, meta,         \
                                                          output + ulong(first) * p.out_stride, p.input_size,        \
                                                          group.y * TileN, rows, stage, tl, simd_lane, simd_group,   \
                                                          p.out_stride, p.out_offset)

#define BENCH_VARIANT(F, f, name, RowsPerSG, Simdgroups, TileN, KS)                                                  \
  [[max_total_threads_per_threadgroup(Simdgroups * 32)]]                                                             \
  kernel void pfb_##f##_##name(BENCH_ARGS) {                                                                         \
    BENCH_BODY(F, RowsPerSG, Simdgroups, TileN, KS);                                                                 \
  }

#define BENCH_VARIANTS(F, f)                                                                                         \
  BENCH_VARIANT(F, f, base, 32, 4, 64, 64)                                                                           \
  BENCH_VARIANT(F, f, sg8, 32, 8, 64, 64)                                                                            \
  BENCH_VARIANT(F, f, sg2, 32, 2, 64, 64)                                                                            \
  BENCH_VARIANT(F, f, n32ks128, 32, 4, 32, 128)                                                                         \
  BENCH_VARIANT(F, f, ks32, 32, 4, 64, 32)                                                                             \
  BENCH_VARIANT(F, f, n32sg8, 32, 8, 32, 64)                                                                           \
  BENCH_VARIANT(F, f, r16sg8, 16, 8, 64, 64)                                                                           \
  BENCH_VARIANT(F, f, r16sg4, 16, 4, 64, 64)                                                                           \
  BENCH_VARIANT(F, f, n32ks32, 32, 4, 32, 32)                                                                          \
  BENCH_VARIANT(F, f, n32sg2, 32, 2, 32, 64)                                                                           \
  BENCH_VARIANT(F, f, n16, 32, 4, 16, 64)                                                                         \
  BENCH_VARIANT(F, f, n32, 32, 4, 32, 64)                                                                            \
  BENCH_VARIANT(F, f, r64sg2, 64, 2, 64, 64)                                                                         \
  [[max_total_threads_per_threadgroup(128)]]                                                                         \
  kernel void pfb_##f##_nodeq(BENCH_ARGS) {                                                                            \
    threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];                         \
    quant_pair_table<F>(tl, simd_group * 32 + simd_lane, 128);                                                         \
    threadgroup half stage[2 * 64 * 64];                                                                               \
    const uint first = group.x * 128, rows = p.rows > first ? p.rows - first : 0;                                      \
    probe_prefill_tile<F, 32, 4, 64, 64, false, true>(input + ulong(first) * p.input_size, w0, w1, meta,               \
                                                      output + ulong(first) * p.out_stride, p.input_size, group.y * 64,\
                                                      rows, stage, tl, simd_lane, simd_group, p.out_stride,            \
                                                      p.out_offset);                                                   \
  }                                                                                                                    \
  [[max_total_threads_per_threadgroup(128)]]                                                                           \
  kernel void pfb_##f##_nomatmul(BENCH_ARGS) {                                                                         \
    threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];                         \
    quant_pair_table<F>(tl, simd_group * 32 + simd_lane, 128);                                                         \
    threadgroup half stage[2 * 64 * 64];                                                                               \
    const uint first = group.x * 128, rows = p.rows > first ? p.rows - first : 0;                                      \
    probe_prefill_tile<F, 32, 4, 64, 64, true, false>(input + ulong(first) * p.input_size, w0, w1, meta,               \
                                                      output + ulong(first) * p.out_stride, p.input_size, group.y * 64,\
                                                      rows, stage, tl, simd_lane, simd_group, p.out_stride,            \
                                                      p.out_offset);                                                   \
  }                                                                                                                    \
  [[max_total_threads_per_threadgroup(128)]]                                                                           \
  kernel void pfb_##f##_es2(BENCH_ARGS) {                                                                            \
    threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];                       \
    quant_pair_table<F>(tl, simd_group * 32 + simd_lane, 128);                                                       \
    threadgroup half stage[2 * 64 * 64];                                                                             \
    const uint first = group.x * 128, rows = p.rows > first ? p.rows - first : 0;                                    \
    es2_prefill_tile<F, 64, 64>(input + ulong(first) * p.input_size, w0, w1, meta,                                   \
                                output + ulong(first) * p.out_stride, p.input_size, group.y * 64, rows, stage, tl,   \
                                simd_lane, simd_group, p.out_stride, p.out_offset);                                  \
  }

BENCH_VARIANTS(FmtNVFP4, nvfp4)
BENCH_VARIANTS(FmtIQ2XXS, iq2xxs)
BENCH_VARIANTS(FmtQ80, q80)
BENCH_VARIANTS(FmtQ4K, q4k)
