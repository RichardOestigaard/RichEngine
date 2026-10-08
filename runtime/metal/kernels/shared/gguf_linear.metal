// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal (MIT notice in THIRD_PARTY_NOTICES). Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/gguf_mxfp4_tile.h"
#include "metal/kernels/common/split_reduce.h"

// ---------------- prefill tiles: a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) skip their matmuls and stores, so a chunk costs its rows rounded up to RowsPerSG rather than
// to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile). Full tiles run this
// loop too: a separate branch-free copy for them, selected per threadgroup, measured up to 4% slower on an M5 Max and
// no faster on an M3 Max.
template <class F, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, GgufEpilogue Ep = EpNone>
inline void gguf_prefill_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint input_size, uint output_origin, uint rows, threadgroup half *stage, threadgroup half2 *tl,
                    uint simd_lane, uint simd_group, uint out_stride, uint out_offset, device bfloat *aux = nullptr) {
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  device bfloat *rows_input = input + ulong(simd_group) * RowsPerSG * input_size;
  auto acc = staged_accumulator<RowsPerSG, TileN, KS>(rows_input, input_size, stage);
  gguf_zero(acc);
  gguf_staged_steps<F, RowsPerSG, TileN, KS, Simdgroups * 32>(rows_input, w0, w1, meta, input_size, output_origin, stage, tl,
                                                              simd_group * 32 + simd_lane, 0, input_size / KS, owns_rows,
                                                              acc);
  if (!owns_rows) return;
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = gguf_epilogue<Ep>(acc[i], aux, o);
  }
}

// The two stages of a prefill tile.
constant constexpr uint kPrefillStages = 2 * GGUF_TILE_COLUMNS * GGUF_PREFILL_STEP;

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows (these kernels and the fused ones): a 3-lane step
// runs the 32-row tile over the storage of four lanes (LinearPlan::storageRows) and the padding lane's rows are
// computed and discarded. Rows are independent, so every active row is the bits of any other tile height
// (gguf-projection full); on a 16-core M5 Pro the 32-row tile at three lanes costs what it costs at four, 3-15% less
// than a 16-row plus an 8-row matmul per stage (0.207 vs 0.218 ms, Q4_K 12288 x 5120, DRAM-cold; 20-core: 0.173 vs
// 0.203).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do. The destination's type Out is bf16, or fp32 for the plain epilogue's
// logits (a_f32, ops::Projection::destination).
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device Out *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / GGUF_STAGED_STEP / p.splits,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per,
                                          (group.y + 1) * per, acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    output[o] = gguf_epilogue<Ep, Out>(v, aux, o);
  });
}
// The staged decode kernels: grid (column tiles, K partitions), two simdgroups.
template <class F, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],
                        device uchar *meta [[buffer(3)]], device Out *output [[buffer(4)]],
                        device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]],
                        device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                        uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  gguf_decode_tile<F, Rows, Ep>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group,
                                stage, tl, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE(F, f, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode<F, R, Ep, Out>;
#define GGUF_DECODE_ROWS(F, f, ep, Ep, Out) \
  GGUF_DECODE(F, f, 8, ep, Ep, Out) GGUF_DECODE(F, f, 16, ep, Ep, Out) GGUF_DECODE(F, f, 32, ep, Ep, Out)
#define GGUF_DECODE_FORMAT(F, f)                                                                                  \
  GGUF_DECODE_ROWS(F, f, a, EpNone, bfloat) GGUF_DECODE_ROWS(F, f, a_f32, EpNone, float)                          \
  GGUF_DECODE_ROWS(F, f, r, EpResidual, bfloat) GGUF_DECODE_ROWS(F, f, g, EpUpWithGate, bfloat)
QUANT_FORMATS(GGUF_DECODE_FORMAT)
// The fused greedy head: the staged decode with an argmax tail instead of
// logits stores — one partial per (row, GGUF_TILE_COLUMNS tile), reduced by
// decode_head_argmax_reduce_tiles_gguf. Same mask semantics as the affine
// fused head: dead rows write -inf partials; the excluded stop tokens never
// win.
template <class F, ushort Rows>
kernel void gguf_decode_amax(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                             device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                             device float *argmax_values [[buffer(4)]], device uint *argmax_indices [[buffer(5)]],
                             device coherent(device) float *partials [[buffer(6)]],
                             device atomic_uint *counters [[buffer(7)]],
                             constant GgufHeadArgmaxParams &hp [[buffer(8)]],
                             uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]],
                             uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS);
  threadgroup half stage[kStagedStages];
  threadgroup uint arrival;
  // One winner per simdgroup per row: the 256-thread table this replaces
  // (GGUF_STAGED_THREADS * Rows float+uint pairs) cost the tile its
  // occupancy and roughly doubled the fused head's time.
  threadgroup float amax_v[GGUF_STAGED_THREADS / 32 * Rows];
  threadgroup uint amax_i[GGUF_STAGED_THREADS / 32 * Rows];
  const uint per = hp.decode.input_size / GGUF_STAGED_STEP / hp.decode.splits,
             origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS,
             column0 = hp.decode.out_offset + origin;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  auto acc = staged_accumulator<Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, hp.decode.input_size, my);
  gguf_zero(acc);
  staged_accumulate<F, Rows, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, w0, w1, meta, hp.decode.input_size, origin, my,
                                                                  tl, simd_lane, group.y * per, (group.y + 1) * per, acc);
  // The thread's per-row best within its own fragment columns; the argmax
  // reducer folds the threads' pairs after the barrier.
  float best[Rows];
  uint besti[Rows];
#pragma unroll
  for (ushort r = 0; r < Rows; ++r) {
    best[r] = -INFINITY;
    besti[r] = 0xffffffffu;
  }
  const uint tiles = hp.decode.out_stride / GGUF_TILE_COLUMNS;
  const uint ptile = hp.decode.out_offset / GGUF_TILE_COLUMNS + group.x;
  gguf_store_sums<Rows>(
      acc, hp.decode.splits, group.y, partials,
      counters + hp.decode.out_offset / GGUF_TILE_COLUMNS + group.x, hp.decode.out_stride, column0,
      simd_group * 32 + simd_lane, &arrival, [&](uint row, uint column, float v) {
        const uint token = column0 + column;
        const uint lane_index = row / 8;
        if (row % 8 >= hp.head.live_rows[lane_index]) return;
        if ((hp.head.exclude_stop_mask & (1u << lane_index)) &&
            (token == hp.head.stop_token_0 || token == hp.head.stop_token_1))
          return;
        if (v > best[row] || (v == best[row] && token < besti[row])) {
          best[row] = v;
          besti[row] = token;
        }
      });
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (ushort r = 0; r < Rows; ++r) {
    float v = best[r];
    uint ix = besti[r];
#pragma unroll
    for (ushort step = 16; step; step >>= 1) {
      const float ov = simd_shuffle_xor(v, step);
      const uint oi = simd_shuffle_xor(ix, step);
      if (ov > v || (ov == v && oi < ix)) {
        v = ov;
        ix = oi;
      }
    }
    if (simd_lane == 0) {
      amax_v[simd_group * Rows + r] = v;
      amax_i[simd_group * Rows + r] = ix;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < Rows) {
    float v = -INFINITY;
    uint ix = 0xffffffffu;
    for (uint t = 0; t < GGUF_STAGED_THREADS / 32; ++t) {
      const float cv = amax_v[t * Rows + tid];
      const uint ci = amax_i[t * Rows + tid];
      if (cv > v || (cv == v && ci < ix)) {
        v = cv;
        ix = ci;
      }
    }
    argmax_values[tid * tiles + ptile] = v;
    argmax_indices[tid * tiles + ptile] = ix;
  }
}
using GgufArgmaxKernel =
    void(device bfloat *, device uchar *, device uchar *, device uchar *, device float *, device uint *,
         device coherent(device) float *, device atomic_uint *, constant GgufHeadArgmaxParams &, uint2, uint, uint,
         uint);
#define GGUF_DECODE_AMAX(F, f, R) \
  template [[host_name("gguf_decode_" #f "_m" #R "_amax")]] kernel GgufArgmaxKernel gguf_decode_amax<F, R>;
#define GGUF_DECODE_AMAX_ROWS(F, f) \
  GGUF_DECODE_AMAX(F, f, 8) GGUF_DECODE_AMAX(F, f, 16) GGUF_DECODE_AMAX(F, f, 32)
QUANT_FORMATS(GGUF_DECODE_AMAX_ROWS)
GGUF_DECODE_AMAX_ROWS(FmtMXFP4N, mxfp4n)
#undef GGUF_DECODE_AMAX_ROWS
#undef GGUF_DECODE_AMAX

// The native MXFP4 decoders, dispatched on Apple GPU family 10 and up only.
GGUF_DECODE_FORMAT(FmtMXFP4N, mxfp4n)
#undef GGUF_DECODE_FORMAT
#undef GGUF_DECODE_ROWS
#undef GGUF_DECODE

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
// MXFP4 on the Metal 4.1 multiplane tensor (ops::Linear's `mxfp4m` decode
// kernels, Apple GPU family 10 and up): the packed-FP4 decode tile of
// kernels/common/gguf_mxfp4_tile.h over one tensor's planes.
template <ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_mxfp4(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                              device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                              device Out *output [[buffer(4)]],
                              device coherent(device) float *partials [[buffer(5)]],
                              device atomic_uint *counters [[buffer(6)]],
                              device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                              uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                              uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[2 * Rows * 32];
  threadgroup uchar scales_plane[2 * Rows + 4];
  threadgroup uint arrival;
  static_cast<void>(w1);   // MXFP4 has no second plane; the slot keeps the shared GgufDecodeKernel signature
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  gguf_decode_mxfp4_tile<Rows, Ep>(input, w0, meta, output, partials,
                                   counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, aux, p.input_size, p.splits,
                                   p.out_stride, origin, p.out_offset + origin, group.y, simd_lane, simd_group,
                                   stage, scales_plane, &arrival);
}
#define GGUF_DECODE_MXFP4(R, ep, Ep, Out) \
  template [[host_name("gguf_decode_mxfp4m_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode_mxfp4<R, Ep, Out>;
#define GGUF_DECODE_MXFP4_ROWS(ep, Ep, Out) \
  GGUF_DECODE_MXFP4(8, ep, Ep, Out) GGUF_DECODE_MXFP4(16, ep, Ep, Out) GGUF_DECODE_MXFP4(32, ep, Ep, Out)
GGUF_DECODE_MXFP4_ROWS(a, EpNone, bfloat) GGUF_DECODE_MXFP4_ROWS(a_f32, EpNone, float)
GGUF_DECODE_MXFP4_ROWS(r, EpResidual, bfloat) GGUF_DECODE_MXFP4_ROWS(g, EpUpWithGate, bfloat)
#undef GGUF_DECODE_MXFP4_ROWS
#undef GGUF_DECODE_MXFP4
#endif

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
// The packed-operand decoders of the formats whose codes matmul2d reads
// directly (ops::Linear's `<f>m` decode kernels, Apple GPU family 10 and
// up): the packed-B tile of kernels/common/gguf_mxfp4_tile.h over plane0,
// the group's (s, m - s * Zero) coefficient applied in the epilogue.
template <class F, class BT, ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_packed(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                               device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                               device Out *output [[buffer(4)]],
                               device coherent(device) float *partials [[buffer(5)]],
                               device atomic_uint *counters [[buffer(6)]],
                               device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                               uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat stage[2 * Rows * 32];
  threadgroup float2 coefs[2 * 64];
  threadgroup float sums[2 * Rows];
  threadgroup uint arrival;
  static_cast<void>(w1);   // the packed formats decode from plane0 alone
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  gguf_decode_packed_tile<F, BT, Rows, Ep>(input, w0, meta, output, partials,
                                          counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, aux, p.input_size,
                                          p.splits, p.out_stride, origin, p.out_offset + origin, group.y, simd_lane,
                                          simd_group, stage, coefs, sums, &arrival);
}
#define GGUF_DECODE_PACKED(F, f, BT, R, ep, Ep, Out) \
  template [[host_name("gguf_decode_" #f "m_m" #R "_" #ep)]] \
  kernel GgufDecodeKernel<Out> gguf_decode_packed<F, BT, R, Ep, Out>;
#define GGUF_DECODE_PACKED_ROWS(F, f, BT, ep, Ep, Out) \
  GGUF_DECODE_PACKED(F, f, BT, 8, ep, Ep, Out) GGUF_DECODE_PACKED(F, f, BT, 16, ep, Ep, Out) \
  GGUF_DECODE_PACKED(F, f, BT, 32, ep, Ep, Out)
#define GGUF_DECODE_PACKED_FORMAT(F, f, BT)                                              \
  GGUF_DECODE_PACKED_ROWS(F, f, BT, a, EpNone, bfloat)                                   \
  GGUF_DECODE_PACKED_ROWS(F, f, BT, a_f32, EpNone, float)                                \
  GGUF_DECODE_PACKED_ROWS(F, f, BT, r, EpResidual, bfloat)                               \
  GGUF_DECODE_PACKED_ROWS(F, f, BT, g, EpUpWithGate, bfloat)
GGUF_DECODE_PACKED_FORMAT(FmtQ40, q40, uint4b_format)
GGUF_DECODE_PACKED_FORMAT(FmtQ41, q41, uint4b_format)
GGUF_DECODE_PACKED_FORMAT(FmtQ4K, q4k, uint4b_format)
GGUF_DECODE_PACKED_FORMAT(FmtQ80, q80, int8_t)
GGUF_DECODE_PACKED_FORMAT(FmtPQ20, pq20, uint2b_format)
#undef GGUF_DECODE_PACKED_FORMAT
#undef GGUF_DECODE_PACKED_ROWS
#undef GGUF_DECODE_PACKED
#endif

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits. The fused and prefill kernels stay plain kernels: as
// template instantiations (weak_odr) the compiler infers fewer parameter attributes and inlines differently.
#define GGUF_SEGMENT(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_FUSED(R, SUFFIX, NATIVE)                                                                       \
  kernel void gguf_decode_fused_m##R##SUFFIX(device bfloat *input [[buffer(0)]], GGUF_SEGMENT(1, w0a, w1a, ma),          \
                                     GGUF_SEGMENT(4, w0b, w1b, mb), GGUF_SEGMENT(7, w0c, w1c, mc),                \
                                     device bfloat *output [[buffer(10)]],                                        \
                                     device coherent(device) float *partials [[buffer(11)]],                      \
                                     device atomic_uint *counters [[buffer(12)]],                                 \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]],                            \
                                     uint2 group [[threadgroup_position_in_grid]],                                \
                                     uint simd_lane [[thread_index_in_simdgroup]],                                \
                                     uint simd_group [[simdgroup_index_in_threadgroup]]) {                        \
    threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries]; threadgroup uint arrival; \
    threadgroup float2 packed_coefs[128]; threadgroup float packed_sums[2 * R];                                 \
    const uint t0 = p.cols[0] / GGUF_TILE_COLUMNS, t1 = t0 + p.cols[1] / GGUF_TILE_COLUMNS;                        \
    const uint s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;                                                       \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                         \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                         \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                          \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / GGUF_STAGED_STEP / p.splits; \
    const uint origin = local * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS, column0 = p.offset[s] + origin; \
    if (NATIVE && gguf_decode_fused_mxfp4<R>(p.fmt[s], input, w0, meta, output, partials,                       \
                      counters + p.offset[s] / GGUF_TILE_COLUMNS + local, p.input_size, p.splits, p.out_stride, \
                      origin, column0, group.y, simd_lane, simd_group, stage, (threadgroup uchar *)tl,        \
                      &arrival))                                                                            \
      return;                                                                                               \
    if (NATIVE && gguf_decode_fused_packed<R>(p.fmt[s], input, w0, meta, output, partials,                    \
                      counters + p.offset[s] / GGUF_TILE_COLUMNS + local, p.input_size, p.splits, p.out_stride, \
                      origin, column0, group.y, simd_lane, simd_group, (threadgroup bfloat *)stage,           \
                      packed_coefs, packed_sums, &arrival))                                                   \
      return;                                                                                               \
    threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;                                            \
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(input, p.input_size, my);                                         \
    gguf_zero(acc);                                                                                               \
    staged_accumulate_any<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, NATIVE>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, \
                                            simd_group * 32 + simd_lane, simd_lane, group.y * per, (group.y + 1) * per, acc); \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / GGUF_TILE_COLUMNS + local,       \
                       p.out_stride, column0, simd_group * 32 + simd_lane, &arrival,                              \
                       [&](uint row, uint column, float v) { output[ulong(row) * p.out_stride + column0 + column] = bfloat(v); }); \
  }
GGUF_DECODE_FUSED(8, , false) GGUF_DECODE_FUSED(16, , false) GGUF_DECODE_FUSED(32, , false)
// The `_n` fused kernels an Apple GPU family 10 host dispatches, whose MXFP4
// segments decode on the native FP4 path.
GGUF_DECODE_FUSED(8, _n, true) GGUF_DECODE_FUSED(16, _n, true) GGUF_DECODE_FUSED(32, _n, true)
#undef GGUF_DECODE_FUSED
#undef GGUF_SEGMENT

// The prefill kernels: grid (GGUF_PREFILL_ROWS-row tiles of the chunk, column tiles of the segment), four
// simdgroups; the residual and gate kernels read aux at buffer 5, the plain one binds none.
#define GGUF_PREFILL_BUFFERS                                                                                       \
  device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]],             \
      device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]]
#define GGUF_PREFILL_THREAD                                                                                        \
  uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
      uint simd_group [[simdgroup_index_in_threadgroup]]
#define GGUF_PREFILL_TABLES(F)                                                                                     \
  threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];                                     \
  quant_pair_table<F>(tl, simd_group * 32 + simd_lane, GGUF_PREFILL_THREADS);                                      \
  threadgroup half stage[kPrefillStages]
#define GGUF_PREFILL(F, f)                                                                                         \
  kernel void gguf_prefill_##f##_a(GGUF_PREFILL_BUFFERS, constant GgufPrefillParams &p [[buffer(5)]],            \
                                   GGUF_PREFILL_THREAD) {                                                          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP>(input + ulong(first) * p.input_size, w0, w1, meta,                        \
                                         output + ulong(first) * p.out_stride, p.input_size,                       \
                                         group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,      \
                                         p.out_stride, p.out_offset);                                              \
  }
#define GGUF_PREFILL_EPILOGUE(F, f, ep, Ep)                                                                        \
  kernel void gguf_prefill_##f##_##ep(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],                    \
                                      constant GgufPrefillParams &p [[buffer(6)]], GGUF_PREFILL_THREAD) {          \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, Ep>(input + ulong(first) * p.input_size, w0, w1, meta,                    \
                                             output + ulong(first) * p.out_stride, p.input_size,                   \
                                             group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane, simd_group,  \
                                             p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);       \
  }
#define GGUF_PREFILL_FORMAT(F, f) \
  GGUF_PREFILL(F, f) GGUF_PREFILL_EPILOGUE(F, f, r, EpResidual) GGUF_PREFILL_EPILOGUE(F, f, g, EpUpWithGate)
QUANT_FORMATS(GGUF_PREFILL_FORMAT)
GGUF_PREFILL_FORMAT(FmtMXFP4N, mxfp4n)
#undef GGUF_PREFILL_FORMAT

// The residual kernels over a view of the leading inputs of wider weight rows, for the Neural Engine FFN split's down
// projection, one per format. Each runs the tile of its residual kernel on the planes advanced past the groups and
// meta units the view leaves unread in the rows of the plane tiles before its column tile, which the tile then
// addresses as rows of input_size inputs.
#define GGUF_PREFILL_LEADING_INPUTS(F, f)                                                                          \
  kernel void gguf_prefill_##f##_r_leading_inputs(GGUF_PREFILL_BUFFERS, device bfloat *aux [[buffer(5)]],          \
                                                  constant GgufPrefillLeadingParams &leading [[buffer(6)]],        \
                                                  GGUF_PREFILL_THREAD) {                                           \
    GGUF_PREFILL_TABLES(F);                                                                                        \
    constant GgufPrefillParams &p = leading.prefill;                                                               \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;                    \
    const ulong before = ulong(group.y * GGUF_TILE_COLUMNS / QUANT_TILE_ROWS) * QUANT_TILE_ROWS;                   \
    const uint groups = p.input_size / 32, plane_groups = leading.plane_input_size / 32;                           \
    const ulong unread = before * (plane_groups - groups),                                                         \
                unread_units = before * (plane_groups / F::MetaGroups - groups / F::MetaGroups);                   \
    gguf_prefill_tile<F, GGUF_PREFILL_SIMDGROUP_ROWS, GGUF_PREFILL_SIMDGROUPS, GGUF_TILE_COLUMNS, GGUF_PREFILL_STEP, \
                      EpResidual>(input + ulong(first) * p.input_size, w0 + unread * F::P0, w1 + unread * F::P1,   \
                                  meta + unread_units * F::MetaBytes, output + ulong(first) * p.out_stride,        \
                                  p.input_size, group.y * GGUF_TILE_COLUMNS, rows, stage, tl, simd_lane,           \
                                  simd_group, p.out_stride, p.out_offset, aux + ulong(first) * p.out_stride);      \
  }
QUANT_FORMATS(GGUF_PREFILL_LEADING_INPUTS)
GGUF_PREFILL_LEADING_INPUTS(FmtMXFP4N, mxfp4n)
#undef GGUF_PREFILL_LEADING_INPUTS
#undef GGUF_PREFILL_EPILOGUE
#undef GGUF_PREFILL
#undef GGUF_PREFILL_TABLES
#undef GGUF_PREFILL_THREAD
#undef GGUF_PREFILL_BUFFERS
