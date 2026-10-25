// The score-only head's single dispatch (RICHENGINE_SCORE_TENSOR and the
// fused RICHENGINE_SCORE_FUSED): a gather over the wanted logits columns.
// Two simdgroups of a threadgroup each stage GGUF_STAGED_COLUMNS of the
// listed columns and run the staged decode tile's dequant + matmul2d on
// them — one simdgroup per 32 columns, its lane staging its own column —
// so a column costs its group loads alone, not its quant tile's. The fused
// variant first runs the decode norm's bitwise math in-kernel (256 threads,
// rms_inverse's reduction, norm_rms_row's store expression) into the
// operand buffer, removing the norm dispatch and its buffer round trip.
// Output rows and the score readback's expectations: lane l's logits are
// row l * lane_rows of the packed destination, as the unrestricted head
// leaves them.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/rms_inverse.h"

constant constexpr uint kScoreRows = 8;   // the tile holding a lane batch
constant constexpr uint kScoreThreads = 256;

template <class F, class W, bool Fuse>
kernel void gguf_decode_score(device bfloat *input [[buffer(0)]],
                              device bfloat *normed [[buffer(1)]],
                              device const W *normw [[buffer(2)]],
                              device uchar *w0 [[buffer(3)]],
                              device uchar *w1 [[buffer(4)]],
                              device uchar *meta [[buffer(5)]],
                              device float *output [[buffer(6)]],
                              device const uint *columns [[buffer(7)]],
                              constant GgufScoreParams &p [[buffer(8)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  constexpr ushort Cols = GGUF_STAGED_COLUMNS, KS = GGUF_STAGED_STEP;
  threadgroup float reductions[8];
  threadgroup half stage[2 * Cols * KS];
  threadgroup half2 tl[F::Kind == QuantCodebook && !F::Native ? kQuantPairTableEntries : 1];
  if constexpr (Fuse) {
    for (uint r = 0; r < p.lanes; ++r) {
      device bfloat *row = input + ulong(r) * p.lane_rows * p.input_size;
      const float inverse = rms_inverse(row, p.input_size, reductions, tid, lane, sg, p.norm_eps);
      for (uint column = tid; column < p.input_size; column += kScoreThreads)
        normed[r * p.input_size + column] = bfloat((float(row[column]) * inverse) * float(normw[column]));
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
  // The tile's two working simdgroups; the rest leave before the pair
  // table's barrier, which only the staging threads reach.
  if (sg >= GGUF_TILE_COLUMNS / Cols) return;
  quant_pair_table<F>(tl, tid, GGUF_TILE_COLUMNS / Cols * 32);
  const uint groups = p.input_size / 32, units = groups / F::MetaGroups;
  const uint base = tg.x * GGUF_TILE_COLUMNS + sg * Cols;
  // The thread's own column of the gather, clamped so a tail group's idle
  // lanes read a valid row; their stores are masked below.
  const uint want = columns[min(base + lane, p.columns - 1)];
  const uint plane_tile = want / QUANT_TILE_ROWS, plane_row = want % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P0;
  device uchar *tw1 = w1 + (ulong(plane_tile) * groups * QUANT_TILE_ROWS + plane_row) * F::P1;
  device uchar *tmeta = meta + (ulong(plane_tile) * units * QUANT_TILE_ROWS + plane_row) * F::MetaBytes;
  const uint a_stride = Fuse ? p.input_size : p.lane_rows * p.input_size;
  device bfloat *a_base = Fuse ? normed : input;
  auto a = tensor(a_base, dextents<int, 2>{int(p.input_size), int(kScoreRows)},
                  array<int, 2>{1, int(a_stride)});
  constexpr auto descriptor = matmul2d_descriptor(kScoreRows, Cols, KS, false, true, false,
                                                  matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  threadgroup half *my = stage + sg * Cols * KS;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(my, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto a0 = a.template slice<KS, kScoreRows>(0, 0);
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
  gguf_zero(acc);
  // One group of 32 inputs per step, this thread staging its column's.
  for (uint step = 0; step < groups; ++step) {
    typename F::Payload packed = F::load(tw0 + ulong(step) * QUANT_TILE_ROWS * F::P0,
                                         tw1 + ulong(step) * QUANT_TILE_ROWS * F::P1);
    const uint unit = step / F::MetaGroups;
    typename F::Meta hdr = F::loadMeta(tmeta + ulong(unit) * QUANT_TILE_ROWS * F::MetaBytes);
    dequant32<F>(packed, hdr, step % F::MetaGroups, tl, my + lane * KS);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto aslice = a.template slice<KS, kScoreRows>(step * KS, 0);
    operation.run(aslice, b0, acc);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  // Row r is lane r; column c of the accumulator is the gather's column
  // base + c. Dead rows and tail columns store nothing.
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    const uint r = index[1], c = index[0];
    if (r >= p.lanes || base + c >= p.columns) continue;
    output[ulong(r) * p.lane_rows * p.out_stride + columns[base + c]] = acc[i];
  }
}

// The bf16-weight fused variant and the unfused one, which never reads its
// normw slot, instantiate against bfloat weights; _fused_f32 takes a GGUF's
// F32 norm weights, as normKernel's _f32 tag does.
#define GGUF_SCORE_KERNEL(F, f, FUSE, W, SUFFIX)                                          \
  template [[host_name("gguf_decode_score_" #f "_m8" SUFFIX)]]                             \
  kernel void gguf_decode_score<F, W, FUSE>(device bfloat *, device bfloat *,              \
                                          device const W *, device uchar *, device uchar *, \
                                          device uchar *, device float *,                  \
                                          device const uint *, constant GgufScoreParams &, \
                                          uint2, uint, uint, uint);
#define GGUF_SCORE_FORMAT(F, f)                                       \
  GGUF_SCORE_KERNEL(F, f, false, bfloat, "")                          \
  GGUF_SCORE_KERNEL(F, f, true, bfloat, "_fused")                     \
  GGUF_SCORE_KERNEL(F, f, true, float, "_fused_f32")
QUANT_FORMATS(GGUF_SCORE_FORMAT)
