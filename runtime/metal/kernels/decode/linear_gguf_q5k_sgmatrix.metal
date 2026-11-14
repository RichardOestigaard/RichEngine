// Q5_K decode on the Metal 4.1 packed uint4b operand (the `q5km` kernels,
// Apple GPU family 10 and up): plane0's low 4 bits feed matmul2d directly,
// as Q4_K's do in kernels/shared/gguf_linear.metal, and plane1's fifth bits
// expand into a threadgroup half B beside the staged activations — the
// packed-B tile of kernels/common/q5k_mpp_tile.h. Each 32-group's (s, m)
// coefficient applies in the epilogue, as the `<f>m` kernels do for the
// single-coefficient formats. Same dispatch as the staged decode: grid
// (64-column tiles, K partitions), two simdgroups of GGUF_STAGED_COLUMNS
// columns per threadgroup. Kept for benchmarks with the other packed
// formats — RICHENGINE_GGUF_PACKED_ON selects it (ops::LinearGguf's
// packedFormat).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/q5k_mpp_tile.h"
#include "metal/kernels/common/split_reduce.h"

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
template <ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_q5k(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                            device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                            device Out *output [[buffer(4)]],
                            device coherent(device) float *partials [[buffer(5)]],
                            device atomic_uint *counters [[buffer(6)]],
                            device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat stage[2 * Rows * 32];
  threadgroup half hi[2 * 2 * GGUF_STAGED_COLUMNS * 32];
  threadgroup float2 coefs[2 * 64];
  threadgroup float sums[2 * Rows];
  threadgroup uint arrival;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  gguf_decode_q5k_tile<Rows, Ep>(input, w0, w1, meta, output, partials,
                               counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, aux, p.input_size,
                               p.splits, p.out_stride, origin, p.out_offset + origin, group.y, simd_lane,
                               simd_group, stage, hi, coefs, sums, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE_Q5K(R, ep, Ep, Out) \
  template [[host_name("gguf_decode_q5km_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode_q5k<R, Ep, Out>;
#define GGUF_DECODE_Q5K_ROWS(ep, Ep, Out) \
  GGUF_DECODE_Q5K(8, ep, Ep, Out) GGUF_DECODE_Q5K(16, ep, Ep, Out) GGUF_DECODE_Q5K(32, ep, Ep, Out)
GGUF_DECODE_Q5K_ROWS(a, EpNone, bfloat) GGUF_DECODE_Q5K_ROWS(a_f32, EpNone, float)
GGUF_DECODE_Q5K_ROWS(r, EpResidual, bfloat) GGUF_DECODE_Q5K_ROWS(g, EpUpWithGate, bfloat)
#undef GGUF_DECODE_Q5K_ROWS
#undef GGUF_DECODE_Q5K
#endif
