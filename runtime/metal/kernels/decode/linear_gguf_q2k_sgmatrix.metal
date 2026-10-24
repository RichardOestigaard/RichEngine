// Q2_K decode on the Metal 4.1 packed uint2b operand (the `q2km` kernels,
// Apple GPU family 10 and up): the packed-B tile of
// kernels/common/q2k_mpp_tile.h over plane0, each 16-group's (s, m)
// coefficient applied in the epilogue, as the `<f>m` kernels of
// kernels/shared/gguf_linear.metal do for the single-coefficient formats.
// Same dispatch as the staged decode: grid (64-column tiles, K partitions),
// two simdgroups of GGUF_STAGED_COLUMNS columns per threadgroup. Kept for
// benchmarks with the other packed formats — RICHENGINE_GGUF_PACKED_ON
// selects it (ops::LinearGguf's packedFormat).
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/q2k_mpp_tile.h"
#include "metal/kernels/common/split_reduce.h"

#if defined(__HAVE_INT4B_FORMAT_TYPE__) && defined(__HAVE_INT2B_FORMAT_TYPE__)
template <ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_q2k(device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                            device uchar *w1 [[buffer(2)]], device uchar *meta [[buffer(3)]],
                            device Out *output [[buffer(4)]],
                            device coherent(device) float *partials [[buffer(5)]],
                            device atomic_uint *counters [[buffer(6)]],
                            device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                            uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup bfloat stage[4 * Rows * 32];
  threadgroup float4 coefs[2 * 64];
  threadgroup float sums[4 * Rows];
  threadgroup uint arrival;
  static_cast<void>(w1);   // Q2_K decodes from plane0 alone; the slot keeps the shared GgufDecodeKernel signature
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  gguf_decode_q2k_tile<Rows, Ep>(input, w0, meta, output, partials,
                               counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, aux, p.input_size,
                               p.splits, p.out_stride, origin, p.out_offset + origin, group.y, simd_lane,
                               simd_group, stage, coefs, sums, &arrival);
}
template <class Out>
using GgufDecodeKernel = void(device bfloat *, device uchar *, device uchar *, device uchar *, device Out *,
                              device coherent(device) float *, device atomic_uint *, device bfloat *,
                              constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE_Q2K(R, ep, Ep, Out) \
  template [[host_name("gguf_decode_q2km_m" #R "_" #ep)]] kernel GgufDecodeKernel<Out> gguf_decode_q2k<R, Ep, Out>;
#define GGUF_DECODE_Q2K_ROWS(ep, Ep, Out) \
  GGUF_DECODE_Q2K(8, ep, Ep, Out) GGUF_DECODE_Q2K(16, ep, Ep, Out) GGUF_DECODE_Q2K(32, ep, Ep, Out)
GGUF_DECODE_Q2K_ROWS(a, EpNone, bfloat) GGUF_DECODE_Q2K_ROWS(a_f32, EpNone, float)
GGUF_DECODE_Q2K_ROWS(r, EpResidual, bfloat) GGUF_DECODE_Q2K_ROWS(g, EpUpWithGate, bfloat)
#undef GGUF_DECODE_Q2K_ROWS
#undef GGUF_DECODE_Q2K
#endif
