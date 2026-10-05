// MXFP4 decode with pre-packed activations (ops::Linear's `mxfp4p` decode
// kernels, Apple GPU family 10 and up): one gguf_pack_half dispatch per step
// converts the bf16 input into a slot-permuted fp16 scratch plane plus a
// per-(row, group) exponent byte plane, then the decode kernels run the
// device-operand tile of kernels/common/gguf_mxfp4p_tile.h — no threadgroup
// staging, no barriers. Weight planes and meta in the MDGG0001 layout
// (metal/abi/QuantFormat.h), the same image gguf_decode_mxfp4m_* reads.
// Keep the source order of float operations, which Metal's default fast math
// lets the compiler reassociate. Set before the includes, so it also holds
// for the shared tile code compiled here.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/gguf_mxfp4p_tile.h"
#include "metal/kernels/common/split_reduce.h"

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
// Pack: one thread per packed element, dispatched over rows * K threads.
// Lane s of the simdgroup covering (row, group) stages slot s of the group —
// input element 16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3) — so a simd_max
// over the lanes is the group's max |a| and the exponent logic matches the
// staged tile's stage32 exactly: e = 0 when the block converts to fp16
// exactly, else the block is written times 2^-e and exponents[row][group]
// holds 127 + e. `packed` is [rows][K] halves, `exponents` [rows][K / 32]
// bytes (kernels/common/gguf_mxfp4p_tile.h documents the layout). The grid's
// thread count is a multiple of 32, so every simdgroup covers exactly one
// (row, group) block.
kernel void gguf_pack_half(device bfloat *input [[buffer(0)]], device half *packed [[buffer(1)]],
                           device uchar *exponents [[buffer(2)]], constant GgufDecodeParams &p [[buffer(8)]],
                           uint i [[thread_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]]) {
  const uint groups = p.input_size / 32;
  const uint s = i & 31, g = i >> 5;
  const uint row = g / groups, slot = g % groups;
  const float a = float(input[ulong(row) * p.input_size + slot * 32 +
                              16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3)]);
  const float m = simd_max(fabs(a));
  const int e = m > 30720.0f || (m > 0.0f && m < 6.1e-5f)
                    ? clamp(int(floor(log2(m))) - 14, -100, 100)
                    : 0;
  packed[i] = half(a * as_type<float>(uint(127 - e) << 23));
  if (s == 0) exponents[row * groups + slot] = uchar(127 + e);
  static_cast<void>(simd_lane);
}

// The decode kernels: grid (64-column tiles, K partitions), two simdgroups
// of 32 columns per threadgroup, exactly as gguf_decode_mxfp4m_*. Buffer 0
// binds the packed half plane (not the bf16 input) and buffer 2 — the w1
// slot an MXFP4 segment never fills — the exponent byte plane.
template <ushort Rows, GgufEpilogue Ep, class Out>
kernel void gguf_decode_mxfp4p(device half *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],
                               device uchar *exponents [[buffer(2)]], device uchar *meta [[buffer(3)]],
                               device Out *output [[buffer(4)]],
                               device coherent(device) float *partials [[buffer(5)]],
                               device atomic_uint *counters [[buffer(6)]],
                               device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]],
                               uint2 group [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                               uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint arrival;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  gguf_decode_mxfp4p_tile<Rows, Ep>(input, exponents, w0, meta, output, partials,
                                    counters + p.out_offset / GGUF_TILE_COLUMNS + group.x, aux, p.input_size,
                                    p.splits, p.out_stride, origin, p.out_offset + origin, group.y, simd_lane,
                                    simd_group, &arrival);
}
template <class Out>
using GgufDecodeMxfp4pKernel = void(device half *, device uchar *, device uchar *, device uchar *, device Out *,
                                    device coherent(device) float *, device atomic_uint *, device bfloat *,
                                    constant GgufDecodeParams &, uint2, uint, uint);
#define GGUF_DECODE_MXFP4P(R, ep, Ep, Out) \
  template [[host_name("gguf_decode_mxfp4p_m" #R "_" #ep)]] \
  kernel GgufDecodeMxfp4pKernel<Out> gguf_decode_mxfp4p<R, Ep, Out>;
#define GGUF_DECODE_MXFP4P_ROWS(ep, Ep, Out) \
  GGUF_DECODE_MXFP4P(8, ep, Ep, Out) GGUF_DECODE_MXFP4P(16, ep, Ep, Out) GGUF_DECODE_MXFP4P(32, ep, Ep, Out)
GGUF_DECODE_MXFP4P_ROWS(a, EpNone, bfloat) GGUF_DECODE_MXFP4P_ROWS(a_f32, EpNone, float)
GGUF_DECODE_MXFP4P_ROWS(r, EpResidual, bfloat) GGUF_DECODE_MXFP4P_ROWS(g, EpUpWithGate, bfloat)
#undef GGUF_DECODE_MXFP4P_ROWS
#undef GGUF_DECODE_MXFP4P

// The prefill kernels (ops::Linear's `mxfp4p` prefill path): the staged
// prefill kernels' grid — (GGUF_PREFILL_ROWS-row tiles of the chunk, column
// tiles of the segment) of four-simdgroup threadgroups — and the same row
// masking, with the packed half plane for A (buffer 0, offset by the tile's
// first row) and the exponent bytes in the w1 slot (buffer 2). The residual
// and gate kernels read aux at buffer 5, the plain one binds none — exactly
// the staged prefill kernels' contract.
#define GGUF_PREFILL_MXFP4P(ep, Ep)                                                                     \
  kernel void gguf_prefill_mxfp4p_##ep(device half *input [[buffer(0)]], device uchar *w0 [[buffer(1)]],  \
                                       device uchar *exponents [[buffer(2)]], device uchar *meta [[buffer(3)]], \
                                       device bfloat *output [[buffer(4)]],                             \
                                       device bfloat *aux [[buffer(5)]],                                \
                                       constant GgufPrefillParams &p [[buffer(6)]],                     \
                                       uint2 group [[threadgroup_position_in_grid]],                    \
                                       uint simd_lane [[thread_index_in_simdgroup]],                    \
                                       uint simd_group [[simdgroup_index_in_threadgroup]]) {            \
    const uint first = group.x * GGUF_PREFILL_ROWS, rows = p.rows > first ? p.rows - first : 0;           \
    const uint groups = p.input_size / 32;                                                              \
    gguf_prefill_mxfp4p_tile<GGUF_PREFILL_SIMDGROUP_ROWS, Ep>(                                           \
        input + ulong(first) * p.input_size, exponents + ulong(first) * groups, w0, meta,               \
        output + ulong(first) * p.out_stride, p.input_size, group.y * GGUF_TILE_COLUMNS, rows,          \
        simd_lane, simd_group, p.out_stride, p.out_offset,                                              \
        aux ? aux + ulong(first) * p.out_stride : aux);                                                 \
  }
GGUF_PREFILL_MXFP4P(a, EpNone)
GGUF_PREFILL_MXFP4P(r, EpResidual)
GGUF_PREFILL_MXFP4P(g, EpUpWithGate)
#undef GGUF_PREFILL_MXFP4P
#endif
