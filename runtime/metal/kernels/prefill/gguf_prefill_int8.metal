// Int8-activation prefill GEMMs for the MLX affine weights (the private AF
// formats of metal/abi/QuantFormat.h): af4, af5, af6 and af8 in meta groups
// of 64 inputs and af4g128, af5g128, af6g128 and af8g128 in groups of 128,
// opted in by RICHENGINE_PREFILL_FAST_INT8 (ops::LinearGguf.cpp's
// addGgufPrefill). Each 64-input group g of a chunk row is quantized to int8
// codes qa with its own scale sa_g, and with w = s * q + z per weight meta
// unit:
//
//   y[m][n] = sum_g ( (sa_g[m] s_g[n]) sum_{k in g} qa[m][k] q[n][k]
//                     + z_g[n] (sa_g[m] sum_{k in g} qa[m][k]) )
//
// The weights' codes feed int8 16x32x16 matmul2d fragments straight from the
// planes; each group's int32 sums fold into fp32 once. The weights are exact;
// only the activations round, per row and group, so the output is not
// bit-identical to the bf16 prefill tile's.
//
// The fragment schedule and the activations' slot order are adapted from
// oMLX's oQ A8 kernel (Apache-2.0 notice in THIRD_PARTY_NOTICES): in fragment
// step t (0-3) of a group, lane column x (coord.x / 4) supplies inputs
// 16x + 4t .. 16x + 4t + 3, 32-group x >> 1, chunk 2 (x & 1) + (t >> 1) of the
// planes, chunk slots {0, 4, 1, 5} or {2, 6, 3, 7} (metal/abi/QuantFormat.h).
// The quantizer writes each 32-group in that order. af4's plane0 holds the
// codes as Q4_0's pair-interleaved words (PAIRS); the oQ formats and af4g128
// hold them as a slot-order bitstream, slot s at bits [s * BITS, (s + 1) *
// BITS) of the group's 4 * BITS bytes, read per chunk below. af8's codes run
// 0..255: they enter the matmul as code - 128 and the epilogue's bias becomes
// z + 128 s, the same products in int32.
//
// The tile itself is kernels/common/gguf_prefill_int8_tile.h, shared with the
// MoE experts' int8 passes (kernels/shared/moe_int8.metal).
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_prefill_int8_tile.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/int8_activations.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// The schedule position, within its 32-group, of the input at slot s of the planes: chunk c = s >> 3 is lane half
// c >> 1 and step half c & 1; chunk slot u holds step bit (u >> 1) & 1 and lane input ((u & 1) << 1) | (u >> 2).
inline uint prefill_int8_position(uint slot) {
  const uint c = slot >> 3, u = slot & 7;
  return 16 * (c >> 1) + 4 * (2 * (c & 1) + ((u >> 1) & 1)) + (((u & 1) << 1) | (u >> 2));
}

// grid (groups / GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS, rows): simdgroup s of threadgroup (b, row) quantizes group
// b * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + s of the row by the int8-activation rule
// (kernels/common/int8_activations.h), lane l its inputs l and 32 + l, and stores the group's scale and the scale
// times its codes' sum.
kernel void gguf_prefill_int8_quantize(device const bfloat *input [[buffer(0)]], device char *codes [[buffer(1)]],
                                       device float2 *scales [[buffer(2)]],
                                       constant GgufPrefillInt8QuantizeParams &p [[buffer(3)]],
                                       uint2 tid [[threadgroup_position_in_grid]],
                                       ushort lane [[thread_index_in_simdgroup]],
                                       ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint g = tid.x * GGUF_PREFILL_INT8_QUANTIZE_SIMDGROUPS + simd_group, row = tid.y;
  if (g >= p.input_size / GGUF_PREFILL_INT8_GROUP || row >= p.rows) return;
  const ulong at = ulong(row) * p.input_size + g * GGUF_PREFILL_INT8_GROUP;
  const bfloat b0 = input[at + lane], b1 = input[at + 32 + lane];
  const float x0 = float(b0), x1 = float(b1);
  const Int8Block block = int8_activation_block(simd_max(max(fabs(x0), fabs(x1))));
  const int q0 = int8_activation_code(x0, block.inverse), q1 = int8_activation_code(x1, block.inverse);
  const uint position = prefill_int8_position(quant_slot(lane));
  codes[at + position] = char(q0);
  codes[at + 32 + position] = char(q1);
  const int sum = simd_sum(q0 + q1);
  const bool finite = simd_all(!int8_activation_not_finite(b0) && !int8_activation_not_finite(b1));
  if (lane != 0) return;
  ((device uint2 *)scales)[ulong(g) * p.storage_rows + row] =
      uint2(int8_activation_bits(block.scale, finite), int8_activation_bits(block.scale * float(sum), finite));
}

// The (column tile, row tile) of tile index `at` of grid.x column tiles by grid.y row tiles:
// GGUF_PREFILL_INT8_WALK row tiles take each column tile in turn before the next column tile, the last block of row
// tiles its remainder.
inline uint2 prefill_int8_walk(uint at, uint2 grid) {
  constexpr uint S = GGUF_PREFILL_INT8_WALK;
  const uint block = at / (S * grid.x), height = min(S, grid.y - block * S), i = at - block * S * grid.x;
  return uint2(i / height, block * S + i % height);
}

// grid (GGUF_PREFILL_INT8_WALK, its tiles / GGUF_PREFILL_INT8_WALK rounded up), GGUF_PREFILL_INT8_THREADS threads:
// threadgroup (x, y) takes tile index y GGUF_PREFILL_INT8_WALK + x of the segment's column tiles by the chunk's row
// tiles of GGUF_PREFILL_INT8_ROWS (prefill_int8_walk), and those past the tiles none. aux (buffer 5) is the
// residual or the gate, the output for the plain epilogue, which reads none.
#define GGUF_PREFILL_INT8_KERNEL(B, R, M, P, f, ep, Ep)                                                              \
  kernel void gguf_prefill_int8_##f##_##ep(                                                                           \
      device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],                              \
      device const uchar *w0 [[buffer(2)]], device const uchar *meta [[buffer(3)]],                                   \
      device bfloat *output [[buffer(4)]], device const bfloat *aux [[buffer(5)]],                                    \
      constant GgufPrefillInt8Params &p [[buffer(6)]],                                                                \
      uint2 tid [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]],                          \
      ushort simd_group [[simdgroup_index_in_threadgroup]]) {                                                         \
    const uint rows = (p.rows + GGUF_PREFILL_INT8_ROWS - 1) / GGUF_PREFILL_INT8_ROWS,                                 \
               at = tid.y * GGUF_PREFILL_INT8_WALK + tid.x;                                                           \
    if (at >= p.column_tiles * rows) return;                                                                          \
    prefill_int8_tile<B, R, M, P, Ep>(codes, scales, w0, meta, output, aux,                                           \
                                      {p.input_size, p.rows, p.storage_rows, p.out_stride, p.out_offset,              \
                                       p.plane_input_size},                                                         \
                                      prefill_int8_walk(at, uint2(p.column_tiles, rows)), lane, simd_group);          \
  }
#define GGUF_PREFILL_INT8_FORMAT(B, R, M, P, f)                                                                       \
  GGUF_PREFILL_INT8_KERNEL(B, R, M, P, f, a, EpNone) GGUF_PREFILL_INT8_KERNEL(B, R, M, P, f, r, EpResidual)          \
      GGUF_PREFILL_INT8_KERNEL(B, R, M, P, f, g, EpUpWithGate)
// af4: pair-interleaved code words, meta groups of 64. The oQ formats and
// af4g128: the slot-order bitstream, REC bytes per (row, group of 32).
GGUF_PREFILL_INT8_FORMAT(4, 16, 2, true, af4)
GGUF_PREFILL_INT8_FORMAT(4, 16, 4, false, af4g128)
GGUF_PREFILL_INT8_FORMAT(5, 20, 2, false, af5)
GGUF_PREFILL_INT8_FORMAT(6, 24, 2, false, af6)
GGUF_PREFILL_INT8_FORMAT(8, 32, 2, false, af8)
GGUF_PREFILL_INT8_FORMAT(5, 20, 4, false, af5g128)
GGUF_PREFILL_INT8_FORMAT(6, 24, 4, false, af6g128)
GGUF_PREFILL_INT8_FORMAT(8, 32, 4, false, af8g128)
GGUF_PREFILL_INT8_FORMAT(2, 8, 2, false, af2)
GGUF_PREFILL_INT8_FORMAT(3, 12, 2, false, af3)
GGUF_PREFILL_INT8_FORMAT(2, 8, 4, false, af2g128)
GGUF_PREFILL_INT8_FORMAT(3, 12, 4, false, af3g128)
#undef GGUF_PREFILL_INT8_FORMAT
#undef GGUF_PREFILL_INT8_KERNEL
