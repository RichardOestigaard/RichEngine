#pragma once
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/q4_sgmatrix.h"

// The eight-row X^T table the Apple9 GGUF register kernel reads
// (LinearInput::Table16, laid out as metal/abi/Gguf.h states). A 64-element
// span occupies 512 bfloat in the q4sg::xt_offset order; its fragments follow the chunk order of the GGUF
// image (metal/abi/QuantFormat.h): fragment 4q + f holds pair f of every
// chunk of the span's 32-element group q, so fragments 4q + 2h and
// 4q + 2h + 1 cover its 16-element group h. Per row, the sum of every 32
// inputs (formats with a min) and the chain seed of every 16 inputs (formats
// with a zero point) are stored row-minor, so a lane reads its two rows at
// once.
namespace gguf_sg {

// Formats with a zero point (Q6_K, Q3_K) enter the MMA as 160 + code - zero,
// exact in bf16 (160 = 128 + 32, Q6_K's zero point), so each 16-input chain
// starts from -160 times the sum of its inputs: the seed the table stores.
constant constexpr uint kZeroPointOffset = 160;

// Physical k inside a span -> (fragment j, row k').
inline uint2 klogical(uint k) {
  const uint q = k >> 5, h = (k >> 4) & 1, a = (k >> 2) & 3, b = (k >> 1) & 1, e = k & 1;
  return uint2(4 * q + 2 * h + b, 2 * a + e);
}

static_assert(GGUF_TABLE16_SPAN_VALUES == q4sg::kXtPerGroup, "a span is one q4sg table group");

// One simdgroup writes one span of one row; the lane holds elements
// 2 lane, 2 lane + 1.
inline void write_input(device bfloat *table, device float *sums, uint width, uint span,
                        uint row, uint lane, bfloat a, bfloat b) {
  const uint2 l = klogical(2 * lane);
  table[span * GGUF_TABLE16_SPAN_VALUES + q4sg::xt_offset(l.x, l.y, row)] = a;
  table[span * GGUF_TABLE16_SPAN_VALUES + q4sg::xt_offset(l.x, l.y + 1, row)] = b;
  float s = float(a) + float(b);
  s += simd_shuffle_xor(s, 1u);
  s += simd_shuffle_xor(s, 2u);
  s += simd_shuffle_xor(s, 4u);
  if ((lane & 7) == 0) sums[(span * (GGUF_TABLE16_SPAN_SEEDS / q4sg::kRows) + lane / 8) * q4sg::kRows + row] = -float(kZeroPointOffset) * s;
  const float s32 = s + simd_shuffle_xor(s, 8u);
  if ((lane & 15) == 0) sums[table16_sums32_offset(width) + (span * (GGUF_TABLE16_SPAN_SUMS / q4sg::kRows) + lane / 16) * q4sg::kRows + row] = s32;
}

struct Table16 {
  static ulong sums_per_tile(uint width) { return table16_sums_per_tile(width); }
  static void write(device bfloat *table, device float *sums, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    write_input(table, sums, width, span, row, lane, a, b);
  }
};

// The mxfp4p A-operand (LinearInput::Packed): a producer drops in Packed for
// Table wherever the caller's lane holds elements 2 * lane, 2 * lane + 1 of a
// 64-column span — lanes 0-15 cover one of the span's two 32-element groups,
// lanes 16-31 the other, so an xor-shuffle reduce over four bits is each
// group's max. `packed` and `exponents` take the table and sums slots,
// indexed by absolute row as the caller's pointer arithmetic provides (a
// tile's first row plus `row`): packed is [rows][width] halves, slot s of
// group g at packed[r * width + g * 32 + s] holding element
// 16 * ((s >> 2) & 1) + 4 * (s >> 3) + (s & 3) of the group scaled by 2^-e,
// exactly as gguf_pack_half writes it (kernels/shared/gguf_mxfp4p.metal);
// exponents is [rows][width / 32] bytes of 127 + e.
struct Packed {
  static ulong sums_per_tile(uint width) { return width / 4; }  // 8 rows * width / 32 bytes
  static void write(device half *packed, device uchar *exponents, uint width, uint span, uint row,
                    uint lane, bfloat a, bfloat b) {
    const float x = float(a), y = float(b);
    float m = fmax(fabs(x), fabs(y));
    // The 16 lanes of this half-simdgroup hold one whole 32-element group.
    m = fmax(m, simd_shuffle_xor(m, 1u));
    m = fmax(m, simd_shuffle_xor(m, 2u));
    m = fmax(m, simd_shuffle_xor(m, 4u));
    m = fmax(m, simd_shuffle_xor(m, 8u));
    const int e = m > 30720.0f || (m > 0.0f && m < 6.1e-5f)
                      ? clamp(int(floor(log2(m))) - 14, -100, 100)
                      : 0;
    const float scale = as_type<float>(uint(127 - e) << 23);
    const uint group = span * 2 + (lane >> 4);
    device half *dst = packed + ulong(row) * width + group * 32;
    // Element j of the group lands in slot (j & 3) | ((j >> 2 & 3) << 3) |
    // ((j >> 4 & 1) << 2) — the inverse of the image's chunk-slot order.
    const uint j = 2 * (lane & 15);
    dst[(j & 3) | ((j & 16) >> 2) | ((j & 12) << 1)] = half(x * scale);
    const uint j1 = j | 1;
    dst[(j1 & 3) | ((j1 & 16) >> 2) | ((j1 & 12) << 1)] = half(y * scale);
    if ((lane & 15) == 0) exponents[ulong(row) * (width / 32) + group] = uchar(127 + e);
  }
};

} // namespace gguf_sg
