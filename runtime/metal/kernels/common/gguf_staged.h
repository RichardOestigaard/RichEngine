#pragma once
#include "metal/kernels/common/quant_formats.h"

// The staged kernels' dequantization (kernels/shared/gguf_linear.metal and
// kernels/shared/moe_gguf.metal, through kernels/common/gguf_staged_tile.h),
// shared with the dequantization test. Includers set `#pragma clang fp
// reassociate(off)` first, so the source order of float operations holds here.
// One thread writes one column's group of 32 (kernels/common/quant_formats.h) as half, each value rounded once;
// chunk c's pairs 0, 1 go to dst + 4c and pairs 2, 3 to dst + 16 + 4c.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, threadgroup half *dst) {
  QuantCoef k;
  if constexpr (F::ScaleInChunk) k = F::coef(meta, F::chunk(w, 0)); else k = F::coef(meta, j);
#pragma unroll
  for (ushort c = 0; c < 4; ++c) {
    const typename F::Chunk q = F::chunk(w, c);
    half4 lo, hi;
    if constexpr (F::Kind == QuantLinear) {
      const uint4 p = F::codes(q);
      lo = half4(staged_linear<F>(p.x, k.s.x, k.m.x), staged_linear<F>(p.y, k.s.x, k.m.x));
      hi = half4(staged_linear<F>(p.z, k.s.y, k.m.y), staged_linear<F>(p.w, k.s.y, k.m.y));
    } else if constexpr (F::Native) {
#if defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
      // The chunk's eight E2M1 nibbles, unpacked in element order; kFP4Values
      // are twice E2M1, so the doubled coefficient keeps one rounding. Index
      // 8 unpacks as -0.0 where the table's 0 is positive, so each product's
      // sign bit is cleared where the value is zero.
      const vec<float, 8> v =
          unpack<float>(packed_metal_fp4_e2m1<8>(as_type<packed_uchar4>(F::indices(q))));
      const float4 a = float4(v[0], v[1], v[2], v[3]) * (2 * k.s.x);
      const float4 b = float4(v[4], v[5], v[6], v[7]) * (2 * k.s.y);
      lo = half4(as_type<float4>(as_type<uint4>(a) & select(uint4(~0u), uint4(0x7FFFFFFFu), a == 0.0f)));
      hi = half4(as_type<float4>(as_type<uint4>(b) & select(uint4(~0u), uint4(0x7FFFFFFFu), b == 0.0f)));
#endif
    } else if constexpr (F::Kind == QuantCodebook) {   // value * s in Scale: one rounding to half either way
      typedef typename F::Scale S;
      const uchar4 b = as_type<uchar4>(F::indices(q));
      lo = half4(half2(vec<S, 2>(tl[b.x]) * S(k.s.x)), half2(vec<S, 2>(tl[b.y]) * S(k.s.x)));
      hi = half4(half2(vec<S, 2>(tl[b.z]) * S(k.s.y)), half2(vec<S, 2>(tl[b.w]) * S(k.s.y)));
    } else if constexpr (F::Kind == QuantInt8) {
      typedef typename F::Scale S;
      const uint2 v = F::values(q);
      lo = half4(vec<S, 4>(as_type<char4>(v.x)) * S(k.s.x));
      hi = half4(vec<S, 4>(as_type<char4>(v.y)) * S(k.s.y));
    } else {
      const uint2 g = F::grid(q); const uint s = F::signs(q);
      lo = half4(float4(as_type<uchar4>(g.x)) * k.s.x);
      hi = half4(float4(as_type<uchar4>(g.y)) * k.s.y);
      lo = select(lo, -lo, bool4(s & 1, s & 2, s & 4, s & 8));
      hi = select(hi, -hi, bool4(s & 16, s & 32, s & 64, s & 128));
    }
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  }
}
