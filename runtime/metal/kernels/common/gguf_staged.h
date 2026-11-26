#pragma once
#include "metal/kernels/common/quant_formats.h"

// The GGUF tiles' dequantization (kernels/shared/gguf_linear.metal and kernels/shared/moe_gguf.metal, through
// kernels/common/gguf_staged_tile.h and kernels/common/gguf_column_tile.h), shared with the dequantization test.
// Includers set `#pragma clang fp reassociate(off)` first, so the source order of float operations holds here.
// One thread decodes one column's group of 32 (kernels/common/quant_formats.h) to half, each value rounded once, and
// hands it to sink(c, lo, hi) by chunk: lo holds chunk c's pairs 0, 1 (elements 4c..4c+3), hi its pairs 2, 3
// (elements 16+4c..16+4c+3). The staged tiles store the group into their stage (dequant32), the column tile keeps it
// in registers.
template <class F>
inline half2 staged_linear(uint pair, float s, float m) {
  // 0x6400 is half 1024, whose ulp is 1: or-ing a code into its mantissa makes 1024 + code, exactly.
  const float2 code = float2(as_type<half2>(pair | 0x64006400u) - half2(half(1024 + F::Zero)));
  if constexpr (F::Zero) return half2(code * s);
  else return half2(fma(code, float2(s), float2(m)));
}
// IQ2_XXS group decode (dequant32_chunks' sink interface), bespoke: the grid magnitudes of kIQ2XXSGrid only ever
// hold bytes 8, 25 and 43, so the group needs its three signed-magnitude products once — half(fl(s * m)) is the
// element's own rounding, s * m commutative — and each element selects among them instead of converting and
// multiplying. The generic chunk path decodes every 7-bit sign index twice (chunks 2c, 2c + 1 halve the same
// index) and re-extracts scale bits per chunk; here the four indices and the four grid entries decode once.
template <class Sink>
inline void dequant32_iq2xxs(uint2 a, ushort meta, Sink sink) {
  const QuantCoef k = FmtIQ2XXS::coef(meta, FmtIQ2XXS::chunk({a}, 0));
  const float s = k.s.x;
  const half h8 = half(s * 8.0f), h25 = half(s * 25.0f), h43 = half(s * 43.0f);
  const uint64_t g0 = kIQ2XXSGrid[a.x & 0xFF], g1 = kIQ2XXSGrid[(a.x >> 8) & 0xFF],
                 g2 = kIQ2XXSGrid[(a.x >> 16) & 0xFF], g3 = kIQ2XXSGrid[a.x >> 24];
  const uint n0 = quant_signs7(a.y & 127), n1 = quant_signs7((a.y >> 7) & 127),
             n2 = quant_signs7((a.y >> 14) & 127), n3 = quant_signs7((a.y >> 21) & 127);
  // Chunk c: elements 4c..4c+3 of half c & 1 of grid entry c >> 1 (lo) and of entry 2 + (c >> 1) (hi), their sign
  // bits at the same positions of the entries' sign bytes. 25 and 43 are the magnitudes with bit 0 set, 43 the one
  // with bit 1.
  const auto emit = [&](ushort c, uint64_t ga, uint64_t gb, uint na, uint nb) __attribute__((always_inline)) {
    const uint h = 32u * (c & 1);
    const uchar4 ma = as_type<uchar4>(uint(ga >> h)), mb = as_type<uchar4>(uint(gb >> h));
    half4 lo = select(select(half4(h8), half4(h25), bool4(ma & uchar4(1))), half4(h43), bool4(ma & uchar4(2)));
    half4 hi = select(select(half4(h8), half4(h25), bool4(mb & uchar4(1))), half4(h43), bool4(mb & uchar4(2)));
    const uint sa = (na >> (4 * (c & 1))) & 0xF, sb = (nb >> (4 * (c & 1))) & 0xF;
    lo = select(lo, -lo, bool4(sa & 1, sa & 2, sa & 4, sa & 8));
    hi = select(hi, -hi, bool4(sb & 1, sb & 2, sb & 4, sb & 8));
    sink(c, lo, hi);
  };
  emit(0, g0, g2, n0, n2);
  emit(1, g0, g2, n0, n2);
  emit(2, g1, g3, n1, n3);
  emit(3, g1, g3, n1, n3);
}
template <class F, class Sink>
inline void dequant32_chunks(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl, Sink sink) {
  if constexpr (F::Id == GGUF_FMT_IQ2XXS) {
    dequant32_iq2xxs(w.a, meta, sink);
    return;
  }
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
    sink(c, lo, hi);
  }
}
// The group into a stage column: chunk c's pairs 0, 1 at dst + 4c, pairs 2, 3 at dst + 16 + 4c.
template <class F>
inline void dequant32(typename F::Payload w, typename F::Meta meta, ushort j, threadgroup half2 *tl,
                      threadgroup half *dst) {
  dequant32_chunks<F>(w, meta, j, tl, [&](ushort c, half4 lo, half4 hi) __attribute__((always_inline)) {
    *((threadgroup half4 *)(dst + 4 * c)) = lo; *((threadgroup half4 *)(dst + 16 + 4 * c)) = hi;
  });
}
