#pragma once

// The oQ (oMLX) affine formats, the private types of metal/abi/QuantFormat.h
// from GGUF_FMT_AF5 on: QuantLinear, value = s * code + bias, meta one bf16
// scale and bias per unit covering MetaGroupsN groups of 32, as AF4's. Their
// plane0 packs a group differently from AF4's pair-interleaved words: the
// slots' codes as one contiguous little-endian bitstream, slot s at bits
// [s * Bits, (s + 1) * Bits) of the group's 4 * Bits bytes, so chunk c is
// bytes [c * Bits, (c + 1) * Bits) — slots 8c..8c + 7 — in the low 8 * Bits
// bits of a uint64. quant_formats.h includes this next to the other Fmt*
// definitions, which it shares the QUANT_FORMAT macro and QuantCoef with.
template <uint Bits, uint MetaGroupsN, uint FmtId> struct FmtOq {
  QUANT_FORMAT(FmtId, QuantLinear, 0, 32, false);
  static_assert(MetaGroupsN == MetaGroups, "MetaGroupsN names the format's kQuantFormats meta_groups");
  struct Payload { uchar bytes[4 * Bits]; };
  typedef uint64_t Chunk;
  typedef uint Meta;
  static Payload load(device uchar *p0, device uchar *) { return *((device Payload *)p0); }
  static Meta loadMeta(device uchar *m) { return *((device uint *)m); }
  static Chunk chunk(Payload w, ushort c) {
    Chunk q = 0;
    for (uint i = 0; i < Bits; ++i) q |= Chunk(w.bytes[c * Bits + i]) << (8 * i);
    return q;
  }
  static Chunk loadChunk(device uchar *p0, device uchar *, ushort c) {
    if constexpr (Bits == 4) return *((device uint *)(p0 + 4 * c));
    if constexpr (Bits == 8) return *((device Chunk *)(p0 + 8 * c));
    Chunk q = 0;
    for (uint i = 0; i < Bits; ++i) q |= Chunk(p0[c * Bits + i]) << (8 * i);
    return q;
  }
  // Pair p's e0 is slot 2p's code at bits 2p * Bits, its e1 slot 2p + 1's at
  // bits (2p + 1) * Bits.
  static uint4 codes(Chunk q) {
    const uint4 m = uint4((1u << Bits) - 1);
    const uint4 e0 = uint4(uint(q), uint(q >> (2 * Bits)), uint(q >> (4 * Bits)), uint(q >> (6 * Bits))) & m;
    const uint4 e1 = uint4(uint(q >> Bits), uint(q >> (3 * Bits)), uint(q >> (5 * Bits)), uint(q >> (7 * Bits))) & m;
    return e0 | (e1 << 16);
  }
  // The unit's bf16 scale at the low half, bias at the high — AF4's coef:
  // every group of the unit shares them, so j is unused.
  static QuantCoef coef(Meta mt, ushort) {
    return {float2(as_type<float>(mt << 16)), float2(as_type<float>(mt & 0xFFFF0000u))};
  }
};

typedef FmtOq<5, 2, GGUF_FMT_AF5> FmtAF5;
typedef FmtOq<6, 2, GGUF_FMT_AF6> FmtAF6;
typedef FmtOq<8, 2, GGUF_FMT_AF8> FmtAF8;
typedef FmtOq<4, 4, GGUF_FMT_AF4G128> FmtAF4G128;
typedef FmtOq<5, 4, GGUF_FMT_AF5G128> FmtAF5G128;
typedef FmtOq<6, 4, GGUF_FMT_AF6G128> FmtAF6G128;
typedef FmtOq<8, 4, GGUF_FMT_AF8G128> FmtAF8G128;
