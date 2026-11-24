#pragma once
// The int8-activation tile for the MLX affine weights whose codes fit int8
// (the private AF formats of metal/abi/QuantFormat.h but AF4P), which the
// dense prefill GEMMs (kernels/prefill/gguf_prefill_int8.metal) and the MoE
// experts' prefill passes (kernels/shared/moe_int8.metal) run: see
// gguf_prefill_int8.metal for its arithmetic and fragment schedule.
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_tile.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;

// What a tile reads and writes: K inputs a row, `rows` rows of codes from
// the codes' start (the scales' row stride storage_rows), destination rows
// of out_stride columns from out_offset on, and planes of
// plane_input_size inputs a row.
struct PrefillInt8Shape {
  uint input_size, rows, storage_rows, out_stride, out_offset, plane_input_size;
  // The routed expert's first row in the planes' tile-major image
  // (moe_gguf_segment's row0); 0 for whole-tensor segments. Weight column
  // addressing alone — output columns stay on col_base.
  uint weight_offset = 0;
};

constant constexpr int kInt8FragM = 16, kInt8FragN = 32, kInt8FragK = 16,
                       kInt8RowFrags = 2, kInt8Elements = 16;

// This lane's (column, row) in matmul2d's int8 16x16 fragment layout, the
// operands' and the destination's.
inline short2 prefill_int8_coord(ushort lane) {
  const short qid = lane >> 2;
  return short2(((qid & 2) | (lane & 1)) * 4, (qid & 4) | ((lane >> 1) & 3));
}

// The four codes of chunk `c`'s slots {0, 4, 1, 5} (step half 0) or {2, 6, 3, 7}
// (step half 1) of a bitstream group record, `qc` its BITS bytes assembled
// little-endian: slot u at bits [u * BITS, (u + 1) * BITS). af8's codes enter
// as code - 128, its bias correction folded into the epilogue instead.
template <int BITS>
inline char4 prefill_int8_oq_quad(ulong qc, int step_half) {
  constexpr uint m = (1u << BITS) - 1;
  constexpr int bias = BITS == 8 ? 128 : 0;
  const int s = 2 * step_half;
  return char4(int((qc >> ((0 + s) * BITS)) & m) - bias, int((qc >> ((4 + s) * BITS)) & m) - bias,
               int((qc >> ((1 + s) * BITS)) & m) - bias, int((qc >> ((5 + s) * BITS)) & m) - bias);
}

// A tile of GGUF_PREFILL_INT8_SIMDGROUPS simdgroups, each 32 rows (two 16-row fragments) by 32 columns, at (column
// tile, row tile) `tile`. REC is the format's plane0 bytes per (row, group of 32) and META_GROUPS its meta unit's
// groups of 32 (kQuantFormats); PAIRS marks af4's pair-interleaved code words.
template <int BITS, int REC, int META_GROUPS, bool PAIRS, GgufEpilogue Ep>
inline void prefill_int8_tile(device const char *codes, device const float2 *scales, device const uchar *w0,
                              device const uchar *meta, device bfloat *output, device const bfloat *aux,
                              PrefillInt8Shape p, uint2 tile, ushort lane, ushort simd_group) {
  using namespace mpp::tensor_ops;
  constexpr int BM = kInt8RowFrags * kInt8FragM, BN = GGUF_TILE_COLUMNS;
  static_assert(BN == kInt8FragN * GGUF_PREFILL_INT8_SIMDGROUPS, "the simdgroups cover the column tile");
  static_assert(BM == GGUF_PREFILL_INT8_ROWS, "a simdgroup holds the tile's rows");
  constexpr uint T = QUANT_TILE_ROWS;
  const int K = int(p.input_size), M = int(p.rows), groups = K / GGUF_PREFILL_INT8_GROUP;
  const uint plane_groups32 = p.plane_input_size / 32, plane_units = p.plane_input_size / (META_GROUPS * 32);
  const int row_base = int(tile.y) * BM, col_base = int(tile.x) * BN + simd_group * kInt8FragN;
  const short2 coord = prefill_int8_coord(lane);
  const int cx = coord.x >> 2;

  constexpr auto desc = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = matmul2d_descriptor(kInt8FragM, kInt8FragN, kInt8FragK, false, true, false,
                                                matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroup> op;
  matmul2d<desc_set, execution_simdgroup> op_set;
  auto ct_a = op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b = op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), int32_t>();

  float cf[kInt8RowFrags][kInt8Elements];
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) cf[i][e] = 0.0f;

  // The lane's weight column n_lane + 8 (q & 1) + 16 (q >> 1) for right-operand run q, all in one plane tile (a
  // 64-column tile never straddles one): plane0 records of REC bytes. The lane reads chunk bytes (x & 1) * 8 .. +7
  // of 32-group 2g + (x >> 1).
  const uint n_lane = uint(col_base + coord.y);
  const ulong record0 = ulong((n_lane + p.weight_offset) / T) * plane_groups32 * T +
                      (n_lane + p.weight_offset) % T;
  const int h32 = cx >> 1;
  // Meta (s, z as bf16 in one uint) of the lane's weight columns n_run + weight_offset + 16h + j:
  // 4 consecutive records.
  const uint n_run = uint(col_base + coord.x);
  const uint n_meta = n_run + p.weight_offset;
  device const uint *ml = (device const uint *)(meta) + ulong(n_meta / T) * plane_units * T + n_meta % T;
  const int m_base = row_base + coord.y;

  for (int g = 0; g < groups; ++g) {
    const ulong grp = ulong(2 * g + h32) * T;
    uint2 wv[4];
    device const uchar *wr[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int off = (q & 1) * 8 + (q >> 1) * 16;
      device const uchar *rec = w0 + (record0 + grp + off) * REC;
      if constexpr (PAIRS)
        wv[q] = *((device const uint2 *)(rec + 8 * (cx & 1)));
      else
        wr[q] = rec;
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int chunk = 2 * (cx & 1) + (t >> 1);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        char4 quad;
        if constexpr (PAIRS) {
          const uint bits = (wv[q][t >> 1] >> (4 * (t & 1))) & 0x0f0f0f0fu;
          quad = as_type<char4>(bits);
        } else {
          ulong qc = 0;
#pragma unroll
          for (int i = 0; i < BITS; ++i) qc |= ulong(wr[q][chunk * BITS + i]) << (8 * i);
          quad = prefill_int8_oq_quad<BITS>(qc, t & 1);
        }
        const int base = (q >> 1) * 8 + (q & 1) * 4;
        ct_b[base + 0] = quad.x; ct_b[base + 1] = quad.y; ct_b[base + 2] = quad.z; ct_b[base + 3] = quad.w;
      }
#pragma unroll
      for (int hf = 0; hf < kInt8RowFrags; ++hf) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + hf * kInt8FragM + r * 8, M - 1);
          const ulong at = ulong(m) * K + g * GGUF_PREFILL_INT8_GROUP + cx * 16 + t * 4;
          const char4 aq = as_type<char4>(*((device const uint *)(codes + at)));
          ct_a[r * 4 + 0] = aq.x; ct_a[r * 4 + 1] = aq.y; ct_a[r * 4 + 2] = aq.z; ct_a[r * 4 + 3] = aq.w;
        }
        if (t == 0) { if (hf == 0) op_set.run(ct_a, ct_b, acc0); else op_set.run(ct_a, ct_b, acc1); }
        else { if (hf == 0) op.run(ct_a, ct_b, acc0); else op.run(ct_a, ct_b, acc1); }
      }
    }
    // The group's correction: s, z of the lane's two runs of four columns, and the scales of its four rows. A
    // meta unit covers META_GROUPS groups of 32 inputs, so the 64-input group g reads unit g / (META_GROUPS / 2).
    const ulong unit = ulong(g) / (META_GROUPS / 2);
    const uint4 mv0 = *((device const uint4 *)(ml + unit * T));
    const uint4 mv1 = *((device const uint4 *)(ml + unit * T + 16));
    float2 sa[kInt8RowFrags][2];
#pragma unroll
    for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
      for (int r = 0; r < 2; ++r)
        sa[i][r] = scales[ulong(g) * p.storage_rows + min(m_base + i * kInt8FragM + r * 8, M - 1)];
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) {
      const int r = (e & 7) >> 2;
      const uint mt = (e >> 3) ? mv1[e & 3] : mv0[e & 3];
      const float s = float(as_type<bfloat>(ushort(mt & 0xFFFF))), z = float(as_type<bfloat>(ushort(mt >> 16)));
      // af8's matmul codes are code - 128: the products lose 128 s per code,
      // restored by the bias z + 128 s.
      const float zb = z + (BITS == 8 ? 128.0f * s : 0.0f);
      cf[0][e] = fma(sa[0][r].x * s, float(acc0[e]), fma(zb, sa[0][r].y, cf[0][e]));
      cf[1][e] = fma(sa[1][r].x * s, float(acc1[e]), fma(zb, sa[1][r].y, cf[1][e]));
    }
  }
#pragma unroll
  for (int i = 0; i < kInt8RowFrags; ++i)
#pragma unroll
    for (int e = 0; e < kInt8Elements; ++e) {
      const int m = m_base + i * kInt8FragM + ((e & 7) >> 2) * 8;
      if (m >= M) continue;
      const ulong o = ulong(m) * p.out_stride + p.out_offset + n_run + (e & 3) + 16 * (e >> 3);
      output[o] = gguf_epilogue<Ep>(cf[i][e], aux, o);
    }
}
