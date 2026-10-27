// GGUF MoE experts on the staged decode tile (kernels/common/gguf_staged_tile.h); the Apple9 register form is in
// kernels/decode/linear_gguf_sgmatrix.metal.
#pragma clang fp reassociate(off)
#include "metal/kernels/common/gguf_column_tile.h"
#include "metal/kernels/common/gguf_staged_tile.h"
#include "metal/kernels/common/gguf_mxfp4_tile.h"
#include "metal/kernels/common/gguf_mxfp4p_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

// MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of grouped tile y with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
// at run time: on a 16-core M5 Pro one run-time-format dispatch over two segments is within -11..+8% of a dispatch
// per format (time-sg at 23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). aux is the gate of the up pass.
// Two simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles
// (decode steps, short prefill chunks) or 32-row tiles (longer chunks, ops::moeGgufPrefillTile), where a tile runs the
// 8-, 16- or 32-row matmul that holds its live rows (moe_live_rows): an expert's last tile is mostly partial. On the
// 35B's real prefill routes (wikitext, chat, code; the three passes of a layer on a 16-core M5 Pro) 32-row tiles take
// 2.42-2.78 ms at 512 rows and 6.70-6.83 ms at 2048 rows against 3.44-3.86 and 8.20-8.23 for 64-row tiles sharing one
// 64-column stage over four 16-row simdgroups, and 1.29-1.71 ms against 1.59-2.60 for 8-row tiles at 128-256 rows,
// all with a 16-row matmul for up to 16 live rows; the 8-row matmul for up to 8 takes another 4-6% off 64- to
// 256-row chunks of gguf-moe-benchmark's uniform routes (40-core M5 Max). An expert's only tile of at most
// MOE_GGUF_COLUMN_ROWS live rows runs the column tile instead (kernels/common/gguf_column_tile.h, moe_column_tile),
// each lane one of the simdgroup's columns over the live rows: routes spread over many experts leave most decode
// tiles at one or two rows, where the staged tile's padded matmul steps bound it.
template <ushort Rows, GgufEpilogue Ep, bool NativeMxfp4>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                 device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device bfloat *output, device bfloat *aux,
                                 constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  if (moe_column_tile(tiles, group.y)) {
    float sums[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(s.format, x, tile.rows, p.input_size, s.w0, s.w1, s.meta, origin + simd_lane,
                                         0, p.input_size / 32, tl, simd_group * 32 + simd_lane, GGUF_STAGED_THREADS,
                                         sums);
    for (uint row = 0; row < tile.rows; ++row) {
      const ulong o = out + ulong(row) * p.output_size + origin + simd_lane;
      output[o] = gguf_epilogue<Ep>(sums[row], aux, o);
    }
    return;
  }
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
    // MXFP4 experts take the multiplane tile (kernels/common/
    // gguf_mxfp4_tile.h): 8 rows live, like the fused kernels; the tile
    // never splits K, so its partials, counter and arrival go unused.
    if constexpr (NativeMxfp4) {
      if (R <= 8 && s.format == GGUF_FMT_MXFP4) {
        gguf_decode_mxfp4_tile<R, Ep>(x, s.w0, s.meta, output + out, (device coherent(device) float *)nullptr,
                                      nullptr, aux + out, p.input_size, 1, p.output_size, origin, origin, 0,
                                      simd_lane, simd_group, stage, (threadgroup uchar *)tl, nullptr);
        return;
      }
    }
#endif
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any_native<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, NativeMxfp4>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl,
                                            simd_group * 32 + simd_lane, simd_lane, 0, p.input_size / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      output[o] = gguf_epilogue<Ep>(v, aux, o);
    });
  };
  moe_live_rows<Rows>(tile.rows, run);
}
template <ushort Rows, GgufEpilogue Ep, bool NativeMxfp4>
kernel void moe_expert_gguf(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                            device const uint *tile_count [[buffer(2)]], device uchar *w0 [[buffer(3)]],
                            device uchar *w1 [[buffer(4)]], device uchar *meta [[buffer(5)]], device uchar *sw0 [[buffer(6)]],
                            device uchar *sw1 [[buffer(7)]], device uchar *smeta [[buffer(8)]],
                            device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],
                            constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]],
                            uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_expert_tile<Rows, Ep, NativeMxfp4>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, p, group,
                                 simd_lane, simd_group, stage, tl);
}
using MoeExpertGgufKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *, device uchar *,
                                 device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                 device bfloat *, device bfloat *, constant MoeGgufExpertParams &, uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpNone, false>;
template [[host_name("moe_expert_gguf_m8_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpUpWithGate, false>;
template [[host_name("moe_expert_gguf_m32_a")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpNone, false>;
template [[host_name("moe_expert_gguf_m32_g")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpUpWithGate, false>;
// The `_n` experts an Apple GPU family 10 host dispatches, whose MXFP4
// segments decode on the native FP4 path.
template [[host_name("moe_expert_gguf_m8_a_n")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpNone, true>;
template [[host_name("moe_expert_gguf_m8_g_n")]] kernel MoeExpertGgufKernel moe_expert_gguf<8, EpUpWithGate, true>;
template [[host_name("moe_expert_gguf_m32_a_n")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpNone, true>;
template [[host_name("moe_expert_gguf_m32_g_n")]] kernel MoeExpertGgufKernel moe_expert_gguf<32, EpUpWithGate, true>;

// The fused gate+up staged tile (ops::MoE's addGgufExperts, unpaced
// non-register plans): each simdgroup accumulates the tile's gate columns
// and up columns back to back on the same staged weights loop, and stores
// silu(gate) * up straight to the intermediate — the split passes'
// EpUpWithGate association (bf16-rounded up times silu of the bf16-rounded
// gate) without the expertOutput round trip or a second dispatch. The
// threadgroup barrier between the two accumulates keeps a format change's
// pair-table refill off the other simdgroup's in-flight dequant reads.
template <ushort Rows>
inline void moe_gguf_fused_tile(device bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                device uchar *gw0, device uchar *gw1, device uchar *gmeta, device uchar *gsw0,
                                device uchar *gsw1, device uchar *gsmeta, device uchar *uw0, device uchar *uw1,
                                device uchar *umeta, device uchar *usw0, device uchar *usw1, device uchar *usmeta,
                                device bfloat *output, constant MoeGgufFusedParams &p, uint2 group, uint simd_lane,
                                uint simd_group, threadgroup half *stage, threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment gate = moe_gguf_segment_fmt(tile.expert, p.experts, p.input_size, p.output_size,
                                                 p.gate_routed_format, p.gate_shared_format, gw0, gw1, gmeta,
                                                 gsw0, gsw1, gsmeta);
  const MoeGgufSegment up = moe_gguf_segment_fmt(tile.expert, p.experts, p.input_size, p.output_size,
                                               p.up_routed_format, p.up_shared_format, uw0, uw1, umeta,
                                               usw0, usw1, usmeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  if (moe_column_tile(tiles, group.y)) {
    float gs[MOE_GGUF_COLUMN_ROWS], us[MOE_GGUF_COLUMN_ROWS];
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(gate.format, x, tile.rows, p.input_size, gate.w0, gate.w1, gate.meta,
                                         origin + simd_lane, 0, p.input_size / 32, tl,
                                         simd_group * 32 + simd_lane, GGUF_STAGED_THREADS, gs);
    column_dot_any<MOE_GGUF_COLUMN_ROWS>(up.format, x, tile.rows, p.input_size, up.w0, up.w1, up.meta,
                                         origin + simd_lane, 0, p.input_size / 32, tl,
                                         simd_group * 32 + simd_lane, GGUF_STAGED_THREADS, us);
    for (uint row = 0; row < tile.rows; ++row) {
      const ulong o = out + ulong(row) * p.output_size + origin + simd_lane;
      output[o] = bfloat(float(bfloat(us[row])) *
                         richengine_silu(float(bfloat(gs[row]))));
    }
    return;
  }
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc_g = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    auto acc_u = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc_g);
    gguf_zero(acc_u);
    staged_accumulate_any_native<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, false>(gate.format, x, gate.w0, gate.w1, gate.meta,
                                            p.input_size, origin, my, tl, simd_group * 32 + simd_lane, simd_lane, 0,
                                            p.input_size / GGUF_STAGED_STEP, 0, acc_g);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    staged_accumulate_any_native<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, false>(up.format, x, up.w0, up.w1, up.meta,
                                            p.input_size, origin, my, tl, simd_group * 32 + simd_lane, simd_lane, 0,
                                            p.input_size / GGUF_STAGED_STEP, 0, acc_u);
#pragma unroll
    for (ushort i = 0; i < acc_g.get_capacity(); ++i) {
      if (!acc_g.is_valid_element(i)) continue;
      const auto index = acc_g.get_multidimensional_index(i);
      const ulong o = out + ulong(index[1]) * p.output_size + origin + uint(index[0]);
      output[o] = bfloat(float(bfloat(float(acc_u[i]))) *
                         richengine_silu(float(bfloat(float(acc_g[i])))));
    }
  };
  moe_live_rows<Rows>(tile.rows, run);
}

template <ushort Rows>
kernel void moe_expert_gguf_fused(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],
                                  device const uint *tile_count [[buffer(2)]], device uchar *gw0 [[buffer(3)]],
                                  device uchar *gw1 [[buffer(4)]], device uchar *gmeta [[buffer(5)]],
                                  device uchar *gsw0 [[buffer(6)]], device uchar *gsw1 [[buffer(7)]],
                                  device uchar *gsmeta [[buffer(8)]], device uchar *uw0 [[buffer(9)]],
                                  device uchar *uw1 [[buffer(10)]], device uchar *umeta [[buffer(11)]],
                                  device uchar *usw0 [[buffer(12)]], device uchar *usw1 [[buffer(13)]],
                                  device uchar *usmeta [[buffer(14)]], device bfloat *output [[buffer(15)]],
                                  constant MoeGgufFusedParams &p [[buffer(16)]],
                                  uint2 group [[threadgroup_position_in_grid]],
                                  uint simd_lane [[thread_index_in_simdgroup]],
                                  uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  moe_gguf_fused_tile<Rows>(input, tiles, tile_count, gw0, gw1, gmeta, gsw0, gsw1, gsmeta, uw0, uw1, umeta, usw0,
                            usw1, usmeta, output, p, group, simd_lane, simd_group, stage, tl);
}
using MoeExpertGgufFusedKernel = void(device bfloat *, device const MoeTileDescriptor *, device const uint *,
                                      device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                      device uchar *, device uchar *, device uchar *, device uchar *, device uchar *,
                                      device uchar *, device uchar *, device bfloat *, constant MoeGgufFusedParams &,
                                      uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_gu")]] kernel MoeExpertGgufFusedKernel moe_expert_gguf_fused<8>;
template [[host_name("moe_expert_gguf_m32_gu")]] kernel MoeExpertGgufFusedKernel moe_expert_gguf_fused<32>;

// The pre-packed decode variant (ops::MoE's packed decode, Apple GPU family
// 10): the gather's slot-permuted fp16 plane and per-(row, group) exponent
// bytes (moe_gather_packed, kernels/shared/moe.metal) let an MXFP4 segment's
// tile read A straight from device memory — gguf_decode_mxfp4p_tile, no
// threadgroup stage, no barriers — while segments in other formats take the
// staged tile of moe_gguf_expert_tile on the bf16 rows the same gather
// still writes. The pass picks per segment, so a model mixing MXFP4
// experts with, say, a Q8_0 shared expert gets the packed path only where
// it applies. The same kernels serve the down pass, whose packed input a
// gguf_pack_half dispatch writes from the bf16 intermediates.
template <ushort Rows, GgufEpilogue Ep>
inline void moe_gguf_packed_tile(device bfloat *input, device half *packed, device uchar *exponents,
                                 device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                 device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0,
                                 device uchar *sw1, device uchar *smeta, device bfloat *output, device bfloat *aux,
                                 device float *partials, device atomic_uint *counters,
                                 constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint tile_index = group.y / p.splits, split = group.y % p.splits;
  if (tile_index >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[tile_index];
  const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(tile_index) * Rows * p.input_size;
  device half *xp = packed + ulong(tile_index) * Rows * p.input_size;
  device uchar *xe = exponents + ulong(tile_index) * Rows * (p.input_size / 32);
  const ulong out = ulong(tile_index) * Rows * p.output_size;
  const uint origin = group.x * GGUF_TILE_COLUMNS + simd_group * GGUF_STAGED_COLUMNS;
  threadgroup half *my = stage + simd_group * kStagedSimdgroupStage;
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
    // The MXFP4 segments the host packed for: the packed plane's rows are
    // the tile's grouped rows and its exponent bytes the same (row, group)
    // index, so the tile is the dense kernels' with per-tile offsets. The
    // K split reuses the dense store: partials stride out_stride per tile,
    // one counter per 64-column segment of it.
    if (s.format == GGUF_FMT_MXFP4) {
      gguf_decode_mxfp4p_tile<R, Ep>(xp, xe, s.w0, s.meta, output + out,
                                     (device coherent(device) float *)(partials + ulong(tile_index) * p.splits * Rows * p.output_size),
                                     counters + tile_index * (p.output_size / GGUF_TILE_COLUMNS) + group.x,
                                     aux + out, p.input_size,
                                     p.splits, p.output_size, origin, origin, split, simd_lane, simd_group, arrival);
      return;
    }
#endif
    // A segment the packed tile does not serve runs its whole K on one
    // partition; the rest exit early.
    if (split) return;
    auto acc = staged_accumulator<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP>(x, p.input_size, my);
    gguf_zero(acc);
    staged_accumulate_any_native<R, GGUF_STAGED_COLUMNS, GGUF_STAGED_STEP, true>(s.format, x, s.w0, s.w1, s.meta, p.input_size,
                                            origin, my, tl, simd_group * 32 + simd_lane, simd_lane, 0,
                                            p.input_size / GGUF_STAGED_STEP, 0, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;
      output[o] = gguf_epilogue<Ep>(v, aux, o);
    });
  };
  moe_live_rows<Rows>(tile.rows, run);
}

template <ushort Rows, GgufEpilogue Ep>
kernel void moe_expert_gguf_packed(device bfloat *input [[buffer(0)]], device half *packed [[buffer(1)]],
                                   device uchar *exponents [[buffer(2)]],
                                   device const MoeTileDescriptor *tiles [[buffer(3)]],
                                   device const uint *tile_count [[buffer(4)]], device uchar *w0 [[buffer(5)]],
                                   device uchar *w1 [[buffer(6)]], device uchar *meta [[buffer(7)]],
                                   device uchar *sw0 [[buffer(8)]], device uchar *sw1 [[buffer(9)]],
                                   device uchar *smeta [[buffer(10)]], device bfloat *output [[buffer(11)]],
                                   device bfloat *aux [[buffer(12)]],
                                   device float *partials [[buffer(13)]],
                                   device atomic_uint *counters [[buffer(14)]],
                                   constant MoeGgufExpertParams &p [[buffer(15)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd_lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup half stage[kStagedStages]; threadgroup half2 tl[kQuantPairTableEntries];
  threadgroup uint arrival;
  moe_gguf_packed_tile<Rows, Ep>(input, packed, exponents, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta,
                                 output, aux, partials, counters, p, group, simd_lane, simd_group, stage, tl,
                                 &arrival);
}
using MoeExpertGgufPackedKernel = void(device bfloat *, device half *, device uchar *,
                                       device const MoeTileDescriptor *, device const uint *, device uchar *,
                                       device uchar *, device uchar *, device uchar *, device uchar *,
                                       device uchar *, device bfloat *, device bfloat *,
                                       device float *, device atomic_uint *,
                                       constant MoeGgufExpertParams &, uint2, uint, uint);
template [[host_name("moe_expert_gguf_m8_a_p")]] kernel MoeExpertGgufPackedKernel moe_expert_gguf_packed<8, EpNone>;
template [[host_name("moe_expert_gguf_m8_g_p")]] kernel MoeExpertGgufPackedKernel moe_expert_gguf_packed<8, EpUpWithGate>;
// The prefill chunks' 32-row tiles on the same packed path.
template [[host_name("moe_expert_gguf_m32_a_p")]] kernel MoeExpertGgufPackedKernel moe_expert_gguf_packed<32, EpNone>;
template [[host_name("moe_expert_gguf_m32_g_p")]] kernel MoeExpertGgufPackedKernel moe_expert_gguf_packed<32, EpUpWithGate>;
