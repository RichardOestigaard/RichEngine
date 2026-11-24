// The MoE experts' prefill passes on int8 activations (ops/MoE.cpp's
// addInt8Experts): the dense prefill's int8 tile
// (kernels/common/gguf_prefill_int8_tile.h) over the grouping's tiles
// (kernels/shared/moe.metal), each tile's expert reached as the staged
// passes reach it (moe_gguf_segment_fmt), in the AF formats the int8 tile
// takes (gguf_prefill_int8_format). Tile t of the tile-strided grouping
// covers grouped rows [t * tile_rows, t * tile_rows + rows); the int8 tile
// reads a tile's live rows alone.
#include "metal/abi/MoE.h"
#include "metal/kernels/common/gguf_prefill_int8_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"

#include <metal_stdlib>
using namespace metal;

// Copies each grouped row's codes and scales from its chunk row's, so every
// expert tile is a dense matrix of its live rows. Threadgroup (tile, part)
// copies the part's 256 codes of the tile's rows, sixteen rows at once, each
// thread sixteen codes of one row, and its four 64-input groups' scales.
kernel void moe_gather_int8(device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],
                            device const uint *grouped_routes [[buffer(2)]],
                            device const MoeTileDescriptor *tiles [[buffer(3)]],
                            device const uint *tile_count [[buffer(4)]], device char *grouped_codes [[buffer(5)]],
                            device float2 *grouped_scales [[buffer(6)]],
                            constant MoeGatherInt8Params &p [[buffer(7)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint thread_index [[thread_index_in_threadgroup]]) {
  if (group.x >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.x];
  const uint lane = thread_index % 16, column = group.y * 256 + lane * 16;
  const uint first_row = group.x * p.tile_rows;
  for (uint row = first_row + thread_index / 16; row < first_row + tile.rows; row += 16) {
    const uint source = grouped_routes[row] / p.routes_per_row;
    *reinterpret_cast<device uint4 *>(grouped_codes + ulong(row) * p.input_size + column) =
        *reinterpret_cast<device const uint4 *>(codes + ulong(source) * p.input_size + column);
    if (lane < 4) {
      const uint g = group.y * 4 + lane;
      grouped_scales[ulong(g) * p.routes + row] = scales[ulong(g) * p.rows + source];
    }
  }
}

// An expert pass of the grouping's tiles: threadgroup (x, y) computes column tile x of tile y with its expert's
// weights, the tile's live rows (at most tile_rows <= GGUF_PREFILL_INT8_ROWS) as the int8 tile's one row tile. The
// up pass (EpUpWithGate) multiplies by the gate the gate pass wrote to `output`, each element in place: the thread
// that writes an element has read its gate there.
template <int BITS, int REC, int META_GROUPS, bool PAIRS, GgufEpilogue Ep, uint Projection>
kernel void moe_expert_int8(device const char *codes [[buffer(0)]], device const float2 *scales [[buffer(1)]],
                            device const MoeTileDescriptor *tiles [[buffer(2)]],
                            device const uint *tile_count [[buffer(3)]], device uchar *w0 [[buffer(4)]],
                            device uchar *w1 [[buffer(5)]], device uchar *meta [[buffer(6)]],
                            device uchar *sw0 [[buffer(7)]], device uchar *sw1 [[buffer(8)]],
                            device uchar *smeta [[buffer(9)]], device bfloat *output [[buffer(10)]],
                            constant MoeInt8ExpertParams &p [[buffer(11)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            ushort lane [[thread_index_in_simdgroup]],
                            ushort simd_group [[simdgroup_index_in_threadgroup]]) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s =
      moe_gguf_segment_fmt(tile.expert, p.experts, p.input_size, p.output_size, p.routed_format, p.shared_format,
                           w0, w1, meta, sw0, sw1, smeta);
  const uint first_row = group.y * p.tile_rows;
  const ulong out = ulong(first_row) * p.output_size;
  prefill_int8_tile<BITS, REC, META_GROUPS, PAIRS, Ep>(
      codes + ulong(first_row) * p.input_size, scales + first_row, s.w0, s.meta, output + out, output + out,
      {p.input_size, tile.rows, p.routes, p.output_size, 0, p.input_size, s.row0}, uint2(group.x, 0), lane, simd_group);
}

using MoeExpertInt8Kernel = void(device const char *, device const float2 *, device const MoeTileDescriptor *,
                                 device const uint *, device uchar *, device uchar *, device uchar *,
                                 device uchar *, device uchar *, device uchar *, device bfloat *,
                                 constant MoeInt8ExpertParams &, uint2, ushort, ushort);
// moe_expert_int8_<format>_<pass>: the gate (plain), up (times the gate's
// SiLU) and down passes. The REC/META_GROUPS/PAIRS triple is the format's
// kQuantFormats row, as gguf_prefill_int8.metal instantiates them.
#define MOE_EXPERT_INT8(B, R, M, P, f)                                                                          \
  template [[host_name("moe_expert_int8_" #f "_gate")]] kernel MoeExpertInt8Kernel                              \
      moe_expert_int8<B, R, M, P, EpNone, 0>;                                                                   \
  template [[host_name("moe_expert_int8_" #f "_up")]] kernel MoeExpertInt8Kernel                                \
      moe_expert_int8<B, R, M, P, EpUpWithGate, 1>;                                                             \
  template [[host_name("moe_expert_int8_" #f "_down")]] kernel MoeExpertInt8Kernel                              \
      moe_expert_int8<B, R, M, P, EpNone, 2>;
MOE_EXPERT_INT8(4, 16, 2, true, af4)
MOE_EXPERT_INT8(4, 16, 4, false, af4g128)
MOE_EXPERT_INT8(5, 20, 2, false, af5)
MOE_EXPERT_INT8(6, 24, 2, false, af6)
MOE_EXPERT_INT8(8, 32, 2, false, af8)
MOE_EXPERT_INT8(5, 20, 4, false, af5g128)
MOE_EXPERT_INT8(6, 24, 4, false, af6g128)
MOE_EXPERT_INT8(8, 32, 4, false, af8g128)
MOE_EXPERT_INT8(2, 8, 2, false, af2)
MOE_EXPERT_INT8(3, 12, 2, false, af3)
MOE_EXPERT_INT8(2, 8, 4, false, af2g128)
MOE_EXPERT_INT8(3, 12, 4, false, af3g128)
#undef MOE_EXPERT_INT8
