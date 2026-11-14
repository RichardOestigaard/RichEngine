#pragma once

#include "metal/abi/KernelABI.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"

// A grouped expert tile of tile_rows rows (kernels/shared/moe.metal) holds
// `rows` live rows first, then padding. An expert pass runs the matmul of the
// smallest of 8, 16 and tile_rows rows that holds the live rows, whose padding
// rows the grouping marks empty, but for the staged passes' column tiles
// (moe_column_tile), which run over their live rows alone. The gather fills
// exactly the rows its passes read: no kernel reads past them.
inline uint moe_matmul_rows(uint rows, uint tile_rows) {
  return min(tile_rows, rows <= 8 ? 8u : rows <= 16 ? 16u : tile_rows);
}
// The staged expert passes run an expert's only tile of at most
// MOE_GGUF_COLUMN_ROWS live rows on the column tile (moe_column_tile in
// abi/MoE.h). The two tiles sum a row's products in different orders, so a
// route's output depends on how many rows the step routes to its expert, up to
// MOE_GGUF_COLUMN_ROWS on the column tile and more on the staged tile: decode
// results of GGUF MoE models on the staged tile are not batch-invariant, since
// a row's experts see the other rows of its step (its lane's draft rows and the
// other lanes'). The grouping's atomics order an expert's routes over its
// tiles, so the tiles of an expert run one tile kind, or a route's output would
// also depend on that order (an expert of nine routes in an 8-row tile and a
// 1-row tile).
// The rows of tile t that the staged expert passes read.
inline uint moe_staged_rows(device const MoeTileDescriptor *tiles, uint t, uint tile_rows) {
  return moe_column_tile(tiles, t) ? tiles[t].rows : moe_matmul_rows(tiles[t].rows, tile_rows);
}

// Runs the pass `run` on the matmul rows of a Max-row tile:
// run(integral_constant<ushort, R>{}) for R = moe_matmul_rows(rows, Max).
template <ushort Max, typename Run>
inline void moe_live_rows(uint rows, Run run) {
  const uint matmul = moe_matmul_rows(rows, Max);
  if constexpr (8 < Max) {
    if (matmul == 8)
      return run(integral_constant<ushort, 8>{});
  }
  if constexpr (16 < Max) {
    if (matmul == 16)
      return run(integral_constant<ushort, 16>{});
  }
  run(integral_constant<ushort, Max>{});
}

// One expert's Q4 slab, [weights][BF16 scales][BF16 biases] in the same
// StorageN=256 affine package as the dense kernels. Routed experts sit at
// their stride in the packed buffer; expert `experts` is the shared expert,
// whose single slab lives in its own buffer.
struct MoeQ4Slab {
  device uchar *weights;
  device bfloat *scales;
  device bfloat *biases;
};

inline MoeQ4Slab moe_q4_slab(device uchar *packed, device uchar *shared,
                             uint expert, uint experts,
                             ulong expert_stride_bytes, uint output_size,
                             uint input_size) {
  device uchar *base = expert == experts
                           ? shared
                           : packed + ulong(expert) * expert_stride_bytes;
  ulong elements = ulong(output_size) * input_size;
  ulong weight_bytes = elements / 2;
  ulong parameter_bytes = elements / 32;
  return {base, reinterpret_cast<device bfloat *>(base + weight_bytes),
          reinterpret_cast<device bfloat *>(base + weight_bytes +
                                            parameter_bytes)};
}

// The expert pass's (column block, tile) pair from the dispatch's group:
// column-major grids arrive as (group.x, group.y); tile-major dispatches
// (RICHENGINE_MOE_TILE_MAJOR) swap the axes so threadgroups sweep an
// expert's row-tiles at one column block, and arrive as (group.y, group.x).
inline uint2 moe_expert_group(constant MoeExpertParams &params, uint2 group) {
  return params.tile_major ? uint2(group.y, group.x) : group;
}

// A GGUF expert pass's weights for one tile (ops/MoE.cpp, metal/abi/MoE.h):
// every routed expert of the projection is one image segment of experts *
// output_size rows, so expert e's planes start at tile e * output_size /
// QUANT_TILE_ROWS (metal/abi/QuantFormat.h); the shared expert (id `experts`) has a segment
// of its own, possibly in another format. w1 is the meta plane for formats
// without a second plane, as the host binds it.
struct MoeGgufSegment {
  device uchar *w0;
  device uchar *w1;
  device uchar *meta;
  uint format;
};

// As moe_gguf_segment, with the routed and shared format ids explicit: the
// fused gate/up pass describes two segments per expert through one params
// struct and calls this for each.
inline MoeGgufSegment moe_gguf_segment_fmt(uint expert, uint experts,
                                           uint input_size, uint output_size,
                                           uint routed_format, uint shared_format,
                                           device uchar *w0, device uchar *w1, device uchar *meta,
                                           device uchar *shared_w0, device uchar *shared_w1,
                                           device uchar *shared_meta) {
  if (expert == experts)
    return {shared_w0, shared_w1, shared_meta, shared_format};
  constant QuantFormat &f = kQuantFormats[routed_format];
  const ulong rows = ulong(expert) * output_size, groups = input_size / 32;
  return {w0 + rows * groups * f.plane0_bytes, w1 + rows * groups * f.plane1_bytes,
          meta + rows * (groups / f.meta_groups) * f.meta_bytes, routed_format};
}

inline MoeGgufSegment moe_gguf_segment(uint expert, constant MoeGgufExpertParams &p,
                                       device uchar *w0, device uchar *w1, device uchar *meta,
                                       device uchar *shared_w0, device uchar *shared_w1,
                                       device uchar *shared_meta) {
  return moe_gguf_segment_fmt(expert, p.experts, p.input_size, p.output_size,
                              p.routed_format, p.shared_format, w0, w1, meta,
                              shared_w0, shared_w1, shared_meta);
}

// The streamed expert pass's weights for one tile (ops/MoE.cpp's
// moe_expert_gguf_streamed_*): the routed experts live in the arena's
// chunks, not resident planes. `slots` is the arena's expert -> global slot
// table (~0u while the expert holds no slot — the tile's pass skips it),
// `chunks` its chunk -> region GPU-address table (MOE_GGUF_STREAM_REGIONS
// uint64s per chunk). A slot resolves to its chunk and local slot exactly
// as moe_expert_chunk does in the staged reference
// (runtime/qwen4/metal/kernels/common/moe_expert_chunk.h); the local slot
// then indexes the chunk region's planes the way an expert indexes a
// resident segment's, so the same tile math applies with `local` for
// `expert`. The shared expert (id `experts`) keeps its resident segment.
// `resident` reports whether the tile has weights to read.
inline MoeGgufSegment moe_gguf_stream_segment(
    uint expert, constant MoeGgufStreamParams &p, device const uint *slots,
    device const ulong *chunks, device uchar *shared_w0, device uchar *shared_w1,
    device uchar *shared_meta, bool &resident) {
  if (expert == p.experts) {
    resident = true;
    return {shared_w0, shared_w1, shared_meta, p.shared_format};
  }
  const uint slot = slots[expert];
  if (slot == 0xffffffffu) {
    resident = false;
    return {nullptr, nullptr, nullptr, p.routed_format};
  }
  resident = true;
  uint chunk = 0, local = slot;
  if (slot >= p.floor_slots) {
    const uint elastic = slot - p.floor_slots;
    chunk = 1 + elastic / p.chunk_slots;
    local = elastic - (chunk - 1) * p.chunk_slots;
  }
  device const ulong *regions =
      chunks + ulong(chunk) * MOE_GGUF_STREAM_REGIONS + ulong(p.projection) * 3;
  device uchar *w0 = reinterpret_cast<device uchar *>(regions[0]);
  device uchar *w1 = reinterpret_cast<device uchar *>(regions[1]);
  device uchar *meta = reinterpret_cast<device uchar *>(regions[2]);
  constant QuantFormat &f = kQuantFormats[p.routed_format];
  const ulong rows = ulong(local) * p.output_size, groups = p.input_size / 32;
  return {w0 + rows * groups * f.plane0_bytes,
          w1 + rows * groups * f.plane1_bytes,
          meta + rows * (groups / f.meta_groups) * f.meta_bytes,
          p.routed_format};
}
