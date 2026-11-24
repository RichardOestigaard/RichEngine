#pragma once

// The Maxwell MBSA indexer (maxwell_moe's multi-block sparse attention,
// kernels/shared/mbsa_indexer.metal): per-token index keys in the pages'
// index plane and a per-KV-group route of each query row, written as the
// sparse attention's per-head block lists. Its projections run through
// ops::Linear; a target composes them.
//
// The block-list indexing differs from the other sparse cores': the
// _k4q8d128 split and reduce kernels resolve a tile's list to index
// (block_list + tile) * kvHeads + kv_head — a lane of `rowTiles` eight-row
// tiles and a prefill sequence's tiles each take kvHeads consecutive lists
// per tile, one per KV group, matching ops::PagedAttention::addVerify's
// block_list + lane * rowTiles base (ops/Mbsa::addRoute writes the same
// indices). The AttentionBlocks the target hands addVerify/addPrefill
// therefore covers kvHeads * rowTiles lists per lane (per sequence's
// tiles), not rowTiles, and blocks.list is the unexpanded base.

#include "metal/CommandGraph.hpp"
#include "metal/abi/Mbsa.h"
#include "ops/BlockIndexer.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace richengine::ops {

// The MBSA indexer's rows of a dispatch's lanes (a verify step's lanes, or
// a prefill sequence as one): `rows` rows of each lane, lane l's from
// logical position firstPositions[l], lane l's index keys through index
// table index.tables[l] a RichKvPage per logical page, `offset` bytes into
// the index plane each entry names (ops::BlockIndex).
struct MbsaRows final {
  uint32_t rows = 0;
  std::span<const uint32_t> firstPositions;
  BlockIndex index;
};

struct Mbsa final {
  // The worst case a (lane, group, tile) list can hold: the tile's rows
  // span at most two route blocks, so their unions reach at most the
  // MBSA_PREFIX_BLOCKS prefix, the MBSA_LOCAL_BLOCKS + 1 local blocks of
  // the two blocks' bands, and each row's MBSA_REMOTE_BLOCKS remote picks —
  // each as MBSA_ROUTE_SUBBLOCKS 4-token entries. A list is usually far
  // shorter: rows share the prefix, and adjacent rows' remote bands
  // overlap. A shorter blocks.stride truncates the list's newest entries.
  static constexpr uint32_t kRouteListEntries =
      (MBSA_PREFIX_BLOCKS + MBSA_LOCAL_BLOCKS + 1 +
       MBSA_ROUTE_ROWS * MBSA_REMOTE_BLOCKS) *
      MBSA_ROUTE_SUBBLOCKS;

  // The index projections (mbsa_index_project, one kernel for both heads
  // and the projections themselves folded in): each (row, head)
  // threadgroup dots `input` — the lanes' fp32 normed rows of inputWidth
  // values — against its head's slice of the fp32 weight planes
  // (queryWeights [heads][dimension][inputWidth], keyWeights
  // [dimension][inputWidth]), normalizes and rotates the projection, and
  // writes the group queries bf16 [lane][row][MBSA_INDEX_HEADS]
  // [MBSA_INDEX_DIMENSION] to `queries` and the shared key head's keys to
  // each row's token slot of the index plane. ropeCos and ropeSin are the
  // main branch's per-row tables of MBSA_ROPE_PAIRS.
  static void addProject(metal::CommandGraph &graph, metal::MetalBuffer input,
                         metal::MetalBuffer queryWeights,
                         metal::MetalBuffer keyWeights,
                         const NormWeights &queryNorm,
                         const NormWeights &keyNorm, metal::MetalBuffer ropeCos,
                         metal::MetalBuffer ropeSin, metal::MetalBuffer queries,
                         const MbsaRows &rows, uint32_t inputWidth);
  // The lanes' routes (mbsa_route): each (lane, group, tile) list at index
  // (list + lane * rowTiles + tile) * MBSA_INDEX_HEADS + group of `lists`,
  // `listStride` entries each, its count in `counts` — the indices the
  // _k4q8d128 sparse splits and reduce read (the file comment). `rowTiles`
  // is the attention plan's eight-row tiles per lane (verify), or the
  // sequence's tiles (prefill, one lane); `list` the block_list base the
  // attention parameters carry; `activeRows` each lane's live rows — the
  // chunk rows of a verify lane or the sequence's rows — which tiles past
  // still get a zero count. Encode the layer's addProject before its
  // addRoute: remote scoring reads index keys the earlier dispatch stored.
  static void addRoute(metal::CommandGraph &graph, metal::MetalBuffer queries,
                       metal::MetalBuffer lists, metal::MetalBuffer counts,
                       uint32_t list, uint32_t listStride, uint32_t rowTiles,
                       std::span<const uint32_t> activeRows,
                       const MbsaRows &rows);
};

} // namespace richengine::ops
