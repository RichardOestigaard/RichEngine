#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <span>

namespace richengine::ops {

class Embedding final {
public:
  // `scale` selects the scaled gather (Gemma's sqrt(hidden) embedding
  // scale); 0 is the unscaled table lookup.
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const EmbeddingWeights &table, metal::MetalBuffer output,
                  uint32_t rows, float scale = 0.0F);
  // A verify step's input tokens, RICHENGINE_TARGET_VERIFY_ROWS per lane: the
  // lane's anchor, row 0 of its draft input rows, then the draft's
  // RICHENGINE_DRAFT_PROPOSAL_TOKENS proposals, each clamped into the
  // vocabulary. liveRows[lane] bounds the live rows (anchor included); a row
  // past it can never be retained, so it repeats the anchor and keeps the
  // dead rows' expert routing on the anchor's paths. An empty span leaves
  // every row live.
  // verifyStride is the lanes' packed row stride: the draft's trained verify
  // depth (draft.verifyRows()), at most RICHENGINE_TARGET_VERIFY_ROWS.
  static void addVerifyInput(metal::CommandGraph &graph,
                             metal::MetalBuffer draftInputTokens,
                             metal::MetalBuffer proposedTokens,
                             metal::MetalBuffer verifyInputTokens,
                             uint32_t vocabulary, uint32_t lanes,
                             uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS,
                             uint32_t draftStride = RICHENGINE_DRAFT_QUERY_ROWS,
                             std::span<const uint32_t> liveRows = {});
  // The RICHENGINE_DEVICE_VERIFY_PARAMS variant (verify_input_tokens_dev):
  // the step pack decode_step_pack wrote from the previous step's
  // acceptance results rides as a buffer binding, and each lane takes its
  // anchor and live count from it when the pack proves to be the lane's
  // own predecessor's (matching request id and — unless the lane is in
  // `deviceLanes` — matching position and live rows against the staged
  // payload). `requestIds` and `positionBases` carry the host's values for
  // that proof, one entry per lane.
  static void addVerifyInputDevice(
      metal::CommandGraph &graph, metal::MetalBuffer draftInputTokens,
      metal::MetalBuffer proposedTokens, metal::MetalBuffer verifyInputTokens,
      metal::MetalBuffer stepPack, std::span<const uint64_t> requestIds,
      std::span<const uint32_t> positionBases, uint32_t deviceLanes,
      uint32_t vocabulary, uint32_t lanes,
      uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS,
      uint32_t draftStride = RICHENGINE_DRAFT_QUERY_ROWS,
      std::span<const uint32_t> liveRows = {});
  // A tree-verify step's inputs, RICHENGINE_TREE_VERIFY_NODES per lane: each
  // node's token, its (t, h, w) rope position (the lane's base triple plus
  // the node's depth) and its ancestor bitmask, from draft_select_tree's
  // tables. `base` holds each lane's three position bases; dead rows take
  // `maskToken`.
  static void addVerifyTreeInput(metal::CommandGraph &graph,
                                 metal::MetalBuffer treeTokens,
                                 metal::MetalBuffer treeNodes,
                                 metal::MetalBuffer treeCounts,
                                 metal::MetalBuffer verifyInputTokens,
                                 metal::MetalBuffer positions,
                                 metal::MetalBuffer masks,
                                 const uint32_t base[][3], uint32_t vocabulary,
                                 uint32_t maskToken, uint32_t lanes);
  // The retained path's captured hidden rows, gathered from the tree's
  // emitted row order into path order, which the draft context commit
  // consumes: `width` elements per row, up to RICHENGINE_TARGET_VERIFY_ROWS.
  static void addTreeCaptureGather(metal::CommandGraph &graph,
                                   metal::MetalBuffer source,
                                   metal::MetalBuffer retainedPath,
                                   metal::MetalBuffer retained,
                                   metal::MetalBuffer destination,
                                   uint32_t width, uint32_t lanes);
};

} // namespace richengine::ops
