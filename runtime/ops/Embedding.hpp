#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace splash::ops {

class Embedding final {
public:
  static void add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                  const EmbeddingWeights &table, metal::MetalBuffer output,
                  uint32_t rows);
  // A verify step's input tokens, SPLASH_TARGET_VERIFY_ROWS per lane: the
  // lane's anchor, row 0 of its draft input rows, then the draft's
  // SPLASH_DRAFT_PROPOSAL_TOKENS proposals, each clamped into the
  // vocabulary.
  static void addVerifyInput(metal::CommandGraph &graph,
                             metal::MetalBuffer draftInputTokens,
                             metal::MetalBuffer proposedTokens,
                             metal::MetalBuffer verifyInputTokens,
                             uint32_t vocabulary, uint32_t lanes);
  // A tree-verify step's inputs, SPLASH_TREE_VERIFY_NODES per lane: each
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
  // consumes: `width` elements per row, up to SPLASH_TARGET_VERIFY_ROWS.
  static void addTreeCaptureGather(metal::CommandGraph &graph,
                                   metal::MetalBuffer source,
                                   metal::MetalBuffer retainedPath,
                                   metal::MetalBuffer retained,
                                   metal::MetalBuffer destination,
                                   uint32_t width, uint32_t lanes);
};

} // namespace splash::ops
