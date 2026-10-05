#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <span>

namespace richengine::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // Tree-verify tables (draft_select_tree): per lane the node descriptors,
  // node tokens and node count of its verify tree
  // (RICHENGINE_TREE_VERIFY_NODES stride). Unused by chain-only selection.
  metal::MetalBuffer treeNodes;
  metal::MetalBuffer treeTokens;
  metal::MetalBuffer treeCounts;
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// A DSpark draft's Markov head: the previous-token feature table
// (markov_w1, [vocabulary][rank]) and the bias projection (markov_w2,
// [vocabulary][rank]), both bf16 row-major.
struct DraftMarkovHead final {
  metal::MetalBuffer embedding;
  metal::MetalBuffer projection;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// RICHENGINE_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// choice, and walks the RICHENGINE_DRAFT_PROPOSAL_TOKENS positions greedily or,
// for a sampling lane, drawing at its temperature.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies) const;
  // The same pipeline plus draft_select_tree, which emits each lane's verify
  // tree tables. treeMask marks the lanes whose comb leaves are emitted
  // (greedy lanes of a tree-capable draft); other lanes get a linear table.
  void addTree(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
               const DraftCodebooks &codebooks,
               std::span<const uint32_t> anchors,
               std::span<const SamplingPolicy> policies,
               uint32_t treeMask) const;
  // The plain DFlash draft's selection (draft_select_top16_sharded +
  // draft_select_plain): per position the best or a drawn candidate of the
  // draft logits, with no predecessor/successor codebooks. The codebookless
  // buffers' selectorHidden and unary are unused.
  void addPlain(metal::CommandGraph &graph,
                const DraftSelectorBuffers &buffers,
                std::span<const uint32_t> anchors,
                std::span<const SamplingPolicy> policies) const;
  // The DSpark draft's selection (dspark_select_top16_sharded +
  // dspark_select_edges + draft_select_dspark): like the plain path, but
  // position p reads logits row p (the anchor row already predicts a token)
  // and each position's candidates are rescored with the Markov bias of the
  // previously sampled token. A parallel pass scores every
  // predecessor/candidate edge in advance — the predecessor set is the
  // previous position's unbiased top-16, not the walk's pick — and the walk
  // chooses the positions serially. selectorHidden is unused; unary holds
  // the merged top-16 logits. There is no tree path.
  void addDSpark(metal::CommandGraph &graph,
                 const DraftSelectorBuffers &buffers,
                 const DraftMarkovHead &markov,
                 std::span<const uint32_t> anchors,
                 std::span<const SamplingPolicy> policies) const;

private:
  uint32_t vocabulary_ = 0;
};

} // namespace richengine::ops
