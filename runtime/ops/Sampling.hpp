#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Sampling.h"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <optional>
#include <span>

namespace richengine::ops {

// The sampling penalties, which rewrite a lane's target logits before its
// policy selects from them: repetition scales the logit of every token the
// prompt or the output holds, presence and frequency lower that of every
// token the output holds. The defaults leave the logits unchanged.
struct SamplingPenalties final {
  float repetition = 1.0F;
  float presence = 0.0F;
  float frequency = 0.0F;

  [[nodiscard]] bool active() const noexcept {
    return repetition != 1.0F || presence != 0.0F || frequency != 0.0F;
  }
};

// One lane's target policy. The defaults are a request's: greedy, and a
// sampling lane keeps every token.
struct SamplingPolicy final {
  // A sampling lane keeps the tokens minP leaves it, then its topK most
  // likely of those, or every one for 0 or a topK past the vocabulary (top-k
  // disabled), then its top-p nucleus of those. Greedy lanes (temperature 0)
  // read none of the three.
  uint32_t topK = 0;
  float temperature = 0.0F;
  float topP = 1.0F;
  bool constrained = false;
  // The lane ignores end-of-sequence: the target never selects a stop token,
  // though the draft may still propose one.
  bool excludesStopTokens = false;
  // Greedy lanes take the argmax of the penalized logits.
  SamplingPenalties penalties{};
  // A sampling lane first drops the tokens less likely than minP times its
  // most likely one; 0 drops none.
  float minP = 0.0F;

  [[nodiscard]] bool samples() const noexcept { return temperature > 0.0F; }
};

// The penalty words of every state lane, one row of vocabulary words each,
// and the row each lane of a dispatch reads. Dispatch lanes follow the batch
// plan, not state lanes, so a lane never binds a row by its own index.
struct PenaltyTable final {
  metal::MetalBuffer words;
  std::span<const uint32_t> rows;
};

struct SamplingWorkspace final {
  uint64_t argmaxValuesBytes = 0;
  uint64_t argmaxIndicesBytes = 0;
  uint64_t partialMassesBytes = 0;
  uint64_t vocabularyRowsBytes = 0;
  uint64_t vocabularyRangesBytes = 0;
  uint64_t vocabularyArrivalsBytes = 0;
};

struct SamplingBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  // Per shard of a sampled row, its share of the row's softmax denominator
  // (metal/abi/Sampling.h TargetShardMass).
  metal::MetalBuffer partialMasses;
  // Per sampled row, where its distribution ends and, for a drafted row,
  // its draft token's probability (TargetVocabularyRow), which acceptance
  // reads.
  metal::MetalBuffer vocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer constraintMasks;
  // The GPU-DFA lanes' admission rows (decode/dfa.metal's next-state table):
  // a lane whose bit is set in dfaLanes reads its selected row's record here
  // instead of constraintMasks.
  metal::MetalBuffer dfaNextStates;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer argmaxValues;
  metal::MetalBuffer argmaxIndices;
  // Verify input tokens [rows]: row 0 of a lane is its anchor, rows 1..7 its
  // draft tokens, which penalized and sampled verify rows read.
  metal::MetalBuffer inputTokens;
  // The draft's candidates and their probabilities at each proposal
  // position (AcceptanceBuffers), which a sampled verify row draws its
  // correction's residual from.
  metal::MetalBuffer draftCandidates;
  metal::MetalBuffer draftProbabilities;
  // Per sampled row, the ranges of its draw (TargetVocabularyRange), which
  // the groups that share the draw sum, and how many of those groups have
  // finished: a count every draw returns to zero, where it starts.
  metal::MetalBuffer vocabularyRanges;
  metal::MetalBuffer vocabularyArrivals;
  // The fused greedy head's per-(row, 128-column tile) argmax partials
  // (decode_head_argmax_q4), bound by the verify path when the head ran
  // fused; empty otherwise.
  metal::MetalBuffer headArgmaxValues;
  metal::MetalBuffer headArgmaxIndices;
  // Lanes whose constraint admission comes from dfaNextStates.
  uint32_t dfaLanes = 0;
  // Constrained lanes the selection nonetheless argmaxes unmasked (a
  // greedy GPU-DFA lane's proposal check): their constrained_mask bit is
  // cleared for the pass while exclude-stop still applies.
  uint32_t unmaskedLanes = 0;
};

struct AcceptanceBuffers final {
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer candidates;
  metal::MetalBuffer proposalProbabilities;
  metal::MetalBuffer targetVocabularyRows;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer outputTokens;
  metal::MetalBuffer retainedCounts;
  metal::MetalBuffer acceptedCounts;
  // Greedy GPU-DFA lanes' truncation rows (decode_dfa_truncate): lanes in
  // truncLanes cap their accepted prefix there.
  metal::MetalBuffer truncRows;
  uint32_t truncLanes = 0;
  // Each lane's verify rows (ops::DraftSelector::addVerifyRows), which bound
  // what it retains; empty verifies every row as before.
  metal::MetalBuffer verifyRows;
};

// Target token policy (penalties, min-p/top-k/top-p, constrained selection,
// stop-token exclusion, greedy argmax) and DFlash acceptance of the draft's
// proposals; the model only supplies policy, buffers, penalty words and its
// stop tokens. The draft's own selector is ops::DraftSelector.
class Sampling final {
public:
  // vocabulary bounds the token ids; logitsStride is a logits row's span
  // (vocabulary padded to the head's section rows).
  explicit Sampling(uint32_t vocabulary, uint32_t logitsStride);

  // Exact scratch/output bytes for the fixed precompiled sampling ABI.
  // Counts may cover one lane or a packed batch; the operator owns sharding.
  [[nodiscard]] static SamplingWorkspace workspace(uint32_t rows);

  // A penalized request's penalty words (metal/abi/Sampling.h), rebuilt
  // when it takes a state lane from the history the lane's prefill consumes:
  // its prompt, then the generatedTokens outputs it emitted. The prompt bit
  // marks every prompt token when markPrompt, as only repetition reads it,
  // and the counts are of every token the target selected: the emitted
  // outputs and the pending anchor. Every token must be inside the
  // vocabulary, one word each, and the history must hold a prompt.
  static void rebuildPenaltyWords(std::span<uint32_t> words,
                                  std::span<const uint32_t> history,
                                  uint64_t generatedTokens,
                                  std::optional<uint32_t> pendingToken,
                                  bool markPrompt);
  // Counts the tokens one step selected into a request's penalty words.
  static void countPenaltyTokens(std::span<uint32_t> words,
                                 std::span<const uint32_t> selected);

  // A lane whose policy has active penalties has its logits rewritten in
  // place first, from its row of the penalty table; the rows must hold the
  // LM head's fresh output. Other lanes dispatch nothing new. A greedy lane
  // then takes each row's argmax and a sampled lane draws from each row's
  // distribution over the whole vocabulary, both among the tokens the lane
  // admits: the first token after a prompt from each lane's row at
  // rowOffset, and every verify row of a batch. Each selected row's token
  // goes to the output tokens in order, one per lane for the first token
  // and eight for verify; acceptance reads a sampled verify row's draft
  // token probability from the vocabulary rows. Both run the same kernels
  // (addSelection).
  void addInitial(metal::CommandGraph &graph,
                  std::span<const SamplingPolicy> policies,
                  SamplingBuffers buffers, uint32_t rowOffset,
                  uint32_t stopToken0, uint32_t stopToken1,
                  const PenaltyTable &penalties) const;
  // liveRows, when nonempty, gives each lane's live verify row count
  // (proposal budget + 1): selection skips rows at or past it. Empty means
  // every lane keeps all RICHENGINE_TARGET_VERIFY_ROWS rows.
  // verifyStride is the lanes' packed verify row stride: the draft's trained
  // verify depth (draft.verifyRows()), at most RICHENGINE_TARGET_VERIFY_ROWS.
  void addVerify(metal::CommandGraph &graph,
                 std::span<const SamplingPolicy> policies,
                 SamplingBuffers buffers, uint32_t stopToken0,
                 uint32_t stopToken1, const PenaltyTable &penalties,
                 std::span<const uint32_t> liveRows = {},
                 uint32_t fusedHead = 0,
                 uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS) const;
  // The tree batch's selection: RICHENGINE_TREE_VERIFY_NODES argmax rows per
  // lane. Tree mode is greedy-only — a sampled, constrained or penalized
  // lane throws (the runtime falls back to the chain batch).
  void addVerifyTree(metal::CommandGraph &graph,
                     std::span<const SamplingPolicy> policies,
                     SamplingBuffers buffers, uint32_t stopToken0,
                     uint32_t stopToken1) const;
  // proposals, when nonempty, caps each lane's accepted draft tokens at its
  // adaptive budget; empty means RICHENGINE_DRAFT_PROPOSAL_TOKENS for all.
  // verifyStride is the lanes' packed verify row stride: the draft's trained
  // verify depth (draft.verifyRows()), at most RICHENGINE_TARGET_VERIFY_ROWS.
  void addAcceptance(
      metal::CommandGraph &graph, AcceptanceBuffers buffers,
      std::span<const uint32_t> maximumRetained,
      std::span<const SamplingPolicy> policies, uint32_t stopToken0,
      uint32_t stopToken1, std::span<const uint32_t> proposals = {},
      uint32_t draftCandidateStride = RICHENGINE_DRAFT_CANDIDATES,
      uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS) const;
  // The greedy GPU-DFA lanes' correction pass: one masked argmax at each
  // lane's truncation row (decode_dfa_truncate's per-lane index), written
  // over that row's unmasked selection. buffers.dfaLanes names the lanes;
  // nothing dispatches when it is zero.
  void addDfaCorrections(metal::CommandGraph &graph,
                       std::span<const SamplingPolicy> policies,
                       const SamplingBuffers &buffers,
                       metal::MetalBuffer truncRows, uint32_t stopToken0,
                       uint32_t stopToken1,
                       std::span<const uint32_t> liveRows,
                       uint32_t verifyStride) const;
  // Greedy acceptance over the lanes' verify trees: the walk emits the
  // retained path's tokens and its emitted-row indices for the KV, GDN and
  // captured-hidden commits.
  void addTreeAcceptance(metal::CommandGraph &graph,
                         metal::MetalBuffer treeTokens,
                         metal::MetalBuffer treeNodes,
                         metal::MetalBuffer treeCounts,
                         metal::MetalBuffer targetTokens,
                         metal::MetalBuffer outputTokens,
                         metal::MetalBuffer retainedCounts,
                         metal::MetalBuffer acceptedCounts,
                         metal::MetalBuffer retainedPath,
                         std::span<const uint32_t> maximumRetained,
                         uint32_t stopToken0, uint32_t stopToken1) const;
  // Splices ANE-produced alternates into every lane's sibling-leaf rows when
  // the predictor job whose serial `expected` holds has published its tokens
  // (metal/abi/Sampling.h TreeLeafPatchParams); a late or absent result keeps
  // the draft's own leaves.
  void addTreeLeafPatch(metal::CommandGraph &graph,
                        metal::MetalBuffer treeTokens,
                        metal::MetalBuffer medusaTokens,
                        metal::MetalBuffer medusaFlag,
                        metal::MetalBuffer treeCounts, uint32_t expected,
                        uint32_t lanes) const;

private:
  // The rows a selection takes from each lane (metal/abi/Sampling.h
  // TargetSamplingParams).
  struct TargetRows final {
    uint32_t rows = 0;
    uint32_t logitsRow = 0;
    uint32_t maskRow = 0;
    uint32_t uniform = 0;
    uint32_t draftedRows = 0;
    // The packed lane stride of the logits and mask rows; zero resolves to
    // RICHENGINE_TARGET_VERIFY_ROWS.
    uint32_t laneStride = 0;
  };

  // Penalizes the row at rowOffset of each penalized lane or, for verify,
  // all its rows, each also counting the draft tokens its context adds.
  // Policies come from validated requests (Model.hpp
  // SamplingParameters::validationError). verifyStride is the packed lane
  // stride of the verify logits and input tokens.
  void addPenalties(metal::CommandGraph &graph,
                    std::span<const SamplingPolicy> policies,
                    const SamplingBuffers &buffers, const PenaltyTable &table,
                    uint32_t rowOffset, bool verify,
                    uint32_t verifyStride = RICHENGINE_TARGET_VERIFY_ROWS) const;
  // The selection kernels' scalar payload for one rows layout
  // (metal/abi/Sampling.h TargetSamplingParams): the policies' masks and
  // per-lane fields, the lanes' live-row bounds, dfaLanes and
  // unmaskedLanes applied from the buffers.
  TargetSamplingParams targetParams(std::span<const SamplingPolicy> policies,
                                    const SamplingBuffers &buffers,
                                    const TargetRows &rows,
                                    uint32_t stopToken0, uint32_t stopToken1,
                                    std::span<const uint32_t> liveRows) const;
  // Selects the rows that rows names of every lane. liveRows bounds each
  // lane's selected rows; empty leaves every lane's full row count.
  void addSelection(metal::CommandGraph &graph,
                    std::span<const SamplingPolicy> policies,
                    const SamplingBuffers &buffers, const TargetRows &rows,
                    uint32_t stopToken0, uint32_t stopToken1,
                    std::span<const uint32_t> liveRows = {},
                    uint32_t fusedHead = 0) const;

  uint32_t vocabulary_ = 0;
  uint32_t logitsStride_ = 0;
  uint32_t maskWords_ = 0;
};

} // namespace richengine::ops
