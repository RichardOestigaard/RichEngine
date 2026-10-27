#include "ops/DraftSelector.hpp"

#include "metal/abi/Sampling.h"
#include "ops/KernelNames.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace richengine::ops {
namespace {

constexpr uint32_t kShards = RICHENGINE_DRAFT_SAMPLING_SHARDS;
constexpr uint32_t kPositions = RICHENGINE_DRAFT_PROPOSAL_TOKENS;
// Each position's group scores its 16 x 16 edge table eight edges per
// simdgroup task; eight simdgroups balance the seven-group B1 dispatch
// against the 28 groups of B4 (wider groups speed up B1 and slow down B4).
constexpr uint32_t kEdgeThreads = 256;

} // namespace

DraftSelector::DraftSelector(uint32_t vocabulary, uint32_t logitsStride)
    : vocabulary_(vocabulary),
      logitsStride_(logitsStride ? logitsStride : vocabulary) {
  if (!vocabulary || logitsStride_ < vocabulary)
    throw std::invalid_argument("invalid draft selector vocabulary");
}

DraftSelectorWorkspace DraftSelector::workspace(uint32_t positions) {
  if (!positions)
    throw std::invalid_argument("invalid draft selector workspace position count");
  const uint64_t candidates = uint64_t{positions} * RICHENGINE_DRAFT_CANDIDATES;
  const uint64_t pool = uint64_t{positions} * RICHENGINE_DSPARK_POOL;
  // The partial values are followed by each position's edge table: the
  // candidates-squared DFlash tables or the pool-squared DSpark ones,
  // whichever is larger.
  return {candidates * kShards * sizeof(uint32_t),
          (candidates * kShards +
           std::max(candidates * RICHENGINE_DRAFT_CANDIDATES,
                    pool * RICHENGINE_DSPARK_POOL)) *
              sizeof(float),
          candidates * sizeof(uint32_t), candidates * sizeof(float),
          pool * sizeof(float)};
}

void DraftSelector::add(metal::CommandGraph &graph,
                        const DraftSelectorBuffers &buffers,
                        const DraftCodebooks &codebooks,
                        std::span<const uint32_t> anchors,
                        std::span<const SamplingPolicy> policies,
                        uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDraftSelectEdges),
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add(std::string(kDraftSelectDflash),
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities, buffers.confidence},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void DraftSelector::addTree(metal::CommandGraph &graph,
                            const DraftSelectorBuffers &buffers,
                            const DraftCodebooks &codebooks,
                            std::span<const uint32_t> anchors,
                            std::span<const SamplingPolicy> policies,
                            uint32_t treeMask, uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  params.tree_mask = treeMask;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDraftSelectEdges),
            {buffers.partialIds, buffers.partialValues, buffers.candidates,
             buffers.unary, buffers.selectorHidden, codebooks.predecessor,
             codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  // The tree comb always emits kDraftQueryRows-deep nodes; on a draft
  // with queryRows() < kDraftQueryRows the rows past its live span are
  // dead compute — clamped and never retained, but still verified.
  graph.add(std::string(kDraftSelectTree),
            {buffers.candidates, buffers.unary, buffers.partialValues,
             buffers.uniforms, buffers.proposedTokens,
             buffers.proposalProbabilities, buffers.treeTokens,
             buffers.treeNodes, buffers.treeCounts},
            params, {lanes, 1, 1}, {1, 1, 1});
}

void DraftSelector::addPlain(metal::CommandGraph &graph,
                             const DraftSelectorBuffers &buffers,
                             std::span<const uint32_t> anchors,
                             std::span<const SamplingPolicy> policies,
                             uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDraftSelectPlain),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.candidates, buffers.proposalProbabilities,
             buffers.proposedTokens, buffers.confidence},
            params, {uint64_t{lanes} * kPositions, 1, 1}, {32, 1, 1});
}

void DraftSelector::addPlainTree(metal::CommandGraph &graph,
                                 const DraftSelectorBuffers &buffers,
                                 std::span<const uint32_t> anchors,
                                 std::span<const ops::SamplingPolicy> policies,
                                 uint32_t treeMask, uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  params.tree_mask = treeMask;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDraftSelectPlainTree),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.candidates, buffers.proposalProbabilities,
             buffers.proposedTokens, buffers.treeTokens, buffers.treeNodes,
             buffers.treeCounts},
            params, {uint64_t{lanes} * kPositions, 1, 1}, {32, 1, 1});
}

void DraftSelector::addDSpark(metal::CommandGraph &graph,
                              const DraftSelectorBuffers &buffers,
                              const DraftMarkovHead &markov,
                              std::span<const uint32_t> anchors,
                              std::span<const SamplingPolicy> policies,
                              uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDsparkSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDsparkSelectEdges),
            {buffers.partialIds, buffers.partialValues, markov.embedding,
             markov.projection, buffers.draftInput},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add(std::string(kDraftSelectDspark),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.proposalProbabilities, buffers.proposedTokens,
             buffers.confidence},
            params, {uint64_t{lanes}, 1, 1}, {32, 1, 1});
}

void DraftSelector::addDSparkTree(metal::CommandGraph &graph,
                              const DraftSelectorBuffers &buffers,
                              const DraftMarkovHead &markov,
                              std::span<const uint32_t> anchors,
                              std::span<const SamplingPolicy> policies,
                              uint32_t treeMask, uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  params.tree_mask = treeMask;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDsparkSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDsparkSelectEdges),
            {buffers.partialIds, buffers.partialValues, markov.embedding,
             markov.projection, buffers.draftInput},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add(std::string(kDraftSelectDsparkTree),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.proposalProbabilities, buffers.proposedTokens,
             buffers.treeTokens, buffers.treeNodes, buffers.treeCounts,
             buffers.draftInput},
            params, {uint64_t{lanes}, 1, 1}, {32, 1, 1});
}

void DraftSelector::addPool(metal::CommandGraph &graph,
                            const DraftSelectorBuffers &buffers,
                            const DraftCodebooks &codebooks,
                            std::span<const uint32_t> anchors,
                            std::span<const SamplingPolicy> policies,
                            uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDflashSelectPoolEdges),
            {buffers.partialIds, buffers.partialValues, buffers.selectorHidden,
             codebooks.predecessor, codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  // The DSpark walk carries no Markov logic of its own — it only reads the
  // pool's shard scores and the edge table — so it walks the DFlash pool
  // table unchanged.
  graph.add(std::string(kDraftSelectDspark),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.proposalProbabilities, buffers.proposedTokens,
             buffers.confidence},
            params, {uint64_t{lanes}, 1, 1}, {32, 1, 1});
}

void DraftSelector::addPoolTree(metal::CommandGraph &graph,
                                const DraftSelectorBuffers &buffers,
                                const DraftCodebooks &codebooks,
                                std::span<const uint32_t> anchors,
                                std::span<const SamplingPolicy> policies,
                                uint32_t treeMask, uint32_t queryRows) const {
  if (anchors.empty() || anchors.size() != policies.size() ||
      anchors.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft selector batch");
  const uint32_t lanes = static_cast<uint32_t>(anchors.size());
  SelectorBatchParams params{};
  params.lanes = lanes;
  params.query_rows = queryRows;
  params.vocabulary = vocabulary_;
  params.tree_mask = treeMask;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    params.anchor[lane] = anchors[lane];
    params.temperature[lane] = policies[lane].temperature;
    if (policies[lane].samples())
      params.sampling_mask |= uint32_t{1} << lane;
  }
  graph.addTail(std::string(kDraftSelectTop16Sharded),
            {buffers.logits, buffers.partialIds, buffers.partialValues},
            DraftTop16Params{vocabulary_, logitsStride_}, queryRows,
            {uint64_t{lanes} * kPositions * kShards, 1, 1});
  graph.add(std::string(kDflashSelectPoolEdges),
            {buffers.partialIds, buffers.partialValues, buffers.selectorHidden,
             codebooks.predecessor, codebooks.successor},
            params, {uint64_t{lanes} * kPositions, 1, 1},
            {kEdgeThreads, 1, 1});
  graph.add(std::string(kDraftSelectDsparkTree),
            {buffers.partialIds, buffers.partialValues, buffers.uniforms,
             buffers.proposalProbabilities, buffers.proposedTokens,
             buffers.treeTokens, buffers.treeNodes, buffers.treeCounts,
             buffers.draftInput},
            params, {uint64_t{lanes}, 1, 1}, {32, 1, 1});
}

void DraftSelector::addVerifyRows(metal::CommandGraph &graph,
                                  const metal::MetalBuffer &confidence,
                                  const metal::MetalBuffer &verifyRows,
                                  const VerifyRowsModel &model,
                                  std::span<const uint32_t> maximumRows,
                                  float confMin) {
  if (maximumRows.empty() || maximumRows.size() > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid draft verify rows batch");
  const uint32_t lanes = static_cast<uint32_t>(maximumRows.size());
  VerifyRowsParams params{};
  params.lanes = lanes;
  params.every_row = model.everyRow ? 1 : 0;
  params.conf_min = confMin;
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    if (!maximumRows[lane] || maximumRows[lane] > RICHENGINE_DRAFT_QUERY_ROWS)
      throw std::invalid_argument("invalid draft verify rows maximum");
    params.maximum[lane] = maximumRows[lane];
  }
  if (!model.everyRow) {
    if (!std::isfinite(model.slope))
      throw std::invalid_argument("invalid draft verify rows model");
    params.slope = model.slope;
    for (uint32_t position = 0; position < kPositions; ++position) {
      if (!std::isfinite(model.intercepts[position]))
        throw std::invalid_argument("invalid draft verify rows model");
      params.intercept[position] = model.intercepts[position];
    }
    // The costs of every total the lanes may verify, from their anchors to
    // sixteen rows each.
    for (uint32_t rows = lanes; rows <= lanes * RICHENGINE_DRAFT_QUERY_ROWS; ++rows) {
      if (!(std::isfinite(model.costs[rows]) && model.costs[rows] > 0.0F))
        throw std::invalid_argument("invalid draft verify rows cost");
      params.cost[rows] = model.costs[rows];
    }
  }
  if (confidence.sizeBytes() < uint64_t{lanes} * kPositions * sizeof(float))
    throw std::invalid_argument("invalid draft proposal confidence");
  if (verifyRows.sizeBytes() < uint64_t{lanes} * sizeof(uint32_t))
    throw std::invalid_argument("invalid draft verify rows buffer");
  graph.add(std::string(kDraftVerifyRows), {confidence, verifyRows}, params,
            {1, 1, 1}, {1, 1, 1});
}

} // namespace richengine::ops
