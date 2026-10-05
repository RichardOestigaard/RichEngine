#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
#include "PlainDraft.hpp"
#include "WeightImages.hpp"
#include "WeightStore.hpp"
#include "ops/DraftAttention.hpp"
#include "ops/DraftSelector.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

namespace richengine::model {

// A DSpark draft (architectures: "Qwen3DSparkModel"/"Lfm2DSparkDraftModel"):
// the plain transformer draft's layer stack — one fused QKV projection,
// per-head Q/K norms, an output projection and a gated MLP per layer, the
// same fc + hidden_norm context fusion injected as every layer's K/V — with
// block-bidirectional attention like DFlash2's. Where DFlash2 carries
// dynamic convolutions and candidate-selector codebooks, DSpark carries a
// sequential Markov head (markov_w1's previous-token features rescored
// through markov_w2 at selection time) and a confidence head, which this
// runtime loads but does not score.
struct DSparkDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<PlainDraftLayerWeights> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  // markov_w1 and markov_w2, [vocabularySize][markovRank] bf16 rows each.
  metal::MetalBuffer markovEmbedding;
  metal::MetalBuffer markovProjection;
  // confidence_head.proj, [1][hiddenSize + markovRank] plus its bias; loaded
  // for completeness, unused this phase.
  metal::MetalBuffer confidenceWeight;
  metal::MetalBuffer confidenceBias;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

inline constexpr std::string_view kDSparkDraftMagic = "MDFS0006";

// An RichEngine package's DSpark-draft files: layer-<N>.bin and model.bin, in the
// section order DSparkDraft.cpp reads.
struct PackedDSparkDraftFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const;
  [[nodiscard]] WeightFile model() const;
};

class DSparkCheckpointLoader;

using DSparkDraftFiles =
    std::variant<PackedDSparkDraftFiles,
                 std::reference_wrapper<DSparkCheckpointLoader>>;

[[nodiscard]] DSparkDraftWeights
loadDSparkDraftWeights(metal::MetalBackend &backend,
                       const DSparkDraftFiles &files,
                       DFlashDraftLayout layout);

// Builds the DSpark draft's layer graph and its Markov selection
// (ops::DraftSelector::addDSpark) from packed buffers and the same
// persistent context ring DFlashDraft uses. The four entry points mirror
// DFlashDraft's so the runtime dispatches either model identically.
class DSparkDraft final {
public:
  DSparkDraft(const DSparkDraftWeights &weights, metal::MetalBackend &backend,
              const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths) const;
  // The DSpark draft has no candidate DAG, so treeMask is ignored: its
  // selector always emits the seven-token chain.
  void addSelection(metal::CommandGraph &graph,
                    const ops::DraftSelectorBuffers &buffers,
                    std::span<const uint32_t> anchors,
                    std::span<const ops::SamplingPolicy> policies,
                    uint32_t treeMask) const;
  void addContextCommit(metal::CommandGraph &graph,
                        DFlashContextBuffers buffers,
                        std::span<const uint32_t> startPositions) const;

private:
  const DSparkDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
};

} // namespace richengine::model
