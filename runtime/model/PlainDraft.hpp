#pragma once

#include "DFlashDraft.hpp"
#include "Model.hpp"
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

// A plain transformer DFlash draft (architectures: ["DFlashDraftModel"]):
// per layer one fused QKV projection, per-head Q/K norms, an output
// projection and a gated MLP — no dynamic convolutions, no candidate
// selector. Its context is the same fc + hidden_norm fusion of captured
// target hidden states, injected as every layer's K/V; the token input is
// the target embedding and the vocabulary head is the target's, shared.
struct PlainDraftLayerWeights final {
  ops::NormWeights inputNorm;
  ops::Projection qkvProjection;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  ops::Projection outputProjection;
  ops::NormWeights postAttentionNorm;
  ops::Projection gateProjection;
  ops::Projection upProjection;
  ops::Projection downProjection;
};

struct PlainDraftWeights final {
  DFlashDraftLayout layout;
  std::vector<PlainDraftLayerWeights> layers;
  ops::Projection contextProjection;
  ops::NormWeights hiddenNorm;
  ops::NormWeights finalNorm;
  std::vector<WeightFileRecord> files;
  uint64_t actualAllocatedBytes = 0;
};

inline constexpr std::string_view kPlainDraftMagic = "MDFP0005";

// An RichEngine package's plain-draft files: layer-<N>.bin and model.bin, in the
// section order PlainDraft.cpp reads.
struct PackedPlainDraftFiles final {
  WeightImages &images;
  std::filesystem::path directory;
  const DFlashDraftLayout &layout;
  [[nodiscard]] WeightFile layer(uint32_t index) const;
  [[nodiscard]] WeightFile model() const;
};

class PlainDraftCheckpointLoader;

using PlainDraftFiles =
    std::variant<PackedPlainDraftFiles,
                 std::reference_wrapper<PlainDraftCheckpointLoader>>;

[[nodiscard]] PlainDraftWeights
loadPlainDraftWeights(metal::MetalBackend &backend,
                      const PlainDraftFiles &files,
                      DFlashDraftLayout layout);

// Builds the plain draft's layer graph and its proposal selection
// (ops::DraftSelector::addPlain) from packed buffers and the same persistent
// context ring DFlashDraft uses. The four entry points mirror
// DFlashDraft's so the runtime dispatches either model identically.
class PlainDraft final {
public:
  PlainDraft(const PlainDraftWeights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators);

  void addContextPrefill(metal::CommandGraph &graph,
                         DFlashPrefillBuffers buffers, uint32_t rows,
                         std::span<const DFlashPrefillSpan> spans) const;
  void addDecode(metal::CommandGraph &graph, DFlashDecodeBuffers buffers,
                 const ops::Projection &vocabularyProjection,
                 std::span<const uint32_t> cacheLengths) const;
  // The plain draft has no candidate DAG, so treeMask is ignored: its
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
  const PlainDraftWeights &weights_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
  ops::DraftSelector selector_;
  // Each layer's key and value rows of its QKV projection, views of its
  // planes, which the context writers project with.
  std::vector<ops::Projection> contextKvProjections_;
};

} // namespace richengine::model
