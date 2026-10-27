#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Granite.hpp"

#include <cmath>

namespace richengine::model {
namespace {

// Granite 4.2 ships no GPU draft (no DFlash/DSpark/MTP checkpoint exists);
// the Null layout satisfies the geometry contract while the n-gram predraft
// supplies every proposal.
DFlashDraftLayout graniteNullDraftLayout(const TargetLayout &target) {
  // The planning paths still size draft workspaces from the layout, so it
  // describes the compiled Q32K8D128 head even though nothing encodes.
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 1280;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 17408;
  layout.attentionHeadDimension = 128;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  return layout;
}

// The granite targets are text-only: no vision layout, no vision weights.
ModelDescriptor graniteDescriptor(std::string name,
                                  const TargetLayout &target) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, graniteNullDraftLayout(target), vision);
  descriptor.visionSource = VisionSource::None;
  // No GPU draft means the host mask round trip never competes with a verify
  // loop, and GPU masks + forced spans measured ~29% faster on the
  // constrained request (dfa-benchmark, 694-state tool-call DFA, B1).
  descriptor.tuning.dfaMode = "full";
  return descriptor;
}

ModelDescriptor granite3BDescriptor(std::string name) {
  return graniteDescriptor(std::move(name), graniteTargetLayout());
}

ModelDescriptor granite8BDescriptor(std::string name) {
  return graniteDescriptor(std::move(name), granite8BTargetLayout());
}

// The granite source's flat config check.
void validateGraniteSourceConfig(id textObj,
                                 const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"max_position_embeddings", layout.maximumContextTokens},
      {"rope_theta", 10000000u}});
  // attention_multiplier is a float the GGUF metadata check verifies.
  if (id value = text[@"attention_multiplier"];
      [value isKindOfClass:[NSNumber class]] &&
      std::abs([value doubleValue] - layout.attentionScale) >
          layout.attentionScale * 1e-6)
    throw std::invalid_argument(
        "unsupported granite attention_multiplier: " +
        std::to_string([value doubleValue]));
}

// The registry's erased-signature entry for the granite target.
TargetWeights loadGraniteTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadGraniteWeights(backend, layout, files);
}

const bool kRegisteredGraniteFamilies = [] {
  registerTargetArch({"granite", &loadGraniteTarget});
  registerModelFamily({.name = "Granite-4.2-3B", .modelType = "granite",
                       .legacyType = "",
                       .makeDescriptor = &granite3BDescriptor,
                       .validateSourceConfig = &validateGraniteSourceConfig});
  registerModelFamily({.name = "Granite-4.2-8B", .modelType = "granite",
                       .legacyType = "",
                       .makeDescriptor = &granite8BDescriptor,
                       .validateSourceConfig = &validateGraniteSourceConfig});
  return true;
}();

} // namespace
} // namespace richengine::model
