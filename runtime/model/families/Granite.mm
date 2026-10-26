#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Granite.hpp"

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

// The registry's erased-signature entry for the granite target.
TargetWeights loadGraniteTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadGraniteWeights(backend, layout, files);
}

const bool kRegisteredGraniteFamilies = [] {
  registerTargetArch({"granite", &loadGraniteTarget});
  registerModelFamily({"Granite-4.2-3B", "granite", "", &granite3BDescriptor,
                       nullptr, nullptr});
  registerModelFamily({"Granite-4.2-8B", "granite", "", &granite8BDescriptor,
                       nullptr, nullptr});
  return true;
}();

} // namespace
} // namespace richengine::model
