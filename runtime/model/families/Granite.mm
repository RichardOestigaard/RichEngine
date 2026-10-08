#include "model/families/FamilyMakers.hpp"

namespace richengine::model {
namespace {

// Granite 4.2 ships no GPU draft (no DFlash/DSpark/MTP checkpoint exists);
// the Null layout satisfies the geometry contract while the n-gram predraft
// supplies every proposal.
DFlashDraftLayout graniteNullDraftLayout(const GraniteLayout &target) {
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
  return layout;
}

// The granite targets are text-only: no vision layout, no vision weights.
ModelDescriptor graniteDescriptor(std::string name, const GraniteLayout &target) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = target.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, graniteNullDraftLayout(target), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace

ModelDescriptor granite3BDescriptor(std::string name) {
  return graniteDescriptor(std::move(name), GraniteLayout{});
}

ModelDescriptor granite8BDescriptor(std::string name) {
  return graniteDescriptor(std::move(name), granite8BLayout());
}

} // namespace richengine::model
