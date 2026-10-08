#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

// The dense target's packed/source draft default: the released DSpark draft
// (openbmb/MiniCPM5-2B-DSpark, five layers of 16x2x128 heads over a
// 2560-wide fused QKV, block_size 7), whose geometry the draft config or
// the packed manifest declares.
DFlashDraftLayout denseDraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DSpark;
  layout.layers = 5;
  layout.hiddenSize = DenseLayout{}.hiddenSize;
  layout.vocabularySize = DenseLayout{}.vocabularySize;
  layout.dynamicSize = 0;
  layout.qkvSize = 2560;
  layout.attentionSize = 2048;
  layout.intermediateSize = 6144;
  layout.attentionHeadDimension = 128;
  layout.rotaryTheta = 5'000'000.0F;
  layout.targetHiddenSize = DenseLayout{}.capturedHiddenSize();
  layout.selectorRank = 0;
  layout.kvHeads = 2;
  layout.markovRank = 256;
  layout.blockSize = 7;
  return layout;
}

// The dense target is text-only: no vision layout, no vision weights.
ModelDescriptor denseDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = DenseLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), DenseLayout{}, denseDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace richengine::model
