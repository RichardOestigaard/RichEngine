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
  layout.headCore = ops::DraftHeadCore::Q16K2D128;
  layout.contextCore = ops::DraftContextCore::Plain2048;
  layout.markovRank = 256;
  layout.blockSize = 7;
  return layout;
}

// A DFlash2 draft for the dense target (Draftfit's MiniCPM5-2B port,
// jayyun98/MiniCPM5-2B-SFT-Draftfit-DFlash2): the DSpark geometry's five
// layers over hidden 2048 with 512-wide dynamic convolutions and the
// 256-rank candidate selector, declared as DFlash2DraftModel.
DFlashDraftLayout denseDFlash2DraftLayout() {
  DFlashDraftLayout layout = denseDraftLayout();
  layout.kind = DraftKind::DFlashTwo;
  layout.dynamicSize = 512;
  layout.contextCore = ops::DraftContextCore::Hidden2048;
  layout.selectorRank = 256;
  layout.markovRank = 0;
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
