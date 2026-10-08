#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

// The LFM2 targets' packed/source draft default: the released DSpark draft
// (LiquidAI/LFM2.5-2.6B-DSpark, five layers of interleaved-rotary 32x8x64
// over 3072, block_size 9), whose geometry the draft config or the packed
// manifest declares.
DFlashDraftLayout lfm2DraftLayout() {
  DFlashDraftLayout layout = denseDraftLayout();
  layout.vocabularySize = Lfm2Layout{}.vocabularySize;
  layout.qkvSize = 3072;
  layout.attentionHeadDimension = 64;
  layout.rotaryTheta = 10'000'000.0F;
  layout.targetHiddenSize = Lfm2Layout{}.capturedHiddenSize();
  layout.kvHeads = 8;
  layout.blockSize = 9;
  layout.ropeInterleaved = 1;
  layout.rmsEpsilon = 1e-5F;
  return layout;
}

// The LFM2.5-8B-A1B target's DSpark draft (LiquidAI/LFM2.5-8B-A1B-DSpark):
// the same five-layer 32x8x64 interleaved-rotary block as LFM2.5-2.6B's,
// over a 3072-wide fused QKV, block_size 9 — at the target's 5e6 RoPE base.
DFlashDraftLayout lfm2moeDraftLayout() {
  DFlashDraftLayout layout = lfm2DraftLayout();
  layout.rotaryTheta = 5'000'000.0F;
  layout.targetHiddenSize = Lfm2MoeLayout{}.capturedHiddenSize();
  return layout;
}

ModelDescriptor lfm2Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Lfm2Layout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), Lfm2Layout{}, lfm2DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

ModelDescriptor lfm2moeDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Lfm2MoeLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), Lfm2MoeLayout{}, lfm2moeDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace richengine::model
