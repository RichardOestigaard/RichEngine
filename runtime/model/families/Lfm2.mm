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
  layout.headCore = ops::DraftHeadCore::Q32K8D64Interleaved;
  layout.contextCore = ops::DraftContextCore::Plain2048;
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

// d1-3B is a decision model: it never decodes, so no DSpark draft is
// installed; the Null layout satisfies the geometry contract while the
// n-gram predraft supplies proposals.
static DFlashDraftLayout d1NullDraftLayout() {
  const Lfm2Layout target;
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
  layout.rotaryTheta = 5'000'000.0F;
  return layout;
}

ModelDescriptor d1Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Lfm2Layout{}.hiddenSize;
  // The d1-3B GGUF declares rope.freq_base 1e6, not LFM2.5's 1e7.
  Lfm2Layout target;
  target.rotaryTheta = 1'000'000.0F;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, d1NullDraftLayout(), vision);
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
