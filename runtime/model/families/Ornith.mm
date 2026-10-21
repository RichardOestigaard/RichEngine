#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

// Ornith 9B declares no DFlash2 draft layout: no compiled context core
// covers hidden 4096/dynamic 1024, so a source assembly or packed manifest
// naming DFlash2DraftModel is refused rather than mis-bound.

// Ornith 1.5 9B's released DFlash draft (ornith-ai/Ornith-1.5-9B-DFlash): a
// plain six-layer transformer over hidden 4096 reading the target's eight
// capture layers — fused fc + hidden_norm features injected as every layer's
// context K/V. Five causal sliding layers, one full-attention layer last.
DFlashDraftLayout ornith9DFlashV1DraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DFlashV1;
  layout.slidingWindow = 4096;
  layout.causalLayers = 0x1F;
  layout.layers = 6;
  layout.hiddenSize = 4096;
  layout.dynamicSize = 0;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 12288;
  layout.targetHiddenSize = 32768;
  layout.selectorRank = 0;
  layout.kvHeads = 8;
  layout.blockSize = 16;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Plain4096;
  return layout;
}

// Ornith is text-only: no vision layout, and no vision weights to load.
ModelDescriptor ornithDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Ornith9BLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), Ornith9BLayout{}, ornith9DFlashV1DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace richengine::model
