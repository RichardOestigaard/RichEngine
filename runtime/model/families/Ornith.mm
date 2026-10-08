#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

// A hypothetical DFlash2 draft for Ornith 9B, kept for source assemblies
// whose draft config declares DFlash2DraftModel: same six layers over
// hidden 4096 and eight captures.
DFlashDraftLayout ornith9DFlash2DraftLayout() {
  DFlashDraftLayout layout;
  layout.layers = 6;
  layout.hiddenSize = 4096;
  layout.dynamicSize = 1024;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 12288;
  layout.targetHiddenSize = 32768;
  layout.kvHeads = 8;
  layout.blockSize = 8;
  return layout;
}

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
