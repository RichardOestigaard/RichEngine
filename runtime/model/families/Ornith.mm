#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Ornith9B.hpp"

namespace richengine::model {
namespace {

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

// The 9B's DFlash2 draft, when a checkpoint lands (dev/distill converts the
// V1 checkpoint's trunk with no-op convs and a zeroed selector): six layers
// over hidden 4096 with 1024-wide dynamic convolutions and the 256-rank
// candidate selector, geometry pinned by the bootstrap fixture.
DFlashDraftLayout ornith9DFlash2DraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DFlashTwo;
  layout.slidingWindow = 2048;
  layout.layers = 6;
  layout.hiddenSize = 4096;
  layout.dynamicSize = 1024;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 12288;
  layout.targetHiddenSize = 32768;
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.blockSize = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Hidden4096;
  return layout;
}

// Ornith is text-only: no vision layout, and no vision weights to load.
ModelDescriptor ornithDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = ornith9TargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), ornith9TargetLayout(),
                          ornith9DFlashV1DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  // GPU masks and forced spans: the host mask round trip stalls the engine
  // outright while a constrained ticket waits, which starves every lane of
  // a tool-call burst. The family's serving block declares the same mode so
  // the server ships the DFA and skips the pushed spans.
  descriptor.tuning.dfaMode = "full";
  return descriptor;
}

// The registry's erased-signature entry for the ornith9 target.
TargetWeights loadOrnith9Target(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadOrnith9BWeights(backend, layout, files);
}

const bool kRegisteredOrnithFamilies = [] {
  registerTargetArch({"ornith9", &loadOrnith9Target});
  registerModelFamily({.name = "Ornith-1.5-9B", .modelType = "qwen3_5_text",
                       .legacyType = "",
                       .makeDescriptor = &ornithDescriptor,
                       .draftLayout = &ornith9DFlash2DraftLayout,
                       .dflashV1Layout = &ornith9DFlashV1DraftLayout,
                       .refineSourceConfig = &refineQwenSourceText});
  return true;
}();

} // namespace
} // namespace richengine::model
