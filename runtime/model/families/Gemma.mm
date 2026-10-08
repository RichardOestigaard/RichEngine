#include "model/families/FamilyMakers.hpp"

namespace richengine::model {
namespace {

// Gemma 4's default: no draft declared, so the Null layout satisfies the
// geometry contract while the n-gram predraft supplies every proposal. The
// planning paths still size draft workspaces from it, so it describes a
// plausible plain block over the target's hidden; a package or assembly that
// declares a draft swaps in gemma4DFlashV1DraftDefaults' geometry instead.
DFlashDraftLayout gemma4DraftDefaults() {
  const Gemma4MoeLayout target;
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 0;
  layout.qkvSize = 5120;
  layout.attentionSize = 4096;
  layout.intermediateSize = 8448;
  layout.selectorRank = 0;
  layout.kvHeads = 8;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.logitSoftcap = target.logitSoftcap;
  return layout;
}

} // namespace

// The same layout as the plain-draft default of a packed gemma4 manifest's
// draft declaration (applyDeclaredDraft): the compiled draft attention cores
// span hidden 2816 since DraftAttention.cpp's Plain2816 layout.
DFlashDraftLayout gemma4DFlashV1DraftDefaults() {
  DFlashDraftLayout layout = gemma4DraftDefaults();
  layout.kind = DraftKind::DFlashV1;
  return layout;
}

// The Gemma 4 target is text-only on this runtime: no vision layout.
ModelDescriptor gemma4Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Gemma4MoeLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), Gemma4MoeLayout{}, gemma4DraftDefaults(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// DiffusionGemma shares the Gemma 4 trunk and its text-only scope; the
// packed package's `diffusion` manifest block carries the canvas schedule,
// which validateDiffusionGemma reads into the layout.
ModelDescriptor diffusionGemmaDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = DiffusionGemmaLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), DiffusionGemmaLayout{}, gemma4DraftDefaults(),
      vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

} // namespace richengine::model
