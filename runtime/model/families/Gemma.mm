#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/Gemma4Moe.hpp"

namespace richengine::model {
namespace {

// Gemma 4's default: no draft declared, so the Null layout satisfies the
// geometry contract while the n-gram predraft supplies every proposal. The
// planning paths still size draft workspaces from it, so it describes a
// plausible plain block over the target's hidden; a package or assembly that
// declares a draft swaps in gemma4DFlashV1DraftDefaults' geometry instead.
DFlashDraftLayout gemma4DraftDefaults() {
  const TargetLayout target = gemma4TargetLayout();
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
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Plain2816;
  return layout;
}

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
  vision.outputHiddenSize = gemma4TargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), gemma4TargetLayout(),
                          gemma4DraftDefaults(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// DiffusionGemma shares the Gemma 4 trunk and its text-only scope; the
// packed package's `diffusion` manifest block carries the canvas schedule,
// which validateDiffusionGemma reads into the layout. Unused while the
// family's registration stays commented out below.
[[maybe_unused]] ModelDescriptor diffusionGemmaDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = diffusionGemmaTargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), diffusionGemmaTargetLayout(),
                          gemma4DraftDefaults(), vision);
  descriptor.visionSource = VisionSource::None;
  // The canvas bundle DiffusionGemma runs under; each knob's own
  // RICHENGINE_CANVAS_* variable still wins over the profile.
  descriptor.tuning.canvasProfile = "paper";
  return descriptor;
}

// The registry's erased-signature entry for the gemma4 target.
TargetWeights loadGemma4Target(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadGemma4MoeWeights(backend, layout, files);
}

// The registry's erased-signature entry for the diffusion_gemma target.
TargetWeights loadDiffusionGemmaTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadDiffusionGemmaWeights(backend, layout, files);
}

const bool kRegisteredGemmaFamilies = [] {
  registerTargetArch({"gemma4", &loadGemma4Target});
  registerTargetArch({"diffusion_gemma", &loadDiffusionGemmaTarget});
  registerModelFamily({"Gemma4-26B-A4B", "gemma4_text", "", &gemma4Descriptor,
                       nullptr, &gemma4DFlashV1DraftDefaults});
  // DiffusionGemma-26B-A4B is disabled: its denoising trunk ships
  // incoherent output pending the numerics bisect in
  // docs/diffusion_issue.md. Re-add its row and the packed-format branch
  // in inspectModelPackage once the trunk matches the torch reference.
  // registerModelFamily({"DiffusionGemma-26B-A4B", "diffusion_gemma_text",
  //                      "", &diffusionGemmaDescriptor, nullptr, nullptr});
  return true;
}();

} // namespace
} // namespace richengine::model
