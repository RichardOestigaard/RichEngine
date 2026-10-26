#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"

namespace richengine::model {
namespace {

DFlashDraftLayout qwen36DraftLayout() {
  DFlashDraftLayout layout;
  layout.layers = 6;
  layout.hiddenSize = 2048;
  layout.dynamicSize = 512;
  layout.intermediateSize = 6144;
  layout.targetHiddenSize = 16384;
  layout.blockSize = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Hidden2048;
  // The pool selector stays off: on the decode benchmark its walk never
  // rescues a rejected position (0/60 target picks at the pool argmax) while
  // the 128-wide edge table costs ~7% of decode throughput re-reading
  // successor codebook rows. RICHENGINE_DFLASH_POOL=1 re-enables it.
  return layout;
}

// Ornith 1.5 35B-A3B's released DFlash draft (ornith-ai/Ornith-1.5-35B-A3B-DFlash):
// the same plain structure over hidden 2048 (attention stays 4096 wide).
DFlashDraftLayout qwen36DFlashV1DraftLayout() {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DFlashV1;
  layout.slidingWindow = 4096;
  layout.causalLayers = 0x1F;
  layout.layers = 6;
  layout.hiddenSize = 2048;
  layout.dynamicSize = 0;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 6144;
  layout.targetHiddenSize = 16384;
  layout.selectorRank = 0;
  layout.kvHeads = 8;
  layout.blockSize = 16;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Plain2048;
  return layout;
}

// The Qwen3.8-27B DFlash2 draft: the default layout's compiled
// Hidden5120/32x8x128 cores over five layers — the signature's scalars
// spelled out so a geometry drift fails loudly instead of silently
// picking another family's defaults.
DFlashDraftLayout qwen38DraftLayout() {
  DFlashDraftLayout layout;
  layout.layers = 5;
  layout.hiddenSize = 5120;
  layout.intermediateSize = 17408;
  layout.targetHiddenSize = 25600;
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.blockSize = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Hidden5120;
  return layout;
}

// The Qwen VL tower both families pack: declared in full — no defaults.
ops::VisionLayout qwenVisionLayout(uint32_t outputHiddenSize) {
  ops::VisionLayout vision;
  vision.depth = 27;
  vision.hiddenSize = 1152;
  vision.patchDimension = 1536;
  vision.intermediateSize = 4304;
  vision.paddedIntermediateSize = 4352;
  vision.mergedHiddenSize = 4608;
  vision.outputHiddenSize = outputHiddenSize;
  vision.heads = 16;
  vision.headDimension = 72;
  vision.positionGridSide = 48;
  vision.patchSize = 16;
  vision.spatialMerge = 2;
  return vision;
}

ModelDescriptor qwen38Descriptor(std::string name) {
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), qwen38TargetLayout(), qwen38DraftLayout(),
      qwenVisionLayout(qwen38TargetLayout().hiddenSize));
  // Adaptive proposal budgets shrink the offered chain toward the
  // acceptance EWMA's +2 margin. On this target a saved verify row costs
  // nearly nothing against a 14.5 GiB weight pass — the draft forward and
  // the verify's fixed work dominate the step — so the shrunk budget only
  // loses tail tokens: measured ~10 tok/s slower at B1 on M5 Pro.
  // RICHENGINE_ADAPTIVE_PROPOSALS=1 restores it.
  descriptor.tuning.adaptiveProposals = false;
  return descriptor;
}

ModelDescriptor qwen36Descriptor(std::string name) {
  const TargetLayout target = qwen36MoeTargetLayout();
  return makeModelDescriptor(std::move(name), target, qwen36DraftLayout(),
                             qwenVisionLayout(target.hiddenSize));
}

// What was kModelFamilies' qwen3_5 rows: the 27B (plus its Bonsai legacy
// alias) and both 35B-A3B family names.
// The registry's erased-signature entry for the qwen3_8 target.
TargetWeights loadQwen38Target(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

// The registry's erased-signature entry for the qwen3_6moe target.
TargetWeights loadQwen36MoeTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

const bool kRegisteredQwenFamilies = [] {
  registerTargetArch({"qwen3_8", &loadQwen38Target});
  registerTargetArch({"qwen3_6moe", &loadQwen36MoeTarget});
  registerModelFamily({"Qwen3.8-27B", "qwen3_5_text", "", &qwen38Descriptor,
                       &qwen38DraftLayout, nullptr});
  // Legacy alias: packages assembled while Bonsai-2-27B was its own
  // family name still resolve — it is the Qwen3.8 architecture and the
  // installer no longer offers it. The parity test exempts this row.
  registerModelFamily({"Bonsai-2-27B", "qwen3_5_text", "", &qwen38Descriptor,
                       &qwen38DraftLayout, nullptr});
  registerModelFamily({"Qwen3.6-35B-A3B", "qwen3_5_moe_text", "",
                       &qwen36Descriptor, &qwen36DraftLayout,
                       &qwen36DFlashV1DraftLayout});
  registerModelFamily({"Ornith-1.5-35B-A3B", "qwen3_5_moe_text", "",
                       &qwen36Descriptor, &qwen36DraftLayout,
                       &qwen36DFlashV1DraftLayout});
  return true;
}();

} // namespace
} // namespace richengine::model
