// The qwen4exp family's registry rows: Qwen3.8-Flash-Next (GGUF
// architecture qwen4exp, text model type qwen4_exp_text). The target
// loads from packed files only and its layer encoder throws named errors
// at the pieces whose kernels richengine does not have yet
// (model/Qwen4Exp.cpp).
#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Qwen4Exp.hpp"

namespace richengine::model {
namespace {

// The family's real DFlash2 draft — incoai/Qwen3.8-Flash-Next-DFlash2:
// three sliding-attention layers over hidden 2560, the shared 32x8x128
// head pattern (fused QKV 6144, attention 4096), intermediate 5120,
// selector rank 256, block size 8 and a 640-wide dynamic convolution
// (hidden/4), bound to DraftContextCore::Hidden2560's compiled
// draft_conv_h2560 kernel. Its context projection reads the target
// layers 3, 17, 31 and 47's stream-mean hidden captures — 4 x 2560 =
// 10240 wide — which Qwen4Exp.cpp's capture path already produces.
DFlashDraftLayout qwen4DraftLayout() {
  const TargetLayout target = qwen4ExpTargetLayout();
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DFlashTwo;
  layout.layers = 3;
  layout.hiddenSize = 2560;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 640;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 5120;
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.attentionHeadDimension = 128;
  layout.rotaryTheta = 1'000'000.0F;
  layout.blockSize = 8;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Hidden2560;
  return layout;
}

// The Null fallback for an assembly that ships no draft/config.json: the
// n-gram predraft still proposes. A declared draft/config.json swaps in
// qwen4DraftLayout through acceptsDeclaredDraft.
DFlashDraftLayout qwen4DraftDefaults() {
  const TargetLayout target = qwen4ExpTargetLayout();
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 0;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 5120;
  layout.selectorRank = 0;
  layout.kvHeads = 8;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.contextCore = ops::DraftContextCore::Unspecified;
  return layout;
}

// The Qwen VL tower (the staged family manifest's vision block is the same
// tower the qwen3 families pack), its output the target's hidden.
ops::VisionLayout qwen4VisionLayout(uint32_t outputHiddenSize) {
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

ModelDescriptor qwen4ExpDescriptor(std::string name) {
  return makeModelDescriptor(std::move(name), qwen4ExpTargetLayout(),
                             qwen4DraftDefaults(),
                             qwen4VisionLayout(qwen4ExpTargetLayout().hiddenSize));
}

// The registry's erased-signature entry for the qwen4exp target.
TargetWeights loadQwen4ExpTarget(metal::MetalBackend &backend,
                               const TargetLayout &layout,
                               const TargetFiles &files) {
  return loadQwen4ExpWeights(backend, layout, files);
}

const bool kRegisteredQwen4Families = [] {
  registerTargetArch({"qwen4exp", &loadQwen4ExpTarget, nullptr,
                      &makeQwen4LayerEncoder});
  registerModelFamily({.name = "Qwen3.8-Flash-Next",
                       .modelType = "qwen4_exp_text", .legacyType = "",
                       .makeDescriptor = &qwen4ExpDescriptor,
                       .draftLayout = &qwen4DraftLayout,
                       .refineSourceConfig = &refineQwenSourceText,
                       .acceptsDeclaredDraft = true,
                       .servesVisionTower = true});
  return true;
}();

} // namespace
} // namespace richengine::model
