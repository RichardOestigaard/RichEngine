#include "model/families/FamilyMakers.hpp"

namespace richengine::model {

// LFM2.5-24B-A2B's released DSpark draft
// (LiquidAI/LFM2.5-24B-A2B-DSpark): the same five-layer interleaved-rotary
// 32x8x64 block over a 3072-wide fused QKV as LFM2.5-2.6B's, block_size 9,
// at the target's 1e7 RoPE base — the layout a declared draft's config or
// packed manifest populates. The family's default stays Null: an assembly
// without draft/ keeps the n-gram predraft.
DFlashDraftLayout lfm2moe24BDSparkDraftLayout() {
  DFlashDraftLayout layout = lfm2DraftLayout();
  layout.targetHiddenSize = Lfm2Moe24BLayout{}.capturedHiddenSize();
  return layout;
}

// LFM2.5-24B-A2B's draft-less default: the Null layout satisfies the
// geometry contract while the n-gram predraft supplies every proposal.
DFlashDraftLayout lfm2moe24BDraftLayout() {
  // The planning paths still size draft workspaces from the layout, so it
  // describes the compiled Q32K8D64 interleaved-rotary head even though
  // nothing encodes.
  const Lfm2Moe24BLayout target;
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 1280;
  layout.qkvSize = 3072;
  layout.attentionSize = 2048;
  layout.intermediateSize = target.intermediateSize;
  layout.attentionHeadDimension = 64;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D64Interleaved;
  layout.ropeInterleaved = 1;
  layout.rmsEpsilon = 1e-5F;
  layout.rotaryTheta = target.rotaryTheta;
  return layout;
}

// The LFM2.5-24B-A2B target is text-only: no vision layout, no vision
// weights.
ModelDescriptor lfm2moe24BDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = Lfm2Moe24BLayout{}.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), Lfm2Moe24BLayout{}, lfm2moe24BDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  // Measured on this target (GGUF Q4_K_M, M5 Pro): the DSpark comb trees
  // lift B1 +55% and B2 +70% (0.19-0.28 acceptance), and the rank-2 leaves
  // land often enough that the MoE union stays cheap; six proposals keep
  // the chain's wins without the tail's dead rows.
  descriptor.tuning.treeVerify = ModelTuning::TreeVerify::On;
  descriptor.tuning.proposalCap = 6;
  return descriptor;
}

} // namespace richengine::model
