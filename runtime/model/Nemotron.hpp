#pragma once

// The nemotron_h_moe target (NVIDIA Nemotron 3.5 Lightning 30B-A3B): a
// tri-hybrid — every block is RMSNorm then exactly one sub-layer, a
// Mamba-2 mixer, a plain GQA attention block or a MoE block, with no
// post-attention norm and no second sub-layer. The arch's deviations from
// the canonical TargetLayout fields it already carries:
//  - The layer schedule is two masks: attentionMask the six attention
//   blocks, moeMask the 23 MoE blocks; the 23 remaining blocks are the
//   Mamba-2 mixers, which take the state cells the GDN hybrids' gdnLayers
//   take (a conv-input FIFO and a heads x headDim x state fp32 matrix).
//  - The Mamba-2 mixer (mamba* fields): an in_proj of packedGdnWidth
//   storage rows splits [gate mambaInnerSize | conv input of
//   convolutionDimension (x | B | C) | dt mambaTimeStepRank] — the GGUF's
//   10304 rows pad to a storage tile. The conv runs convolutionTaps
//   depthwise taps; the scan keeps mambaHeads scalar decays (ssm_a, the
//   kernels apply -exp), a D skip, mambaGroups shared B/C groups of
//   mambaStateSize values, and normalizes each group's span before the
//   sigmoid gate (norm-before-gate).
//  - The attention blocks are plain NoPE GQA: no q/k norms, no
//   interleaved gate row, no positional embedding at all — HF's
//   NemotronHAttention never applies rotary (position flows through the
//   SSM) and llama.cpp binds LLAMA_ROPE_TYPE_NONE. The GGUF's
//   rope.dimension_count 84 is a converter artifact nothing reads.
//  - The MoE blocks (FfnKind::SparseMoe): a sigmoid router with an F32
//   per-expert selection bias (exp_probs_b), ungated relu-squared
//   experts (up -> relu(x)^2 -> down — no gate slab) and an ungated
//   shared expert of sharedIntermediateSize at unit weight. The 2.5x
//   routed scaling factor (expert_weights_scale) applies in the select's
//   norm_topk_prob weights (MoeRouteParams::weight_scale).
// Its weights load from a nemotron_h_moe GGUF only: no packed file
// schema or safetensors loader maps the tensor set.

#include "model/TargetLayout.hpp"
#include "model/TargetModel.hpp"
#include "model/TargetFiles.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "ops/Linear.hpp"
#include "ops/Mamba2.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace richengine::model {

// Nemotron 3.5 Lightning's canonical layout. Its layer images carry the
// family's magic (the section order Nemotron.cpp reads: the block's
// attn_norm, then the Mamba-2 mixer's ssm_* sections, the attention
// block's attn_* sections or the MoE block's ffn_* ones; head.bin the
// final norm and the untied logits).
[[nodiscard]] constexpr TargetLayout nemotronTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "nemotron";
  layout.ggufArch = "nemotron_h_moe";
  layout.ffnKind = FfnKind::SparseMoe;
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = false;
  layout.maximumContextTokens = 262144;
  layout.layers = 52;
  layout.hiddenSize = 2688;
  layout.vocabularySize = 131072;
  // The attention blocks (5, 12, 19, 26, 33, 42) and the MoE blocks:
  // every block that is neither runs a Mamba-2 mixer.
  layout.attentionMask = (uint64_t{1} << 5) | (uint64_t{1} << 12) |
                         (uint64_t{1} << 19) | (uint64_t{1} << 26) |
                         (uint64_t{1} << 33) | (uint64_t{1} << 42);
  layout.moeMask = (uint64_t{1} << 1) | (uint64_t{1} << 3) |
                   (uint64_t{1} << 6) | (uint64_t{1} << 8) |
                   (uint64_t{1} << 10) | (uint64_t{1} << 13) |
                   (uint64_t{1} << 15) | (uint64_t{1} << 17) |
                   (uint64_t{1} << 20) | (uint64_t{1} << 22) |
                   (uint64_t{1} << 24) | (uint64_t{1} << 27) |
                   (uint64_t{1} << 29) | (uint64_t{1} << 31) |
                   (uint64_t{1} << 34) | (uint64_t{1} << 36) |
                   (uint64_t{1} << 38) | (uint64_t{1} << 40) |
                   (uint64_t{1} << 43) | (uint64_t{1} << 45) |
                   (uint64_t{1} << 47) | (uint64_t{1} << 49) |
                   (uint64_t{1} << 51);
  // qkv 4608 = 32 * 128 queries + 2 * 2 * 128 keys and values. The
  // in_proj's logical 10304 rows (4096 gate + 6144 conv + 64 dt) pad to
  // one storage tile.
  layout.packedFullWidth = 4608;
  layout.packedGdnWidth = 10496;
  layout.convolutionTaps = 4;
  layout.convolutionDimension = 6144;
  layout.mambaHeads = 64;
  layout.mambaHeadDimension = 64;
  layout.mambaGroups = 8;
  layout.mambaStateSize = 128;
  layout.mambaInnerSize = 4096;
  layout.mambaTimeStepRank = 64;
  layout.mambaDtMinimum = 0.001f;
  // HF NemotronHMamba2Mixer clamps dt to (time_step_min, inf) — 0.1 is the
  // dt_bias initialization bound (time_step_max), not a runtime cap.
  layout.mambaDtMaximum = std::numeric_limits<float>::infinity();
  layout.mambaCore = ops::Mamba2Core::Nemotron64;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 32;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 128;
  // NoPE attention: no rotary pairs, no rope tables.
  layout.rotaryPairs = 0;
  layout.stopTokens = {11, 11};
  layout.attentionCore = kv::Core::Kv2Group16D128;
  layout.experts = 128;
  layout.expertsPerToken = 6;
  layout.expertIntermediateSize = 1856;
  layout.packedExpertWidth = 1856;
  layout.sharedIntermediateSize = 3712;
  layout.packedSharedWidth = 3712;
  layout.moeSharedExpert = true;
  layout.moeUngatedExperts = true;
  layout.moeRoutedScale = 2.5F;
  layout.rmsEpsilon = 1e-5F;
  layout.maskToken = 990;
  layout.hiddenCaptureLayers = {51};
  layout.captureLayerCount = 1;
  return layout;
}

// Nemotron 3.5 Bolt 74B-A9B's layout: the same tri-hybrid arch at 88
// blocks — attention at 7, 16, 25, 36, 47, 58, 69, 80, MoE at 40 blocks,
// Mamba-2 elsewhere — with the mixer's doubled inner size (128 heads) and
// the heterogeneous Star Elastic MoE: per-layer expert counts (240..496,
// `experts` their maximum) and expert widths (1536..2560), LatentMoE
// projections into a 1024-wide latent the routed experts read and write,
// and a constant 5376-wide shared expert in the full hidden space.
[[nodiscard]] constexpr TargetLayout boltTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "nemotron";
  layout.ggufArch = "nemotron_h_moe";
  layout.ffnKind = FfnKind::SparseMoe;
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = false;
  layout.maximumContextTokens = 262144;
  layout.layers = 88;
  layout.hiddenSize = 4096;
  layout.vocabularySize = 131072;
  const auto bit = [](uint32_t layer) constexpr {
    return LayerMask{1} << layer;
  };
  layout.attentionMask = bit(7) | bit(16) | bit(25) | bit(36) | bit(47) |
                         bit(58) | bit(69) | bit(78);
  layout.moeMask = bit(1) | bit(3) | bit(5) | bit(8) | bit(10) | bit(12) |
                   bit(14) | bit(17) | bit(19) | bit(21) | bit(23) | bit(26) |
                   bit(28) | bit(30) | bit(32) | bit(34) | bit(37) | bit(39) |
                   bit(41) | bit(43) | bit(45) | bit(48) | bit(50) | bit(52) |
                   bit(54) | bit(56) | bit(59) | bit(61) | bit(63) | bit(65) |
                   bit(67) | bit(70) | bit(72) | bit(74) | bit(76) | bit(79) |
                   bit(81) | bit(83) | bit(85) | bit(87);
  // qkv 4608 = 32 * 128 queries + 2 * 2 * 128 keys and values. The
  // in_proj's logical 18560 rows (8192 gate + 10240 conv + 128 dt) pad to
  // the 18688-row storage tile.
  layout.packedFullWidth = 4608;
  layout.packedGdnWidth = 18688;
  layout.convolutionTaps = 4;
  layout.convolutionDimension = 10240;
  layout.mambaHeads = 128;
  layout.mambaHeadDimension = 64;
  layout.mambaGroups = 8;
  layout.mambaStateSize = 128;
  layout.mambaInnerSize = 8192;
  layout.mambaTimeStepRank = 128;
  layout.mambaDtMinimum = 0.001f;
  layout.mambaDtMaximum = std::numeric_limits<float>::infinity();
  layout.mambaCore = ops::Mamba2Core::Bolt128;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 32;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 0;
  layout.stopTokens = {11, 11};
  layout.attentionCore = kv::Core::Kv2Group16D128;
  // The Star Elastic maximums size the scratch; the GGUF's per-layer
  // tensors carry each block's own count and width.
  layout.experts = 496;
  layout.expertsPerToken = 9;
  layout.expertIntermediateSize = 2560;
  layout.packedExpertWidth = 2560;
  layout.sharedIntermediateSize = 5376;
  layout.packedSharedWidth = 5376;
  layout.moeSharedExpert = true;
  layout.moeUngatedExperts = true;
  layout.moeRoutedScale = 5.0F;
  layout.moeLatentSize = 1024;
  layout.rmsEpsilon = 1e-5F;
  layout.hiddenCaptureLayers = {87};
  layout.captureLayerCount = 1;
  return layout;
}

[[nodiscard]] TargetModelWeights
loadNemotronWeights(metal::MetalBackend &backend, const TargetLayout &layout,
                    const TargetFiles &files);

// The nemotron target's layer encoder (Nemotron.cpp): the one-sub-layer
// blocks — Mamba-2 mixer, GQA attention or ungated relu-squared MoE —
// and the Mamba-2 state commits the verify step's retained rows feed.
[[nodiscard]] std::unique_ptr<TargetLayerEncoder>
makeNemotronLayerEncoder(const TargetModel &model);

} // namespace richengine::model
