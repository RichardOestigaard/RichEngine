#pragma once

// The maxwell_moe target (Maxwell-V1.0-Flash; config model_type
// maxwell_moe): a pure-attention MoE — every layer runs block-sparse
// attention (MBSA) over a fine-grained MoE, the residual is mhcStreams
// fp32 hyper-connected streams and a learned contraction (HCHead) feeds
// the final norm and head. The arch's deviations from the canonical
// TargetLayout fields it already carries:
//  - The MBSA indexer (layout's indexer* fields): one index query head
//    per KV group and one shared index key head, both of
//    indexerHeadDimension values, per-head RMS norms and the main
//    branch's RoPE. Every token keeps its post-RoPE index key in the
//    page's index plane (indexerPerTokenKeys): a row's route scores each
//    token's key against its group's query, takes each block's maximum
//    (indexerBlockTokens tokens) and keeps indexerRemoteBlocks blocks of
//    the eligible band — after the indexerPrefixBlocks prefix blocks and
//    before the indexerLocalBlocks local blocks ending at its own — for
//    an independent route per KV group of at most prefix + remote +
//    local blocks.
//  - The manifold-constrained hyper-connection (mhc* fields) replaces
//    the qwen4exp mixes: one projection of the streams per sub-block
//    maps to mhcMixWidth logits (streams pre-weights, streams post-
//    weights and a streams x streams residual matrix the Sinkhorn
//    passes of mhcSinkhornIterations make doubly stochastic); the
//    pre-weights aggregate the streams into the block's normed input
//    and the post-weights and mixed residual absorb its output.
//  - HCHead (mhcHead): the learned streams-to-one contraction before
//    the final norm and untied lm_head.
//  - A sigmoid router with an F32 per-expert selection bias and an
//    ungated shared expert, which runs as a dense SwiGLU beside the
//    routed block (the MoE block itself routes sharedExpert=false).
// Its weights load from packed files only: no GGUF image plan or
// safetensors loader maps the maxwell_moe tensor set.

#include "model/TargetLayout.hpp"
#include "model/TargetModel.hpp"
#include "model/TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace richengine::model {

// Maxwell-V1.0-Flash's canonical layout. Its packed files carry the
// family's own magics (layer sections in the order MaxwellMoe.cpp reads:
// the input norm, the fused QKV and its per-head norms, the out
// projection, the indexer, the attention mHC, the post-attention norm,
// the dense or routed FFN, the shared expert's, the FFN mHC; head.bin
// the HC head, the final norm and the logits).
[[nodiscard]] constexpr TargetLayout maxwellMoeTargetLayout() noexcept {
  TargetLayout layout;
  layout.arch = "maxwell_moe";
  layout.layerMagic = "MDFM0001";
  layout.headMagic = "MDFM0002";
  layout.ffnKind = FfnKind::SparseMoe;
  layout.attentionQueryStride = 1;
  layout.attentionQkNorm = true;
  // GGUF installs (the maxwell-moe arch) load through the image plan
  // GgufImage's Maxwell schema builds; safetensors installs stay packed.
  layout.ggufArch = "maxwell-moe";
  layout.denseLayers = 3;
  layout.maximumContextTokens = 262144;
  layout.layers = 45;
  layout.hiddenSize = 2048;
  layout.vocabularySize = 200704;
  // qkv 5120 = 32 * 128 queries + 2 * 4 * 128 keys and values.
  layout.packedFullWidth = 5120;
  layout.attentionWidth = 4096;
  layout.attentionQueryHeads = 32;
  layout.attentionKvHeads = 4;
  layout.attentionHeadDimension = 128;
  layout.rotaryPairs = 64;
  layout.rotaryTheta = 20'000'000.0F;
  layout.fullAttentionPeriod = 1;
  layout.stopTokens = {200001, 2};
  layout.attentionCore = kv::Core::Kv4Group8D128;
  layout.intermediateSize = 6144;
  layout.experts = 256;
  layout.expertsPerToken = 12;
  layout.expertIntermediateSize = 512;
  layout.packedExpertWidth = 512;
  layout.sharedIntermediateSize = 1024;
  layout.packedSharedWidth = 1024;
  // The shared expert runs beside the routed block, not through it.
  layout.moeSharedExpert = false;
  layout.rmsEpsilon = 1e-6F;
  // The residual: four fp32 streams of hiddenSize.
  layout.hyperConnections = 4;
  // The MBSA indexer: 4 group heads and one key head of 128, per-token
  // index keys, 128-token blocks, a route of 8 prefix, up to 16 remote
  // and 8 local blocks per KV group.
  layout.indexerHeads = 4;
  layout.indexerHeadDimension = 128;
  layout.indexerBlockTokens = 128;
  layout.indexerRouteGroups = 4;
  layout.indexerPrefixBlocks = 8;
  layout.indexerLocalBlocks = 8;
  layout.indexerRemoteBlocks = 16;
  layout.indexerPerTokenKeys = true;
  // The mHC mapping: streams * streams + 2 * streams logits of a row,
  // the residual matrix Sinkhorn-projected through 20 iterations.
  layout.mhcMixWidth = 24;
  layout.mhcSinkhornIterations = 20;
  layout.mhcHead = true;
  // The MTP draft's capture: the final layer's full stream state, which
  // the captured buffer holds bf16 and streamWidth wide (captureStreams).
  layout.hiddenCaptureLayers = {44};
  layout.captureLayerCount = 1;
  layout.captureStreams = true;
  return layout;
}

// One layer's MBSA indexer: the fp32 group-query [indexerHeads *
// indexerHeadDimension][hidden] and key [indexerHeadDimension][hidden]
// projections and their per-head norms.
struct MaxwellIndexerWeights final {
  ops::Projection query;
  ops::Projection key;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
};

// One hyper-connection module (the attention block's or the FFN's): the
// fp32 mapping [mhcMixWidth][streamWidth], the three F32 gating factors
// (pre, post and residual alphas) and the mhcMixWidth F32 bias.
struct MaxwellMhcWeights final {
  ops::Projection mapping;
  metal::MetalBuffer alphas;
  metal::MetalBuffer bias;
};

// The ungated shared expert of a MoE layer, which runs as an ordinary
// SwiGLU beside the routed block.
struct MaxwellSharedFfn final {
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;
};

// What a maxwell_moe layer carries past the shared TargetLayerWeights
// record (its attention mixer, its norms and its dense or routed FFN):
// the indexer, both mHC modules and the MoE layers' shared expert.
struct MaxwellLayerExtras final {
  MaxwellIndexerWeights indexer;
  MaxwellMhcWeights attentionMhc;
  std::optional<MaxwellSharedFfn> sharedFfn;
  MaxwellMhcWeights ffnMhc;
};

// The model's learned streams-to-one contraction (HCHead): the fp32
// projection [streams][streamWidth], the streams F32 base and the F32
// scale the sigmoid gate applies.
struct MaxwellHcHeadWeights final {
  ops::Projection fn;
  metal::MetalBuffer base;
  metal::MetalBuffer scale;
};

// The maxwell_moe package's weights: the shared record's layers plus,
// in extras, the same layers' indexers, mHC modules and shared experts
// in the same order, and the head's HC contraction.
struct MaxwellMoeWeights final : TargetModelWeights {
  std::vector<MaxwellLayerExtras> extras;
  MaxwellHcHeadWeights hcHead;
};

[[nodiscard]] MaxwellMoeWeights
loadMaxwellMoeWeights(metal::MetalBackend &backend,
                      const TargetLayout &layout, const TargetFiles &files);

// The geometry the layers' and head's projections dispatch, the base's
// plus the extras' (the indexers', the mHC mappings' and the HC head's).
[[nodiscard]] TargetModelGeometry
targetModelGeometry(const MaxwellMoeWeights &weights);

} // namespace richengine::model
