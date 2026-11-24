// The maxwell family's registry row: Maxwell-V1.0-Flash (model_type
// maxwell_moe). The target loads from packed files only; its layer
// encoder throws named errors at the mHC and MBSA pieces whose kernels
// richengine does not have yet (model/MaxwellMoe.cpp).
#import <Foundation/Foundation.h>

#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/MaxwellMoe.hpp"
#include "model/Model.hpp"

#include <stdexcept>

namespace richengine::model {
namespace {

// An NSNumber that is a JSON boolean of `expected`.
void requireBoolean(id value, bool expected, std::string_view label) {
  if (![value isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID() ||
      [value boolValue] != static_cast<BOOL>(expected))
    throw std::invalid_argument(std::string(label) +
                                (expected ? " must be true" : " must be false"));
}

// The maxwell_moe source's config check: the geometry and the
// block-sparse, mHC and router settings the released config.json fixes.
void validateMaxwellSourceConfig(id textObj,
                                 const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize},
      {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"attention_head_size", layout.attentionHeadDimension},
      {"intermediate_size", layout.intermediateSize},
      {"moe_intermediate_size", layout.expertIntermediateSize},
      {"shared_expert_intermediate_size", layout.sharedIntermediateSize},
      {"num_experts", layout.experts},
      {"num_experts_per_tok", layout.expertsPerToken},
      {"block_sparse_gqa_indexer_n_heads", layout.indexerHeads},
      {"block_sparse_gqa_indexer_head_dim", layout.indexerHeadDimension},
      {"block_sparse_gqa_block_size", layout.indexerBlockTokens},
      {"block_sparse_gqa_prefix_window_size",
       layout.indexerPrefixBlocks * layout.indexerBlockTokens},
      {"block_sparse_gqa_local_window_size",
       layout.indexerLocalBlocks * layout.indexerBlockTokens},
      {"block_sparse_gqa_num_remote_blocks", layout.indexerRemoteBlocks},
      {"block_sparse_gqa_total_support_blocks",
       layout.indexerPrefixBlocks + layout.indexerRemoteBlocks +
           layout.indexerLocalBlocks},
      {"num_residual_streams", layout.hyperConnections},
      {"mhc_sinkhorn_iterations", layout.mhcSinkhornIterations},
      {"max_position_embeddings", 524288}});
  requireNumber(text[@"rope_theta"], layout.rotaryTheta, "rope_theta");
  requireNumber(text[@"rms_norm_eps"], layout.rmsEpsilon, "rms_norm_eps");
  requireBoolean(text[@"qk_layernorm"], true, "qk_layernorm");
  requireBoolean(text[@"enable_hyper_connections"], true,
                 "enable_hyper_connections");
  requireBoolean(text[@"moe_router_enable_expert_bias"], true,
                 "moe_router_enable_expert_bias");
  requireBoolean(text[@"use_shared_expert_gate"], false,
                 "use_shared_expert_gate");
  requireBoolean(text[@"tie_word_embeddings"], false, "tie_word_embeddings");
  // The released config declares one MTP layer; serving defers its draft
  // (the Null layout below) rather than refuse the package.
  requireNumber(text[@"mtp_num_layers"], 1, "mtp_num_layers");
  id reduction = text[@"block_sparse_gqa_group_reduction"];
  requireEqual([reduction isKindOfClass:[NSString class]]
                   ? static_cast<NSString *>(reduction).UTF8String
                   : "",
               "per_group_max", "block_sparse_gqa_group_reduction");
  id score = text[@"router_score_function"];
  requireEqual([score isKindOfClass:[NSString class]]
                   ? static_cast<NSString *>(score).UTF8String
                   : "",
               "sigmoid", "router_score_function");
  id act = text[@"hidden_act"];
  requireEqual([act isKindOfClass:[NSString class]]
                   ? static_cast<NSString *>(act).UTF8String
                   : "",
               "silu", "hidden_act");
  // Every layer is block-sparse; the leading mlp_only_layers are the
  // dense FFNs.
  NSArray *mlp = requireArray(text, @"mlp_only_layers", "mlp_only_layers");
  requireEqual(mlp.count, uint64_t{layout.denseLayers}, "mlp_only_layers");
  for (NSUInteger index = 0; index < mlp.count; ++index)
    requireNumber(mlp[index], index, "mlp_only_layers entry");
  NSArray *freq = requireArray(text, @"block_sparse_gqa_layer_freq",
                               "block_sparse_gqa_layer_freq");
  requireEqual(freq.count, uint64_t{layout.layers},
               "block_sparse_gqa_layer_freq count");
  for (NSUInteger index = 0; index < freq.count; ++index)
    requireNumber(freq[index], 1, "block_sparse_gqa_layer_freq entry");
}

// The checkpoint's internal MTP layer (blk.45 + nextn.* of the GGUF): one
// hyper-connected decoder layer drafting fifteen chained proposals. Its
// layout fields carry the dense GQA attention, the mHC mix width in
// dynamicSize and the shared expert's intermediate in intermediateSize;
// targetHiddenSize is the full stream capture width.
DFlashDraftLayout maxwellDraftDefaults() {
  const TargetLayout target = maxwellMoeTargetLayout();
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Mtp;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = target.mhcMixWidth;
  layout.qkvSize = target.packedFullWidth;
  layout.attentionSize = target.attentionWidth;
  layout.intermediateSize = target.sharedIntermediateSize;
  layout.selectorRank = 0;
  layout.kvHeads = target.attentionKvHeads;
  layout.attentionHeadDimension = target.attentionHeadDimension;
  layout.rotaryTheta = target.rotaryTheta;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.blockSize = ExecutionLimits::targetVerifyRows;
  layout.rmsEpsilon = target.rmsEpsilon;
  layout.headCore = ops::DraftHeadCore::Unspecified;
  layout.contextCore = ops::DraftContextCore::Unspecified;
  return layout;
}

// Maxwell is text-only: no vision layout, and no vision weights to load.
ModelDescriptor maxwellDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = maxwellMoeTargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), maxwellMoeTargetLayout(),
                          maxwellDraftDefaults(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// The registry's erased-signature entry for the maxwell_moe target.
TargetWeights loadMaxwellMoeTarget(metal::MetalBackend &backend,
                                   const TargetLayout &layout,
                                   const TargetFiles &files) {
  return loadMaxwellMoeWeights(backend, layout, files);
}

const bool kRegisteredMaxwellFamilies = [] {
  registerTargetArch({"maxwell_moe", &loadMaxwellMoeTarget, nullptr,
                      &makeMaxwellLayerEncoder});
  registerModelFamily({.name = "Maxwell-V1.0-Flash",
                       .modelType = "maxwell_moe", .legacyType = "",
                       .makeDescriptor = &maxwellDescriptor,
                       .draftLayout = &maxwellDraftDefaults,
                       .validateSourceConfig = &validateMaxwellSourceConfig});
  return true;
}();

} // namespace
} // namespace richengine::model
