// The nemotron family's registry row: NVIDIA Nemotron 3.5 Lightning
// 30B-A3B (config model_type nemotron_h; the GGUF declares arch
// nemotron_h_moe, which the target's TargetLayout::arch "nemotron" maps).
// The target loads from a nemotron_h_moe GGUF only; the n-gram predraft
// supplies every proposal (NVIDIA's DFlash/DSpark drafts are not wired
// yet).
#import <Foundation/Foundation.h>

#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Nemotron.hpp"
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

// Nemotron ships no wired GPU draft; the Null layout satisfies the
// geometry contract while the n-gram predraft supplies every proposal.
DFlashDraftLayout nemotronNullDraftLayout(const TargetLayout &target) {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.targetHiddenSize = target.capturedHiddenSize();
  // DraftStateLayout::valid() wants a nonzero ring shape even though a
  // Null draft never allocates one.
  layout.kvHeads = 1;
  layout.attentionHeadDimension = 1;
  layout.slidingWindow = 1;
  // contextKvSize() is qkvSize - attentionSize: the defaults' 6144 - 4096
  // would plan context K/V scratch for a draft that never runs.
  layout.qkvSize = 0;
  layout.attentionSize = 0;
  return layout;
}

// The nemotron source's config check: the geometry and the mamba/MoE
// settings the released config.json fixes.
void validateNemotronSourceConfig(id textObj,
                                  const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize},
      {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"head_dim", layout.attentionHeadDimension},
      {"mamba_num_heads", layout.mambaHeads},
      {"mamba_head_dim", layout.mambaHeadDimension},
      {"mamba_groups", layout.mambaGroups},
      {"ssm_state_size", layout.mambaStateSize},
      {"ssm_inner_size", layout.mambaInnerSize},
      {"time_step_rank", layout.mambaTimeStepRank},
      {"conv_kernel", layout.convolutionTaps},
      {"num_experts", layout.experts},
      {"num_experts_per_tok", layout.expertsPerToken},
      {"moe_intermediate_size", layout.expertIntermediateSize},
      {"shared_expert_intermediate_size",
       layout.sharedIntermediateSize},
      {"max_position_embeddings", layout.maximumContextTokens}});
  requireNumber(text[@"rms_norm_eps"], 1e-5, "rms_norm_eps");
  requireNumber(text[@"expert_weights_scale"], layout.moeRoutedScale,
                "expert_weights_scale");
  id act = text[@"hidden_act"];
  requireEqual([act isKindOfClass:[NSString class]]
                   ? static_cast<NSString *>(act).UTF8String
                   : "",
               "relu2", "hidden_act");
  requireBoolean(text[@"tie_word_embeddings"], false,
                 "tie_word_embeddings");
  requireBoolean(text[@"expert_weights_norm"], true,
                 "expert_weights_norm");
  // The block schedule the masks fix: layer_types names each block
  // "mamba", "moe" or "attention".
  NSArray *kinds =
      requireArray(text, @"layer_types", "layer_types");
  requireEqual(kinds.count, uint64_t{layout.layers},
               "layer_types count");
  for (NSUInteger index = 0; index < kinds.count; ++index) {
    id kind = kinds[index];
    const char *name =
        [kind isKindOfClass:[NSString class]]
            ? static_cast<NSString *>(kind).UTF8String
            : "";
    const bool attention = layout.isFullAttentionLayer(index);
    const bool moe = layout.isMoeLayer(index);
    requireEqual(
        std::string_view(name),
        attention ? std::string_view("attention")
                  : moe ? std::string_view("moe")
                        : std::string_view("mamba"),
        "layer_types entry");
  }
}

// Nemotron is text-only: no vision layout, and no vision weights to load.
ModelDescriptor nemotronDescriptorFor(std::string name,
                                      const TargetLayout &layout) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = layout.hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), layout,
                          nemotronNullDraftLayout(layout),
                          vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

ModelDescriptor nemotronDescriptor(std::string name) {
  return nemotronDescriptorFor(std::move(name), nemotronTargetLayout());
}

// Bolt's descriptor: the 88-block layout — its discrimination from
// Lightning is the family the install's signature match names, which the
// per-layer GGUF metadata (88 blocks, moe_latent_size) then verifies.
ModelDescriptor boltDescriptor(std::string name) {
  return nemotronDescriptorFor(std::move(name), boltTargetLayout());
}

// The registry's erased-signature entry for the nemotron target.
TargetWeights loadNemotronTarget(metal::MetalBackend &backend,
                                 const TargetLayout &layout,
                                 const TargetFiles &files) {
  return loadNemotronWeights(backend, layout, files);
}

const bool kRegisteredNemotronFamilies = [] {
  registerTargetArch({"nemotron", &loadNemotronTarget, nullptr,
                      &makeNemotronLayerEncoder});
  registerModelFamily(
      {.name = "Nemotron-3.5-Lightning-30B-A3B",
       .modelType = "nemotron_h_moe_text", .legacyType = "nemotron_h",
       .makeDescriptor = &nemotronDescriptor,
       .validateSourceConfig = &validateNemotronSourceConfig,
       .acceptsDeclaredDraft = true});
  registerModelFamily(
      {.name = "Nemotron-3.5-Bolt-74B-A9B",
       .modelType = "nemotron_h_moe_text", .legacyType = "nemotron_h",
       .makeDescriptor = &boltDescriptor,
       .validateSourceConfig = &validateNemotronSourceConfig,
       .acceptsDeclaredDraft = true});
  return true;
}();

} // namespace
} // namespace richengine::model
