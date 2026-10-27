#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Lfm2Moe24B.hpp"

namespace richengine::model {

// Defined in families/Lfm2.mm: the LFM2 targets' DSpark default the 24B's
// DSpark layout derives from (FamilyMakers.hpp is gone — extern here
// instead of a shared header).
DFlashDraftLayout lfm2DraftLayout();

namespace {

using affine::copy;
using affine::embeddingSection;
using affine::image;
using affine::Image;
using affine::lfm2MoeFfn;
using affine::projection;

// LFM2.5-24B-A2B's released DSpark draft
// (LiquidAI/LFM2.5-24B-A2B-DSpark): the same five-layer interleaved-rotary
// 32x8x64 block over a 3072-wide fused QKV as LFM2.5-2.6B's, block_size 9,
// at the target's 1e7 RoPE base — the layout a declared draft's config or
// packed manifest populates. The family's default stays Null: an assembly
// without draft/ keeps the n-gram predraft.
DFlashDraftLayout lfm2moe24BDSparkDraftLayout() {
  DFlashDraftLayout layout = lfm2DraftLayout();
  layout.targetHiddenSize = lfm2Moe24BTargetLayout().capturedHiddenSize();
  return layout;
}

// LFM2.5-24B-A2B's draft-less default: the Null layout satisfies the
// geometry contract while the n-gram predraft supplies every proposal.
DFlashDraftLayout lfm2moe24BDraftLayout() {
  // The planning paths still size draft workspaces from the layout, so it
  // describes the compiled Q32K8D64 interleaved-rotary head even though
  // nothing encodes.
  const TargetLayout target = lfm2Moe24BTargetLayout();
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
  vision.outputHiddenSize = lfm2Moe24BTargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), lfm2Moe24BTargetLayout(),
                          lfm2moe24BDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  // Measured on this target (GGUF Q4_K_M, M5 Pro): the DSpark comb trees
  // lift B1 +55% and B2 +70% (0.19-0.28 acceptance), and the rank-2 leaves
  // land often enough that the MoE union stays cheap; six proposals keep
  // the chain's wins without the tail's dead rows.
  descriptor.tuning.treeVerify = ModelTuning::TreeVerify::On;
  descriptor.tuning.proposalCap = 6;
  // The 0.15 break-even was measured on a dense 2B draft. On this sparse-MoE
  // target a 16-row verify saturates the 64-expert union — the drafted step
  // costs ~6x an anchor step — so a lane whose acceptance EWMA can't promise
  // at least ~2 tokens decodes anchor-only (exact greedy) instead.
  descriptor.tuning.draftBypassExpect = 2.0;
  return descriptor;
}

// ---- LFM2.5-24B-A2B ("lfm2_moe_24b") ----

// The lfm2_moe_24b source's config check: the 8B's schedule at the 24B
// geometry, conv_dim optional (it is the hidden size upstream).
void validateLfm2Moe24BSourceConfig(id textObj,
                                    const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"conv_L_cache", layout.convolutionTaps},
      {"intermediate_size", layout.intermediateSize},
      {"num_experts", layout.experts}, {"num_experts_per_tok", layout.expertsPerToken},
      {"moe_intermediate_size", layout.expertIntermediateSize},
      {"num_dense_layers", layout.denseLayers}});
  // conv_dim is the hidden size upstream; MLX exports may omit it.
  if (text[@"conv_dim"])
    requireGeometryNumbers(text, {{"conv_dim", layout.convolutionDimension}});
  // Exports disagree on the trained context (131072 vs 128000).
  requireAtMost(text, @"max_position_embeddings", layout.maximumContextTokens,
                "max_position_embeddings");
  // The conv/full_attention layer schedule of layer_types.
  NSArray *types = requireArray(text, @"layer_types", "target layer_types");
  requireEqual(types.count, layout.layers, "target layer_types count");
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    id value = types[layer];
    if (![value isKindOfClass:[NSString class]])
      throw std::invalid_argument("target layer type must be a string");
    const std::string expected =
        layout.isFullAttentionLayer(layer) ? "full_attention" : "conv";
    const char *actual = static_cast<NSString *>(value).UTF8String;
    requireEqual(actual ? actual : "", expected,
                 "target layer " + std::to_string(layer));
  }
}

// The LFM2.5-24B-A2B source (MLX layout, "lfm2_moe" model_type): the 8B's
// validation at the 24B geometry — 40 layers, dense intermediate 11776, 64
// experts of 1536, RoPE base 1e7.
void validateLfm2Moe24BConfiguration(const SafetensorsCheckpoint &source, const TargetLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_L_cache", layout.convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  // conv_dim is the hidden size upstream; MLX exports may omit it.
  source.requireConfigNumberOrAbsent("conv_dim", layout.convolutionDimension);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2_moe");
  source.requireLayerTypeMask(layout.layers, layout.attentionMask, "full_attention", "conv");
}

// The 24B's layer image: the 8B's section order and tensor names at the
// 24B's geometry and its own image magic — the dense projections keep the
// gate/up/down names, the experts the switch_mlp ones.
Image lfm2Moe24BLayerImage(const TargetLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", layout.layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  copy(result, prefix + "operator_norm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows = uint64_t{layout.attentionKvHeads} * layout.attentionHeadDimension;
    projection(result, {{attention + "q_proj", layout.attentionWidth},
                        {attention + "k_proj", kvRows},
                        {attention + "v_proj", kvRows}},
               layout.packedFullWidth, layout.hiddenSize, 4, 1, resolve, false, true);
    copy(result, attention + "q_layernorm.weight", {layout.attentionHeadDimension});
    copy(result, attention + "k_layernorm.weight", {layout.attentionHeadDimension});
    projection(result, {{attention + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth,
               4, 1, resolve, false, true);
  } else {
    const std::string conv = prefix + "conv.";
    projection(result, {{conv + "in_proj", layout.packedGdnWidth}}, layout.packedGdnWidth, layout.hiddenSize,
               4, 1, resolve, false, true);
    copy(result, conv + "conv.weight", {layout.convolutionDimension, 1, layout.convolutionTaps});
    projection(result, {{conv + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.convolutionDimension,
               4, 1, resolve, false, true);
  }
  copy(result, prefix + "ffn_norm.weight", {layout.hiddenSize});
  const std::string ffn = prefix + "feed_forward.";
  if (!layout.isMoeLayer(layer)) {
    projection(result, {{ffn + "gate_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
               resolve, false, true);
    projection(result, {{ffn + "up_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
               resolve, false, true);
    projection(result, {{ffn + "down_proj", layout.hiddenSize}}, layout.hiddenSize, layout.intermediateSize, 4, 1,
               resolve, false, true);
  } else {
    lfm2MoeFfn(result, layout, ffn, ffn + "switch_mlp.gate_proj", ffn + "switch_mlp.up_proj",
               ffn + "switch_mlp.down_proj", resolve);
  }
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image lfm2Moe24BHeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve, false, true);
  return result;
}

Image lfm2Moe24BEmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string prefix = "model.embed_tokens";
  if (resolve) {
    embeddingSection(result, prefix, layout.vocabularySize, layout.hiddenSize, resolve);
    return result;
  }
  result.quantized.emplace_back(prefix, 4);
  copy(result, prefix + ".weight", {layout.vocabularySize, layout.hiddenSize / 8}, "U32");
  copy(result, prefix + ".scales", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  copy(result, prefix + ".biases", {layout.vocabularySize, layout.hiddenSize / kQ4GroupElements});
  return result;
}

std::vector<Image> lfm2Moe24BImages(const SafetensorsCheckpoint *,
                                    const TargetLayout &layout,
                                    const QuantResolver &resolve) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(lfm2Moe24BLayerImage(layout, layer, resolve));
  result.push_back(lfm2Moe24BHeadImage(layout, resolve));
  result.push_back(lfm2Moe24BEmbeddingImage(layout, resolve));
  return result;
}

constexpr AffineArchHooks kLfm2Moe24BHooks{&lfm2Moe24BImages,
                                         &validateLfm2Moe24BConfiguration};

// The registry's erased-signature entry for the lfm2_moe_24b target.
TargetWeights loadLfm2Moe24BTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadLfm2Moe24BWeights(backend, layout, files);
}

const bool kRegisteredLfm2Moe24BFamilies = [] {
  registerTargetArch({"lfm2_moe_24b", &loadLfm2Moe24BTarget,
                      &kLfm2Moe24BHooks});
  registerModelFamily({.name = "LFM2.5-24B-A2B", .modelType = "lfm2_moe",
                       .legacyType = "",
                       .makeDescriptor = &lfm2moe24BDescriptor,
                       .draftLayout = &lfm2moe24BDSparkDraftLayout,
                       .validateSourceConfig = &validateLfm2Moe24BSourceConfig,
                       .dsparkArchName = "Lfm2DSparkDraftModel",
                       .draftNormEpsilon = 1e-5,
                       .acceptsDeclaredDraft = true});
  return true;
}();

} // namespace
} // namespace richengine::model
