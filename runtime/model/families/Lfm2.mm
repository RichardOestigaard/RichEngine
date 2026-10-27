#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Lfm2.hpp"
#include "model/Lfm2Moe.hpp"

namespace richengine::model {

// Defined in families/MiniCpm.mm: the dense target's DSpark default the
// LFM2 draft layouts derive from (FamilyMakers.hpp is gone — extern here
// instead of a shared header).
DFlashDraftLayout denseDraftLayout();

// The LFM2 targets' packed/source draft default: the released DSpark draft
// (LiquidAI/LFM2.5-2.6B-DSpark, five layers of interleaved-rotary 32x8x64
// over 3072, block_size 9), whose geometry the draft config or the packed
// manifest declares. Not file-local: families/Lfm2Moe24B.mm derives its
// DSpark layout from it and declares it extern there.
DFlashDraftLayout lfm2DraftLayout() {
  DFlashDraftLayout layout = denseDraftLayout();
  const TargetLayout target = lfm2TargetLayout();
  layout.vocabularySize = target.vocabularySize;
  layout.qkvSize = 3072;
  layout.attentionHeadDimension = 64;
  layout.rotaryTheta = 10'000'000.0F;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.kvHeads = 8;
  layout.blockSize = 9;
  layout.ropeInterleaved = 1;
  layout.headCore = ops::DraftHeadCore::Q32K8D64Interleaved;
  layout.contextCore = ops::DraftContextCore::Plain2048;
  layout.rmsEpsilon = 1e-5F;
  return layout;
}

namespace {

using affine::copy;
using affine::embeddingSection;
using affine::image;
using affine::Image;
using affine::lfm2MoeFfn;
using affine::projection;

// The LFM2.5-8B-A1B target's DSpark draft (LiquidAI/LFM2.5-8B-A1B-DSpark):
// the same five-layer 32x8x64 interleaved-rotary block as LFM2.5-2.6B's,
// over a 3072-wide fused QKV, block_size 9 — at the target's 5e6 RoPE base.
DFlashDraftLayout lfm2moeDraftLayout() {
  DFlashDraftLayout layout = lfm2DraftLayout();
  layout.rotaryTheta = 5'000'000.0F;
  layout.targetHiddenSize = lfm2MoeTargetLayout().capturedHiddenSize();
  return layout;
}

ModelDescriptor lfm2Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = lfm2TargetLayout().hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), lfm2TargetLayout(), lfm2DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// The d1 decision models never decode, so no DSpark draft is installed;
// the Null layout satisfies the geometry contract while the n-gram
// predraft supplies proposals.
DFlashDraftLayout d1NullDraftLayout(const TargetLayout &target) {
  DFlashDraftLayout layout;
  layout.kind = DraftKind::Null;
  layout.layers = 1;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 1280;
  layout.qkvSize = 6144;
  layout.attentionSize = 4096;
  layout.intermediateSize = 17408;
  layout.attentionHeadDimension = 128;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.selectorRank = 256;
  layout.kvHeads = 8;
  layout.headCore = ops::DraftHeadCore::Q32K8D128;
  layout.rotaryTheta = 5'000'000.0F;
  return layout;
}

ModelDescriptor d1Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = lfm2TargetLayout().hiddenSize;
  // The d1-3B GGUF declares rope.freq_base 1e6, not LFM2.5's 1e7.
  TargetLayout target = lfm2TargetLayout();
  target.rotaryTheta = 1'000'000.0F;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, d1NullDraftLayout(target), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// d1-omni-600M (LiquidAI/d1-omni-600M-GGUF): the LFM2.5-Encoder-350M
// backbone post-trained for the System One readout — 18 blocks over
// embedding 1024 and vocab 65536 at rope base 1e6: 10 convs, 6 attention
// layers of 16x8 heads of 64, then two decision blocks of 16x16 heads with
// a narrower ungated FFN, finished by token-type embeddings and a cls
// pooling head. Bidirectional: it never decodes. The descriptor records
// the geometry the assembly validates against; loading still refuses —
// the encoder's attention geometry, its full-prompt mask and the decision
// blocks' fused biased layers have no compiled kernels.
ModelDescriptor d1omniDescriptor(std::string name) {
  ops::VisionLayout vision;
  TargetLayout target = lfm2TargetLayout();
  vision.outputHiddenSize = target.hiddenSize = 1024;
  target.maximumContextTokens = 128000;
  target.layers = 18;
  target.vocabularySize = 65536;
  target.packedGdnWidth = 3 * target.hiddenSize;
  // 1024 query rows + 2*512 key/value; the decision blocks' fused QKV is a
  // different tensor the loaders never reach.
  target.packedFullWidth = 2048;
  target.convolutionDimension = 1024;
  target.attentionWidth = 1024;
  target.attentionQueryHeads = 16;
  target.attentionKvHeads = 8;
  target.attentionMask = (uint64_t{1} << 2) | (uint64_t{1} << 5) |
                         (uint64_t{1} << 8) | (uint64_t{1} << 10) |
                         (uint64_t{1} << 12) | (uint64_t{1} << 14) |
                         (uint64_t{1} << 16) | (uint64_t{1} << 17);
  target.rotaryTheta = 1'000'000.0F;
  target.maskToken = 7;
  target.stopTokens = {7, 7};
  target.hiddenCaptureLayers = {2, 5, 10, 14, 17};
  target.intermediateSize = 4608;
  target.encoder = true;
  target.decisionBlocks = 2;
  target.altAttentionKvHeads = 16;
  target.altIntermediateSize = 4096;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, d1NullDraftLayout(target), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

ModelDescriptor lfm2moeDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = lfm2MoeTargetLayout().hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), lfm2MoeTargetLayout(), lfm2moeDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// ---- LFM2 ("lfm2") ----

// The lfm2 source's config check, shared by the d1 variants' targets: the
// conv/full_attention layer schedule plus the encoder decision models'
// per-layer geometry.
void validateLfm2SourceConfig(id textObj,
                              const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", layout.convolutionTaps},
      {"intermediate_size", layout.intermediateSize}});
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
  // An encoder decision model (d1-omni) states encoder plus the
  // per-layer fields its non-uniform geometry needs; a causal target
  // states none of them.
  if (id encoder = text[@"encoder"];
      encoder != nil &&
      ![encoder isEqual:layout.encoder ? @YES : @NO])
    throw std::invalid_argument(
        "target encoder flag does not match the family's");
  if (NSArray *kvs = text[@"layer_kv_heads"];
      [kvs isKindOfClass:[NSArray class]]) {
    requireEqual(kvs.count, layout.layers, "target layer_kv_heads count");
    for (uint32_t layer = 0; layer < layout.layers; ++layer)
      requireNumber(
          kvs[layer],
          layout.isFullAttentionLayer(layer) ? layout.kvHeadsAt(layer) : 0,
          "target layer_kv_heads " + std::to_string(layer));
  }
  if (NSArray *widths = text[@"layer_intermediate_sizes"];
      [widths isKindOfClass:[NSArray class]]) {
    requireEqual(widths.count, layout.layers,
                 "target layer_intermediate_sizes count");
    for (uint32_t layer = 0; layer < layout.layers; ++layer)
      requireNumber(widths[layer], layout.intermediateSizeAt(layer),
                    "target layer_intermediate_sizes " +
                        std::to_string(layer));
  }
}

// The lfm2_moe source's config check: lfm2's schedule plus the MoE block's
// fields.
void validateLfm2MoeSourceConfig(id textObj,
                                 const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize}, {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", layout.convolutionTaps},
      {"intermediate_size", layout.intermediateSize},
      {"num_experts", layout.experts}, {"num_experts_per_tok", layout.expertsPerToken},
      {"moe_intermediate_size", layout.expertIntermediateSize},
      {"num_dense_layers", layout.denseLayers}});
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

// The LFM2 source (MLX layout, "lfm2" model_type): `language_model.model.`
// tensors, conv and full_attention layer types and tied embeddings.
void validateLfm2Configuration(const SafetensorsCheckpoint &source, const TargetLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", layout.convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2");
  source.requireLayerTypeMask(layout.layers, layout.attentionMask, "full_attention", "conv");
}

// LFM2's layer image (language_model.model.layers.N.): the conv layers'
// [B|C|x] in_proj, [channels,1,taps] kernel and out_proj, or the attention
// layers' fused QKV with per-head norms.
Image lfm2LayerImage(const TargetLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", layout.layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
  copy(result, prefix + "operator_norm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows = uint64_t{layout.attentionKvHeads} * layout.attentionHeadDimension;
    projection(result, {{attention + "q_proj", layout.attentionWidth},
                        {attention + "k_proj", kvRows},
                        {attention + "v_proj", kvRows}},
               layout.packedFullWidth, layout.hiddenSize, 4, 1, resolve);
    copy(result, attention + "q_layernorm.weight", {layout.attentionHeadDimension});
    copy(result, attention + "k_layernorm.weight", {layout.attentionHeadDimension});
    projection(result, {{attention + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth,
               4, 1, resolve);
  } else {
    const std::string conv = prefix + "conv.";
    projection(result, {{conv + "in_proj", layout.packedGdnWidth}}, layout.packedGdnWidth, layout.hiddenSize,
               4, 1, resolve);
    copy(result, conv + "conv.weight", {layout.convolutionDimension, 1, layout.convolutionTaps});
    projection(result, {{conv + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.convolutionDimension,
               4, 1, resolve);
  }
  copy(result, prefix + "ffn_norm.weight", {layout.hiddenSize});
  const std::string ffn = prefix + "feed_forward.";
  projection(result, {{ffn + "w1", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
             resolve);
  projection(result, {{ffn + "w3", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
             resolve);
  projection(result, {{ffn + "w2", layout.hiddenSize}}, layout.hiddenSize, layout.intermediateSize, 4, 1,
             resolve);
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image lfm2HeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "language_model.model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve);
  return result;
}

Image lfm2EmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string prefix = "language_model.model.embed_tokens";
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

// The lfm2 affine hook — the d1 variants' targets share it.
std::vector<Image> lfm2Images(const SafetensorsCheckpoint *,
                              const TargetLayout &layout,
                              const QuantResolver &resolve) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(lfm2LayerImage(layout, layer, resolve));
  result.push_back(lfm2HeadImage(layout, resolve));
  result.push_back(lfm2EmbeddingImage(layout, resolve));
  return result;
}

constexpr AffineArchHooks kLfm2Hooks{&lfm2Images, &validateLfm2Configuration};

// ---- LFM2-MoE ("lfm2_moe") ----

// The LFM2.5-8B-A1B source (MLX layout, "lfm2_moe" model_type): LFM2's
// mixers, a dense FFN on the leading num_dense_layers and the MoE block —
// router, per-expert selection bias and expert slabs — after them.
void validateLfm2MoeConfiguration(const SafetensorsCheckpoint &source, const TargetLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", layout.convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2_moe");
  source.requireLayerTypeMask(layout.layers, layout.attentionMask, "full_attention", "conv");
}

// LFM2-MoE's layer image: LFM2's mixers, then the dense FFN below
// denseLayers or the router, bias and expert slabs of its MoE block.
Image lfm2MoeLayerImage(const TargetLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", layout.layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
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
    projection(result, {{ffn + "w1", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
               resolve, false, true);
    projection(result, {{ffn + "w3", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize, 4, 1,
               resolve, false, true);
    projection(result, {{ffn + "w2", layout.hiddenSize}}, layout.hiddenSize, layout.intermediateSize, 4, 1,
               resolve, false, true);
  } else {
    lfm2MoeFfn(result, layout, ffn, ffn + "experts.w1", ffn + "experts.w3", ffn + "experts.w2",
               resolve);
  }
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image lfm2MoeHeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "language_model.model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve, false, true);
  return result;
}

Image lfm2MoeEmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string prefix = "language_model.model.embed_tokens";
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

std::vector<Image> lfm2MoeImages(const SafetensorsCheckpoint *,
                                 const TargetLayout &layout,
                                 const QuantResolver &resolve) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(lfm2MoeLayerImage(layout, layer, resolve));
  result.push_back(lfm2MoeHeadImage(layout, resolve));
  result.push_back(lfm2MoeEmbeddingImage(layout, resolve));
  return result;
}

constexpr AffineArchHooks kLfm2MoeHooks{&lfm2MoeImages, &validateLfm2MoeConfiguration};

// The registry's erased-signature entry for the lfm2 target.
TargetWeights loadLfm2Target(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadLfm2Weights(backend, layout, files);
}

// The registry's erased-signature entry for the lfm2_moe target.
TargetWeights loadLfm2MoeTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadLfm2MoeWeights(backend, layout, files);
}

const bool kRegisteredLfm2Families = [] {
  registerTargetArch({"lfm2", &loadLfm2Target, &kLfm2Hooks});
  registerTargetArch({"lfm2_moe", &loadLfm2MoeTarget, &kLfm2MoeHooks});
  registerModelFamily({.name = "LFM2.5-2.6B", .modelType = "lfm2",
                       .legacyType = "", .makeDescriptor = &lfm2Descriptor,
                       .draftLayout = &lfm2DraftLayout,
                       .validateSourceConfig = &validateLfm2SourceConfig,
                       .dsparkArchName = "Lfm2DSparkDraftModel",
                       .draftNormEpsilon = 1e-5});
  registerModelFamily({.name = "d1-3B", .modelType = "lfm2",
                       .legacyType = "", .makeDescriptor = &d1Descriptor,
                       .validateSourceConfig = &validateLfm2SourceConfig,
                       .dsparkArchName = "Lfm2DSparkDraftModel",
                       .draftNormEpsilon = 1e-5});
  registerModelFamily({.name = "d1-omni", .modelType = "lfm2",
                       .legacyType = "", .makeDescriptor = &d1omniDescriptor,
                       .validateSourceConfig = &validateLfm2SourceConfig,
                       .dsparkArchName = "Lfm2DSparkDraftModel",
                       .draftNormEpsilon = 1e-5});
  registerModelFamily({.name = "LFM2.5-8B-A1B", .modelType = "lfm2_moe",
                       .legacyType = "", .makeDescriptor = &lfm2moeDescriptor,
                       .draftLayout = &lfm2moeDraftLayout,
                       .validateSourceConfig = &validateLfm2MoeSourceConfig,
                       .dsparkArchName = "Lfm2DSparkDraftModel",
                       .draftNormEpsilon = 1e-5});
  return true;
}();

} // namespace
} // namespace richengine::model
