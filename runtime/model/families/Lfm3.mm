// The LFM3 family: the tri-hybrid text target (LFM3-1.7B) and the d1-2B
// decision model's LFM3 text tower. The layer image carries the sandwich
// norms and the unquantized gate_proj beside LFM2's sections.
#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetLayerEncoder.hpp"
#include "model/TargetWeights.hpp"
#include "model/Lfm3.hpp"

namespace richengine::model {

namespace {

using affine::copy;
using affine::embeddingSection;
using affine::image;
using affine::Image;
using affine::projection;

// No draft ships with the LFM3 family; the Null layout satisfies the
// geometry contract while the n-gram predraft supplies proposals.
DFlashDraftLayout lfm3NullDraftLayout(const TargetLayout &target) {
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

ModelDescriptor lfm3Descriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = lfm3TargetLayout().hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), lfm3TargetLayout(), lfm3NullDraftLayout(lfm3TargetLayout()),
      vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// The LFM3 text tower at the narrower 128000-token context: the d1-2B
// decision model (LiquidAI/d1-2B, a lfm3_vl config whose text_config
// states model_type lfm3), a causal decoder run score-only by the
// decision pathway — one forward per judgment set, never decoded; and
// LFM3-Omni-2B's generative text tower, whose GGUF declares the same
// 128000-token context. The vision tower's tensors are outside the
// packed target.
ModelDescriptor d1v3Descriptor(std::string name) {
  ops::VisionLayout vision;
  TargetLayout target = lfm3TargetLayout();
  target.maximumContextTokens = 128000;
  vision.outputHiddenSize = target.hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), target, lfm3NullDraftLayout(target), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// LFM3-Omni-2B: the d1v3 text tower plus the lfm3vl vision tower its
// mmproj carries — fourteen hybrid blocks at hidden 1024, a 2x2 merge and
// the GELU-erf projector into the 2048-wide text hidden.
ModelDescriptor lfm3OmniDescriptor(std::string name) {
  ModelDescriptor descriptor = d1v3Descriptor(std::move(name));
  ops::LfmVisionLayout &v = descriptor.lfmVision;
  v.depth = 14;
  v.hiddenSize = 1024;
  v.patchDimension = 1536;
  v.intermediateSize = 2560;
  v.mergedHiddenSize = 4096;
  v.outputHiddenSize = descriptor.target.hiddenSize;
  v.heads = 16;
  v.kvHeads = 8;
  v.headDimension = 64;
  v.patchSize = 16;
  v.spatialMerge = 2;
  // The same mmproj's lfm3a audio encoder: seventeen hybrid blocks at
  // hidden 768 over mel 128, stride 8 in time, sliding-window-32 causal
  // attention, and the LayerNorm + GELU-erf projector into the text
  // hidden. The GGUF's clip.audio.feed_forward_length (1792) is the
  // intermediate width; the HF config's 2560 belongs to a different module.
  ops::LfmAudioLayout &a = descriptor.lfmAudio;
  a.depth = 17;
  a.hiddenSize = 768;
  a.intermediateSize = 1792;
  a.outputHiddenSize = descriptor.target.hiddenSize;
  a.heads = 8;
  a.kvHeads = 4;
  a.headDimension = 96;
  a.melBins = 128;
  a.convChannels = 256;
  a.window = 32;
  a.tokenStride = 8;
  return descriptor;
}

// The layer_types entry each layer kind declares.
std::string lfm3LayerType(const TargetLayout &layout, uint32_t layer) {
  if (!layout.isFullAttentionLayer(layer)) return "conv";
  return layout.isGlobalAttentionLayer(layer) ? "full_attention"
                                              : "sliding_attention";
}

// The lfm3 source's config check (a d1-2B config's resolved text object
// carries the same fields): geometry plus the tri-hybrid layer_types.
void validateLfm3SourceConfig(id textObj,
                              const ModelDescriptor &descriptor) {
  NSDictionary *text = static_cast<NSDictionary *>(textObj);
  const TargetLayout &layout = descriptor.target;
  requireGeometryNumbers(text, {{"hidden_size", layout.hiddenSize},
      {"num_hidden_layers", layout.layers},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"conv_L_cache", layout.convolutionTaps},
      {"intermediate_size", layout.intermediateSize},
      {"sliding_window", layout.slidingWindowTokens}});
  // Conv channels equal hidden_size; use_sandwich_norm is declared.
  id sandwich = text[@"use_sandwich_norm"];
  if (![sandwich isKindOfClass:[NSNumber class]] ||
      CFGetTypeID((__bridge CFTypeRef)sandwich) != CFBooleanGetTypeID() ||
      ![sandwich boolValue])
    throw std::invalid_argument("target use_sandwich_norm must be true");
  requireAtMost(text, @"max_position_embeddings", layout.maximumContextTokens,
                "max_position_embeddings");
  NSArray *types = requireArray(text, @"layer_types", "target layer_types");
  requireEqual(types.count, layout.layers, "target layer_types count");
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    id value = types[layer];
    if (![value isKindOfClass:[NSString class]])
      throw std::invalid_argument("target layer type must be a string");
    const std::string expected = lfm3LayerType(layout, layer);
    const char *actual = static_cast<NSString *>(value).UTF8String;
    requireEqual(actual ? actual : "", expected,
                 "target layer " + std::to_string(layer));
  }
}

// The LFM3 source (MLX/safetensors layout, "lfm3" model_type):
// `language_model.model.` tensors, the conv/sliding/full layer types and
// tied embeddings.
void validateLfm3Configuration(const SafetensorsCheckpoint &source,
                               const TargetLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_L_cache", layout.convolutionTaps},
      {"sliding_window", layout.slidingWindowTokens},
      {"use_sandwich_norm", 1},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigNumberAtMost("max_position_embeddings",
                                   layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm3");
  // The checkpoint API's mask check covers two layer kinds; the tri-hybrid
  // schedule's per-type check runs in validateLfm3SourceConfig.
}

// LFM3's layer image (language_model.model.layers.N.): the conv layers'
// [B|C|x] in_proj, [channels,1,taps] kernel and out_proj, or the attention
// layers' fused QKV with per-head norms and the unquantized gate_proj —
// then the FFN and the two sandwich norms. Section order is the loader's
// read order (TargetLoader.hpp's positional reads).
Image lfm3LayerImage(const TargetLayout &layout, uint32_t layer,
                     const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin",
                       layout.layerMagic, layer, full ? 1u : 0u);
  const std::string prefix =
      "language_model.model.layers." + std::to_string(layer) + ".";
  copy(result, prefix + "operator_norm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    const uint64_t kvRows =
        uint64_t{layout.attentionKvHeads} * layout.attentionHeadDimension;
    projection(result, {{attention + "q_proj", layout.attentionWidth},
                        {attention + "k_proj", kvRows},
                        {attention + "v_proj", kvRows}},
               layout.packedFullWidth, layout.hiddenSize, 4, 1, resolve);
    copy(result, attention + "q_norm.weight",
         {layout.attentionHeadDimension});
    copy(result, attention + "k_norm.weight",
         {layout.attentionHeadDimension});
    projection(result, {{attention + "o_proj", layout.hiddenSize}},
               layout.hiddenSize, layout.attentionWidth, 4, 1, resolve);
    // gate_proj is 16 rows — too narrow for a Q4 tile on its own: the
    // image pads it to one 256-row tile (the zero rows never reach a
    // kernel).
    projection(result, {{attention + "gate_proj", layout.attentionQueryHeads}},
               layout.attentionGateRows, layout.hiddenSize, 4, 1, resolve,
               /*forceFloat*/ false, /*preferAffine*/ false);
  } else {
    const std::string conv = prefix + "conv.";
    projection(result, {{conv + "in_proj", layout.packedGdnWidth}},
               layout.packedGdnWidth, layout.hiddenSize, 4, 1, resolve);
    copy(result, conv + "conv.weight",
         {layout.convolutionDimension, 1, layout.convolutionTaps});
    projection(result, {{conv + "out_proj", layout.hiddenSize}},
               layout.hiddenSize, layout.convolutionDimension, 4, 1, resolve);
  }
  copy(result, prefix + "ffn_norm.weight", {layout.hiddenSize});
  const std::string ffn = prefix + "feed_forward.";
  projection(result, {{ffn + "w1", layout.intermediateSize}},
             layout.intermediateSize, layout.hiddenSize, 4, 1, resolve);
  projection(result, {{ffn + "w3", layout.intermediateSize}},
             layout.intermediateSize, layout.hiddenSize, 4, 1, resolve);
  projection(result, {{ffn + "w2", layout.hiddenSize}},
             layout.hiddenSize, layout.intermediateSize, 4, 1, resolve);
  copy(result, prefix + "post_operator_norm.weight", {layout.hiddenSize});
  copy(result, prefix + "post_ffn_norm.weight", {layout.hiddenSize});
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image lfm3HeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "language_model.model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve);
  return result;
}

Image lfm3EmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
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

std::vector<Image> lfm3Images(const SafetensorsCheckpoint *,
                              const TargetLayout &layout,
                              const QuantResolver &resolve) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    result.push_back(lfm3LayerImage(layout, layer, resolve));
  result.push_back(lfm3HeadImage(layout, resolve));
  result.push_back(lfm3EmbeddingImage(layout, resolve));
  return result;
}

constexpr AffineArchHooks kLfm3Hooks{&lfm3Images, &validateLfm3Configuration};

// The registry's erased-signature entry for the lfm3 target.
TargetWeights loadLfm3Target(metal::MetalBackend &backend,
                             const TargetLayout &layout,
                             const TargetFiles &files) {
  return loadLfm3Weights(backend, layout, files);
}

const bool kRegisteredLfm3Families = [] {
  registerTargetArch({"lfm3", &loadLfm3Target, &kLfm3Hooks,
                      &makeLfm3LayerEncoder});
  registerModelFamily({.name = "LFM3-1.7B", .modelType = "lfm3",
                       .legacyType = "", .makeDescriptor = &lfm3Descriptor,
                       .validateSourceConfig = &validateLfm3SourceConfig,
                       .draftNormEpsilon = 1e-5, .takesNoDraft = true});
  // Legacy alias: the base checkpoint's retired family name resolves to
  // the LFM3-1.7B descriptor; the installer never offers it.
  registerModelFamily({.name = "LFM3-1.7B-Base", .modelType = "lfm3",
                       .legacyType = "", .makeDescriptor = &lfm3Descriptor,
                       .validateSourceConfig = &validateLfm3SourceConfig,
                       .draftNormEpsilon = 1e-5, .takesNoDraft = true});
  // d1-2B's structured-decision serving rides the score-only pathway; the
  // descriptor is the LFM3 text tower its one forward pass runs. The check
  // reads the text_config's model_type (lfm3), not the wrapper's lfm3_vl.
  registerModelFamily({.name = "d1-2B", .modelType = "lfm3",
                       .legacyType = "", .makeDescriptor = &d1v3Descriptor,
                       .validateSourceConfig = &validateLfm3SourceConfig,
                       .draftNormEpsilon = 1e-5, .takesNoDraft = true});
  // LFM3-Omni-2B's generative text tower: the same tri-hybrid target as
  // LFM3-1.7B at the 128000-token context its GGUF declares, served with
  // the lfm3vl vision tower the mmproj carries (its a.* audio sections
  // stay for the audio loaders).
  registerModelFamily({.name = "LFM3-Omni-2B", .modelType = "lfm3",
                       .legacyType = "", .makeDescriptor = &lfm3OmniDescriptor,
                       .validateSourceConfig = &validateLfm3SourceConfig,
                       .draftNormEpsilon = 1e-5, .takesNoDraft = true,
                       .servesLfmVisionTower = true,
                       .servesAudioTower = true,
                       .servesSpeechTower = true});
  return true;
}();

} // namespace
} // namespace richengine::model
