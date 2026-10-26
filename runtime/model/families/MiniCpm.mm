#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/Dense.hpp"

namespace richengine::model {

// The dense target's packed/source draft default: the released DSpark draft
// (openbmb/MiniCPM5-2B-DSpark, five layers of 16x2x128 heads over a
// 2560-wide fused QKV, block_size 7), whose geometry the draft config or
// the packed manifest declares. Not file-local: families/Lfm2.mm derives
// its draft layouts from it and declares it extern there (FamilyMakers.hpp
// is gone — no shared header).
DFlashDraftLayout denseDraftLayout() {
  const TargetLayout target = denseTargetLayout();
  DFlashDraftLayout layout;
  layout.kind = DraftKind::DSpark;
  layout.layers = 5;
  layout.hiddenSize = target.hiddenSize;
  layout.vocabularySize = target.vocabularySize;
  layout.dynamicSize = 0;
  layout.qkvSize = 2560;
  layout.attentionSize = 2048;
  layout.intermediateSize = 6144;
  layout.attentionHeadDimension = 128;
  layout.rotaryTheta = 5'000'000.0F;
  layout.targetHiddenSize = target.capturedHiddenSize();
  layout.selectorRank = 0;
  layout.kvHeads = 2;
  layout.headCore = ops::DraftHeadCore::Q16K2D128;
  layout.contextCore = ops::DraftContextCore::Plain2048;
  layout.markovRank = 256;
  layout.blockSize = 7;
  return layout;
}

namespace {

using affine::copy;
using affine::embeddingSection;
using affine::image;
using affine::Image;
using affine::projection;

// A DFlash2 draft for the dense target (Draftfit's MiniCPM5-2B port,
// jayyun98/MiniCPM5-2B-SFT-Draftfit-DFlash2): the DSpark geometry's five
// layers over hidden 2048 with 512-wide dynamic convolutions and the
// 256-rank candidate selector, declared as DFlash2DraftModel.
DFlashDraftLayout denseDFlash2DraftLayout() {
  DFlashDraftLayout layout = denseDraftLayout();
  layout.kind = DraftKind::DFlashTwo;
  layout.dynamicSize = 512;
  layout.contextCore = ops::DraftContextCore::Hidden2048;
  layout.selectorRank = 256;
  layout.markovRank = 0;
  return layout;
}

// The dense target is text-only: no vision layout, no vision weights.
ModelDescriptor denseDescriptor(std::string name) {
  ops::VisionLayout vision;
  vision.outputHiddenSize = denseTargetLayout().hiddenSize;
  ModelDescriptor descriptor = makeModelDescriptor(
      std::move(name), denseTargetLayout(), denseDraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  return descriptor;
}

// ---- The dense target's affine plan ----

// The MiniCPM5-2B source (MLX layout, "llama" model_type): flat `model.`
// tensors and a separate lm_head.
void validateDenseConfiguration(const SafetensorsCheckpoint &source, const TargetLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize}, {"head_dim", layout.attentionHeadDimension},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize}, {"rms_norm_eps", 1e-6},
      {"rope_theta", layout.rotaryTheta},
      {"tie_word_embeddings", 0},
      {"max_position_embeddings", layout.maximumContextTokens}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  // The released MiniCPM5-2B-MLX config omits attention_bias (llama's false
  // default); a stated true still fails.
  source.requireConfigNumberOrAbsent("attention_bias", 0);
  source.requireConfigString("model_type", "llama");
  source.requireConfigString("hidden_act", "silu");
}

// The dense target's layer image (model.layers.N.): q/k/v fused, no norms.
Image denseLayerImage(const TargetLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  Image result = image("layer-" + std::to_string(layer) + ".bin", layout.layerMagic, layer, 1u);
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  copy(result, prefix + "input_layernorm.weight", {layout.hiddenSize});
  const std::string attention = prefix + "self_attn.";
  const uint64_t kvRows = uint64_t{layout.attentionKvHeads} * layout.attentionHeadDimension;
  projection(result, {{attention + "q_proj", layout.attentionWidth},
                      {attention + "k_proj", kvRows},
                      {attention + "v_proj", kvRows}},
             layout.packedFullWidth, layout.hiddenSize, 4, 1, resolve);
  projection(result, {{attention + "o_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth,
             4, 1, resolve);
  copy(result, prefix + "post_attention_layernorm.weight", {layout.hiddenSize});
  const std::string mlp = prefix + "mlp.";
  projection(result, {{mlp + "gate_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize,
             4, 1, resolve);
  projection(result, {{mlp + "up_proj", layout.intermediateSize}}, layout.intermediateSize, layout.hiddenSize,
             4, 1, resolve);
  projection(result, {{mlp + "down_proj", layout.hiddenSize}}, layout.hiddenSize, layout.intermediateSize,
             4, 1, resolve);
  return result;
}

Image denseHeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "model.norm.weight", {layout.hiddenSize});
  projection(result, {{"lm_head", layout.vocabularySize}}, layout.vocabularySize, layout.hiddenSize, 4, 1,
             resolve);
  return result;
}

Image denseEmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
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

// The dense target's affine hook: the source's tensors are flat `model.*`
// names no checkpoint check refines.
std::vector<Image> denseImages(const SafetensorsCheckpoint *,
                               const TargetLayout &layout,
                               const QuantResolver &resolve) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(denseLayerImage(layout, layer, resolve));
  result.push_back(denseHeadImage(layout, resolve));
  result.push_back(denseEmbeddingImage(layout, resolve));
  return result;
}

constexpr AffineArchHooks kDenseHooks{&denseImages, &validateDenseConfiguration};

// The registry's erased-signature entry for the dense target.
TargetWeights loadDenseTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadDenseWeights(backend, layout, files, false);
}

const bool kRegisteredDenseFamilies = [] {
  registerTargetArch({"dense", &loadDenseTarget, &kDenseHooks});
  // MiniCPM5's early releases stated model_type "minicpm".
  registerModelFamily({"MiniCPM5-2B", "llama", "minicpm", &denseDescriptor,
                       &denseDFlash2DraftLayout, nullptr});
  return true;
}();

} // namespace
} // namespace richengine::model
