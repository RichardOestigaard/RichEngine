#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/DescriptorChecks.hpp"
#include "model/ModelRegistry.hpp"
#include "model/TargetLoader.hpp"
#include "model/TargetWeights.hpp"
#include "model/MiniCpmV47.hpp"

namespace richengine::model {
namespace {

using affine::append;
using affine::copy;
using affine::embeddingSection;
using affine::fusedWindow;
using affine::image;
using affine::Image;
using affine::projection;
using affine::Section;
using affine::SectionKind;

// The vision tower is a two-stage SigLIP merger this runtime's vision
// pipeline does not serve: text and draft only.
ModelDescriptor minicpmV47Descriptor(std::string name) {
  // The MiniCPM-V SigLIP tower is not served; only the descriptor's
  // vision/draft hidden-size contract needs the output width.
  ops::VisionLayout vision;
  vision.outputHiddenSize = minicpmV47MoeTargetLayout().hiddenSize;
  ModelDescriptor descriptor =
      makeModelDescriptor(std::move(name), minicpmV47MoeTargetLayout(),
                          minicpmV47DraftLayout(), vision);
  descriptor.visionSource = VisionSource::None;
  // Measured on the Q4_K_M install (decode bench, widths 1-4, M5 Pro):
  // 0.5 prunes ~35% of drafted rows at identical output for +5-10% decode
  // tokens/s; 0.7 prunes deeper and loses it back.
  descriptor.tuning.draftConfMin = 0.5;
  return descriptor;
}

// ---- MiniCPM-V 4.7 35B-A3B ----
// The checkpoint keeps the language model under a base prefix (HF's
// model.language_model., an MLX conversion's language_model.model.) and
// stores each layer's routed gate and up projections fused: the experts'
// [256, 1024, hidden] gate_up_proj carries gate rows [0, 512) then up rows
// [512, 1024) of every expert. Each exported projection section reads its
// window through the part's expertStride/expertOffset.

Image minicpmV47LayerImage(const TargetLayout &layout, const std::string &base,
                           uint32_t layer, const QuantResolver &resolve,
                           bool fusedExperts) {
  // A SparseMoe target: the affine-tile projections win where the oQ block
  // decoders measured slowest.
  const bool denseAffine = true;
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin",
                       layout.layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = base + "layers." + std::to_string(layer) + ".";
  copy(result, prefix + "input_layernorm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    projection(result, {{attention + "q_proj", 2 * layout.attentionWidth},
                                {attention + "k_proj", layout.attentionKvHeads * layout.attentionHeadDimension},
                                {attention + "v_proj", layout.attentionKvHeads * layout.attentionHeadDimension}},
               layout.packedFullWidth, layout.hiddenSize, 4, 1, resolve, false, denseAffine);
    copy(result, attention + "q_norm.weight", {layout.attentionHeadDimension});
    copy(result, attention + "k_norm.weight", {layout.attentionHeadDimension});
    projection(result, {{attention + "o_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth,
               4, 1, resolve, false, denseAffine);
  } else {
    const std::string gdn = prefix + "linear_attn.";
    projection(result, {{gdn + "in_proj_qkv", layout.convolutionDimension},
                                {gdn + "in_proj_z", layout.attentionWidth},
                                {gdn + "in_proj_b", layout.gdnValueHeads},
                                {gdn + "in_proj_a", layout.gdnValueHeads}},
               layout.packedGdnWidth, layout.hiddenSize, 4, 1, resolve, false, denseAffine);
    copy(result, gdn + "conv1d.weight", {layout.convolutionDimension, layout.convolutionTaps, 1});
    Section decay;
    decay.kind = SectionKind::Decay;
    decay.input = {gdn + "A_log", {"BF16", "F32"}, {layout.gdnValueHeads}};
    decay.bytes = uint64_t(layout.gdnValueHeads) * sizeof(float);
    append(result, std::move(decay));
    copy(result, gdn + "dt_bias", {layout.gdnValueHeads});
    copy(result, gdn + "norm.weight", {layout.gdnHeadDimension});
    projection(result, {{gdn + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth,
               4, 1, resolve, false, denseAffine);
  }
  copy(result, prefix + "post_attention_layernorm.weight", {layout.hiddenSize});
  const std::string mlp = prefix + "mlp.";
  projection(result, {{mlp + "gate", layout.experts}}, layout.experts, layout.hiddenSize, 8, 1, resolve,
             true);
  const uint32_t width = layout.expertIntermediateSize;
  if (fusedExperts) {
    // HF's fused block: each expert's gate rows lead its up rows in the
    // per-expert stride of the fused slab's 2 * expertIntermediateSize rows.
    const std::string gateUp = mlp + "experts.gate_up_proj";
    const uint32_t fusedGateUpRows = 2 * width;
    fusedWindow(result, gateUp, width, layout.hiddenSize, 4, layout.experts,
                fusedGateUpRows, 0, resolve);
    fusedWindow(result, gateUp, width, layout.hiddenSize, 4, layout.experts,
                fusedGateUpRows, width, resolve);
    projection(result, {{mlp + "experts.down_proj", layout.hiddenSize}}, layout.hiddenSize, width, 4,
               layout.experts, resolve, false);
  } else {
    // The MLX conversions' switch_mlp schema: one expert slab per
    // projection, as Qwen3.6's own checkpoints ship them.
    const std::string experts = mlp + "switch_mlp.";
    projection(result, {{experts + "gate_proj", width}}, width, layout.hiddenSize, 4,
               layout.experts, resolve, false);
    projection(result, {{experts + "up_proj", width}}, width, layout.hiddenSize, 4,
               layout.experts, resolve, false);
    projection(result, {{experts + "down_proj", layout.hiddenSize}}, layout.hiddenSize, width, 4,
               layout.experts, resolve, false);
  }
  const std::string shared = mlp + "shared_expert.";
  projection(result, {{shared + "gate_proj", width}}, width, layout.hiddenSize, 4, 1, resolve, false);
  projection(result, {{shared + "up_proj", width}}, width, layout.hiddenSize, 4, 1, resolve, false);
  projection(result, {{shared + "down_proj", layout.hiddenSize}}, layout.hiddenSize, width, 4, 1,
             resolve, false);
  projection(result, {{mlp + "shared_expert_gate", 1}}, kQ4StorageN, layout.hiddenSize, 8, 1, resolve,
             true);
  return result;
}

Image minicpmV47HeadImage(const TargetLayout &layout, const std::string &base,
                          const std::string &head, const QuantResolver &resolve) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, base + "norm.weight", {layout.hiddenSize});
  projection(result, {{head, layout.vocabularySize}},
             paddedVocabularySize(layout.vocabularySize), layout.hiddenSize, 4, 1, resolve, false,
             true);
  return result;
}

Image minicpmV47EmbeddingImage(const TargetLayout &layout, const std::string &base,
                               const QuantResolver &resolve) {
  Image result = image("embedding.bin", kEmbeddingMagic, layout.vocabularySize, layout.hiddenSize);
  const std::string prefix = base + "embed_tokens";
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

// The checkpoint's tensor names: HF ships model.language_model.* with the
// routed experts' gate and up fused in experts.gate_up_proj; the MLX
// conversions keep language_model.model.* and the switch_mlp.* split
// experts (the head is language_model.lm_head or lm_head).
struct MinicpmV47Names {
  std::string base;
  std::string head;
  bool fusedExperts = false;
};

MinicpmV47Names minicpmV47Names(const SafetensorsCheckpoint &source) {
  MinicpmV47Names result;
  if (source.find("model.language_model.embed_tokens.weight")) {
    result.base = "model.language_model.";
    result.head = "lm_head";
  } else {
    result.base = "language_model.model.";
    result.head = source.find("language_model.lm_head.weight") ? "language_model.lm_head" : "lm_head";
  }
  result.fusedExperts =
      static_cast<bool>(source.find(result.base + "layers.0.mlp.experts.gate_up_proj.weight"));
  return result;
}

std::vector<Image> minicpmV47Images(const TargetLayout &layout, const MinicpmV47Names &names,
                                    const QuantResolver &resolve = {}) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    result.push_back(
        minicpmV47LayerImage(layout, names.base, layer, resolve, names.fusedExperts));
  result.push_back(minicpmV47HeadImage(layout, names.base, names.head, resolve));
  result.push_back(minicpmV47EmbeddingImage(layout, names.base, resolve));
  return result;
}

// The minicpm_v47_moe affine hook: the image plan is built from the
// checkpoint's own tensor names, which vary with the conversion.
std::vector<Image> minicpmV47PlanImages(const SafetensorsCheckpoint *source,
                                        const TargetLayout &layout,
                                        const QuantResolver &resolve) {
  // Byte budgeting only (a null source): the split-expert sections carry
  // the same bytes as the fused ones, so the default MLX names suffice.
  return minicpmV47Images(layout,
                          source ? minicpmV47Names(*source)
                                 : MinicpmV47Names{"language_model.model.", "lm_head", false},
                          resolve);
}

constexpr AffineArchHooks kMiniCpmV47Hooks{&minicpmV47PlanImages};

// The registry's erased-signature entry for the minicpm_v47_moe target.
TargetWeights loadMinicpmV47MoeTarget(metal::MetalBackend &backend,
                     const TargetLayout &layout,
                     const TargetFiles &files) {
  return loadMiniCpmV47MoeWeights(backend, layout, files);
}

const bool kRegisteredMiniCpmVFamilies = [] {
  registerTargetArch({"minicpm_v47_moe", &loadMinicpmV47MoeTarget,
                      &kMiniCpmV47Hooks});
  registerModelFamily({.name = "MiniCPM-V-4.7-35B-A3B",
                       .modelType = "qwen3_5_moe_text", .legacyType = "",
                       .makeDescriptor = &minicpmV47Descriptor,
                       .draftLayout = &minicpmV47DraftLayout,
                       .refineSourceConfig = &refineQwenSourceText});
  return true;
}();

} // namespace
} // namespace richengine::model
