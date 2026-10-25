#include "model/AffineTarget.hpp"
#include "model/AffineToBlock.hpp"
#include "model/Granite.hpp"
#include "Checked.hpp"
#include "model/AffinePlan.hpp"
#include "model/Dense.hpp"
#include "model/Lfm2.hpp"
#include "model/Lfm2Moe.hpp"
#include "model/Lfm2Moe24B.hpp"
#include "model/MiniCpmV47.hpp"
#include "model/OqWrite.hpp"
#include "model/Ornith9B.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"
#include <fstream>
#include <sstream>

namespace richengine::model {
namespace {

using affine::append;
using affine::copy;
using affine::image;
using affine::Image;
using affine::ProjectionPart;
using affine::QuantSpec;
using affine::Section;
using affine::SectionKind;

// Projection parts, stacked in row order, padded with zero rows to `rows`.
// With a resolver the section is an Oq one: each part takes the spec the
// resolver resolves — the `bits` parameter is today's fallback when it is
// empty — and a part the resolver leaves unquantized keeps its plain float
// weight (an F32 float segment).
void projection(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts,
                uint32_t rows, uint32_t columns, uint32_t bits = 4, uint32_t experts = 1,
                const QuantResolver &resolve = {}, bool forceFloat = false,
                bool preferAffine = false) {
  validateQ4Layout(rows, columns);
  const bool oq = static_cast<bool>(resolve);
  Section section;
  section.kind = oq ? SectionKind::Oq : SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = bits;
  section.experts = experts;
  uint64_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    const std::optional<affine::QuantSpec> spec =
        oq ? resolve(name) : std::optional<affine::QuantSpec>(affine::QuantSpec{bits});
    image.quantized.emplace_back(name, spec);
    ProjectionPart part{count, {}};
    if (spec && oq && forceFloat) {
      // A quantized module the reader wants as F32 anyway (the MoE router
      // and scalar gate): the writer dequantizes its codes.
      part.dequantize = true;
      part.bits = spec->bits;
      part.groupSize = spec->groupSize;
      for (size_t field = 0; field < 3; ++field) {
        std::vector<uint64_t> shape{count,
                                    field ? columns / spec->groupSize : columns * spec->bits / 32};
        part.fields.push_back(
            {name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
             field == 0 ? std::vector<std::string>{"U32"}
                        : std::vector<std::string>{"BF16", "F16"},
             std::move(shape)});
      }
    } else if (spec) {
      part.bits = spec->bits;
      part.groupSize = spec->groupSize;
      for (size_t field = 0; field < 3; ++field) { // weight, scales, biases
        std::vector<uint64_t> shape{count,
                                    field ? columns / spec->groupSize : columns * spec->bits / 32};
        if (experts > 1) shape.insert(shape.begin(), experts);
        // oQ sources may store the meta as F16; the affine path is BF16.
        part.fields.push_back(
            {name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
             field == 0   ? std::vector<std::string>{"U32"}
             : oq         ? std::vector<std::string>{"BF16", "F16"}
                          : std::vector<std::string>{"BF16"},
             std::move(shape)});
      }
    } else {
      part.unquantized = true;
      std::vector<uint64_t> shape{count, columns};
      if (experts > 1) shape.insert(shape.begin(), experts);
      part.fields.push_back({name + ".weight", {"BF16", "F16", "F32"}, std::move(shape)});
    }
    sourceRows += count;
    section.parts.push_back(std::move(part));
  }
  if (!experts || sourceRows > rows || rows - sourceRows >= kQ4StorageN)
    throw WeightStoreError("invalid affine projection padding");
  // preferAffine marks a projection the oQ reader takes through
  // OqTargetFormat::projection/fused — not a section readQuantizedSegment
  // reads (routers, expert slabs, biases). When every part's spec is the
  // affine tile's own (4-bit over 64) the section writes as OqAffine: the
  // descriptor still identifies it, but the Affine64 kernels run it.
  const bool affine = preferAffine && oq && experts == 1 && !forceFloat &&
                      std::all_of(section.parts.begin(), section.parts.end(),
                                  [](const ProjectionPart &part) {
                                    return !part.unquantized && !part.dequantize &&
                                           !part.nativeRows && part.bits == 4 &&
                                           part.groupSize == 64;
                                  });
  if (affine) section.kind = SectionKind::OqAffine;
  const uint64_t affineBytes = checkedMultiply<WeightStoreError>(
      uint64_t(rows) * columns * bits / 8 + uint64_t(rows) * columns / 16,
      experts, "affine projection");
  section.bytes = section.kind == SectionKind::OqAffine
                      ? kWeightFileAlignment + affineBytes
                      : oq ? affine::oq::sectionBytes(section) : affineBytes;
  if (oq) image.type |= kBlockWeightsImageType;
  append(image, std::move(section));
}

// The token table's Oq section: a descriptor and the native rows of its
// spec, or the widened F32 rows when the checkpoint leaves it unquantized.
void embeddingSection(Image &image, const std::string &prefix, uint32_t rows, uint32_t columns,
                      const QuantResolver &resolve) {
  const std::optional<affine::QuantSpec> spec = resolve(prefix);
  image.quantized.emplace_back(prefix, spec);
  Section section;
  section.kind = SectionKind::Oq;
  section.rows = rows;
  section.columns = columns;
  ProjectionPart part{rows, {}};
  part.nativeRows = true;
  if (spec) {
    part.bits = spec->bits;
    part.groupSize = spec->groupSize;
    part.fields.push_back({prefix + ".weight", {"U32"}, {rows, columns * spec->bits / 32}});
    part.fields.push_back({prefix + ".scales", {"BF16", "F16"}, {rows, columns / spec->groupSize}});
    part.fields.push_back({prefix + ".biases", {"BF16", "F16"}, {rows, columns / spec->groupSize}});
  } else {
    part.unquantized = true;
    part.fields.push_back({prefix + ".weight", {"BF16", "F16", "F32"}, {rows, columns}});
  }
  section.parts.push_back(std::move(part));
  section.bytes = affine::oq::sectionBytes(section);
  append(image, std::move(section));
}

template<class Layout>
Image layerImage(const Layout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  // The affine-tile projections win where the oQ block decoders measured
  // slowest — the SparseMoe layouts — and lose on the dense ones.
  constexpr bool denseAffine = Layout::ffnKind == FfnKind::SparseMoe;
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", Layout::layerMagic, layer, full ? 1u : 0u);
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
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
    copy(result, gdn + "conv1d.weight", {layout.convolutionDimension, kGdnConvolutionTaps, 1});
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
  const auto ffn = [&](const std::string &name, uint32_t intermediate, uint32_t experts = 1,
                      bool dense = false) {
    for (const std::string projectionName : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = projectionName == "down_proj";
      const uint32_t n = down ? layout.hiddenSize : intermediate;
      const uint32_t k = down ? intermediate : layout.hiddenSize;
      projection(result, {{name + projectionName, n}}, n, k, 4, experts, resolve,
                 false, dense);
    }
  };
  if constexpr (Layout::ffnKind == FfnKind::SparseMoe) {
    // The router and the shared-expert scalar gate: the block MoE reads
    // them F32, so an oQ plan emits float segments — widened when the source
    // keeps them float, dequantized when it quantizes them.
    projection(result, {{mlp + "gate", layout.experts}}, layout.experts, layout.hiddenSize, 8, 1, resolve,
               true);
    ffn(mlp + "switch_mlp.", layout.expertIntermediateSize, layout.experts);
    ffn(mlp + "shared_expert.", layout.expertIntermediateSize);
    projection(result, {{mlp + "shared_expert_gate", 1}}, kQ4StorageN, layout.hiddenSize, 8, 1, resolve,
               true);
  } else {
    ffn(mlp, layout.intermediateSize);
  }
  return result;
}

template<class Layout>
Image headImage(const Layout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", Layout::headMagic, layout.layers, 2);
  copy(result, "language_model.model.norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.lm_head", layout.vocabularySize}},
             paddedVocabularySize(layout.vocabularySize), layout.hiddenSize, 4, 1, resolve, false,
             Layout::ffnKind == FfnKind::SparseMoe);
  return result;
}

// The token rows as stored: 4-bit codes, scales and biases; an oQ plan's
// native rows instead.
template<class Layout>
Image embeddingImage(const Layout &layout, const QuantResolver &resolve = {}) {
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

template<class Layout>
std::vector<Image> images(const Layout &layout, const QuantResolver &resolve = {}) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer, resolve));
  result.push_back(headImage(layout, resolve));
  result.push_back(embeddingImage(layout, resolve));
  return result;
}

// ---- MiniCPM-V 4.7 35B-A3B ----
// The checkpoint keeps the language model under a base prefix (HF's
// model.language_model., an MLX conversion's language_model.model.) and
// stores each layer's routed gate and up projections fused: the experts'
// [256, 1024, hidden] gate_up_proj carries gate rows [0, 512) then up rows
// [512, 1024) of every expert. Each exported projection section reads its
// window through the part's expertStride/expertOffset.

// Defined below; MiniCPM's oQ plan resolves against the same spec matrix.
QuantResolver oqResolver(const SafetensorsCheckpoint &source);

// One fused window of `name`: the source block of `stride` rows per expert
// contributes `rows` starting at `offset`.
void fusedWindow(Image &image, const std::string &name, uint32_t rows, uint32_t columns,
                 uint32_t bits, uint32_t experts, uint32_t stride, uint32_t offset,
                 const QuantResolver &resolve) {
  validateQ4Layout(rows, columns);
  const bool oq = static_cast<bool>(resolve);
  const std::optional<affine::QuantSpec> spec =
      oq ? resolve(name) : std::optional<affine::QuantSpec>(affine::QuantSpec{bits});
  image.quantized.emplace_back(name, spec);
  Section section;
  section.kind = oq ? SectionKind::Oq : SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = bits;
  section.experts = experts;
  ProjectionPart part{rows, {}};
  part.expertStride = stride;
  part.expertOffset = offset;
  if (spec) {
    part.bits = spec->bits;
    part.groupSize = spec->groupSize;
    for (size_t field = 0; field < 3; ++field) { // weight, scales, biases
      part.fields.push_back(
          {name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
           field == 0 ? std::vector<std::string>{"U32"}
                      : oq ? std::vector<std::string>{"BF16", "F16"}
                           : std::vector<std::string>{"BF16"},
           {experts, stride,
            field ? columns / spec->groupSize : columns * spec->bits / 32}});
    }
  } else {
    part.unquantized = true;
    part.fields.push_back(
        {name + ".weight", {"BF16", "F16", "F32"}, {experts, stride, columns}});
  }
  section.parts.push_back(std::move(part));
  const uint64_t affineBytes = checkedMultiply<WeightStoreError>(
      uint64_t(rows) * columns * bits / 8 + uint64_t(rows) * columns / 16,
      experts, "affine projection");
  section.bytes = oq ? affine::oq::sectionBytes(section) : affineBytes;
  if (oq) image.type |= kBlockWeightsImageType;
  append(image, std::move(section));
}

Image minicpmV47LayerImage(const MiniCpmV47MoeLayout &layout, const std::string &base,
                           uint32_t layer, const QuantResolver &resolve,
                           bool fusedExperts) {
  // A SparseMoe target: the affine-tile projections win where the oQ block
  // decoders measured slowest.
  constexpr bool denseAffine = true;
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin",
                       MiniCpmV47MoeLayout::layerMagic, layer, full ? 1u : 0u);
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
    copy(result, gdn + "conv1d.weight", {layout.convolutionDimension, kGdnConvolutionTaps, 1});
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
    // per-expert stride of `fusedGateUpRows`.
    const std::string gateUp = mlp + "experts.gate_up_proj";
    fusedWindow(result, gateUp, width, layout.hiddenSize, 4, layout.experts,
                layout.fusedGateUpRows, 0, resolve);
    fusedWindow(result, gateUp, width, layout.hiddenSize, 4, layout.experts,
                layout.fusedGateUpRows, width, resolve);
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

Image minicpmV47HeadImage(const MiniCpmV47MoeLayout &layout, const std::string &base,
                          const std::string &head, const QuantResolver &resolve) {
  Image result = image("head.bin", MiniCpmV47MoeLayout::headMagic, layout.layers, 2);
  copy(result, base + "norm.weight", {layout.hiddenSize});
  projection(result, {{head, layout.vocabularySize}},
             paddedVocabularySize(layout.vocabularySize), layout.hiddenSize, 4, 1, resolve, false,
             true);
  return result;
}

Image minicpmV47EmbeddingImage(const MiniCpmV47MoeLayout &layout, const std::string &base,
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

std::vector<Image> minicpmV47Images(const MiniCpmV47MoeLayout &layout, const MinicpmV47Names &names,
                                    const QuantResolver &resolve = {}) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    result.push_back(
        minicpmV47LayerImage(layout, names.base, layer, resolve, names.fusedExperts));
  result.push_back(minicpmV47HeadImage(layout, names.base, names.head, resolve));
  result.push_back(minicpmV47EmbeddingImage(layout, names.base, resolve));
  return result;
}

std::shared_ptr<affine::PlannedCheckpoint> planMinicpmV47(const std::filesystem::path &directory,
                                                        const MiniCpmV47MoeLayout &layout,
                                                        const QuantResolver &resolve = {}) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  planned->images = minicpmV47Images(layout, minicpmV47Names(planned->source), resolve);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

std::shared_ptr<affine::PlannedCheckpoint> planMinicpmV47Oq(const std::filesystem::path &directory,
                                                          const MiniCpmV47MoeLayout &layout) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  planned->images = minicpmV47Images(layout, minicpmV47Names(planned->source),
                                     oqResolver(planned->source));
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

// The MiniCPM5-2B source (MLX layout, "llama" model_type): flat `model.`
// tensors and a separate lm_head.
void validateConfiguration(const SafetensorsCheckpoint &source, const DenseLayout &layout) {
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

// The LFM2 source (MLX layout, "lfm2" model_type): `language_model.model.`
// tensors, conv and full_attention layer types and tied embeddings.
void validateConfiguration(const SafetensorsCheckpoint &source, const Lfm2Layout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", Lfm2Layout::convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2");
  source.requireLayerTypeMask(layout.layers, layout.attentionMask, "full_attention", "conv");
}

// The dense target's layer image (model.layers.N.): q/k/v fused, no norms.
Image layerImage(const DenseLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  Image result = image("layer-" + std::to_string(layer) + ".bin", DenseLayout::layerMagic, layer, 1u);
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

Image headImage(const DenseLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", DenseLayout::headMagic, layout.layers, 2);
  copy(result, "model.norm.weight", {layout.hiddenSize});
  projection(result, {{"lm_head", layout.vocabularySize}}, layout.vocabularySize, layout.hiddenSize, 4, 1,
             resolve);
  return result;
}

Image embeddingImage(const DenseLayout &layout, const QuantResolver &resolve = {}) {
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

// LFM2's layer image (language_model.model.layers.N.): the conv layers'
// [B|C|x] in_proj, [channels,1,taps] kernel and out_proj, or the attention
// layers' fused QKV with per-head norms.
Image layerImage(const Lfm2Layout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", Lfm2Layout::layerMagic, layer, full ? 1u : 0u);
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
    copy(result, conv + "conv.weight", {layout.convolutionDimension, 1, Lfm2Layout::convolutionTaps});
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

// The LFM2.5-8B-A1B source (MLX layout, "lfm2_moe" model_type): LFM2's
// mixers, a dense FFN on the leading num_dense_layers and the MoE block —
// router, per-expert selection bias and expert slabs — after them.
void validateConfiguration(const SafetensorsCheckpoint &source, const Lfm2MoeLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_dim", layout.convolutionDimension}, {"conv_L_cache", Lfm2MoeLayout::convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2_moe");
  source.requireLayerTypeMask(layout.layers, Lfm2MoeLayout::attentionMask, "full_attention", "conv");
}

// LFM2-MoE's layer image: LFM2's mixers, then the dense FFN below
// denseLayers or the router, bias and expert slabs of its MoE block.
Image layerImage(const Lfm2MoeLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", Lfm2MoeLayout::layerMagic, layer, full ? 1u : 0u);
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
    copy(result, conv + "conv.weight", {layout.convolutionDimension, 1, Lfm2MoeLayout::convolutionTaps});
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
    // The router's rows: the block MoE reads it F32, so an oQ plan emits a
    // float segment — widened or dequantized as the source holds it.
    projection(result, {{ffn + "gate", layout.experts}}, kQ4StorageN, layout.hiddenSize, 8, 1, resolve,
               true);
    // The sigmoid gate's per-expert selection bias, F32 (exp_probs_b): a
    // bare copy for the affine reader, a one-row float segment for the
    // oQ one.
    if (resolve) {
      Section bias;
      bias.kind = SectionKind::Oq;
      bias.rows = 1;
      bias.columns = layout.experts;
      ProjectionPart part{1, {}};
      part.unquantized = true;
      part.fields.push_back({ffn + "expert_bias", {"F32", "BF16", "F16"}, {layout.experts}});
      bias.parts.push_back(std::move(part));
      bias.bytes = affine::oq::sectionBytes(bias);
      append(result, std::move(bias));
    } else {
      Section bias;
      bias.input = {ffn + "expert_bias", {"F32"}, {layout.experts}};
      bias.bytes = uint64_t{layout.experts} * sizeof(float);
      append(result, std::move(bias));
    }
    const uint32_t width = layout.expertIntermediateSize;
    projection(result, {{ffn + "experts.w1", width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
    projection(result, {{ffn + "experts.w3", width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
    projection(result, {{ffn + "experts.w2", layout.hiddenSize}}, layout.hiddenSize, width, 4, layout.experts,
               resolve);
  }
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image headImage(const Lfm2MoeLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", Lfm2MoeLayout::headMagic, layout.layers, 2);
  copy(result, "language_model.model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve, false, true);
  return result;
}

Image embeddingImage(const Lfm2MoeLayout &layout, const QuantResolver &resolve = {}) {
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

// The LFM2.5-24B-A2B source (MLX layout, "lfm2_moe" model_type): the 8B's
// validation at the 24B geometry — 40 layers, dense intermediate 11776, 64
// experts of 1536, RoPE base 1e7.
void validateConfiguration(const SafetensorsCheckpoint &source, const Lfm2Moe24BLayout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize},
      {"num_attention_heads", layout.attentionQueryHeads},
      {"num_key_value_heads", layout.attentionKvHeads},
      {"intermediate_size", layout.intermediateSize},
      {"conv_L_cache", Lfm2Moe24BLayout::convolutionTaps},
      {"norm_eps", 1e-5}, {"tie_word_embeddings", 1},
      {"rope_parameters.rope_theta", layout.rotaryTheta}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  // conv_dim is the hidden size upstream; MLX exports may omit it.
  source.requireConfigNumberOrAbsent("conv_dim", layout.convolutionDimension);
  source.requireConfigNumberAtMost("max_position_embeddings", layout.maximumContextTokens);
  source.requireConfigString("model_type", "lfm2_moe");
  source.requireLayerTypeMask(layout.layers, Lfm2Moe24BLayout::attentionMask, "full_attention", "conv");
}

// The 24B's layer image: the 8B's section order and tensor names at the
// 24B's geometry and its own image magic.
Image layerImage(const Lfm2Moe24BLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", Lfm2Moe24BLayout::layerMagic, layer, full ? 1u : 0u);
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
    copy(result, conv + "conv.weight", {layout.convolutionDimension, 1, Lfm2Moe24BLayout::convolutionTaps});
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
    // The router's rows: the block MoE reads it F32, so an oQ plan emits a
    // float segment — widened or dequantized as the source holds it.
    projection(result, {{ffn + "gate", layout.experts}}, kQ4StorageN, layout.hiddenSize, 8, 1, resolve,
               true);
    // The sigmoid gate's per-expert selection bias, F32 (exp_probs_b): a
    // bare copy for the affine reader, a one-row float segment for the
    // oQ one.
    if (resolve) {
      Section bias;
      bias.kind = SectionKind::Oq;
      bias.rows = 1;
      bias.columns = layout.experts;
      ProjectionPart part{1, {}};
      part.unquantized = true;
      part.fields.push_back({ffn + "expert_bias", {"F32", "BF16", "F16"}, {layout.experts}});
      bias.parts.push_back(std::move(part));
      bias.bytes = affine::oq::sectionBytes(bias);
      append(result, std::move(bias));
    } else {
      Section bias;
      bias.input = {ffn + "expert_bias", {"F32"}, {layout.experts}};
      bias.bytes = uint64_t{layout.experts} * sizeof(float);
      append(result, std::move(bias));
    }
    const uint32_t width = layout.expertIntermediateSize;
    projection(result, {{ffn + "switch_mlp.gate_proj", width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
    projection(result, {{ffn + "switch_mlp.up_proj", width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
    projection(result, {{ffn + "switch_mlp.down_proj", layout.hiddenSize}}, layout.hiddenSize, width, 4, layout.experts,
               resolve);
  }
  return result;
}

// The tied LM head shares the token embedding's quantized rows.
Image headImage(const Lfm2Moe24BLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", Lfm2Moe24BLayout::headMagic, layout.layers, 2);
  copy(result, "model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve, false, true);
  return result;
}

Image embeddingImage(const Lfm2Moe24BLayout &layout, const QuantResolver &resolve = {}) {
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

// The tied LM head shares the token embedding's quantized rows.
Image headImage(const Lfm2Layout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", Lfm2Layout::headMagic, layout.layers, 2);
  copy(result, "language_model.model.embedding_norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.model.embed_tokens", layout.vocabularySize}},
             layout.vocabularySize, layout.hiddenSize, 4, 1, resolve);
  return result;
}

Image embeddingImage(const Lfm2Layout &layout, const QuantResolver &resolve = {}) {
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

template<class Layout>
std::shared_ptr<affine::PlannedCheckpoint> planImages(const std::filesystem::path &directory,
                                                    const Layout &layout) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  validateConfiguration(planned->source, layout);
  std::vector<Image> list;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) list.push_back(layerImage(layout, layer));
  list.push_back(headImage(layout));
  list.push_back(embeddingImage(layout));
  planned->images = std::move(list);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

// The checkpoint at directory with every image of layout bound to it. Its
// config.json is the model's, which inspectModelPackage checks.
template<class Layout>
std::shared_ptr<affine::PlannedCheckpoint> plan(const std::filesystem::path &directory, const Layout &layout) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  planned->images = images(layout);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

// The oQ resolver of a checkpoint: the spec the configuration declares for
// a module the tensors mark quantized, nullopt for a plain float weight;
// the oQ formats are the affine-mode ones of 4..8 bits over groups of 64 or
// 128.
QuantResolver oqResolver(const SafetensorsCheckpoint &source) {
  return [&source](std::string_view module) -> std::optional<QuantSpec> {
    const std::optional<QuantSpec> spec = source.quantization(module);
    if (!spec) return std::nullopt;
    if ((spec->bits != 4 && spec->bits != 5 && spec->bits != 6 && spec->bits != 8) ||
        (spec->groupSize != 64 && spec->groupSize != 128) ||
        (!spec->mode.empty() && spec->mode != "affine"))
      throw WeightStoreError("unsupported oQ quantization for " + std::string(module) + ": " +
                             std::to_string(spec->bits) + " bits, group " +
                             std::to_string(spec->groupSize) + ", " + spec->mode);
    return spec;
  };
}

// plan with the oQ resolver: the layouts without a validateConfiguration.
template<class Layout>
std::shared_ptr<affine::PlannedCheckpoint> planOq(const std::filesystem::path &directory, const Layout &layout) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  const QuantResolver resolve = oqResolver(planned->source);
  planned->images = images(layout, resolve);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

// planImages with the oQ resolver.
template<class Layout>
std::shared_ptr<affine::PlannedCheckpoint> planOqImages(const std::filesystem::path &directory,
                                                      const Layout &layout) {
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  validateConfiguration(planned->source, layout);
  const QuantResolver resolve = oqResolver(planned->source);
  std::vector<Image> list;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) list.push_back(layerImage(layout, layer, resolve));
  list.push_back(headImage(layout, resolve));
  list.push_back(embeddingImage(layout, resolve));
  planned->images = std::move(list);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

uint64_t plannedImageBytes(const std::shared_ptr<affine::PlannedCheckpoint> &planned) {
  uint64_t total = 0;
  for (const Image &image : planned->images) total += image.bytes;
  return total;
}

} // namespace

std::vector<Image> affineTargetImages(const Qwen3_8Layout &layout) { return images(layout); }
std::vector<Image> affineTargetImages(const Ornith9BLayout &layout) { return images(layout); }
std::vector<Image> affineTargetImages(const Qwen3_6MoeLayout &layout) { return images(layout); }
std::vector<Image> affineTargetImages(const MiniCpmV47MoeLayout &layout) {
  // Byte budgeting only: the split-expert sections carry the same bytes as
  // the fused ones, so the default MLX names suffice.
  return minicpmV47Images(layout, {"language_model.model.", "lm_head", false});
}
std::vector<Image> affineTargetImages(const DenseLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(headImage(layout));
  result.push_back(embeddingImage(layout));
  return result;
}
std::vector<Image> affineTargetImages(const Lfm2Layout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(headImage(layout));
  result.push_back(embeddingImage(layout));
  return result;
}
std::vector<Image> affineTargetImages(const Lfm2MoeLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(headImage(layout));
  result.push_back(embeddingImage(layout));
  return result;
}
std::vector<Image> affineTargetImages(const Lfm2Moe24BLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(headImage(layout));
  result.push_back(embeddingImage(layout));
  return result;
}

std::vector<Image> affineTargetImages(const GraniteLayout &) {
  throw std::invalid_argument("Granite targets load from packed or GGUF sources only");
}

AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Qwen3_8Layout &layout)
    : images_(images), directory_(directory), planned_(plan(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Ornith9BLayout &layout)
    : images_(images), directory_(directory), planned_(plan(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Qwen3_6MoeLayout &layout)
    : images_(images), directory_(directory), planned_(plan(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const MiniCpmV47MoeLayout &layout)
    : images_(images), directory_(directory), planned_(planMinicpmV47(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const DenseLayout &layout, bool preferBlockWeights)
    : images_(images), directory_(directory), planned_(planImages(directory, layout)) {
  // The sections keep their bound inputs; only their bytes and writer change.
  if (preferBlockWeights) planned_->images = affine::blockImages(std::move(planned_->images));
}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Lfm2Layout &layout)
    : images_(images), directory_(directory), planned_(planImages(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Lfm2MoeLayout &layout)
    : images_(images), directory_(directory), planned_(planImages(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const Lfm2Moe24BLayout &layout)
    : images_(images), directory_(directory), planned_(planImages(directory, layout)) {}
AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &,
                                       const GraniteLayout &)
    : images_(images) {
  throw std::invalid_argument("Granite targets load from packed or GGUF sources only");
}

std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Qwen3_8Layout &layout) {
  return planOq(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Ornith9BLayout &layout) {
  return planOq(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Qwen3_6MoeLayout &layout) {
  return planOq(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const MiniCpmV47MoeLayout &layout) {
  return planMinicpmV47Oq(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const DenseLayout &layout) {
  return planOqImages(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Lfm2Layout &layout) {
  return planOqImages(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Lfm2MoeLayout &layout) {
  return planOqImages(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const Lfm2Moe24BLayout &layout) {
  return planOqImages(directory, layout);
}
std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &,
                                                             const GraniteLayout &) {
  throw std::invalid_argument("Granite targets load from packed or GGUF sources only");
}

uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Qwen3_8Layout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Ornith9BLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Qwen3_6MoeLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const MiniCpmV47MoeLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const DenseLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2Layout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2MoeLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const Lfm2Moe24BLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}
uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const GraniteLayout &layout) {
  return plannedImageBytes(oqPlannedCheckpoint(directory, layout));
}

AffineTargetLoader::~AffineTargetLoader() = default;
ImagePlan AffineTargetLoader::imagePlan(size_t index, std::string_view directory) {
  const affine::Image &image = planned_->images.at(index);
  // The packed cache is checked once, lazily: the source plan the stamp
  // embeds must exist first.
  if (packed_ < 0)
    packed_ = affine::packedCacheUsable(directory_, planned_->images) ? 1 : 0;
  if (packed_)
    return packedImage(directory_ / image.name,
                       std::string(directory) + "/" + image.name, image.magic,
                       image.layer, image.type);
  ImagePlan result = affine::imagePlan(planned_, index, directory);
  const std::filesystem::path cached = directory_ / image.name;
  const std::vector<Image> *plan_images = &planned_->images;
  result.write = [write = std::move(result.write), cached, plan_images,
                  directory = directory_](std::span<uint8_t> destination,
                                          const metal::MetalBuffer &buffer) {
    write(destination, buffer);
    // Persist the image beside the sources: the next load reads it
    // verbatim. Cache writes are best effort — an unwritable package
    // just loads the slow way every time.
    affine::cachePackedImage(cached, destination);
    affine::refreshPackedStamp(directory, *plan_images);
  };
  return result;
}
WeightFile AffineTargetLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 2) throw WeightStoreError("target layer is out of range");
  return images_.load(imagePlan(index, "target"));
}
WeightFile AffineTargetLoader::head() {
  return images_.load(imagePlan(planned_->images.size() - 2, "target"));
}
WeightFile AffineTargetLoader::embedding() {
  return images_.load(imagePlan(planned_->images.size() - 1, "target"));
}

} // namespace richengine::model
