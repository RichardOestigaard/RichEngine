#include "model/AffineTarget.hpp"
#include "model/AffineToBlock.hpp"
#include "Checked.hpp"
#include "model/AffinePlan.hpp"
#include "model/ModelRegistry.hpp"
#include "model/OqWrite.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"
#include <fstream>
#include <sstream>

namespace richengine::model {
namespace affine {

// Projection parts, stacked in row order, padded with zero rows to `rows`.
// With a resolver the section is an Oq one: each part takes the spec the
// resolver resolves — the `bits` parameter is today's fallback when it is
// empty — and a part the resolver leaves unquantized keeps its plain float
// weight (an F32 float segment).
void projection(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts,
                uint32_t rows, uint32_t columns, uint32_t bits, uint32_t experts,
                const QuantResolver &resolve, bool forceFloat,
                bool preferAffine) {
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

// One fused window of `name`: the source block of `stride` rows per expert
// contributes `rows` starting at `offset`. MiniCPM-V's hook reads its
// experts.gate_up_proj through it.
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

// The MoE FFN's router, selection bias and expert slabs; the LFM2-MoE 8B's
// and 24B's layer images share the block, differing only in the expert
// tensors' names (the 8B's experts.w1/w3/w2, the 24B's
// switch_mlp.gate/up/down_proj).
void lfm2MoeFfn(Image &result, const TargetLayout &layout, const std::string &ffn,
                const std::string &gate, const std::string &up, const std::string &down,
                const QuantResolver &resolve) {
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
  projection(result, {{gate, width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
  projection(result, {{up, width}}, width, layout.hiddenSize, 4, layout.experts, resolve);
  projection(result, {{down, layout.hiddenSize}}, layout.hiddenSize, width, 4, layout.experts,
             resolve);
}

} // namespace affine

namespace {

using affine::append;
using affine::copy;
using affine::embeddingSection;
using affine::image;
using affine::Image;
using affine::projection;
using affine::ProjectionPart;
using affine::QuantSpec;
using affine::Section;
using affine::SectionKind;

// ---- The GDN hybrids (qwen3_8, ornith9, qwen3_6moe) ----
// Their checkpoints share one schema: language_model.model.* tensors, the
// linear_attn/self_attn mixers and the mlp. FFN — dense or the switch_mlp
// experts with a shared one as ffnKind says. Every arch whose registration
// carries no AffineArchHooks plans this default.

Image gdnLayerImage(const TargetLayout &layout, uint32_t layer, const QuantResolver &resolve = {}) {
  // The affine-tile projections win where the oQ block decoders measured
  // slowest — the SparseMoe layouts — and lose on the dense ones.
  const bool denseAffine = layout.ffnKind == FfnKind::SparseMoe;
  const bool full = layout.isFullAttentionLayer(layer);
  Image result = image("layer-" + std::to_string(layer) + ".bin", layout.layerMagic, layer, full ? 1u : 0u);
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
  if (layout.ffnKind == FfnKind::SparseMoe) {
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

Image gdnHeadImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  Image result = image("head.bin", layout.headMagic, layout.layers, 2);
  copy(result, "language_model.model.norm.weight", {layout.hiddenSize});
  projection(result, {{"language_model.lm_head", layout.vocabularySize}},
             paddedVocabularySize(layout.vocabularySize), layout.hiddenSize, 4, 1, resolve, false,
             layout.ffnKind == FfnKind::SparseMoe);
  return result;
}

// The token rows as stored: 4-bit codes, scales and biases; an oQ plan's
// native rows instead.
Image gdnEmbeddingImage(const TargetLayout &layout, const QuantResolver &resolve = {}) {
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

std::vector<Image> gdnImages(const TargetLayout &layout, const QuantResolver &resolve = {}) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(gdnLayerImage(layout, layer, resolve));
  result.push_back(gdnHeadImage(layout, resolve));
  result.push_back(gdnEmbeddingImage(layout, resolve));
  return result;
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

uint64_t plannedImageBytes(const std::shared_ptr<affine::PlannedCheckpoint> &planned) {
  uint64_t total = 0;
  for (const Image &image : planned->images) total += image.bytes;
  return total;
}

// The targets no MLX or oQ source can plan: a packed-only layout (Gemma 4,
// DiffusionGemma) and one whose family ships no MLX schema (Granite's
// sources are packed or GGUF — mlxImageable stays false there).
void requireImageableLayout(const TargetLayout &layout) {
  if (!layout.mlxImageable)
    throw std::invalid_argument("Granite targets load from packed or GGUF sources only");
  if (layout.packedOnly)
    throw std::invalid_argument("Gemma 4 targets load from packed files only");
}

// The checkpoint at directory with every image of layout bound to it. Its
// config.json is the model's, which inspectModelPackage checks. `oq` plans
// every projection against the checkpoint's per-module specs. The arch's
// registered hooks validate and plan it; an arch without hooks takes the
// GDN schema and no validation.
std::shared_ptr<affine::PlannedCheckpoint> plan(const std::filesystem::path &directory,
                                                const TargetLayout &layout, bool oq = false) {
  requireImageableLayout(layout);
  auto planned = std::make_shared<affine::PlannedCheckpoint>(directory);
  const AffineArchHooks *hooks = affineHooksFor(layout);
  if (hooks && hooks->validateConfiguration)
    hooks->validateConfiguration(planned->source, layout);
  const QuantResolver resolve = oq ? oqResolver(planned->source) : QuantResolver{};
  planned->images = hooks && hooks->planImages
                        ? hooks->planImages(&planned->source, layout, resolve)
                        : gdnImages(layout, resolve);
  for (Image &image : planned->images) affine::bind(image, planned->source);
  return planned;
}

} // namespace

std::vector<Image> affineTargetImages(const TargetLayout &layout) {
  requireImageableLayout(layout);
  const AffineArchHooks *hooks = affineHooksFor(layout);
  // Byte budgeting only, no checkpoint: a hook that derives its tensor
  // names from the source plans its default schema (the sections carry
  // the same bytes either way).
  if (hooks && hooks->planImages) return hooks->planImages(nullptr, layout, {});
  return gdnImages(layout);
}

AffineTargetLoader::AffineTargetLoader(WeightImages &images, const std::filesystem::path &directory,
                                       const TargetLayout &layout, bool preferBlockWeights)
    : images_(images), directory_(directory), planned_(plan(directory, layout)) {
  // The sections keep their bound inputs; only their bytes and writer change.
  if (preferBlockWeights) planned_->images = affine::blockImages(std::move(planned_->images));
}

std::shared_ptr<affine::PlannedCheckpoint> oqPlannedCheckpoint(const std::filesystem::path &directory,
                                                             const TargetLayout &layout) {
  return plan(directory, layout, true);
}

uint64_t oqTargetImageBytes(const std::filesystem::path &directory, const TargetLayout &layout) {
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
