#include "model/DraftCheckpoint.hpp"
#include "model/AffinePlan.hpp"
#include "model/AffineTarget.hpp"
#include "model/AffineToBlock.hpp"
#include "model/WeightLayout.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DFlashTwoDraft.hpp"
#include "model/DSparkDraft.hpp"
#include "model/DFlashV1Draft.hpp"
#include "model/DSplineDraft.hpp"
#include "model/NemotronMtpDraft.hpp"

#include <string>
#include <utility>
#include <vector>

namespace richengine::model {
namespace {

using affine::copy;
using affine::embeddingSection;
using affine::Image;

// A projection's BF16 parts, stacked in row order, which the writer quantizes;
// they fill its rows.
void quantized(Image &image, std::initializer_list<std::pair<std::string, uint32_t>> parts, uint32_t rows,
               uint32_t columns) {
  validateQ4Layout(rows, columns);
  affine::Section section;
  section.kind = affine::SectionKind::Quantize;
  section.rows = rows;
  section.columns = columns;
  section.bytes = uint64_t(rows) * columns / 2 + uint64_t(rows) * columns / 16;
  uint32_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    affine::ProjectionPart part{count, {}};
    part.fields.push_back({name + ".weight", {"BF16"}, {count, columns}});
    section.parts.push_back(std::move(part));
    sourceRows += count;
  }
  if (sourceRows != rows) throw WeightStoreError("draft projection parts do not fill its rows");
  affine::append(image, std::move(section));
}

// The resolver that leaves every part unquantized: the oQ writer's F32
// float segments hold exact widened rows at any extent
// (nemotronMtpImage's precedent).
const QuantResolver &keepFloat() {
  static const QuantResolver resolver = [](std::string_view)
      -> std::optional<SafetensorsCheckpoint::QuantSpec> {
    return std::nullopt;
  };
  return resolver;
}

// A draft whose projection dims are off every quantized tile — the
// nemotron DSpark draft's hidden 2688 tiles no StorageN row and no GGUF
// 256-column block — writes every projection as float segments. The
// file's type flag is per-image, so a draft is all-quantized or
// all-float; any off-tile projection forces the rest off the tiles too.
bool floatProjections(const DFlashDraftLayout &layout) {
  const auto q4 = [](uint32_t rows, uint32_t columns) {
    return rows % kQ4StorageN == 0 && columns % kQ4GroupElements == 0;
  };
  return !(q4(layout.qkvSize, layout.hiddenSize) &&
           q4(layout.hiddenSize, layout.attentionSize) &&
           q4(layout.intermediateSize, layout.hiddenSize) &&
           q4(layout.hiddenSize, layout.intermediateSize) &&
           q4(layout.hiddenSize, layout.targetHiddenSize));
}

// The sections of each file in the order DFlashDraft.cpp reads them.
Image layerImage(const DFlashDraftLayout &layout, uint32_t layer) {
  Image result = affine::image("layer-" + std::to_string(layer) + ".bin", kDFlashLayerMagic, layer, 0);
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  const auto convolution = [&](const std::string &name) {
    copy(result, name + ".base_kernel", {2, 2, hidden});
    quantized(result, {{name + ".kernel_projection", layout.dynamicSize}}, layout.dynamicSize, hidden);
  };
  copy(result, prefix + "input_layernorm.weight", {hidden});
  convolution(prefix + "attention_conv");
  quantized(result, {{attention + "q_proj", layout.attentionSize}, {attention + "k_proj", kv}, {attention + "v_proj", kv}},
            layout.qkvSize, hidden);
  copy(result, attention + "q_norm.weight", {layout.attentionHeadDimension});
  copy(result, attention + "k_norm.weight", {layout.attentionHeadDimension});
  quantized(result, {{attention + "o_proj", hidden}}, hidden, layout.attentionSize);
  copy(result, prefix + "post_attention_layernorm.weight", {hidden});
  convolution(prefix + "mlp_conv");
  quantized(result, {{prefix + "mlp.gate_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  quantized(result, {{prefix + "mlp.up_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  quantized(result, {{prefix + "mlp.down_proj", hidden}}, hidden, layout.intermediateSize);
  return result;
}

Image modelImage(const DFlashDraftLayout &layout) {
  Image result = affine::image("model.bin", kDFlashLayerMagic, layout.layers, 1);
  const std::string selector = "candidate_selector.";
  quantized(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize, layout.targetHiddenSize);
  copy(result, "hidden_norm.weight", {layout.hiddenSize});
  copy(result, "norm.weight", {layout.hiddenSize});
  quantized(result, {{selector + "hidden_projection", layout.selectorRank}}, layout.selectorRank, layout.hiddenSize);
  copy(result, selector + "predecessor_codebook", {layout.vocabularySize, layout.selectorRank});
  copy(result, selector + "successor_codebook", {layout.vocabularySize, layout.selectorRank});
  return result;
}

// The plain transformer draft's files, in the order DFlashV1Draft.cpp reads
// them: the same fused QKV tile split as DFlash2, minus its convolutions.
// `floats` plans every projection as an oQ float segment instead —
// DSparkDraft.cpp reads both forms back (the off-tile nemotron draft).
Image dflashV1LayerImage(const DFlashDraftLayout &layout, uint32_t layer, bool floats = false) {
  Image result = affine::image("layer-" + std::to_string(layer) + ".bin", kDFlashV1DraftMagic, layer, 0);
  const std::string prefix = "layers." + std::to_string(layer) + ".";
  const std::string attention = prefix + "self_attn.";
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  const auto projection = [&](std::initializer_list<std::pair<std::string, uint32_t>> parts,
                              uint32_t rows, uint32_t columns) {
    if (floats)
      affine::projection(result, parts, rows, columns, 4, 1, keepFloat(), true);
    else
      quantized(result, parts, rows, columns);
  };
  copy(result, prefix + "input_layernorm.weight", {hidden});
  projection({{attention + "q_proj", layout.attentionSize}, {attention + "k_proj", kv}, {attention + "v_proj", kv}},
             layout.qkvSize, hidden);
  copy(result, attention + "q_norm.weight", {layout.attentionHeadDimension});
  copy(result, attention + "k_norm.weight", {layout.attentionHeadDimension});
  projection({{attention + "o_proj", hidden}}, hidden, layout.attentionSize);
  copy(result, prefix + "post_attention_layernorm.weight", {hidden});
  projection({{prefix + "mlp.gate_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  projection({{prefix + "mlp.up_proj", layout.intermediateSize}}, layout.intermediateSize, hidden);
  projection({{prefix + "mlp.down_proj", hidden}}, hidden, layout.intermediateSize);
  return result;
}

Image dflashV1ModelImage(const DFlashDraftLayout &layout, bool floats = false) {
  Image result = affine::image("model.bin", kDFlashV1DraftMagic, layout.layers, 1);
  if (floats)
    affine::projection(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize,
                       layout.targetHiddenSize, 4, 1, keepFloat(), true);
  else
    quantized(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize, layout.targetHiddenSize);
  copy(result, "hidden_norm.weight", {layout.hiddenSize});
  copy(result, "norm.weight", {layout.hiddenSize});
  return result;
}

// The DSpark draft's files: the layers are the plain draft's (both carry a
// fused QKV tile, per-head norms and a gated MLP); model.bin adds the Markov
// head's tables and the confidence head.
Image dsparkModelImage(const DFlashDraftLayout &layout, bool floats) {
  Image result = affine::image("model.bin", kDSparkDraftMagic, layout.layers, 1);
  if (floats)
    affine::projection(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize,
                       layout.targetHiddenSize, 4, 1, keepFloat(), true);
  else
    quantized(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize, layout.targetHiddenSize);
  copy(result, "hidden_norm.weight", {layout.hiddenSize});
  copy(result, "norm.weight", {layout.hiddenSize});
  copy(result, "markov_head.markov_w1.weight", {layout.vocabularySize, layout.markovRank});
  copy(result, "markov_head.markov_w2.weight", {layout.vocabularySize, layout.markovRank});
  // The confidence head ships only on drafts that declare one
  // (enable_confidence_head); Nemotron's DSpark draft drops it.
  if (layout.confidenceHead) {
    copy(result, "confidence_head.proj.weight", {1, layout.hiddenSize + layout.markovRank});
    copy(result, "confidence_head.proj.bias", {1});
  }
  // A draft carrying its own embed_tokens (Nemotron's trained mask row):
  // unquantized, the oQ writer widens the bf16 rows to the F32 native table
  // readOqEmbedding binds for gguf_embed_f32.
  if (layout.tokenEmbedding)
    embeddingSection(result, "embed_tokens", layout.vocabularySize,
                     layout.hiddenSize, keepFloat());
  return result;
}

// The DSpline draft's model.bin: the plain draft's fc and norms, then the
// selected-prefix code bridge's tensors, bf16 as stored, and the four
// I64 format sentinels the loader checks. Section order is the read order
// of DSplineDraft.cpp's readDraft.
Image dsplineModelImage(const DFlashDraftLayout &layout) {
  Image result = affine::image("model.bin", kDSplineDraftMagic, layout.layers, 1);
  const std::string bridge = "selected_prefix_bridge.";
  quantized(result, {{"fc", layout.hiddenSize}}, layout.hiddenSize, layout.targetHiddenSize);
  copy(result, "hidden_norm.weight", {layout.hiddenSize});
  copy(result, "norm.weight", {layout.hiddenSize});
  copy(result, bridge + "codebook", {layout.vocabularySize, layout.bridgeCodeSize});
  copy(result, bridge + "early_read_weights",
       {layout.layers - 1, layout.bridgeCodeSize, layout.hiddenSize});
  copy(result, bridge + "read_weight", {layout.bridgeCodeSize, layout.hiddenSize});
  copy(result, bridge + "feature_write_weights",
       {layout.layers, layout.hiddenSize, layout.bridgeRank});
  copy(result, bridge + "gate.weight", {layout.bridgeRank, layout.bridgeCodeSize});
  copy(result, bridge + "predecessor_gate.weight",
       {layout.bridgeRank, layout.bridgeCodeSize});
  copy(result, bridge + "prev.weight", {layout.bridgeRank, layout.bridgeCodeSize});
  copy(result, bridge + "gate.bias", {layout.bridgeRank});
  copy(result, bridge + "out.weight", {layout.bridgeCodeSize, layout.bridgeRank});
  copy(result, bridge + "independent_write_version", {}, "I64");
  copy(result, bridge + "terminal_disabled_version", {}, "I64");
  copy(result, "draft_block_causal_version", {}, "I64");
  copy(result, "last_write_refine_version", {}, "I64");
  return result;
}

// The nemotron MTP block's mtp.bin, in loadNemotronMtpDraftWeights' read
// order: the nextn norms, the eh and attention projections as F32 float
// segments (hidden 2688 is off every quantized tile: no AF4 256-row, no
// GGUF 256-column group — the float segments take any extent), the MoE's
// bf16 tensors the draft's nmtp_moe_* kernels read as stored, and the
// output and shared-head norms.
Image nemotronMtpImage(const DFlashDraftLayout &layout) {
  Image result = affine::image("mtp.bin", kNemotronMtpDraftMagic, layout.layers, 1);
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t kv = layout.kvHeads * layout.attentionHeadDimension;
  // Every projection unquantized: the oQ writer's F32 float segments hold
  // exact widened bf16 rows at any extent — the draft's eh (2688x5376),
  // attention (2688 into and out of 4096/256) projections.
  const auto floatProjection = [&](const std::string &name, uint32_t rows,
                                   uint32_t columns) {
    affine::projection(result, {{name, rows}}, rows, columns, 4, 1,
                       keepFloat(), true);
  };
  copy(result, "mtp.nextn.enorm.weight", {hidden});
  copy(result, "mtp.nextn.hnorm.weight", {hidden});
  floatProjection("mtp.nextn.eh_proj.weight", hidden, 2 * hidden);
  copy(result, "mtp.attn_norm.weight", {hidden});
  floatProjection("mtp.self_attn.q_proj.weight", layout.attentionSize, hidden);
  floatProjection("mtp.self_attn.k_proj.weight", kv, hidden);
  floatProjection("mtp.self_attn.v_proj.weight", kv, hidden);
  floatProjection("mtp.self_attn.o_proj.weight", hidden, layout.attentionSize);
  copy(result, "mtp.post_attention_norm.weight", {hidden});
  copy(result, "mtp.moe.gate_inp.weight", {128, hidden});
  copy(result, "mtp.moe.exp_probs_b.bias", {128});
  copy(result, "mtp.moe.up_exps.weight", {128, 1856, hidden});
  copy(result, "mtp.moe.down_exps.weight", {128, hidden, 1856});
  copy(result, "mtp.moe.up_shexp.weight", {3712, hidden});
  copy(result, "mtp.moe.down_shexp.weight", {hidden, 3712});
  copy(result, "mtp.output_norm.weight", {hidden});
  copy(result, "mtp.nextn.shared_head_norm.weight", {hidden});
  return result;
}

} // namespace

std::vector<Image> draftCheckpointImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) result.push_back(layerImage(layout, layer));
  result.push_back(modelImage(layout));
  return result;
}

DFlash2CheckpointLoader::DFlash2CheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                                             const DFlashDraftLayout &layout)
    : images_(images), planned_(std::make_shared<affine::PlannedCheckpoint>(directory)) {
  planned_->images = affine::blockImages(draftCheckpointImages(layout));
  for (Image &image : planned_->images) affine::bind(image, planned_->source);
}
DFlash2CheckpointLoader::~DFlash2CheckpointLoader() = default;
WeightFile DFlash2CheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return images_.load(affine::imagePlan(planned_, index, "draft"));
}
WeightFile DFlash2CheckpointLoader::model() {
  return images_.load(affine::imagePlan(planned_, planned_->images.size() - 1, "draft"));
}

std::vector<Image> dflashV1DraftCheckpointImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  const bool floats = floatProjections(layout);
  for (uint32_t layer = 0; layer < layout.layers; ++layer)
    result.push_back(dflashV1LayerImage(layout, layer, floats));
  result.push_back(dflashV1ModelImage(layout, floats));
  return result;
}

DFlashV1DraftCheckpointLoader::DFlashV1DraftCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                                                     const DFlashDraftLayout &layout)
    : images_(images), planned_(std::make_shared<affine::PlannedCheckpoint>(directory)) {
  planned_->images = affine::blockImages(dflashV1DraftCheckpointImages(layout));
  for (Image &image : planned_->images) affine::bind(image, planned_->source);
}
DFlashV1DraftCheckpointLoader::~DFlashV1DraftCheckpointLoader() = default;
WeightFile DFlashV1DraftCheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return images_.load(affine::imagePlan(planned_, index, "draft"));
}
WeightFile DFlashV1DraftCheckpointLoader::model() {
  return images_.load(affine::imagePlan(planned_, planned_->images.size() - 1, "draft"));
}

std::vector<Image> dsparkDraftCheckpointImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  const bool floats = floatProjections(layout);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    Image layerImage = dflashV1LayerImage(layout, layer, floats);
    layerImage.magic = std::string(kDSparkDraftMagic);
    // The per-head attention-sink bias trails the layer's sections — the
    // same tail position DSparkDraft.cpp reads it from.
    if (layout.attentionSinks)
      copy(layerImage,
           "layers." + std::to_string(layer) +
               ".self_attn.attention_sink_bias",
           {layout.attentionSize / layout.attentionHeadDimension});
    result.push_back(std::move(layerImage));
  }
  result.push_back(dsparkModelImage(layout, floats));
  return result;
}

DSparkCheckpointLoader::DSparkCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                                               const DFlashDraftLayout &layout)
    : images_(images), planned_(std::make_shared<affine::PlannedCheckpoint>(directory)) {
  planned_->images = affine::blockImages(dsparkDraftCheckpointImages(layout));
  for (Image &image : planned_->images) affine::bind(image, planned_->source);
}
DSparkCheckpointLoader::~DSparkCheckpointLoader() = default;
WeightFile DSparkCheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return images_.load(affine::imagePlan(planned_, index, "draft"));
}
WeightFile DSparkCheckpointLoader::model() {
  return images_.load(affine::imagePlan(planned_, planned_->images.size() - 1, "draft"));
}

std::vector<Image> dsplineDraftCheckpointImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    Image layerImage = dflashV1LayerImage(layout, layer);
    layerImage.magic = std::string(kDSplineDraftMagic);
    result.push_back(std::move(layerImage));
  }
  result.push_back(dsplineModelImage(layout));
  return result;
}

DSplineCheckpointLoader::DSplineCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                                               const DFlashDraftLayout &layout)
    : images_(images), planned_(std::make_shared<affine::PlannedCheckpoint>(directory)) {
  planned_->images = affine::blockImages(dsplineDraftCheckpointImages(layout));
  for (Image &image : planned_->images) affine::bind(image, planned_->source);
}
DSplineCheckpointLoader::~DSplineCheckpointLoader() = default;
WeightFile DSplineCheckpointLoader::layer(uint32_t index) {
  if (index >= planned_->images.size() - 1) throw WeightStoreError("draft layer is out of range");
  return images_.load(affine::imagePlan(planned_, index, "draft"));
}
WeightFile DSplineCheckpointLoader::model() {
  return images_.load(affine::imagePlan(planned_, planned_->images.size() - 1, "draft"));
}

std::vector<Image> nemotronMtpDraftCheckpointImages(const DFlashDraftLayout &layout) {
  std::vector<Image> result;
  result.push_back(nemotronMtpImage(layout));
  return result;
}

NemotronMtpCheckpointLoader::NemotronMtpCheckpointLoader(WeightImages &images, const std::filesystem::path &directory,
                                             const DFlashDraftLayout &layout)
    : images_(images), planned_(std::make_shared<affine::PlannedCheckpoint>(directory)) {
  planned_->images = nemotronMtpDraftCheckpointImages(layout);
  for (Image &image : planned_->images) affine::bind(image, planned_->source);
}
NemotronMtpCheckpointLoader::~NemotronMtpCheckpointLoader() = default;
WeightFile NemotronMtpCheckpointLoader::mtp() {
  return images_.load(affine::imagePlan(planned_, 0, "draft"));
}

} // namespace richengine::model
