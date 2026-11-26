#include "model/LiLiCorrGgufDraft.hpp"

#include "metal/abi/Gguf.h"
#include "Checked.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/WeightLayout.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace richengine::model {
namespace {

using gguf::Conversion;
using gguf::Copy;
using gguf::Image;
using gguf::Repack;
using gguf::TensorRows;

// The single .gguf in a draft directory.
std::filesystem::path findDraftGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty())
      throw GgufError("draft directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error)
    throw GgufError("cannot list draft directory: " + directory.string());
  if (found.empty())
    throw GgufError("draft directory holds no GGUF: " + directory.string());
  return found;
}

TensorRows tensorRows(const GgufTensor &tensor, uint64_t rows, uint64_t rowBytes) {
  if (!rows || rows * rowBytes != tensor.bytes)
    throw GgufError("unexpected size for " + tensor.name);
  return {tensor.name, tensor.type, tensor.offset, rows, rowBytes, {}};
}

// Plans the single image in exactly the order loadLiLiCorrGgufDraftWeights
// reads it. Only the trunk tensors the bring-up forward runs are written.
class Builder {
public:
  explicit Builder(const GgufFile &file) : file_(file) {
    image_.name = "lilicorr.bin";
    image_.magic = std::string(kLiLiCorrGgufDraftMagic);
    image_.layer = 0;
    image_.type = kBlockWeightsImageType;
    const auto header = weightFileHeader(kLiLiCorrGgufDraftMagic, 0, image_.type);
    image_.fills.push_back({0, {header.begin(), header.end()}});
    cursor_ = header.size();
  }

  const GgufTensor &requireTensor(const std::string &name) const {
    return file_.require(name);
  }

  // A vector or matrix copied without a descriptor, narrowed to bf16 when
  // the draft kernel reads bf16. GGUF stores these tensors F32.
  void copyBf16(const std::string &name, uint64_t elements) {
    const GgufTensor &tensor = requireTensor(name);
    if (tensor.type != ggml::kF32 || tensor.elements() != elements)
      throw GgufError("expected an F32 vector of " + std::to_string(elements) +
                      " values: " + name);
    image_.copies.push_back({section(elements * kBFloat16Bytes),
                             tensorRows(tensor, 1, tensor.bytes),
                             Conversion::RoundToBfloat16});
  }

  // An F32 vector or row-major matrix copied without conversion. The
  // scorer's small dense layers read these rows directly.
  void copyF32(const std::string &name, uint64_t elements) {
    const GgufTensor &tensor = requireTensor(name);
    if (tensor.type != ggml::kF32 || tensor.elements() != elements)
      throw GgufError("expected an F32 tensor of " + std::to_string(elements) +
                      " values: " + name);
    image_.copies.push_back({section(elements * sizeof(float)),
                             tensorRows(tensor, 1, tensor.bytes),
                             Conversion::None});
  }

  void copyF32Matrix(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor &tensor = requireTensor(name);
    if (!rows || !columns || tensor.type != ggml::kF32 ||
        tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("unexpected F32 matrix shape or type: " + name);
    copyF32(name, rows * columns);
  }

  // A raw block_nvfp4 matrix, rows x columns, with no engine-plane repack:
  // the scorer kernels decode its 36-byte groups directly.
  void copyNvfp4Matrix(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor &tensor = requireTensor(name);
    if (tensor.type != 40 || tensor.rows() != rows || tensor.columns() != columns ||
        columns % 64)
      throw GgufError("unexpected NVFP4 tensor shape or type: " + name);
    image_.copies.push_back({section(rows * (columns / 64) * 36),
                             tensorRows(tensor, 1, tensor.bytes),
                             Conversion::None});
  }

  // A BF16 [rows, columns] projection as a bf16 float segment.
  void bfloatProjection(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor &tensor = requireTensor(name);
    if (tensor.type != ggml::kBF16 || tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("expected a BF16 [" + std::to_string(rows) + ", " +
                      std::to_string(columns) + "] tensor: " + name);
    const uint64_t bytes = rows * columns * kBFloat16Bytes;
    descriptor(ggml::kBF16, rows, columns, {}, {bytes, 0, 0}, name);
    image_.copies.push_back({section(bytes), tensorRows(tensor, rows, columns * kBFloat16Bytes),
                             Conversion::None});
  }

  // A quantized [rows, columns] projection repacked into engine planes.
  void quantizedProjection(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor &tensor = requireTensor(name);
    const uint32_t format = gguf_format_of(tensor.type);
    if (format == GGUF_FMT_COUNT || tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("unexpected quantized tensor shape or type: " + name);
    Repack repack = planes(format, rows, columns, name);
    if (format == GGUF_FMT_NVFP4) {
      std::string scaleName(name);
      if (scaleName.size() > 7 && scaleName.compare(scaleName.size() - 7, 7, ".weight") == 0)
        scaleName.replace(scaleName.size() - 7, 7, ".scale");
      if (const GgufTensor *scaleTensor = file_.find(scaleName)) {
        if (scaleTensor->type != ggml::kF32 || scaleTensor->elements() != 1)
          throw GgufError("unexpected shape for " + scaleName);
        float scale = 1.0F;
        file_.source().readData(scaleTensor->offset,
                                {reinterpret_cast<uint8_t *>(&scale), sizeof(scale)});
        repack.scale = scale;
      }
    }
    repack.sources.push_back(tensorRows(
        tensor, rows, ggufRowBytes(kQuantFormats[format], columns)));
    image_.repacks.push_back(std::move(repack));
  }

  Image finish() {
    image_.bytes = alignWeightOffset(cursor_);
    return std::move(image_);
  }

private:
  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty LiLiCorr image section");
    const uint64_t start = alignWeightOffset(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  Repack planes(uint32_t format, uint64_t rows, uint64_t columns,
                const std::string &name) {
    if (!rows || columns % quant_register_unit(format))
      throw GgufError("tensor is not tile aligned: " + name);
    const uint64_t storedRows =
        (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
    const QuantFormat &layout = kQuantFormats[format];
    const GgufPlaneBytes bytes = ggufPlaneBytes(layout, storedRows, columns);
    descriptor(layout.ggml_type, rows, columns, layout, bytes, name);
    Repack repack;
    repack.format = format;
    repack.rows = storedRows;
    repack.columns = columns;
    repack.plane0 = section(bytes.plane0);
    repack.plane1 = bytes.plane1 ? section(bytes.plane1) : 0;
    repack.meta = section(bytes.meta);
    return repack;
  }

  void descriptor(uint32_t type, uint64_t rows, uint64_t columns,
                  const QuantFormat &format, const GgufPlaneBytes &bytes,
                  const std::string &name) {
    if (rows > std::numeric_limits<uint32_t>::max() ||
        columns > std::numeric_limits<uint32_t>::max())
      throw GgufError("tensor is too large for its descriptor: " + name);
    GgufTensorDescriptor d{};
    d.type = type;
    d.outputSize = static_cast<uint32_t>(rows);
    d.inputSize = static_cast<uint32_t>(columns);
    d.p0 = format.plane0_bytes;
    d.p1 = format.plane1_bytes;
    d.metaBytes = format.meta_bytes;
    d.metaGroups = format.meta_groups;
    d.plane0Bytes = bytes.plane0;
    d.plane1Bytes = bytes.plane1;
    d.metaTotalBytes = bytes.meta;
    std::vector<uint8_t> encoded(sizeof d);
    std::memcpy(encoded.data(), &d, sizeof d);
    image_.fills.push_back({section(encoded.size()), std::move(encoded)});
  }

  const GgufFile &file_;
  Image image_;
  uint64_t cursor_ = 0;
};

void requireMetadata(const GgufFile &file, const DFlashDraftLayout &layout) {
  if (file.architecture() != "dflash")
    throw GgufError("LiLiCorr draft GGUF is not arch dflash");
  const auto expect = [&](std::string_view key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(key);
    if (found != value)
      throw GgufError("unexpected LiLiCorr metadata " + std::string(key));
  };
  expect("dflash.block_count", layout.layers);
  expect("dflash.embedding_length", layout.hiddenSize);
  expect("dflash.feed_forward_length", layout.intermediateSize);
  expect("dflash.attention.head_count", layout.attentionSize / layout.attentionHeadDimension);
  expect("dflash.attention.head_count_kv", layout.kvHeads);
  expect("dflash.attention.key_length", layout.attentionHeadDimension);
  expect("dflash.attention.value_length", layout.attentionHeadDimension);
  expect("dflash.attention.sliding_window", layout.slidingWindow);
  expect("dflash.block_size", layout.blockSize);
  expect("dflash.conv_group_size", kDraftConvolutionGroup);
  expect("dflash.conv_kernel_size", kDraftConvolutionTaps);
  if (file.unsignedValue("dflash.attention.causal") != std::optional<uint64_t>{0})
    throw GgufError("LiLiCorr draft attention must be non-causal");
  expect("lilicorr.hidden_size", 1024);
  expect("lilicorr.candidate_topk", 8);
  expect("lilicorr.num_heads", 8);
  expect("lilicorr.num_layers", 2);
  expect("lilicorr.factor_dim", 1024);
  expect("lilicorr.mlp_hidden_size", 2048);
  if (!file.floatValue("lilicorr.vector_eps") ||
      !file.floatValue("lilicorr.logit_scale"))
    throw GgufError("LiLiCorr scorer metadata is missing");
}

float nvfp4Scale(const GgufFile &file, const std::string &weightName) {
  std::string scaleName(weightName);
  if (scaleName.size() < 7 || scaleName.compare(scaleName.size() - 7, 7, ".weight") != 0)
    throw GgufError("NVFP4 weight name does not end in .weight: " + weightName);
  scaleName.replace(scaleName.size() - 7, 7, ".scale");
  const GgufTensor &tensor = file.require(scaleName);
  if (tensor.type != ggml::kF32 || tensor.elements() != 1)
    throw GgufError("unexpected shape for " + scaleName);
  float scale = 1.0F;
  file.source().readData(tensor.offset,
                         {reinterpret_cast<uint8_t *>(&scale), sizeof(scale)});
  return scale;
}

Image planImage(const GgufFile &file, const DFlashDraftLayout &layout) {
  Builder builder(file);
  const uint32_t hidden = layout.hiddenSize;
  const uint32_t dynamic = layout.dynamicSize;
  const uint32_t qHeads = layout.attentionSize / layout.attentionHeadDimension;
  const uint32_t kvWidth = layout.kvHeads * layout.attentionHeadDimension;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    builder.copyBf16(prefix + "attn_norm.weight", hidden);
    builder.copyBf16(prefix + "attn_conv_base", uint64_t{4} * hidden);
    builder.bfloatProjection(prefix + "attn_conv_proj.weight", dynamic, hidden);
    builder.bfloatProjection(prefix + "attn_q.weight", layout.attentionSize, hidden);
    builder.bfloatProjection(prefix + "attn_k.weight", kvWidth, hidden);
    builder.bfloatProjection(prefix + "attn_v.weight", kvWidth, hidden);
    builder.copyBf16(prefix + "attn_q_norm.weight", layout.attentionHeadDimension);
    builder.copyBf16(prefix + "attn_k_norm.weight", layout.attentionHeadDimension);
    builder.bfloatProjection(prefix + "attn_output.weight", hidden, layout.attentionSize);
    builder.copyBf16(prefix + "ffn_norm.weight", hidden);
    builder.copyBf16(prefix + "ffn_conv_base", uint64_t{4} * hidden);
    builder.bfloatProjection(prefix + "ffn_conv_proj.weight", dynamic, hidden);
    builder.quantizedProjection(prefix + "ffn_gate.weight", layout.intermediateSize, hidden);
    builder.quantizedProjection(prefix + "ffn_up.weight", layout.intermediateSize, hidden);
    builder.quantizedProjection(prefix + "ffn_down.weight", hidden, layout.intermediateSize);
    builder.copyBf16(prefix + "attn_sinks.weight", qHeads);
  }
  builder.quantizedProjection("fc.weight", hidden, layout.targetHiddenSize);
  builder.copyBf16("enc.output_norm.weight", hidden);
  builder.copyBf16("output_norm.weight", hidden);

  constexpr uint32_t scorerHidden = 1024;
  constexpr uint32_t scorerTargetHidden = 4096;
  constexpr uint32_t scorerMlp = 2048;
  constexpr uint32_t scorerNodes = 56;
  constexpr uint32_t scorerHeads = 8;
  builder.copyNvfp4Matrix("lilicorr.token_proj.weight", scorerHidden, scorerTargetHidden);
  builder.copyF32("lilicorr.token_proj.bias", scorerHidden);
  builder.copyNvfp4Matrix("lilicorr.pass_hidden_proj.weight", scorerHidden, scorerTargetHidden);
  builder.copyF32("lilicorr.pass_hidden_proj.bias", scorerHidden);
  builder.copyNvfp4Matrix("lilicorr.context_proj.weight", scorerHidden, scorerTargetHidden);
  builder.copyF32("lilicorr.context_proj.bias", scorerHidden);
  const uint32_t featureLayer = layout.layers + 2;
  const std::string featurePrefix = "blk." + std::to_string(featureLayer) + ".";
  builder.copyF32(featurePrefix + "ffn_norm.weight", 5);
  builder.copyF32(featurePrefix + "ffn_norm.bias", 5);
  builder.copyF32Matrix(featurePrefix + "ffn_up.weight", scorerHidden, 5);
  builder.copyF32(featurePrefix + "ffn_up.bias", scorerHidden);
  builder.copyF32Matrix(featurePrefix + "ffn_down.weight", scorerHidden, scorerHidden);
  builder.copyF32(featurePrefix + "ffn_down.bias", scorerHidden);
  builder.copyF32("lilicorr.slot_embedding.weight", scorerHidden * 7);
  builder.copyF32("lilicorr.rank_embedding.weight", scorerHidden * 8);
  builder.copyF32("lilicorr.attention_bias.bias", scorerNodes * scorerNodes * scorerHeads);
  builder.copyF32("lilicorr.output_norm.weight", scorerHidden);
  builder.copyF32("lilicorr.anchor_norm.weight", scorerHidden);
  builder.copyF32Matrix("lilicorr.factor_input_proj.weight", scorerHidden, 3 * scorerHidden);
  builder.copyF32("lilicorr.factor_input_proj.bias", scorerHidden);
  builder.copyF32Matrix("lilicorr.out_head.weight", scorerHidden, scorerHidden);
  builder.copyF32("lilicorr.out_head.bias", scorerHidden);
  builder.copyF32Matrix("lilicorr.in_head.weight", scorerHidden, scorerHidden);
  builder.copyF32("lilicorr.in_head.bias", scorerHidden);
  builder.copyF32Matrix("lilicorr.anchor_out_head.weight", scorerHidden, scorerHidden);
  builder.copyF32("lilicorr.anchor_out_head.bias", scorerHidden);
  for (uint32_t scorerLayer = 0; scorerLayer < 2; ++scorerLayer) {
    const std::string trunkPrefix = "blk." + std::to_string(layout.layers + scorerLayer) + ".";
    const std::string inPrefix = "lilicorr.layers." + std::to_string(scorerLayer) + ".";
    builder.copyF32(trunkPrefix + "attn_norm.weight", scorerHidden);
    builder.copyF32Matrix(inPrefix + "attn.in_proj.weight", 3 * scorerHidden, scorerHidden);
    builder.copyF32(inPrefix + "attn.in_proj.bias", 3 * scorerHidden);
    builder.copyNvfp4Matrix(trunkPrefix + "attn_output.weight", scorerHidden, scorerHidden);
    builder.copyF32(trunkPrefix + "attn_output.bias", scorerHidden);
    builder.copyF32(trunkPrefix + "ffn_norm.weight", scorerHidden);
    builder.copyNvfp4Matrix(trunkPrefix + "ffn_up.weight", scorerMlp, scorerHidden);
    builder.copyF32(trunkPrefix + "ffn_up.bias", scorerMlp);
    builder.copyNvfp4Matrix(trunkPrefix + "ffn_down.weight", scorerHidden, scorerMlp);
    builder.copyF32(trunkPrefix + "ffn_down.bias", scorerHidden);
  }
  return builder.finish();
}

} // namespace

uint64_t liLiCorrGgufDraftImageBytes(
    const std::filesystem::path &directory, const DFlashDraftLayout &layout) {
  WeightSource source(findDraftGguf(directory));
  const GgufFile file(source);
  requireMetadata(file, layout);
  return planImage(file, layout).bytes;
}

DFlashDraftWeights loadLiLiCorrGgufDraftWeights(
    metal::MetalBackend &backend, WeightImages &images,
    const std::filesystem::path &directory, DFlashDraftLayout layout) {
  if (layout.kind != DraftKind::LiLiCorr)
    throw WeightStoreError("a LiLiCorr GGUF requires a LiLiCorr layout");
  const std::filesystem::path path = findDraftGguf(directory);
  auto planned = std::make_shared<WeightSource>(path);
  const GgufFile file(*planned);
  requireMetadata(file, layout);
  const Image plan = planImage(file, layout);

  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  WeightFile image = images.load({
      "draft/" + plan.name, plan.magic, plan.layer, plan.type, plan.bytes,
      [&backend, planned, plan](std::span<uint8_t>, const metal::MetalBuffer &buffer) {
        writeGgufImage(backend, *planned, buffer, plan);
        planned->checkUnchanged();
      }});

  DFlashDraftWeights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);
  const uint64_t convolutionBytes = uint64_t{4} * layout.hiddenSize * kBFloat16Bytes;
  const uint64_t headNormBytes = uint64_t{layout.attentionHeadDimension} * kBFloat16Bytes;
  const uint32_t kvWidth = layout.kvHeads * layout.attentionHeadDimension;
  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    DFlashDraftLayerWeights layer;
    layer.inputNorm = readNorm(image, layout.hiddenSize, false, "input-norm");
    layer.inputNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.attentionConvolution = image.section(convolutionBytes, "attention-convolution");
    layer.attentionDynamic = readBlockProjection(image, layout.dynamicSize, layout.hiddenSize,
                                                 "attention-dynamic");
    ops::QuantizedSegment q = readQuantizedSegment(image, "query");
    ops::QuantizedSegment k = readQuantizedSegment(image, "key");
    ops::QuantizedSegment v = readQuantizedSegment(image, "value");
    if (q.outputSize != layout.attentionSize || q.inputSize != layout.hiddenSize ||
        k.outputSize != kvWidth || k.inputSize != layout.hiddenSize ||
        v.outputSize != kvWidth || v.inputSize != layout.hiddenSize)
      throw WeightStoreError("LiLiCorr QKV segments do not match the layout");
    q.columnOffset = 0;
    k.columnOffset = layout.attentionSize;
    v.columnOffset = layout.attentionSize + kvWidth;
    layer.qkvProjection = ops::Projection(
        layout.qkvSize, layout.hiddenSize,
        ops::BlockWeights{{std::move(q), std::move(k), std::move(v)}});
    layer.queryNorm = image.section(headNormBytes, "query-norm");
    layer.keyNorm = image.section(headNormBytes, "key-norm");
    layer.outputProjection = readBlockProjection(image, layout.hiddenSize, layout.attentionSize,
                                                 "attention-output");
    layer.postAttentionNorm = readNorm(image, layout.hiddenSize, false, "post-attention-norm");
    layer.postAttentionNorm.rmsEpsilon = layout.rmsEpsilon;
    layer.mlpConvolution = image.section(convolutionBytes, "mlp-convolution");
    layer.mlpDynamic = readBlockProjection(image, layout.dynamicSize, layout.hiddenSize,
                                           "mlp-dynamic");
    layer.gateProjection = readBlockProjection(image, layout.intermediateSize, layout.hiddenSize,
                                               "mlp-gate");
    layer.upProjection = readBlockProjection(image, layout.intermediateSize, layout.hiddenSize,
                                             "mlp-up");
    layer.downProjection = readBlockProjection(image, layout.hiddenSize, layout.intermediateSize,
                                               "mlp-down");
    layer.attentionSinks = image.section(
        uint64_t{layout.attentionSize / layout.attentionHeadDimension} * kBFloat16Bytes,
        "attention-sinks");
    result.layers.push_back(std::move(layer));
  }
  result.contextProjection = readBlockProjection(image, layout.hiddenSize, layout.targetHiddenSize,
                                                 "context-projection");
  result.hiddenNorm = readNorm(image, layout.hiddenSize, false, "hidden-norm");
  result.hiddenNorm.rmsEpsilon = layout.rmsEpsilon;
  result.finalNorm = readNorm(image, layout.hiddenSize, false, "final-norm");
  result.finalNorm.rmsEpsilon = layout.rmsEpsilon;

  constexpr uint32_t scorerHidden = 1024;
  constexpr uint32_t scorerTargetHidden = 4096;
  constexpr uint32_t scorerMlp = 2048;
  constexpr uint32_t scorerNodes = 56;
  constexpr uint32_t scorerHeads = 8;
  const auto f32Bytes = [](uint64_t elements) { return elements * sizeof(float); };
  const auto nvBytes = [](uint64_t rows, uint64_t columns) {
    return rows * (columns / 64) * 36;
  };
  LiLiCorrScorerWeights &scorer = result.lilicorr;
  scorer.present = true;
  scorer.vectorEpsilon = static_cast<float>(*file.floatValue("lilicorr.vector_eps"));
  scorer.logitScale = static_cast<float>(*file.floatValue("lilicorr.logit_scale"));
  scorer.rmsEpsilon = layout.rmsEpsilon;
  scorer.tokenProjection = image.section(nvBytes(scorerHidden, scorerTargetHidden), "lilicorr-token-proj");
  scorer.tokenBias = image.section(f32Bytes(scorerHidden), "lilicorr-token-bias");
  scorer.passProjection = image.section(nvBytes(scorerHidden, scorerTargetHidden), "lilicorr-pass-proj");
  scorer.passBias = image.section(f32Bytes(scorerHidden), "lilicorr-pass-bias");
  scorer.contextProjection = image.section(nvBytes(scorerHidden, scorerTargetHidden), "lilicorr-context-proj");
  scorer.contextBias = image.section(f32Bytes(scorerHidden), "lilicorr-context-bias");
  scorer.featureNorm = image.section(f32Bytes(5), "lilicorr-feature-norm");
  scorer.featureNormBias = image.section(f32Bytes(5), "lilicorr-feature-norm-bias");
  scorer.featureUp = image.section(f32Bytes(uint64_t{scorerHidden} * 5), "lilicorr-feature-up");
  scorer.featureUpBias = image.section(f32Bytes(scorerHidden), "lilicorr-feature-up-bias");
  scorer.featureDown = image.section(f32Bytes(uint64_t{scorerHidden} * scorerHidden), "lilicorr-feature-down");
  scorer.featureDownBias = image.section(f32Bytes(scorerHidden), "lilicorr-feature-down-bias");
  scorer.slotEmbedding = image.section(f32Bytes(uint64_t{scorerHidden} * 7), "lilicorr-slot-embedding");
  scorer.rankEmbedding = image.section(f32Bytes(uint64_t{scorerHidden} * 8), "lilicorr-rank-embedding");
  scorer.attentionBias = image.section(f32Bytes(uint64_t{scorerNodes} * scorerNodes * scorerHeads),
                                       "lilicorr-attention-bias");
  scorer.outputNorm = image.section(f32Bytes(scorerHidden), "lilicorr-output-norm");
  scorer.anchorNorm = image.section(f32Bytes(scorerHidden), "lilicorr-anchor-norm");
  scorer.factorInput = image.section(f32Bytes(uint64_t{scorerHidden} * 3 * scorerHidden),
                                     "lilicorr-factor-input");
  scorer.factorInputBias = image.section(f32Bytes(scorerHidden), "lilicorr-factor-input-bias");
  scorer.outHead = image.section(f32Bytes(uint64_t{scorerHidden} * scorerHidden), "lilicorr-out-head");
  scorer.outHeadBias = image.section(f32Bytes(scorerHidden), "lilicorr-out-head-bias");
  scorer.inHead = image.section(f32Bytes(uint64_t{scorerHidden} * scorerHidden), "lilicorr-in-head");
  scorer.inHeadBias = image.section(f32Bytes(scorerHidden), "lilicorr-in-head-bias");
  scorer.anchorOutHead = image.section(f32Bytes(uint64_t{scorerHidden} * scorerHidden),
                                       "lilicorr-anchor-out-head");
  scorer.anchorOutHeadBias = image.section(f32Bytes(scorerHidden), "lilicorr-anchor-out-head-bias");
  scorer.tokenScale = nvfp4Scale(file, "lilicorr.token_proj.weight");
  scorer.passScale = nvfp4Scale(file, "lilicorr.pass_hidden_proj.weight");
  scorer.contextScale = nvfp4Scale(file, "lilicorr.context_proj.weight");
  scorer.layers.reserve(2);
  for (uint32_t scorerLayer = 0; scorerLayer < 2; ++scorerLayer) {
    LiLiCorrScorerLayerWeights layer;
    layer.attentionNorm = image.section(f32Bytes(scorerHidden), "lilicorr-attn-norm");
    layer.attentionIn = image.section(f32Bytes(uint64_t{3} * scorerHidden * scorerHidden),
                                      "lilicorr-attn-in");
    layer.attentionInBias = image.section(f32Bytes(3 * scorerHidden), "lilicorr-attn-in-bias");
    layer.attentionOut = image.section(nvBytes(scorerHidden, scorerHidden), "lilicorr-attn-out");
    layer.attentionOutBias = image.section(f32Bytes(scorerHidden), "lilicorr-attn-out-bias");
    layer.mlpNorm = image.section(f32Bytes(scorerHidden), "lilicorr-mlp-norm");
    layer.mlpUp = image.section(nvBytes(scorerMlp, scorerHidden), "lilicorr-mlp-up");
    layer.mlpUpBias = image.section(f32Bytes(scorerMlp), "lilicorr-mlp-up-bias");
    layer.mlpDown = image.section(nvBytes(scorerHidden, scorerMlp), "lilicorr-mlp-down");
    layer.mlpDownBias = image.section(f32Bytes(scorerHidden), "lilicorr-mlp-down-bias");
    const std::string prefix = "blk." + std::to_string(layout.layers + scorerLayer) + ".";
    layer.attentionOutScale = nvfp4Scale(file, prefix + "attn_output.weight");
    layer.mlpUpScale = nvfp4Scale(file, prefix + "ffn_up.weight");
    layer.mlpDownScale = nvfp4Scale(file, prefix + "ffn_down.weight");
    scorer.layers.push_back(std::move(layer));
  }
  image.finish();
  result.files.push_back(image.record());
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace richengine::model
