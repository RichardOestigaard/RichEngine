#include "model/GgufImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/GgufImageLayout.hpp"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>

namespace richengine::model::gguf {
namespace {

static_assert([] {
  for (const QuantFormat &format : kQuantFormats) {
    const GgmlTypeTraits *type = ggmlTypeTraits(format.ggml_type);
    if (!type || type->blockElements != format.block_elements || type->blockBytes != format.block_bytes)
      return false;
  }
  return true;
}(), "format table types are the GGUF types, block for block");
// The repack and the reference find meta unit u in native block u.
static_assert([] {
  for (const QuantFormat &format : kQuantFormats)
    if (format.meta_groups * 32 != format.block_elements) return false;
  return true;
}(), "a meta unit is one native block");

bool quantizedType(uint32_t type) { return gguf_format_of(type) != GGUF_FMT_COUNT; }
bool floatType(uint32_t type) { return type == ggml::kF32; }
// The token rows the embedding kernel gathers.
bool embeddingType(uint32_t type) { return gguf_embedding_format(gguf_format_of(type)); }
// alpha/beta run as one segment of their shared type: any format of the
// projections (one repacked tensor) or F32 (one float tensor), which BF16
// becomes exactly.
bool alphaBetaType(uint32_t type) { return quantizedType(type) || type == ggml::kF32 || type == ggml::kBF16; }

// Plans one image. A missing tensor or one of a type this build cannot load
// is added to `problems` and left out of the image, so the planner can name
// every such tensor at once; a tensor of the wrong shape throws.
class Builder {
public:
  Builder(const GgufFile &file, const TargetGeometry &geometry, std::vector<std::string> &problems,
          std::string name, uint32_t layer, uint32_t type)
      : file_(file), geometry_(geometry), problems_(problems) {
    image_.name = std::move(name);
    image_.magic = kGgufImageMagic;
    image_.layer = layer;
    image_.type = type;
    const auto header = weightFileHeader(image_.magic, layer, type);
    image_.fills.push_back({0, {header.begin(), header.end()}});
    cursor_ = header.size();
  }

  // A norm as stored: F32, which the norm kernels read unrounded, as
  // llama.cpp does (ops::NormWeights).
  void floatNorm(const std::string &name, uint64_t elements) {
    if (const GgufTensor *tensor = floatVector(name, elements)) copy(tensorRows(*tensor, 1, tensor->bytes));
  }

  // Quantized rows [N, K] repacked into planes, rows in `order`. `rows`
  // may exceed the tensor's own within one tail tile: a vocabulary that is
  // not a tile multiple pads its head section with zero rows.
  void quantized(const std::string &name, uint64_t rows, uint64_t columns, RowOrder order = {}) {
    const GgufTensor *tensor = find(name, quantizedType);
    if (!tensor) return;
    if (tensor->rows() > rows || rows - tensor->rows() >= QUANT_TILE_ROWS ||
        tensor->columns() != columns)
      throw GgufError("unexpected shape for " + name);
    const uint32_t format = gguf_format_of(tensor->type);
    Repack repack = planes(format, rows, columns, name);
    repack.sources.push_back(
        tensorRows(*tensor, tensor->rows(), ggufRowBytes(kQuantFormats[format], columns), order));
    image_.repacks.push_back(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows), rows in grouped head
  // order: one 256-row tensor of their format padded with zero rows, or F32 as
  // one float tensor, which BF16 is widened to exactly.
  void alphaBeta(const std::string &betaName, const std::string &alphaName) {
    const GgufTensor *beta = file_.find(betaName), *alpha = file_.find(alphaName);
    if (!beta || !alpha || beta->type != alpha->type || !alphaBetaType(beta->type)) {
      reject(betaName, beta);
      reject(alphaName, alpha);
      return;
    }
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {beta, alpha})
      if (t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
    if (beta->type == ggml::kF32 || beta->type == ggml::kBF16) {
      const uint64_t widening = beta->type == ggml::kBF16 ? 2 : 1;
      const uint64_t bytes = (beta->bytes + alpha->bytes) * widening;
      descriptor(ggml::kF32, 2ull * heads, hidden, {}, {bytes, 0, 0}, betaName);
      uint64_t destination = section(bytes);
      for (const GgufTensor *t : {beta, alpha}) {
        image_.copies.push_back({destination, tensorRows(*t, heads, t->bytes / heads, grouped(0, 1)),
                                 widening == 2 ? Conversion::WidenToFloat32 : Conversion::None});
        destination += t->bytes * widening;
      }
      return;
    }
    if (2 * heads > QUANT_TILE_ROWS) throw GgufError("alpha/beta rows exceed one 256-row tile");
    const uint32_t format = gguf_format_of(beta->type);
    const uint64_t rowBytes = ggufRowBytes(kQuantFormats[format], hidden);
    Repack repack = planes(format, QUANT_TILE_ROWS, hidden, alphaName);
    for (const GgufTensor *t : {beta, alpha}) repack.sources.push_back(tensorRows(*t, heads, rowBytes, grouped(0, 1)));
    image_.repacks.push_back(std::move(repack));
  }

  // The convolution taps of every channel, q and k channels as stored, then
  // value channels in grouped head order, as the exact bf16 values the GDN
  // kernels read.
  void convolution(const std::string &name, uint64_t keyRows) {
    const uint32_t channels = geometry_.convolutionDimension;
    if (const GgufTensor *tensor = floatVector(name, uint64_t{channels} * kGdnConvolutionTaps))
      copy(tensorRows(*tensor, channels, tensor->bytes / channels, grouped(keyRows, geometry_.gdnHeadDimension)),
           Conversion::NarrowToBfloat16);
  }

  // The depthwise conv's per-channel bias, as the bf16 values the conv
  // kernels read (Nemotron's ssm_conv1d.bias).
  void convBias(const std::string &name) {
    if (const GgufTensor *tensor = floatVector(name, geometry_.convolutionDimension))
      copy(tensorRows(*tensor, 1, tensor->bytes),
           Conversion::RoundToBfloat16);
  }

  // LFM2's conv kernel F32, as the bf16 values the conv kernel reads, in
  // the stored channel-major [channels, taps] order. The taps are a few
  // thousand values: rounding them is how every serving stack loads the
  // QAD builds, whose distilled F32 taps do not narrow exactly.
  void convTaps(const std::string &name) {
    const uint32_t channels = geometry_.convolutionDimension;
    const uint64_t taps = geometry_.convolutionTaps ? geometry_.convolutionTaps : kGdnConvolutionTaps;
    if (const GgufTensor *tensor = floatVector(name, channels * taps))
      copy(tensorRows(*tensor, taps, tensor->bytes / taps),
           Conversion::RoundToBfloat16);
  }

  // A per value head F32 vector in grouped head order: as stored, or as the
  // exact bf16 values the kernels read.
  void headVector(const std::string &name, Conversion conversion) {
    const uint32_t heads = geometry_.gdnValueHeads;
    if (const GgufTensor *tensor = floatVector(name, heads))
      copy(tensorRows(*tensor, heads, tensor->bytes / heads, grouped(0, 1)), conversion);
  }

  // A per Mamba-2 head F32 vector in stored order — nemotron's heads are
  // already group-major, unlike the GDN vectors headVector regroups: as
  // stored, or the exact bf16 values the kernels read.
  void mambaHeadVector(const std::string &name, Conversion conversion) {
    const uint32_t heads = geometry_.mambaHeads;
    if (const GgufTensor *tensor = floatVector(name, heads))
      copy(tensorRows(*tensor, heads, tensor->bytes / heads), conversion);
  }

  // Native token rows, gathered by the embedding kernel.
  void embeddingRows(const std::string &name) {
    const GgufTensor *tensor = find(name, embeddingType);
    if (!tensor) return;
    if (tensor->rows() != geometry_.vocabularySize || tensor->columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + name);
    copiedRows(*tensor);
  }

  // An F32 tensor [rows, columns] as stored (the MoE router and the
  // shared-expert scalar gate, which llama.cpp keeps unquantized).
  void floatTensor(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = find(name, floatType);
    if (!tensor) return;
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected an F32 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    copiedRows(*tensor);
  }

  // A [rows, columns] float segment the kernels read as fp32: F32 as
  // stored, or BF16 widened exactly — the tensors llama.cpp keeps BF16
  // (maxwell's mHC mappings and MBSA indexer projections) while the fp32
  // segments of the packed target keep them unrounded.
  void floatSegment(const std::string &name, uint64_t rows, uint64_t columns) {
    const GgufTensor *tensor = file_.find(name);
    if (!tensor || (tensor->type != ggml::kF32 && tensor->type != ggml::kBF16)) {
      reject(name, tensor);
      return;
    }
    if (tensor->rows() != rows || tensor->columns() != columns)
      throw GgufError("expected a float [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + name);
    const uint64_t widening = tensor->type == ggml::kBF16 ? 2 : 1;
    descriptor(ggml::kF32, rows, columns, {}, {rows * columns * sizeof(float), 0, 0}, name);
    copy(tensorRows(*tensor, rows, tensor->bytes / rows),
         widening == 2 ? Conversion::WidenToFloat32 : Conversion::None);
  }

  // Value heads of headRows rows from row `from` on, in grouped order.
  RowOrder grouped(uint64_t from, uint32_t headRows) const {
    return {from, headRows, geometry_.gdnKeyHeads, geometry_.gdnValueHeads / geometry_.gdnKeyHeads};
  }

  Image finish() {
    image_.bytes = alignWeightOffset(cursor_);
    return std::move(image_);
  }

private:
  void reject(const std::string &name, const GgufTensor *tensor) {
    problems_.push_back(name + (tensor ? " (" + ggmlTypeName(tensor->type) + ")" : " (missing)"));
  }

  // The tensor `name` when it is present and of an accepted type.
  const GgufTensor *find(const std::string &name, bool (*accepted)(uint32_t type)) {
    const GgufTensor *tensor = file_.find(name);
    if (tensor && accepted(tensor->type)) return tensor;
    reject(name, tensor);
    return nullptr;
  }

  const GgufTensor *floatVector(const std::string &name, uint64_t elements) {
    const GgufTensor *tensor = find(name, floatType);
    if (tensor && tensor->elements() != elements) throw GgufError("unexpected shape for " + name);
    return tensor;
  }

  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty image section in " + image_.name);
    const uint64_t start = alignWeightOffset(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  static TensorRows tensorRows(const GgufTensor &tensor, uint64_t count, uint64_t rowBytes, RowOrder order = {}) {
    if (!count || count * rowBytes != tensor.bytes) throw GgufError("unexpected size for " + tensor.name);
    return {tensor.name, tensor.type, tensor.offset, count, rowBytes, order};
  }

  // Rows written as stored, narrowed to bf16, or widened to F32, into
  // their own section.
  void copy(TensorRows source, Conversion conversion = Conversion::None) {
    uint64_t bytes = source.rows * source.rowBytes;
    if (conversion == Conversion::NarrowToBfloat16 || conversion == Conversion::RoundToBfloat16)
      bytes /= 2;
    else if (conversion == Conversion::WidenToFloat32)
      bytes *= 2;
    image_.copies.push_back({section(bytes), std::move(source), conversion});
  }

  // A tensor's rows as stored, after their descriptor.
  void copiedRows(const GgufTensor &tensor) {
    descriptor(tensor.type, tensor.rows(), tensor.columns(), {}, {tensor.bytes, 0, 0}, tensor.name);
    copy(tensorRows(tensor, tensor.rows(), tensor.bytes / tensor.rows()));
  }

  // The descriptor and planes of a [rows, columns] quantized tensor. The
  // column unit is the format's register-tile span — 64 inputs for the
  // one-meta-group formats (Q4_0, Q8_0, MXFP4 — nemotron's 2688-wide
  // hidden fills forty-two), a whole 256-superblock for the K-quants. The
  // planes store whole 256-row tiles: a tensor whose rows are not a tile
  // multiple (nemotron's 2688-wide outputs) declares its logical rows and
  // pads its stored tiles; the tail rows' stale bytes are never read.
  Repack planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
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

  // The descriptor of a tensor of `type`, quantized in `format` (a float
  // tensor has neither per-group nor meta bytes).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, const QuantFormat &format,
                  const GgufPlaneBytes &bytes, const std::string &name) {
    if (rows > std::numeric_limits<uint32_t>::max() || columns > std::numeric_limits<uint32_t>::max())
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
  const TargetGeometry &geometry_;
  std::vector<std::string> &problems_;
  Image image_;
  uint64_t cursor_ = 0;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

// The dense ("llama" arch) and LFM2 GGUFs' metadata: their fields differ
// from the qwen35 family (no ssm/full_attention_interval keys; LFM2's
// head_count_kv is a per-layer array). Optional keys are checked only when
// the file declares them.
void requireNewTargetMetadata(const GgufFile &file, const TargetGeometry &geometry,
                              const std::string &arch) {
  // A bidirectional target — the d1-omni decision encoder states
  // attention.causal false — has no compiled kernels: refuse by name
  // rather than report its first differing field. An lfm3 decision model's
  // GGUF declares the flag with its lfm3.decision.type (d1-2B): the
  // decision pathway runs the tower causally, score-only, so the flag does
  // not refuse a file that states one.
  if (file.unsignedValue(arch + ".attention.causal") ==
          std::optional<uint64_t>{0} &&
      !(arch == "lfm3" && file.stringValue(arch + ".decision.type")))
    throw GgufError(
        "this GGUF is a bidirectional encoder (attention.causal false); "
        "RichEngine does not serve it");
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  const auto expectOptional = [&](const char *key, uint64_t value) {
    if (const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
        found && *found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    std::to_string(*found) + " (expected " + std::to_string(value) + ")";
  };
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers);
  expect("embedding_length", geometry.hiddenSize);
  expect("feed_forward_length", geometry.intermediateSize);
  // LFM3's sliding-window pattern marks its local attention layers: true
  // exactly on the layers that attend but are not global.
  if (arch == "lfm3") {
    if (const std::optional<std::span<const double>> pattern =
            file.numericArray(arch + ".attention.sliding_window_pattern")) {
      for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
        if (layer >= pattern->size()) break;
        const bool expected = geometry.isFullAttentionLayer(layer) &&
                              !geometry.isGlobalAttentionLayer(layer);
        if (static_cast<bool>((*pattern)[layer]) != expected)
          mismatched += (mismatched.empty() ? "" : ", ") +
                        std::string("attention.sliding_window_pattern[") +
                        std::to_string(layer) + "]";
      }
    }
    if (geometry.slidingWindowTokens)
      expect("attention.sliding_window", geometry.slidingWindowTokens);
    if (geometry.altRotaryTheta)
      expectFloat("rope.freq_base_swa", geometry.rotaryTheta);
  }
  // LFM2 writes per-layer head arrays; a scalar means the whole model.
  if (const std::optional<std::span<const double>> heads =
          file.numericArray(arch + ".attention.head_count")) {
    for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
      if (layer >= heads->size()) break;
      const uint64_t expected =
          geometry.isFullAttentionLayer(layer)
              ? geometry.attentionWidth / geometry.attentionHeadDimension
              : 0;
      if (uint64_t((*heads)[layer]) != expected)
        mismatched += (mismatched.empty() ? "" : ", ") +
                      std::string("attention.head_count[") + std::to_string(layer) + "]";
    }
  } else {
    expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  }
  if (const std::optional<std::span<const double>> kvs =
          file.numericArray(arch + ".attention.head_count_kv")) {
    for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
      if (layer >= kvs->size()) break;
      const uint64_t expected =
          geometry.isFullAttentionLayer(layer) ? geometry.attentionKvHeads : 0;
      if (uint64_t((*kvs)[layer]) != expected)
        mismatched += (mismatched.empty() ? "" : ", ") +
                      std::string("attention.head_count_kv[") + std::to_string(layer) + "]";
    }
  } else {
    expect("attention.head_count_kv", geometry.attentionKvHeads);
  }
  expectOptional("attention.key_length", geometry.attentionHeadDimension);
  expectOptional("attention.value_length", geometry.attentionHeadDimension);
  expectOptional("rope.dimension_count", 2ull * geometry.rotaryPairs);
  // An lfm3's global layers carry the declared rope.freq_base; the sliding
  // layers' rope.freq_base_swa is checked above.
  expectFloat("rope.freq_base",
              geometry.altRotaryTheta ? geometry.altRotaryTheta
                                      : geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", geometry.rmsEpsilon);
  if (arch == "lfm2" || arch == "lfm2moe" || arch == "lfm3")
    expect("shortconv.l_cache", geometry.convolutionTaps ? geometry.convolutionTaps : 3);
  if (arch == "granite") {
    // Granite's fixed softmax multiplier, and the unit scales we do not
    // support: a 1.0 default may still be declared.
    expectFloat("attention.scale", geometry.attentionScale);
    expectFloat("embedding_scale", 1.0);
    expectFloat("residual_scale", 1.0);
    expectFloat("logit_scale", 1.0);
  }
  if (arch == "lfm2moe") {
    // The sigmoid-gated MoE without a shared expert.
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("leading_dense_block_count", geometry.leadingDenseLayers);
    expect("expert_gating_func", 2); // sigmoid
  }
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

void requireMaxwellMetadata(const GgufFile &file, const TargetGeometry &geometry,
                            const std::string &arch);
void requireNemotronMetadata(const GgufFile &file, const TargetGeometry &geometry,
                             const std::string &arch);

// The target geometry the metadata declares, with the rotary embedding and
// norms the kernels compute: the RoPE base and rotated dimensions, the RMS
// epsilon and no RoPE scaling. One error names every mismatch.
void requireMetadata(const GgufFile &file, const TargetGeometry &geometry) {
  const std::string arch = geometry.architecture();
  if (file.architecture() != arch)
    throw GgufError("GGUF architecture is " + file.architecture() + ", but the package's target is " + arch);
  if (arch == "maxwell-moe") {
    requireMaxwellMetadata(file, geometry, arch);
    return;
  }
  if (arch == "nemotron_h_moe") {
    requireNemotronMetadata(file, geometry, arch);
    return;
  }
  if (arch == "llama" || arch == "lfm2" || arch == "lfm2moe" ||
      arch == "lfm3" || arch == "granite") {
    requireNewTargetMetadata(file, geometry, arch);
    return;
  }
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  // A float equal to value up to its F32 rounding.
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("rope.dimension_count", 2ull * geometry.rotaryPairs);
  expectFloat("rope.freq_base", geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", 1e-6);
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  expect("full_attention_interval", geometry.fullAttentionPeriod);
  expect("ssm.conv_kernel", kGdnConvolutionTaps);
  expect("ssm.group_count", geometry.gdnKeyHeads);
  expect("ssm.time_step_rank", geometry.gdnValueHeads);
  expect("ssm.state_size", geometry.gdnHeadDimension);
  expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  if (geometry.sparseMoe()) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length", geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

// The maxwell-moe GGUF's metadata: the geometry the Maxwell target
// compiles, the MTP block it ships (nextn_predict_layers adds a layer the
// images do not plan), its indexer and hyper-connection parameters and its
// sigmoid MoE with per-expert bias and an ungated shared expert.
void requireMaxwellMetadata(const GgufFile &file, const TargetGeometry &geometry,
                            const std::string &arch) {
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("feed_forward_length", geometry.intermediateSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("attention.indexer.head_count", geometry.indexerHeads);
  expect("attention.indexer.key_length", geometry.indexerHeadDimension);
  expect("attention.indexer.block_size", geometry.indexerBlockTokens);
  expect("attention.indexer.prefix_blocks", geometry.indexerPrefixBlocks);
  expect("attention.indexer.local_blocks", geometry.indexerLocalBlocks);
  expect("attention.indexer.top_k", geometry.indexerRemoteBlocks);
  expect("hyper_connection.count", geometry.mhcStreams);
  expect("hyper_connection.sinkhorn_iterations", geometry.mhcSinkhornIterations);
  expectFloat("hyper_connection.epsilon", geometry.rmsEpsilon);
  expect("expert_count", geometry.experts);
  expect("expert_used_count", geometry.expertsPerToken);
  expect("expert_feed_forward_length", geometry.expertIntermediateSize);
  expect("expert_shared_count", 1);
  expect("expert_shared_feed_forward_length", geometry.sharedIntermediateSize);
  expect("expert_gating_func", 2); // sigmoid
  expect("expert_weights_norm", 1);
  expect("leading_dense_block_count", geometry.leadingDenseLayers);
  expectFloat("rope.freq_base", geometry.rotaryTheta);
  expectFloat("attention.layer_norm_rms_epsilon", geometry.rmsEpsilon);
  expect("vocab_size", geometry.vocabularySize);
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

// The nemotron_h_moe GGUF's metadata: the geometry the Nemotron target
// compiles, the per-layer arrays the two block masks mirror, the Mamba-2
// and MoE dimensions and the rope/norm settings.
void requireNemotronMetadata(const GgufFile &file,
                             const TargetGeometry &geometry,
                             const std::string &arch) {
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  const auto expectFloat = [&](const char *key, double value) {
    const std::optional<double> found = file.floatValue(arch + "." + key);
    if (found && std::abs(*found - value) <= value * 1e-6) return;
    std::ostringstream text;
    text << (mismatched.empty() ? "" : ", ") << key << ' ';
    if (found) text << *found;
    else text << "missing";
    text << " (expected " << value << ')';
    mismatched += text.str();
  };
  expect("block_count", geometry.layers);
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  // The nemotron attention blocks are NoPE — the released GGUF's
  // rope.dimension_count 84 and freq_base are converter artifacts the
  // kernels never read, so they are not pinned here.
  expectFloat("attention.layer_norm_rms_epsilon", geometry.rmsEpsilon);
  expect("ssm.conv_kernel", 4);
  expect("ssm.state_size", geometry.mambaStateSize);
  expect("ssm.group_count", geometry.mambaGroups);
  expect("ssm.inner_size", geometry.mambaInnerSize);
  expect("ssm.time_step_rank", geometry.mambaTimeStepRank);
  expect("expert_count", geometry.experts);
  expect("expert_used_count", geometry.expertsPerToken);
  expect("expert_feed_forward_length", geometry.expertIntermediateSize);
  expect("expert_shared_feed_forward_length", geometry.sharedIntermediateSize);
  expect("expert_shared_count", 1);
  expect("expert_weights_norm", 1);
  expectFloat("expert_weights_scale", 2.5);
  // The per-layer arrays name the block schedule: head_count_kv 2 on the
  // attention blocks, feed_forward_length the experts' width on the MoE
  // blocks and 0 elsewhere.
  if (const std::optional<std::span<const double>> kvs =
          file.numericArray(arch + ".attention.head_count_kv")) {
    for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
      if (layer >= kvs->size()) break;
      const uint64_t expected =
          geometry.isFullAttentionLayer(layer) ? geometry.attentionKvHeads : 0;
      if (uint64_t((*kvs)[layer]) != expected)
        mismatched += (mismatched.empty() ? "" : ", ") +
                      std::string("attention.head_count_kv[") + std::to_string(layer) + "]";
    }
  }
  if (const std::optional<std::span<const double>> ffl =
          file.numericArray(arch + ".feed_forward_length")) {
    for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
      if (layer >= ffl->size()) break;
      const uint64_t expected =
          geometry.isMoeLayer(layer) ? geometry.expertIntermediateSize : 0;
      if (uint64_t((*ffl)[layer]) != expected)
        mismatched += (mismatched.empty() ? "" : ", ") +
                      std::string("feed_forward_length[") + std::to_string(layer) + "]";
    }
  }
  if (const auto scaling = file.stringValue(arch + ".rope.scaling.type"); scaling && *scaling != "none")
    mismatched += (mismatched.empty() ? "" : ", ") + std::string("rope.scaling.type ") + *scaling + " (expected none)";
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
}

// One nemotron_h_moe block's image, in loadNemotronWeights' read order:
// the input norm, then the one sub-layer's sections — the Mamba-2
// mixer's ssm_*, the attention block's attn_* or the MoE block's ffn_*.
Image nemotronLayerImage(const GgufFile &file, const TargetGeometry &g,
                         std::vector<std::string> &problems,
                         uint32_t index) {
  const std::string p = prefix(index);
  const uint32_t kind = g.isFullAttentionLayer(index) ? 1u
                        : g.isMoeLayer(index)         ? 2u
                                                      : 0u;
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin",
            index, kind);
  const uint64_t hidden = g.hiddenSize;
  b.floatNorm(p + "attn_norm.weight", hidden);
  if (kind == 1) {
    const uint64_t kvRows =
        uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(p + "attn_q.weight", g.attentionWidth, hidden);
    b.quantized(p + "attn_k.weight", kvRows, hidden);
    b.quantized(p + "attn_v.weight", kvRows, hidden);
    b.quantized(p + "attn_output.weight", hidden, g.attentionWidth);
  } else if (kind == 2) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, hidden);
    b.floatTensor(p + "exp_probs_b.bias", 1, routed);
    b.quantized(p + "ffn_up_exps.weight", routed * width, hidden);
    b.quantized(p + "ffn_down_exps.weight", routed * hidden, width);
    const uint64_t shared = g.sharedIntermediateSize;
    b.quantized(p + "ffn_up_shexp.weight", shared, hidden);
    b.quantized(p + "ffn_down_shexp.weight", hidden, shared);
  } else {
    // The Mamba-2 mixer: the in_proj (its stored rows pad to the image's
    // storage tile), the taps-major conv kernel and bias, the raw ssm_a
    // (the kernels apply -exp), the D skip and dt bias as the bf16
    // values the kernels read, the grouped gated norm and the out
    // projection.
    b.quantized(p + "ssm_in.weight", g.mambaInputRows, hidden);
    b.convTaps(p + "ssm_conv1d.weight");
    b.convBias(p + "ssm_conv1d.bias");
    b.floatNorm(p + "ssm_a", g.mambaHeads);
    // D the kernels read as fp32; dt bias the bf16 rows they read.
    b.floatNorm(p + "ssm_d", g.mambaHeads);
    b.mambaHeadVector(p + "ssm_dt.bias", Conversion::RoundToBfloat16);
    b.floatNorm(p + "ssm_norm.weight", g.mambaInnerSize);
    b.quantized(p + "ssm_out.weight", hidden, g.mambaInnerSize);
  }
  return b.finish();
}

// One maxwell-moe layer's image, in loadMaxwellMoeWeights' read order
// (runtime/model/MaxwellMoe.cpp): the input norm, the attention mixer's
// QKV, per-head norms and output, the MBSA indexer's float projections
// and norms, the post-attention norm, the attention mHC, the dense or
// MoE FFN and the FFN mHC.
Image maxwellLayerImage(const GgufFile &file, const TargetGeometry &g,
                        std::vector<std::string> &problems, uint32_t index) {
  const std::string p = prefix(index);
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, 1u);
  const uint64_t hidden = g.hiddenSize;
  const uint64_t streams = g.mhcStreams, mix = g.mhcMixWidth;
  b.floatNorm(p + "attn_norm.weight", hidden);
  const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
  b.quantized(p + "attn_q.weight", g.attentionWidth, hidden);
  b.quantized(p + "attn_k.weight", kvRows, hidden);
  b.quantized(p + "attn_v.weight", kvRows, hidden);
  b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
  b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
  b.quantized(p + "attn_output.weight", hidden, g.attentionWidth);
  b.floatSegment(p + "indexer.q_proj.weight",
                 uint64_t{g.indexerHeads} * g.indexerHeadDimension, hidden);
  b.floatSegment(p + "indexer.k_proj.weight", g.indexerHeadDimension, hidden);
  b.floatNorm(p + "indexer.q_norm.weight", g.indexerHeadDimension);
  b.floatNorm(p + "indexer.k_norm.weight", g.indexerHeadDimension);
  b.floatNorm(p + "ffn_norm.weight", hidden);
  b.floatSegment(p + "hc_attn_fn.weight", mix, streams * hidden);
  b.floatNorm(p + "hc_attn_scale.weight", 3);
  b.floatNorm(p + "hc_attn_base.weight", mix);
  if (index >= g.leadingDenseLayers) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, hidden);
    b.floatTensor(p + "exp_probs_b.bias", 1, routed);
    b.quantized(p + "ffn_gate_exps.weight", routed * width, hidden);
    b.quantized(p + "ffn_up_exps.weight", routed * width, hidden);
    b.quantized(p + "ffn_down_exps.weight", routed * hidden, width);
    const uint64_t shared = g.sharedIntermediateSize;
    b.quantized(p + "ffn_gate_shexp.weight", shared, hidden);
    b.quantized(p + "ffn_up_shexp.weight", shared, hidden);
    b.quantized(p + "ffn_down_shexp.weight", hidden, shared);
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, hidden);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, hidden);
    b.quantized(p + "ffn_down.weight", hidden, g.intermediateSize);
  }
  b.floatSegment(p + "hc_ffn_fn.weight", mix, streams * hidden);
  b.floatNorm(p + "hc_ffn_scale.weight", 3);
  b.floatNorm(p + "hc_ffn_base.weight", mix);
  return b.finish();
}

// The maxwell-moe MTP block's image, in loadMaxwellMtpWeights' read order:
// the nextn input norms and fused eh projection, the dense layer's
// attention and its mHC modules, the sigmoid MoE with its shared expert,
// the nextn HC head and the layer output norm. The MTP block shares the
// target's token table and vocabulary head.
Image maxwellMtpImage(const GgufFile &file, const TargetGeometry &g,
                      std::vector<std::string> &problems) {
  const std::string p = prefix(g.layers);
  Builder b(file, g, problems, "mtp.bin", g.layers, 1u);
  const uint64_t hidden = g.hiddenSize;
  const uint64_t streams = g.mhcStreams, mix = g.mhcMixWidth;
  const std::string nextn = p + "nextn.";
  b.floatNorm(nextn + "enorm.weight", hidden);
  b.floatNorm(nextn + "hnorm.weight", hidden);
  b.quantized(nextn + "eh_proj.weight", hidden, 2 * hidden);
  b.floatNorm(p + "attn_norm.weight", hidden);
  const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
  b.quantized(p + "attn_q.weight", g.attentionWidth, hidden);
  b.quantized(p + "attn_k.weight", kvRows, hidden);
  b.quantized(p + "attn_v.weight", kvRows, hidden);
  b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
  b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
  b.quantized(p + "attn_output.weight", hidden, g.attentionWidth);
  b.floatNorm(p + "ffn_norm.weight", hidden);
  b.floatSegment(p + "hc_attn_fn.weight", mix, streams * hidden);
  b.floatNorm(p + "hc_attn_scale.weight", 3);
  b.floatNorm(p + "hc_attn_base.weight", mix);
  const uint64_t routed = g.experts, width = g.expertIntermediateSize;
  b.floatSegment(p + "ffn_gate_inp.weight", routed, hidden);
  b.floatSegment(p + "exp_probs_b.bias", 1, routed);
  b.quantized(p + "ffn_gate_exps.weight", routed * width, hidden);
  b.quantized(p + "ffn_up_exps.weight", routed * width, hidden);
  b.quantized(p + "ffn_down_exps.weight", routed * hidden, width);
  const uint64_t shared = g.sharedIntermediateSize;
  b.quantized(p + "ffn_gate_shexp.weight", shared, hidden);
  b.quantized(p + "ffn_up_shexp.weight", shared, hidden);
  b.quantized(p + "ffn_down_shexp.weight", hidden, shared);
  b.floatSegment(p + "hc_ffn_fn.weight", mix, streams * hidden);
  b.floatNorm(p + "hc_ffn_scale.weight", 3);
  b.floatNorm(p + "hc_ffn_base.weight", mix);
  b.floatSegment(nextn + "hc_head_fn.weight", streams, streams * hidden);
  b.floatNorm(nextn + "hc_head_base.weight", streams);
  b.floatNorm(nextn + "hc_head_scale.weight", 1);
  b.floatNorm(p + "layer_output_norm.weight", hidden);
  return b.finish();
}

Image layerImage(const GgufFile &file, const TargetGeometry &g, std::vector<std::string> &problems,
                 uint32_t index) {
  if (g.architecture() == "maxwell-moe")
    return maxwellLayerImage(file, g, problems, index);
  if (g.architecture() == "nemotron_h_moe")
    return nemotronLayerImage(file, g, problems, index);
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  const std::string arch = g.architecture();
  Builder b(file, g, problems, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
  if (full) {
    // A "llama" GGUF stores each rotated head's q and k rows interleaved in
    // rope pairs; the image deinterleaves them into richengine's head-major
    // order (gguf::RowOrder::rotaryInterleaved).
    const RowOrder rope = arch == "llama" || arch == "granite"
        ? RowOrder{0, g.attentionHeadDimension, 0, 0, true}
        : RowOrder{};
    b.quantized(p + "attn_q.weight",
                (g.attentionQueryGate ? 2ull : 1ull) * g.attentionHeadDimension *
                    (g.attentionWidth / g.attentionHeadDimension),
                g.hiddenSize, rope);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(p + "attn_k.weight", kvRows, g.hiddenSize, rope);
    b.quantized(p + "attn_v.weight", kvRows, g.hiddenSize);
    if (g.attentionQkNorm) {
      b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
      b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
    }
    b.quantized(p + "attn_output.weight", g.hiddenSize, g.attentionWidth);
    // LFM3's per-head gate is a few stored rows inside one quantized tile.
    if (g.attentionGateRows)
      b.quantized(p + "attn_gate.weight", g.attentionGateRows, g.hiddenSize);
  } else if (arch == "lfm2" || arch == "lfm2moe" || arch == "lfm3") {
    // The double-gated short convolution: in_proj's [B|C|x] rows, the
    // channel-major kernel and out_proj.
    b.quantized(p + "shortconv.in_proj.weight", 3ull * g.convolutionDimension, g.hiddenSize);
    b.convTaps(p + "shortconv.conv.weight");
    b.quantized(p + "shortconv.out_proj.weight", g.hiddenSize, g.convolutionDimension);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows; // q and k
    b.quantized(p + "attn_qkv.weight", g.convolutionDimension, g.hiddenSize, b.grouped(keyRows, g.gdnHeadDimension));
    b.quantized(p + "attn_gate.weight", valueRows, g.hiddenSize, b.grouped(0, g.gdnHeadDimension));
    b.alphaBeta(p + "ssm_beta.weight", p + "ssm_alpha.weight");
    b.convolution(p + "ssm_conv1d.weight", keyRows);
    b.headVector(p + "ssm_a", Conversion::None);
    b.headVector(p + "ssm_dt.bias", Conversion::NarrowToBfloat16);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(p + "ssm_out.weight", g.hiddenSize, valueRows);
  }
  b.floatNorm(p + (arch == "qwen35" || arch == "qwen35moe"
                       ? "post_attention_norm.weight"
                       : "ffn_norm.weight"),
              g.hiddenSize);
  if (g.sparseMoe() && index >= g.leadingDenseLayers) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(p + "ffn_gate_inp.weight", routed, g.hiddenSize);
    if (!g.sharedExpert)
      // The sigmoid gate's per-expert selection bias (exp_probs_b).
      b.floatTensor(p + "exp_probs_b.bias", 1, routed);
    b.quantized(p + "ffn_gate_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_up_exps.weight", routed * width, g.hiddenSize);
    b.quantized(p + "ffn_down_exps.weight", routed * g.hiddenSize, width);
    if (g.sharedExpert) {
      b.quantized(p + "ffn_gate_shexp.weight", width, g.hiddenSize);
      b.quantized(p + "ffn_up_shexp.weight", width, g.hiddenSize);
      b.quantized(p + "ffn_down_shexp.weight", g.hiddenSize, width);
      b.floatTensor(p + "ffn_gate_inp_shexp.weight", 1, g.hiddenSize);
    }
  } else {
    b.quantized(p + "ffn_gate.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_up.weight", g.intermediateSize, g.hiddenSize);
    b.quantized(p + "ffn_down.weight", g.hiddenSize, g.intermediateSize);
  }
  // LFM3's sandwich norms: post-mixer, then post-FFN, after the FFN
  // section as the layer image's section order states them.
  if (g.sandwichNorm) {
    b.floatNorm(p + "post_attention_norm.weight", g.hiddenSize);
    b.floatNorm(p + "post_ffw_norm.weight", g.hiddenSize);
  }
  return b.finish();
}

// A rotated GGUF (GgufRotation) must name exactly what the loader rotates:
// every tensor the images of a dense target repack, whose segments read
// H (D x) (ops::InputRotation) while float segments (F32 or BF16 alpha/beta)
// read x as it is, and the token table, which the rotated gather decodes from
// PQ2_0 rows, with the GDN value heads of the rotated inputs grouped.
void requireRotation(const GgufFile &file, const TargetGeometry &g, const std::vector<Image> &images) {
  const GgufRotation &rotation = *file.rotation();
  if (g.sparseMoe()) throw GgufError("rotated weights are supported for dense targets only");
  if (!rotation.valueHeadsGrouped)
    throw GgufError("rotated GDN inputs must keep their value heads grouped (prism.hadamard.gdn_v_grouped)");
  std::set<std::string, std::less<>> repacked;
  for (const Image &image : images)
    for (const Repack &repack : image.repacks)
      for (const TensorRows &source : repack.sources) repacked.insert(source.name);
  if (rotation.weights != repacked)
    throw GgufError("the rotation must name every quantized tensor of the target and nothing else");
  if (rotation.tables != std::set<std::string, std::less<>>{"token_embd.weight"} ||
      file.require("token_embd.weight").type != ggml::kPQ2_0)
    throw GgufError("the rotation's one token table must be token_embd.weight in PQ2_0");
}

} // namespace

std::vector<Image> planImages(const GgufFile &file, const TargetGeometry &geometry) {
  requireMetadata(file, geometry);
  std::vector<std::string> problems;
  std::vector<Image> images;
  for (uint32_t layer = 0; layer < geometry.layers; ++layer)
    images.push_back(layerImage(file, geometry, problems, layer));
  Builder head(file, geometry, problems, "head.bin", geometry.layers, 2);
  // maxwell-moe's head image leads with the HC head — its fp32 mapping
  // projection (BF16 stored), base and scale — before the final norm.
  if (geometry.mhcHead) {
    head.floatSegment("output_hc_fn.weight", geometry.mhcStreams,
                      uint64_t{geometry.mhcStreams} * geometry.hiddenSize);
    head.floatNorm("output_hc_base.weight", geometry.mhcStreams);
    head.floatNorm("output_hc_scale.weight", 1);
  }
  // The LFM2 targets name their final norm token_embd_norm and tie their
  // output head to the token embedding; an lfm3 GGUF ties the head but
  // keeps the dense targets' output_norm name.
  head.floatNorm(geometry.tiedOutput && geometry.architecture() != "lfm3"
                     ? "token_embd_norm.weight"
                     : "output_norm.weight",
                 geometry.hiddenSize);
  head.quantized(geometry.tiedOutput ? "token_embd.weight" : "output.weight",
                 paddedVocabularySize(geometry.vocabularySize), geometry.hiddenSize);
  images.push_back(head.finish());
  Builder embedding(file, geometry, problems, "embedding.bin", geometry.vocabularySize, geometry.hiddenSize);
  embedding.embeddingRows("token_embd.weight");
  images.push_back(embedding.finish());
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems) names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("GGUF tensors this build cannot load: " + names);
  }
  if (file.rotation()) requireRotation(file, geometry, images);
  return images;
}

Image planMtpImage(const GgufFile &file, const TargetGeometry &geometry) {
  if (geometry.architecture() != "maxwell-moe" || !geometry.mtpLayers)
    throw GgufError("this target's GGUF ships no MTP block");
  std::vector<std::string> problems;
  Image image = maxwellMtpImage(file, geometry, problems);
  if (!problems.empty()) {
    std::string names;
    for (const std::string &problem : problems)
      names += (names.empty() ? "" : ", ") + problem;
    throw GgufError("GGUF tensors this build cannot load: " + names);
  }
  return image;
}

} // namespace richengine::model::gguf
