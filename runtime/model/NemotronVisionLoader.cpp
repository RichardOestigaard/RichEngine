#include "model/NemotronVisionLoader.hpp"
#include "model/GgufFile.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <functional>
#include <span>
#include <vector>

namespace richengine::model {
namespace {

using vision::Input;
using vision::Plan;
using vision::Section;

// The sections of the packed Nemotron Omni image, in the order the weight
// reader (NemotronVision.cpp) consumes them: patch embedding, position table
// and register embeddings, then per block its LayerNorms, separate biased
// attention projections and biased GELU(erf) FFN, then the projector's
// RMSNorm weight and two bias-free maps.
Plan nemotronPlan(const ops::NemotronVisionLayout &l) {
  Plan result{l.depth, l.patchSize, 0, {}};
  auto &sections = result.sections;
  const auto add = [&](std::string gguf, uint32_t rows, uint32_t columns = 1) {
    sections.push_back({"", std::move(gguf), rows, columns, rows, columns});
  };
  const uint32_t positions = l.positionGridSide * l.positionGridSide;
  add("v.patch_embd.weight", l.hiddenSize, l.patchDimension);
  sections.back().patch = true;
  add("v.position_embd.weight", positions, l.hiddenSize);
  add("v.class_embd", l.registers, l.hiddenSize);
  for (uint32_t i = 0; i < l.depth; ++i) {
    const std::string blk = "v.blk." + std::to_string(i) + ".";
    add(blk + "ln1.weight", l.hiddenSize);
    add(blk + "ln1.bias", l.hiddenSize);
    add(blk + "attn_q.weight", l.hiddenSize, l.hiddenSize);
    add(blk + "attn_q.bias", l.hiddenSize);
    add(blk + "attn_k.weight", l.hiddenSize, l.hiddenSize);
    add(blk + "attn_k.bias", l.hiddenSize);
    add(blk + "attn_v.weight", l.hiddenSize, l.hiddenSize);
    add(blk + "attn_v.bias", l.hiddenSize);
    add(blk + "attn_out.weight", l.hiddenSize, l.hiddenSize);
    add(blk + "attn_out.bias", l.hiddenSize);
    add(blk + "ln2.weight", l.hiddenSize);
    add(blk + "ln2.bias", l.hiddenSize);
    add(blk + "ffn_up.weight", l.intermediateSize, l.hiddenSize);
    add(blk + "ffn_up.bias", l.intermediateSize);
    add(blk + "ffn_down.weight", l.hiddenSize, l.intermediateSize);
    add(blk + "ffn_down.bias", l.hiddenSize);
  }
  add("mm.model.mlp.0.weight", l.mergedHiddenSize);
  add("mm.model.mlp.1.weight", l.projectorHiddenSize, l.mergedHiddenSize);
  add("mm.model.mlp.3.weight", l.outputHiddenSize, l.projectorHiddenSize);
  uint64_t at = kWeightFileAlignment;
  for (auto &s : sections) {
    s.offset = at;
    at = alignWeightOffset(at + uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  }
  result.bytes = at;
  return result;
}

Input nemotronInput(std::string name, SourceTensor tensor) {
  if (tensor.dtype != "BF16" && tensor.dtype != "F32" && tensor.dtype != "F16")
    throw WeightStoreError("vision tensor " + name + " in " + tensor.file->path().string() +
                           " is " + tensor.dtype + "; preparation reads BF16, F32 or F16");
  return {std::move(name), std::move(tensor)};
}

void requireNemotronMmprojMetadata(const GgufFile &gguf,
                                   const ops::NemotronVisionLayout &layout) {
  const std::optional<std::string> projector =
      gguf.stringValue("clip.projector_type").has_value()
          ? gguf.stringValue("clip.projector_type")
          : gguf.stringValue("clip.vision.projector_type");
  if (gguf.architecture() != "clip" || projector != "nemotron_omni")
    throw WeightStoreError("unsupported vision GGUF architecture");
  for (const auto &[key, expected] : std::initializer_list<std::pair<const char *, uint64_t>>{
           {"clip.vision.projection_dim", layout.outputHiddenSize},
           {"clip.vision.patch_size", layout.patchSize},
           {"clip.vision.embedding_length", layout.hiddenSize},
           {"clip.vision.feed_forward_length", layout.intermediateSize},
           {"clip.vision.block_count", layout.depth},
           {"clip.vision.attention.head_count", layout.heads},
           {"clip.vision.projector.scale_factor", layout.spatialMerge},
           {"clip.use_gelu", 1},
           {"clip.vision.nemotron_omni", 1}})
    if (gguf.unsignedValue(key) != expected)
      throw WeightStoreError(std::string("vision metadata mismatch: ") + key);
  const auto epsilon = gguf.floatValue("clip.vision.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) || std::abs(*epsilon - 1e-6) > 1e-12)
    throw WeightStoreError("vision norm epsilon mismatch");
  const auto mean = gguf.numericArray("clip.vision.image_mean");
  const auto stddev = gguf.numericArray("clip.vision.image_std");
  constexpr double expectedMean[3] = {0.48145466, 0.4578275, 0.40821073};
  constexpr double expectedStd[3] = {0.26862954, 0.26130258, 0.27577711};
  const auto close = [](const auto &values, const double expected[3]) {
    return values && values->size() == 3 &&
           std::all_of(values->begin(), values->end(), [](double x) { return x > 0; }) &&
           std::abs((*values)[0] - expected[0]) < 1e-6 &&
           std::abs((*values)[1] - expected[1]) < 1e-6 &&
           std::abs((*values)[2] - expected[2]) < 1e-6;
  };
  if (!close(mean, expectedMean) || !close(stddev, expectedStd))
    throw WeightStoreError("vision image normalization mismatch");
  const auto minPixels = gguf.unsignedValue("clip.vision.image_min_pixels");
  const auto maxPixels = gguf.unsignedValue("clip.vision.image_max_pixels");
  if (!minPixels || !maxPixels || !*minPixels || *maxPixels < *minPixels)
    throw WeightStoreError("vision image pixel bounds mismatch");
}

void bindNemotronMmproj(const GgufFile &gguf, const ops::NemotronVisionLayout &layout,
                        Plan &plan) {
  requireNemotronMmprojMetadata(gguf, layout);
  const uint64_t p = layout.patchSize;
  for (auto &s : plan.sections) {
    const GgufTensor &t = gguf.require(s.gguf);
    std::vector<uint64_t> shape;
    if (s.patch)
      shape = {p, p, 3, s.rows};
    else if (s.gguf == "v.position_embd.weight" || s.gguf == "v.class_embd")
      shape = {s.columns, s.rows};
    else if (s.columns == 1)
      shape = {s.rows};
    else
      shape = {s.columns, s.rows};
    if (t.dims != shape) throw WeightStoreError("vision tensor shape mismatch: " + s.gguf);
    s.inputs.push_back(nemotronInput(s.gguf, {&gguf.source(), ggmlTypeName(t.type), t.dims, t.offset, t.bytes}));
  }
}

// Round-to-nearest-even BF16. The mmproj's F32 position table, norms, biases
// and patch kernel are source-precision values, not widened BF16: the tower
// runs in BF16, so preparation rounds them once here.
uint16_t roundBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
}

void convertRow(const SourceTensor &tensor, const uint8_t *source, uint16_t *destination,
                uint64_t count) {
  if (tensor.dtype == "BF16") {
    std::memcpy(destination, source, count * kBFloat16Bytes);
    return;
  }
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    if (tensor.dtype == "F32") {
      std::memcpy(&value, source + i * 4, 4);
    } else {
      _Float16 half;
      std::memcpy(&half, source + i * 2, 2);
      value = static_cast<float>(half);
    }
    destination[i] = roundBfloat16(value);
  }
}

uint64_t sourceRowBytes(const SourceTensor &tensor, uint64_t columns) {
  return columns * (tensor.dtype == "F32" ? 4 : 2);
}

struct NemotronStaging {
  std::vector<uint8_t> source;
};

// Every section is already in the packed row order: matrices are output rows
// of contiguous input columns, the position table and register embeddings are
// token rows of contiguous hidden channels, and the still patch kernel's
// channel/row/column order is the patch row the encoder reads.
void addNemotronSectionTasks(uint8_t *image, const Section &s,
                             std::vector<std::function<void(NemotronStaging &)>> &tasks) {
  const Input &input = s.inputs.front();
  const uint64_t bytes = sourceRowBytes(input.tensor, s.columns);
  const auto batchRows = static_cast<uint32_t>(
      std::clamp<uint64_t>(kLoadStepBytes / std::max<uint64_t>(bytes, 1), 1, s.rows));
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) + uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, &input, bytes, row, count, output](NemotronStaging &staging) {
      if (staging.source.size() < count * bytes) staging.source.resize(count * bytes);
      input.tensor.read(row * bytes, std::span(staging.source).first(count * bytes));
      for (uint32_t r = 0; r < count; ++r)
        convertRow(input.tensor, staging.source.data() + r * bytes,
                   output + uint64_t(r) * s.storedColumns, s.columns);
    });
  }
}

void writeNemotronVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes)
    throw WeightStoreError("nemotron vision image destination size differs");
  const auto header = weightFileHeader(kNemotronVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset, uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(NemotronStaging &)>> tasks;
  for (const auto &s : plan.sections)
    addNemotronSectionTasks(destination.data(), s, tasks);
  std::vector<NemotronStaging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace

struct NemotronVisionLoader::Planned {
  ops::NemotronVisionLayout layout;
  std::unique_ptr<WeightSource> mmproj;
  Plan plan;
  void checkUnchanged() const { mmproj->checkUnchanged(); }
};

NemotronVisionLoader::NemotronVisionLoader(const std::filesystem::path &directory,
                                           const ops::NemotronVisionLayout &layout) {
  requireNemotronVisionLayout(layout);
  auto planned = std::make_shared<Planned>();
  planned->layout = layout;
  planned->mmproj = std::make_unique<WeightSource>(directory / "mmproj.gguf");
  const GgufFile gguf(*planned->mmproj);
  planned->plan = nemotronPlan(layout);
  bindNemotronMmproj(gguf, layout, planned->plan);
  planned_ = std::move(planned);
}

NemotronVisionLoader::~NemotronVisionLoader() = default;

const ops::NemotronVisionLayout &NemotronVisionLoader::layout() const noexcept {
  return planned_->layout;
}

ImagePlan NemotronVisionLoader::image() const {
  return {"vision/nemotron.bin", std::string(kNemotronVisionMagic), planned_->layout.depth, 0,
          planned_->plan.bytes,
          [planned = planned_](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
            writeNemotronVision(bytes, planned->plan);
            planned->checkUnchanged();
          }};
}

uint64_t nemotronVisionImageBytes(const ops::NemotronVisionLayout &layout) {
  return nemotronPlan(layout).bytes;
}

void requireNemotronVisionLayout(const ops::NemotronVisionLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.patchDimension || !layout.intermediateSize ||
      !layout.mergedHiddenSize || !layout.projectorHiddenSize || !layout.outputHiddenSize ||
      !layout.heads || !layout.headDimension || !layout.positionGridSide || !layout.patchSize ||
      !layout.spatialMerge || !layout.registers ||
      layout.heads * layout.headDimension != layout.hiddenSize ||
      layout.mergedHiddenSize != layout.hiddenSize * layout.spatialMerge * layout.spatialMerge ||
      layout.patchDimension != 3 * layout.patchSize * layout.patchSize)
    throw WeightStoreError("Nemotron vision layout is inconsistent");
}

} // namespace richengine::model
