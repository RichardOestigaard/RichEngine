#include "model/LfmVisionLoader.hpp"
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

// The sections of the packed LFM3-VL image, in the order the weight reader
// (LfmVision.cpp) consumes them: patch embedding, embedding norm, then per
// block its operator norm, its conv or attention tensors (the mmproj's
// per-block kind, so every planned section is bound), ffn norm and silu
// FFN, then the projector's input LayerNorm and two affine maps.
Plan lfmPlan(const ops::LfmVisionLayout &l, const std::vector<bool> &conv) {
  Plan result{l.depth, l.patchSize, 0, {}};
  auto &sections = result.sections;
  const auto add = [&](std::string gguf, uint32_t rows, uint32_t columns = 1) {
    sections.push_back({"", std::move(gguf), rows, columns, rows, columns});
  };
  const uint32_t kv = l.kvHeads * l.headDimension;
  add("v.patch_embd", l.hiddenSize, l.patchDimension);
  sections.back().patch = true;
  add("v.embedding_norm.weight", l.hiddenSize);
  for (uint32_t i = 0; i < l.depth; ++i) {
    const std::string blk = "v.blk." + std::to_string(i) + ".";
    add(blk + "ln1.weight", l.hiddenSize);
    if (conv[i]) {
      add(blk + "conv2d_in_proj.weight", 3 * l.hiddenSize, l.hiddenSize);
      add(blk + "conv2d_spatial.weight", l.hiddenSize, 9);
      add(blk + "conv2d_out_proj.weight", l.hiddenSize, l.hiddenSize);
    } else {
      add(blk + "attn_q.weight", l.hiddenSize, l.hiddenSize);
      add(blk + "attn_k.weight", kv, l.hiddenSize);
      add(blk + "attn_v.weight", kv, l.hiddenSize);
      add(blk + "attn_q_norm.weight", l.headDimension);
      add(blk + "attn_k_norm.weight", l.headDimension);
      add(blk + "attn_out.weight", l.hiddenSize, l.hiddenSize);
    }
    add(blk + "ln2.weight", l.hiddenSize);
    add(blk + "ffn_gate.weight", l.intermediateSize, l.hiddenSize);
    add(blk + "ffn_up.weight", l.intermediateSize, l.hiddenSize);
    add(blk + "ffn_down.weight", l.hiddenSize, l.intermediateSize);
  }
  add("mm.input_norm.weight", l.hiddenSize);
  add("mm.input_norm.bias", l.hiddenSize);
  add("mm.0.weight", l.mergedHiddenSize, l.mergedHiddenSize);
  add("mm.0.bias", l.mergedHiddenSize);
  add("mm.2.weight", l.outputHiddenSize, l.mergedHiddenSize);
  add("mm.2.bias", l.outputHiddenSize);
  uint64_t at = kWeightFileAlignment;
  for (auto &s : sections) {
    s.offset = at;
    at = alignWeightOffset(at + uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  }
  result.bytes = at;
  return result;
}

// A source tensor of a dtype the LFM writer reads: the block projections
// are Q8_0 (dequantized at write), the norms, depthwise kernels, patch
// embeddings and projector F32, F16 or BF16.
Input lfmInput(std::string name, SourceTensor tensor) {
  if (tensor.dtype != "Q8_0" && tensor.dtype != "BF16" && tensor.dtype != "F16" &&
      tensor.dtype != "F32")
    throw WeightStoreError("vision tensor " + name + " in " + tensor.file->path().string() +
                           " is " + tensor.dtype + "; preparation reads Q8_0, BF16, F16 or F32");
  return {std::move(name), std::move(tensor)};
}

// The metadata of a clip lfm3vl mmproj of this layout.
void requireLfmMmprojMetadata(const GgufFile &gguf, const ops::LfmVisionLayout &layout) {
  // A mixed-modality mmproj names the tower by its modality key (llama.cpp
  // falls back from clip.projector_type to clip.vision.projector_type).
  const std::optional<std::string> projector =
      gguf.stringValue("clip.projector_type").has_value()
          ? gguf.stringValue("clip.projector_type")
          : gguf.stringValue("clip.vision.projector_type");
  if (gguf.architecture() != "clip" || projector != "lfm3vl")
    throw WeightStoreError("unsupported vision GGUF architecture");
  for (const auto &[key, expected] : std::initializer_list<std::pair<const char *, uint64_t>>{
           {"clip.vision.projection_dim", layout.outputHiddenSize},
           {"clip.vision.patch_size", layout.patchSize},
           {"clip.vision.embedding_length", layout.hiddenSize},
           {"clip.vision.feed_forward_length", layout.intermediateSize},
           {"clip.vision.block_count", layout.depth},
           {"clip.vision.attention.head_count", layout.heads},
           {"clip.vision.attention.head_count_kv", layout.kvHeads},
           {"clip.vision.spatial_merge_size", layout.spatialMerge},
           {"clip.use_silu", 1}})
    if (gguf.unsignedValue(key) != expected) throw WeightStoreError(std::string("vision metadata mismatch: ") + key);
  const auto epsilon = gguf.floatValue("clip.vision.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) || *epsilon <= 0.0 || *epsilon > 1e-5)
    throw WeightStoreError("vision norm epsilon mismatch");
  if (gguf.stringValue("clip.vision.image_resize_algo") != "bicubic")
    throw WeightStoreError("vision image resize algorithm mismatch");
  for (const char *key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto values = gguf.numericArray(key);
    if (!values || values->size() != 3 ||
        !std::all_of(values->begin(), values->end(), [](double x) { return x == 0.5; }))
      throw WeightStoreError("vision image normalization mismatch");
  }
  const auto minPixels = gguf.unsignedValue("clip.vision.image_min_pixels");
  const auto maxPixels = gguf.unsignedValue("clip.vision.image_max_pixels");
  if (!minPixels || !maxPixels || !*minPixels || *maxPixels < *minPixels)
    throw WeightStoreError("vision image pixel bounds mismatch");
}

// The per-block kind an lfm3vl mmproj declares by the tensors it carries:
// conv2d_in_proj marks a conv block, attn_q an attention block; a block
// holding both or neither is malformed.
std::vector<bool> lfmConvBlocks(const GgufFile &gguf, uint32_t depth) {
  std::vector<bool> conv(depth);
  for (uint32_t i = 0; i < depth; ++i) {
    const std::string blk = "v.blk." + std::to_string(i) + ".";
    conv[i] = gguf.find(blk + "conv2d_in_proj.weight") != nullptr;
    const bool attn = gguf.find(blk + "attn_q.weight") != nullptr;
    if (conv[i] == attn)
      throw WeightStoreError("lfm3vl block " + std::to_string(i) +
                             " is neither a complete conv nor a complete attention block");
  }
  return conv;
}

// Binds every planned section to its mmproj tensor, leaving the file's other
// sections (a.*, mm.a.*, a.gen.*) to the audio loaders that share it.
void bindLfmMmproj(const GgufFile &gguf, const ops::LfmVisionLayout &layout, Plan &plan) {
  requireLfmMmprojMetadata(gguf, layout);
  const uint64_t p = layout.patchSize;
  const auto bind = [&](Section &s, const std::string &name, const std::vector<uint64_t> &shape) {
    const GgufTensor &t = gguf.require(name);
    if (t.dims != shape) throw WeightStoreError("vision tensor shape mismatch: " + name);
    s.inputs.push_back(lfmInput(name, {&gguf.source(), ggmlTypeName(t.type), t.dims, t.offset, t.bytes}));
  };
  for (auto &s : plan.sections) {
    if (s.patch) {
      for (uint32_t frame = 0; frame < 2; ++frame)
        bind(s, s.gguf + ".weight" + (frame ? ".1" : ""), {p, p, 3, s.rows});
      continue;
    }
    if (s.columns == 1) {
      bind(s, s.gguf, {s.rows});
    } else if (s.gguf.ends_with("conv2d_spatial.weight")) {
      bind(s, s.gguf, {3, 3, 1, s.rows});
    } else {
      bind(s, s.gguf, {s.columns, s.rows});
    }
  }
}

// Round-to-nearest-even BF16. The lfm3vl mmproj's weights are already
// quantized (Q8_0 blocks, F32 kernels, F16 patch embeddings): dequantizing
// into the BF16 image rounds once more on top of the source's own rounding.
uint16_t roundBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
}

// Converts count values of one source tensor row to BF16 at destination.
// A Q8_0 row is 34-byte blocks of an F16 scale and 32 int8 values.
void convertRow(const SourceTensor &tensor, const uint8_t *source, uint16_t *destination,
                uint64_t count) {
  if (tensor.dtype == "BF16") {
    std::memcpy(destination, source, count * kBFloat16Bytes);
    return;
  }
  if (tensor.dtype == "Q8_0") {
    const uint8_t *block = source;
    for (uint64_t i = 0; i < count; i += 32) {
      _Float16 half;
      std::memcpy(&half, block, 2);
      const float scale = static_cast<float>(half);
      for (uint64_t j = 0; j < 32 && i + j < count; ++j)
        destination[i + j] = roundBfloat16(scale * float(static_cast<int8_t>(block[2 + j])));
      block += 34;
    }
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
  return tensor.dtype == "Q8_0" ? columns / 32 * 34 : columns * (tensor.dtype == "F32" ? 4 : 2);
}

struct LfmStaging {
  std::vector<uint8_t> source;
  std::vector<uint16_t> values;
};

// The tasks that write one section in batches of whole rows, as
// VisionPreparation's do: the patch section interleaves the two temporal
// frame kernels into (channel, frame, pixel) columns like the Qwen plan.
void addLfmSectionTasks(uint8_t *image, const Section &s, uint32_t pixels,
                        std::vector<std::function<void(LfmStaging &)>> &tasks) {
  const auto frames = static_cast<uint32_t>(s.inputs.size());
  const uint32_t columns = s.columns / frames;
  uint64_t widest = 0;
  for (const auto &in : s.inputs)
    widest = std::max(widest, sourceRowBytes(in.tensor, columns));
  const auto batchRows = static_cast<uint32_t>(
      std::clamp<uint64_t>(kLoadStepBytes / std::max<uint64_t>(widest, 1), 1, s.rows));
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) + uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, pixels, frames, columns, row, count, output](LfmStaging &staging) {
      const auto convertRows = [&](const Input &in, uint16_t *to, uint32_t stride) {
        const uint64_t bytes = sourceRowBytes(in.tensor, columns);
        if (staging.source.size() < count * bytes) staging.source.resize(count * bytes);
        in.tensor.read(row * bytes, std::span(staging.source).first(count * bytes));
        for (uint32_t r = 0; r < count; ++r)
          convertRow(in.tensor, staging.source.data() + r * bytes, to + uint64_t(r) * stride, columns);
      };
      if (!s.patch) {
        convertRows(s.inputs.front(), output, s.storedColumns);
      } else {
        if (staging.values.size() < uint64_t(count) * s.columns)
          staging.values.resize(uint64_t(count) * s.columns);
        uint16_t *values = staging.values.data();
        for (uint32_t frame = 0; frame < frames; ++frame)
          convertRows(s.inputs[frame], values + uint64_t(frame) * count * columns, columns);
        for (uint32_t r = 0; r < count; ++r)
          for (uint32_t c = 0; c < s.columns; ++c) {
            const uint32_t channel = c / (2 * pixels), frame = c / pixels % 2, pixel = c % pixels;
            output[uint64_t(r) * s.storedColumns + c] =
                values[(uint64_t(frame) * count + r) * columns + uint64_t(channel) * pixels + pixel];
          }
      }
    });
  }
}

void writeLfmVision(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes)
    throw WeightStoreError("lfm vision image destination size differs");
  const auto header = weightFileHeader(kLfmVisionMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset, uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(LfmStaging &)>> tasks;
  for (const auto &s : plan.sections)
    addLfmSectionTasks(destination.data(), s, plan.patchSize * plan.patchSize, tasks);
  std::vector<LfmStaging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace

struct LfmVisionLoader::Planned {
  ops::LfmVisionLayout layout;
  std::unique_ptr<WeightSource> mmproj;
  std::vector<bool> conv;
  Plan plan;
  void checkUnchanged() const { mmproj->checkUnchanged(); }
};

LfmVisionLoader::LfmVisionLoader(const std::filesystem::path &directory,
                                 const ops::LfmVisionLayout &layout) {
  requireLfmVisionLayout(layout);
  auto planned = std::make_shared<Planned>();
  planned->layout = layout;
  planned->mmproj = std::make_unique<WeightSource>(directory / "mmproj.gguf");
  const GgufFile gguf(*planned->mmproj);
  planned->conv = lfmConvBlocks(gguf, layout.depth);
  planned->plan = lfmPlan(layout, planned->conv);
  bindLfmMmproj(gguf, layout, planned->plan);
  planned_ = std::move(planned);
}

LfmVisionLoader::~LfmVisionLoader() = default;

const ops::LfmVisionLayout &LfmVisionLoader::layout() const noexcept { return planned_->layout; }

// The per-block conv flags the weight reader follows over the image's
// sections: true at the conv blocks' indices.
const std::vector<bool> &LfmVisionLoader::convBlocks() const noexcept { return planned_->conv; }

ImagePlan LfmVisionLoader::image() const {
  return {"vision/lfm.bin", std::string(kLfmVisionMagic), planned_->layout.depth, 0,
          planned_->plan.bytes,
          [planned = planned_](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
            writeLfmVision(bytes, planned->plan);
            planned->checkUnchanged();
          }};
}

uint64_t lfmVisionImageBytes(const ops::LfmVisionLayout &layout) {
  // An upper bound the memory plan tolerates: every block planned with both
  // operator sets, so the real image is never larger.
  const std::vector<bool> allConv(layout.depth, true), allAttn(layout.depth, false);
  uint64_t conv = lfmPlan(layout, allConv).bytes, attn = lfmPlan(layout, allAttn).bytes;
  return std::max(conv, attn);
}

void requireLfmVisionLayout(const ops::LfmVisionLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.patchDimension || !layout.intermediateSize ||
      !layout.mergedHiddenSize || !layout.outputHiddenSize || !layout.heads || !layout.kvHeads ||
      !layout.headDimension || !layout.patchSize || !layout.spatialMerge ||
      layout.heads % layout.kvHeads ||
      layout.heads * layout.headDimension != layout.hiddenSize ||
      layout.mergedHiddenSize != layout.hiddenSize * layout.spatialMerge * layout.spatialMerge ||
      layout.patchDimension != 3 * 2 * layout.patchSize * layout.patchSize)
    throw WeightStoreError("LFM vision layout is inconsistent");
}

} // namespace richengine::model
