#include "model/LfmAudioLoader.hpp"
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

// The sections of the packed lfm3a image, in the order the weight reader
// (LfmAudio.cpp) consumes them: the conv frontend's three stages and the
// lin_out projection, then per block its operator norm, its conv or
// attention tensors (the mmproj's per-block kind, so every planned section
// is bound), ffn norm and silu FFN, the final RMSNorm and the projector's
// LayerNorm and two affine maps.
Plan lfmAudioPlan(const ops::LfmAudioLayout &l, const std::vector<bool> &conv) {
  Plan result{l.depth, 0, 0, {}};
  auto &sections = result.sections;
  const auto add = [&](std::string gguf, uint32_t rows, uint32_t columns = 1) {
    sections.push_back({"", std::move(gguf), rows, columns, rows, columns});
  };
  const uint32_t convChannels = l.convChannels;
  const uint32_t hidden = l.hiddenSize;
  const uint32_t kv = l.kvHeads * l.headDimension;
  add("a.conv1d.0.weight", convChannels, 9);
  add("a.conv1d.0.bias", convChannels);
  add("a.conv1d.2.weight", convChannels, 9);
  add("a.conv1d.2.bias", convChannels);
  add("a.conv1d.3.weight", convChannels, convChannels);
  add("a.conv1d.3.bias", convChannels);
  add("a.conv1d.5.weight", convChannels, 9);
  add("a.conv1d.5.bias", convChannels);
  add("a.conv1d.6.weight", convChannels, convChannels);
  add("a.conv1d.6.bias", convChannels);
  add("a.pre_encode.out.weight", hidden,
      l.melBins / l.tokenStride * convChannels);
  add("a.pre_encode.out.bias", hidden);
  for (uint32_t i = 0; i < l.depth; ++i) {
    const std::string blk = "a.blk." + std::to_string(i) + ".";
    add(blk + "ln1.weight", hidden);
    if (conv[i]) {
      add(blk + "conv_pw1.weight", 3 * hidden, hidden);
      add(blk + "conv_dw.weight", hidden, 3);
      add(blk + "conv_pw2.weight", hidden, hidden);
    } else {
      add(blk + "attn_q.weight", hidden, hidden);
      add(blk + "attn_k.weight", kv, hidden);
      add(blk + "attn_v.weight", kv, hidden);
      add(blk + "attn_q_norm.weight", l.headDimension);
      add(blk + "attn_k_norm.weight", l.headDimension);
      add(blk + "attn_out.weight", hidden, hidden);
    }
    add(blk + "ln2.weight", hidden);
    add(blk + "ffn_gate.weight", l.intermediateSize, hidden);
    add(blk + "ffn_up.weight", l.intermediateSize, hidden);
    add(blk + "ffn_down.weight", hidden, l.intermediateSize);
  }
  add("a.post_ln.weight", hidden);
  add("mm.a.mlp.0.weight", hidden);
  add("mm.a.mlp.0.bias", hidden);
  add("mm.a.mlp.1.weight", l.outputHiddenSize, hidden);
  add("mm.a.mlp.1.bias", l.outputHiddenSize);
  add("mm.a.mlp.3.weight", l.outputHiddenSize, l.outputHiddenSize);
  add("mm.a.mlp.3.bias", l.outputHiddenSize);
  uint64_t at = kWeightFileAlignment;
  for (auto &s : sections) {
    s.offset = at;
    at = alignWeightOffset(at + uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  }
  result.bytes = at;
  return result;
}

// A source tensor of a dtype the LFM writer reads: the block projections
// are Q8_0 (dequantized at write), the norms, depthwise kernels, conv
// kernels and projector biases F32.
Input lfmAudioInput(std::string name, SourceTensor tensor) {
  if (tensor.dtype != "Q8_0" && tensor.dtype != "BF16" && tensor.dtype != "F16" &&
      tensor.dtype != "F32")
    throw WeightStoreError("audio tensor " + name + " in " + tensor.file->path().string() +
                           " is " + tensor.dtype + "; preparation reads Q8_0, BF16, F16 or F32");
  return {std::move(name), std::move(tensor)};
}

// The metadata of a clip lfm3a (audio encoder) section of the shared
// LFM3-Omni mmproj.
void requireLfmAudioMetadata(const GgufFile &gguf, const ops::LfmAudioLayout &layout) {
  if (gguf.architecture() != "clip" ||
      gguf.stringValue("clip.audio.projector_type") != "lfm3a")
    throw WeightStoreError("unsupported audio GGUF architecture");
  if (gguf.unsignedValue("clip.has_audio_encoder") != 1)
    throw WeightStoreError("audio GGUF declares no audio encoder");
  for (const auto &[key, expected] :
       std::initializer_list<std::pair<const char *, uint64_t>>{
           {"clip.audio.projection_dim", layout.outputHiddenSize},
           {"clip.audio.embedding_length", layout.hiddenSize},
           {"clip.audio.feed_forward_length", layout.intermediateSize},
           {"clip.audio.block_count", layout.depth},
           {"clip.audio.attention.head_count", layout.heads},
           {"clip.audio.attention.head_count_kv", layout.kvHeads},
           {"clip.audio.num_mel_bins", layout.melBins},
           {"clip.audio.window_size", layout.window}})
    if (gguf.unsignedValue(key) != expected)
      throw WeightStoreError(std::string("audio metadata mismatch: ") + key);
  const auto epsilon = gguf.floatValue("clip.audio.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) || *epsilon <= 0.0 || *epsilon > 1e-5)
    throw WeightStoreError("audio norm epsilon mismatch");
}

// The per-block kind an lfm3a mmproj declares by the tensors it carries:
// conv_pw1 marks a conv block, attn_q an attention block; a block holding
// both or neither is malformed.
std::vector<bool> lfmAudioConvBlocks(const GgufFile &gguf, uint32_t depth) {
  std::vector<bool> conv(depth);
  for (uint32_t i = 0; i < depth; ++i) {
    const std::string blk = "a.blk." + std::to_string(i) + ".";
    conv[i] = gguf.find(blk + "conv_pw1.weight") != nullptr;
    const bool attn = gguf.find(blk + "attn_q.weight") != nullptr;
    if (conv[i] == attn)
      throw WeightStoreError("lfm3a block " + std::to_string(i) +
                             " is neither a complete conv nor a complete attention block");
  }
  return conv;
}

// Binds every planned section to its mmproj tensor, leaving the file's
// other sections (v.*, mm.*, a.gen.*) to the loaders that share it.
void bindLfmAudioMmproj(const GgufFile &gguf, const ops::LfmAudioLayout &layout,
                        Plan &plan) {
  requireLfmAudioMetadata(gguf, layout);
  const auto bind = [&](Section &s, const std::string &name,
                        const std::vector<uint64_t> &shape) {
    const GgufTensor &t = gguf.require(name);
    if (t.dims != shape)
      throw WeightStoreError("audio tensor shape mismatch: " + name);
    s.inputs.push_back(lfmAudioInput(
        name, {&gguf.source(), ggmlTypeName(t.type), t.dims, t.offset, t.bytes}));
  };
  for (auto &s : plan.sections) {
    if (s.columns == 1) {
      if (s.gguf.starts_with("a.conv1d."))
        // conv1d biases carry (1, 1, channels) dims.
        bind(s, s.gguf, {1, 1, s.rows});
      else
        bind(s, s.gguf, {s.rows});
    } else if (s.gguf.starts_with("a.conv1d.") && s.columns == 9) {
      // Dense and depthwise 3x3 conv kernels alike: (kx, ky, 1, out).
      bind(s, s.gguf, {3, 3, 1, s.rows});
    } else if (s.gguf.starts_with("a.conv1d.")) {
      // The 1x1 mixing convs: (kx, ky, in, out).
      bind(s, s.gguf, {1, 1, s.columns, s.rows});
    } else if (s.gguf.ends_with("conv_dw.weight")) {
      // The taps-3 causal depthwise kernel: (tap, channel).
      bind(s, s.gguf, {s.columns, s.rows});
    } else {
      bind(s, s.gguf, {s.columns, s.rows});
    }
  }
}

// Round-to-nearest-even BF16.
uint16_t roundBfloat16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
}

void convertAudioRow(const SourceTensor &tensor, const uint8_t *source,
                     uint16_t *destination, uint64_t count) {
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

uint64_t audioSourceRowBytes(const SourceTensor &tensor, uint64_t columns) {
  return tensor.dtype == "Q8_0" ? columns / 32 * 34
                              : columns * (tensor.dtype == "F32" ? 4 : 2);
}

struct AudioStaging {
  std::vector<uint8_t> source;
};

// The tasks that write one section in batches of whole rows, as
// VisionPreparation's do: every audio section keeps its rows as bound.
void addAudioSectionTasks(uint8_t *image, const Section &s,
                          std::vector<std::function<void(AudioStaging &)>> &tasks) {
  const uint64_t widest = audioSourceRowBytes(s.inputs.front().tensor, s.columns);
  const auto batchRows = static_cast<uint32_t>(std::clamp<uint64_t>(
      kLoadStepBytes / std::max<uint64_t>(widest, 1), 1, s.rows));
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    const uint32_t count = std::min(batchRows, s.rows - row);
    auto *output = reinterpret_cast<uint16_t *>(image + s.offset) +
                   uint64_t(row) * s.storedColumns;
    tasks.push_back([&s, row, count, output](AudioStaging &staging) {
      const Input &in = s.inputs.front();
      const uint64_t bytes = audioSourceRowBytes(in.tensor, s.columns);
      if (staging.source.size() < count * bytes)
        staging.source.resize(count * bytes);
      in.tensor.read(row * bytes,
                     std::span(staging.source).first(count * bytes));
      for (uint32_t r = 0; r < count; ++r)
        convertAudioRow(in.tensor, staging.source.data() + r * bytes,
                        output + uint64_t(r) * s.storedColumns, s.columns);
    });
  }
}

void writeLfmAudio(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes)
    throw WeightStoreError("lfm audio image destination size differs");
  const auto header = weightFileHeader(kLfmAudioMagic, plan.depth, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset,
                         uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(AudioStaging &)>> tasks;
  for (const auto &s : plan.sections)
    addAudioSectionTasks(destination.data(), s, tasks);
  std::vector<AudioStaging> staging(loadThreads());
  parallelFor(tasks.size(),
              [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace

struct LfmAudioLoader::Planned {
  ops::LfmAudioLayout layout;
  std::unique_ptr<WeightSource> mmproj;
  std::vector<bool> conv;
  Plan plan;
  void checkUnchanged() const { mmproj->checkUnchanged(); }
};

LfmAudioLoader::LfmAudioLoader(const std::filesystem::path &directory,
                               const ops::LfmAudioLayout &layout) {
  requireLfmAudioLayout(layout);
  auto planned = std::make_shared<Planned>();
  planned->layout = layout;
  planned->mmproj = std::make_unique<WeightSource>(directory / "mmproj.gguf");
  const GgufFile gguf(*planned->mmproj);
  planned->conv = lfmAudioConvBlocks(gguf, layout.depth);
  planned->plan = lfmAudioPlan(layout, planned->conv);
  bindLfmAudioMmproj(gguf, layout, planned->plan);
  planned_ = std::move(planned);
}

LfmAudioLoader::~LfmAudioLoader() = default;

const ops::LfmAudioLayout &LfmAudioLoader::layout() const noexcept {
  return planned_->layout;
}

const std::vector<bool> &LfmAudioLoader::convBlocks() const noexcept {
  return planned_->conv;
}

ImagePlan LfmAudioLoader::image() const {
  return {"audio/lfm.bin", std::string(kLfmAudioMagic), planned_->layout.depth, 0,
          planned_->plan.bytes,
          [planned = planned_](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
            writeLfmAudio(bytes, planned->plan);
            planned->checkUnchanged();
          }};
}

uint64_t lfmAudioImageBytes(const ops::LfmAudioLayout &layout) {
  // An upper bound the memory plan tolerates: every block planned with
  // both operator sets, so the real image is never larger.
  const std::vector<bool> allConv(layout.depth, true), allAttn(layout.depth, false);
  uint64_t conv = lfmAudioPlan(layout, allConv).bytes,
           attn = lfmAudioPlan(layout, allAttn).bytes;
  return std::max(conv, attn);
}

void requireLfmAudioLayout(const ops::LfmAudioLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.intermediateSize ||
      !layout.outputHiddenSize || !layout.heads || !layout.kvHeads ||
      !layout.headDimension || !layout.melBins || !layout.convChannels ||
      !layout.window || !layout.tokenStride ||
      layout.melBins % layout.tokenStride ||
      layout.heads % layout.kvHeads ||
      layout.heads * layout.headDimension != layout.hiddenSize)
    throw WeightStoreError("LFM audio layout is inconsistent");
}

} // namespace richengine::model
