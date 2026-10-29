#include "model/LfmGenLoader.hpp"
#include "model/GgufFile.hpp"
#include "model/VisionPreparation.hpp"
#include "model/WeightLayout.hpp"
#include "ops/LfmSpeechGen.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <functional>
#include <span>
#include <vector>

// Section order must match decodeLfmSpeechWeights' take() sequence.
namespace richengine::model {
namespace {

using vision::Input;
using vision::Plan;
using vision::Section;

inline constexpr std::string_view kLfmGenMagic = "LFGN0001";

Plan genPlan(const std::vector<bool> &conv) {
  Plan result{0, 0, 0, {}};
  auto &sections = result.sections;
  const auto add = [&](std::string gguf, uint64_t values) {
    sections.push_back({"", std::move(gguf), 1, static_cast<uint32_t>(values),
                        1, static_cast<uint32_t>(values)});
  };
  add("a.gen.code.proj_in.weight", 2048ULL * 12288);
  add("a.gen.code.proj_in.bias", 12288);
  add("a.gen.code.embd.weight", 2017ULL * 12 * 1024);
  add("a.gen.code.mask_embd", 1024);
  add("a.gen.code.out_embd.weight", 24204ULL * 2048);
  add("a.gen.code.output_norm.weight", 1024);
  add("a.gen.code.head_norm.weight", 12ULL * 1024);
  for (int i = 0; i < 3; ++i) {
    const std::string blk = "a.gen.code.blk." + std::to_string(i) + ".";
    add(blk + "ln1.weight", 1024);
    add(blk + "attn_q.weight", 1024ULL * 1024);
    add(blk + "attn_k.weight", 256ULL * 1024);
    add(blk + "attn_v.weight", 256ULL * 1024);
    add(blk + "attn_q_norm.weight", 32);
    add(blk + "attn_k_norm.weight", 32);
    add(blk + "attn_gate.weight", 32ULL * 1024);
    add(blk + "attn_out.weight", 1024ULL * 1024);
    add(blk + "ln2.weight", 1024);
    add(blk + "ffn_gate.weight", 2816ULL * 1024);
    add(blk + "ffn_up.weight", 2816ULL * 1024);
    add(blk + "ffn_down.weight", 1024ULL * 2816);
  }
  add("a.gen.wav.quant.first.codebook.weight", 2016ULL * 4);
  add("a.gen.wav.quant.first.out_proj.weight", 48ULL * 256);
  add("a.gen.wav.quant.first.out_proj.bias", 256);
  add("a.gen.wav.tfm.in_proj.weight", 256ULL * 512);
  add("a.gen.wav.tfm.in_proj.bias", 512);
  add("a.gen.wav.tfm.out_proj.weight", 512ULL * 1282);
  add("a.gen.wav.tfm.out_proj.bias", 1282);
  add("a.gen.wav.tfm.output_norm.weight", 512);
  for (int i = 0; i < 8; ++i) {
    const std::string blk = "a.gen.wav.tfm.blk." + std::to_string(i) + ".";
    add(blk + "ln1.weight", 512);
    add(blk + "attn_post_norm.weight", 512);
    if (conv[i]) {
      add(blk + "conv_pw1.weight", 512ULL * 1536);
      add(blk + "conv_dw.weight", 7ULL * 512);
      add(blk + "conv_pw2.weight", 512ULL * 512);
    } else {
      add(blk + "attn_q.weight", 512ULL * 512);
      add(blk + "attn_k.weight", 128ULL * 512);
      add(blk + "attn_v.weight", 128ULL * 512);
      add(blk + "attn_q_norm.weight", 64);
      add(blk + "attn_k_norm.weight", 64);
      add(blk + "attn_gate.weight", 8ULL * 512);
      add(blk + "attn_out.weight", 512ULL * 512);
    }
    add(blk + "ln2.weight", 512);
    add(blk + "ffn_gate.weight", 2304ULL * 512);
    add(blk + "ffn_up.weight", 2304ULL * 512);
    add(blk + "ffn_down.weight", 512ULL * 2304);
    add(blk + "ffn_post_norm.weight", 512);
  }
  uint64_t at = kWeightFileAlignment;
  for (auto &s : sections) {
    s.offset = at;
    at = alignWeightOffset(at + uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  }
  result.bytes = at;
  return result;
}

Input genInput(std::string name, SourceTensor tensor) {
  if (tensor.dtype != "BF16" && tensor.dtype != "F16" && tensor.dtype != "F32" &&
      tensor.dtype != "Q8_0")
    throw WeightStoreError("lfm-gen tensor " + name + " in " +
                           tensor.file->path().string() + " is " + tensor.dtype +
                           "; preparation reads Q8_0, BF16, F16 or F32");
  return {std::move(name), std::move(tensor)};
}

void requireGenMetadata(const GgufFile &gguf) {
  if (gguf.architecture() != "clip" ||
      gguf.stringValue("clip.gen.audio.projector_type") != "lfm3a_gen")
    throw WeightStoreError("unsupported lfm-gen GGUF metadata");
  const auto epsilon = gguf.floatValue("clip.gen.audio.attention.layer_norm_epsilon");
  if (!epsilon || !std::isfinite(*epsilon) || *epsilon <= 0.0 || *epsilon > 1e-5)
    throw WeightStoreError("lfm-gen norm epsilon mismatch");
}

std::vector<bool> genConvBlocks(const GgufFile &gguf) {
  std::vector<bool> conv(8);
  for (int i = 0; i < 8; ++i) {
    const std::string blk = "a.gen.wav.tfm.blk." + std::to_string(i) + ".";
    conv[i] = gguf.find(blk + "conv_pw1.weight") != nullptr;
    const bool attn = gguf.find(blk + "attn_q.weight") != nullptr;
    if (conv[i] == attn)
      throw WeightStoreError("lfm-gen wav block " + std::to_string(i) +
                             " is neither a complete conv nor a complete attention block");
  }
  const std::vector<bool> expected{true, true, false, true, false, true, false, true};
  if (conv != expected)
    throw WeightStoreError("lfm-gen wav layer schedule mismatch");
  return conv;
}

// GGUF dims are innermost-first; a 2-D weight arrives as {in, out} while a
// norm/bias vector arrives as {out}. The plan's stored layout is
// [out x in] rows of bf16 values — the same flat row stream the reader
// consumes (matvec over contiguous rows).
void bindGenMmproj(const GgufFile &gguf, const std::vector<bool> &conv, Plan &plan) {
  requireGenMetadata(gguf);
  for (auto &s : plan.sections) {
    const GgufTensor &t = gguf.require(s.gguf);
    s.inputs.push_back(genInput(s.gguf, {&gguf.source(), ggmlTypeName(t.type), t.dims, t.offset, t.bytes}));
    uint64_t values = 1;
    for (const uint64_t d : t.dims) values *= d;
    if (values != s.columns)
      throw WeightStoreError("lfm-gen tensor size mismatch: " + s.gguf);
  }
  (void)conv;
}

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

struct GenStaging {
  std::vector<uint8_t> source;
};

void addGenSectionTasks(uint8_t *image, const Section &s,
                        std::vector<std::function<void(GenStaging &)>> &tasks) {
  const uint64_t bytes = sourceRowBytes(s.inputs.front().tensor, s.columns);
  auto *output = reinterpret_cast<uint16_t *>(image + s.offset);
  tasks.push_back([&s, bytes, output](GenStaging &staging) {
    const Input &in = s.inputs.front();
    if (staging.source.size() < bytes) staging.source.resize(bytes);
    in.tensor.read(0, std::span(staging.source).first(bytes));
    convertRow(in.tensor, staging.source.data(), output, s.columns);
  });
}

void writeLfmGen(std::span<uint8_t> destination, const Plan &plan) {
  if (destination.size() != plan.bytes)
    throw WeightStoreError("lfm-gen image destination size differs");
  const auto header = weightFileHeader(kLfmGenMagic, 0, 0);
  std::vector<std::pair<uint64_t, uint64_t>> extents{{0, header.size()}};
  for (const auto &s : plan.sections)
    extents.emplace_back(s.offset, uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes);
  zeroUnwritten(destination, std::move(extents));
  std::memcpy(destination.data(), header.data(), header.size());
  std::vector<std::function<void(GenStaging &)>> tasks;
  for (const auto &s : plan.sections) addGenSectionTasks(destination.data(), s, tasks);
  std::vector<GenStaging> staging(loadThreads());
  parallelFor(tasks.size(), [&](size_t index, unsigned thread) { tasks[index](staging[thread]); });
}

} // namespace

struct LfmGenLoader::Planned {
  std::unique_ptr<WeightSource> mmproj;
  std::vector<bool> conv;
  Plan plan;
  void checkUnchanged() const { mmproj->checkUnchanged(); }
};

LfmGenLoader::LfmGenLoader(const std::filesystem::path &directory) {
  auto planned = std::make_shared<Planned>();
  planned->mmproj = std::make_unique<WeightSource>(directory / "mmproj.gguf");
  const GgufFile gguf(*planned->mmproj);
  planned->conv = genConvBlocks(gguf);
  planned->plan = genPlan(planned->conv);
  bindGenMmproj(gguf, planned->conv, planned->plan);
  planned_ = std::move(planned);
}

LfmGenLoader::~LfmGenLoader() = default;

const std::vector<bool> &LfmGenLoader::wavConvBlocks() const noexcept { return planned_->conv; }

std::vector<uint64_t> LfmGenLoader::sectionOffsets() const {
  std::vector<uint64_t> offsets;
  offsets.reserve(planned_->plan.sections.size());
  for (const Section &s : planned_->plan.sections)
    offsets.push_back(s.offset / kBFloat16Bytes);
  return offsets;
}

ImagePlan LfmGenLoader::image() const {
  return {"audio/lfm-gen.bin", std::string(kLfmGenMagic), 0, 0,
          planned_->plan.bytes,
          [planned = planned_](std::span<uint8_t> bytes, const metal::MetalBuffer &) {
            writeLfmGen(bytes, planned->plan);
            planned->checkUnchanged();
          }};
}

uint64_t lfmGenImageBytes() {
  return genPlan({true, true, false, true, false, true, false, true}).bytes;
}

} // namespace richengine::model
