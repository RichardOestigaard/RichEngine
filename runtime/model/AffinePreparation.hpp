#pragma once

// The affine images of a checkpoint in the existing packed ABI, as
// model/AffineTarget.cpp plans an MLX target's and model/DraftCheckpoint.cpp a
// DFlash2 draft's, and their writer. An MLX projection's codes, scales and
// biases are reordered into 256-row tiles without requantization, a BF16
// projection is quantized into the same tiles as MLX's affine quantization
// rounds it, the GDN decay becomes float(-exp(double(A_log))), and every
// other tensor is copied as stored.

#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightSource.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace richengine::model::affine {

using QuantSpec = SafetensorsCheckpoint::QuantSpec;

// Block is a Projection planned into the block image layout instead of the
// affine tiles (model/AffineToBlock.cpp): a GgufTensorDescriptor, the AF4
// plane0 tiles and the meta plane, each at a 16 KiB boundary within the
// section. Oq is an oMLX mixed-precision checkpoint's projections (model/
// OqWrite.cpp): a run of GgufTensorDescriptor + plane0/meta segments at the
// parts' own formats, or F32 float segments for the parts left unquantized.
// OqAffine is an oQ section whose parts the affine tiles encode as well (all
// 4-bit over groups of 64): a GgufTensorDescriptor typed ggml::kAffineQ4, then
// the same planes a Projection section writes, so the oQ reader can pick the
// Affine64 kernels where they measure faster (model/OqWrite.cpp).
enum class SectionKind { Copy, Decay, Projection, Quantize, Block, Oq, OqAffine };

// A checkpoint tensor a section reads: its name, the dtypes it is read in and
// its shape, planned from the layout; tensor is bound once the checkpoint is
// opened.
struct Input {
  std::string name;
  std::vector<std::string> dtypes;
  std::vector<uint64_t> shape;
  const SourceTensor *tensor = nullptr;
};

// Source rows of a fused projection: its MLX weight, scales and biases
// (Projection), or its BF16 weight (Quantize). Oq sections plan each part's
// own spec: bits and groupSize of the checkpoint's, unquantized for a plain
// float weight, and nativeRows for the token table's gathered rows.
struct ProjectionPart {
  uint32_t rows = 0;
  std::vector<Input> fields;
  uint32_t bits = 4, groupSize = 64;
  bool unquantized = false;
  // Quantized in the source but written dequantized into an F32 float
  // segment: the block MoE reads its router and scalar gate as F32.
  bool dequantize = false;
  bool nativeRows = false;
  // A window of a fused source's rows: MiniCPM-V's experts.gate_up_proj
  // stores each expert's gate and up rows interleaved [gate|up], so its
  // sections read expertStride-row source blocks at a row offset. Zero
  // declares the part's rows contiguous — expertStride defaults to rows.
  uint32_t expertStride = 0, expertOffset = 0;
  [[nodiscard]] uint32_t sourceExpertRows() const noexcept {
    return expertStride ? expertStride : rows;
  }
  // The source row of destination row `flat` ([0, experts * rows)).
  [[nodiscard]] uint64_t sourceRow(uint64_t flat) const noexcept {
    return expertStride ? (flat / rows) * expertStride + expertOffset + flat % rows
                        : flat;
  }
};

struct Section {
  SectionKind kind = SectionKind::Copy;
  uint64_t offset = 0, bytes = 0;
  Input input; // Copy and Decay
  // Projection and Quantize: the parts in row order. A Projection's rows past
  // them are zero; a Quantize section has none, nor experts, and 4 bits.
  std::vector<ProjectionPart> parts;
  uint32_t rows = 0, columns = 0, experts = 1, bits = 4;
};

// A 16-byte header (magic, layer, type) in a 16 KiB block, then 16 KiB-aligned
// sections; quantized lists the affine modules it reads and the spec the
// planner expects of each, nullopt for a module left unquantized.
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
  std::vector<std::pair<std::string, std::optional<QuantSpec>>> quantized;
};

// Quantizes one group of 64 BF16 weights to the codes, scale and bias the
// affine (and AF4 block) stores keep, as mlx.core.quantize rounds them.
void quantizeGroup(const uint8_t *weights, uint8_t *codes, uint16_t &scale, uint16_t &bias);

// Writes every byte of a bound image into destination, which is its size.
void writeAffineImage(std::span<uint8_t> destination, const Image &image);

// The packed cache of a safetensors load: every image written verbatim to
// directory/<name>.bin, and .packed listing the safetensors shards it came
// from and the plan that produced it ("name size mtime-ns" per shard, then
// "name magic layer type bytes" per image). A matching stamp with every
// image file present lets later loads read the images as packed files —
// identical bytes, no transpose, no quantize.
std::string packedStamp(const std::filesystem::path &directory,
                        const std::vector<Image> &images);
bool packedCacheUsable(const std::filesystem::path &directory,
                       const std::vector<Image> &images);
// Writes destination to path (a tmp file, then rename) unless it already
// holds the bytes' size — best effort: the cache only speeds later loads.
void cachePackedImage(const std::filesystem::path &path, std::span<const uint8_t> bytes);
// The stamp lands once every image file is present; a partial cache keeps
// failing the check and fills in on later loads.
void refreshPackedStamp(const std::filesystem::path &directory,
                        const std::vector<Image> &images);

} // namespace richengine::model::affine
