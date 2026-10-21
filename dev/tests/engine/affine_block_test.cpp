// The block-layout form of an affine Q4 projection (model/AffineToBlock.cpp,
// preferBlockWeights): the image plans an AF4 descriptor and planes that hold
// the same weights, scale for scale and bias for bias, as the affine tiles.
// Runs without Metal: the writers work on plain memory.

#include "TestChecks.hpp"
#include "TestFiles.hpp"
#include "model/AffinePlan.hpp"
#include "model/AffineToBlock.hpp"
#include "model/DFlashDraft.hpp"
#include "model/DraftCheckpoint.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace richengine::model;
using richengine::test::rejects;
using richengine::test::require;
namespace {

constexpr uint32_t kRows = 512, kColumns = 256, kPartRows = 256;
// The fused-window source's per-expert block and the window's share of it
// (the up half of a fused gate_up block: rows 256-511 of a 512-row block).
constexpr uint32_t kWindowExperts = 2, kWindowStride = 512, kWindowRows = 256,
                   kWindowOffset = 256;

uint32_t mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352d;
  value ^= value >> 15;
  value *= 0x846ca68b;
  return value ^ (value >> 16);
}
uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  return uint16_t((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
}
float fromBf16(uint16_t bits) {
  const uint32_t wide = uint32_t(bits) << 16;
  float value;
  std::memcpy(&value, &wide, 4);
  return value;
}

// A safetensors shard with the tensors of one fused projection: two Q4 parts
// of kPartRows rows and kColumns columns each, and a bf16 norm.
void writeFixture(const std::filesystem::path &root) {
  richengine::test::writeFile(
      root / "config.json", R"({"quantization":{"bits":4,"group_size":64}})");
  std::vector<uint8_t> data;
  const auto tensor = [&](std::vector<uint8_t> bytes) {
    data.insert(data.end(), bytes.begin(), bytes.end());
    return data.size() - bytes.size();
  };
  struct Part {
    uint64_t weight, scales, biases;
  } parts[2];
  for (uint32_t part = 0; part < 2; ++part) {
    std::vector<uint8_t> weight(kPartRows * kColumns / 2),
        scales(kPartRows * (kColumns / 64) * 2), biases(scales.size());
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = uint8_t(mix(uint32_t(i) + 17 * part));
    for (size_t i = 0; i < scales.size() / 2; ++i) {
      const float scale = 0.004F + float(mix(uint32_t(i) + 31 * part) % 17) * 0.0001F;
      const uint16_t s = bf16(scale), b = bf16(-7.5F * scale);
      std::memcpy(scales.data() + 2 * i, &s, 2);
      std::memcpy(biases.data() + 2 * i, &b, 2);
    }
    parts[part] = {tensor(std::move(weight)), tensor(std::move(scales)), tensor(std::move(biases))};
  }
  // A fused window source: [kWindowExperts][kWindowStride][columns] as the
  // fused gate_up_proj checkpoints keep them — each window's rows sit
  // `expertOffset` into the source block's per-expert rows.
  std::vector<uint8_t> windowWeight(kWindowExperts * kWindowStride * kColumns / 2),
      windowScales(kWindowExperts * kWindowStride * (kColumns / 64) * 2),
      windowBiases(windowScales.size());
  for (size_t i = 0; i < windowWeight.size(); ++i) windowWeight[i] = uint8_t(mix(uint32_t(i) + 97));
  for (size_t i = 0; i < windowScales.size() / 2; ++i) {
    const float scale = 0.005F + float(mix(uint32_t(i) + 43) % 13) * 0.0002F;
    const uint16_t s = bf16(scale), b = bf16(-3.25F * scale);
    std::memcpy(windowScales.data() + 2 * i, &s, 2);
    std::memcpy(windowBiases.data() + 2 * i, &b, 2);
  }
  const uint64_t windowWeightOffset = tensor(std::move(windowWeight));
  const uint64_t windowScalesOffset = tensor(std::move(windowScales));
  const uint64_t windowBiasesOffset = tensor(std::move(windowBiases));
  const uint64_t norm = tensor(std::vector<uint8_t>(128, 0x3C));
  char header[2048];
  const int length = std::snprintf(
      header, sizeof header,
      R"({"a.weight":{"dtype":"U32","shape":[%u,%u],"data_offsets":[0,%u]},)"
      R"("a.scales":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("a.biases":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("b.weight":{"dtype":"U32","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("b.scales":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("b.biases":{"dtype":"BF16","shape":[%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("window.weight":{"dtype":"U32","shape":[%u,%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("window.scales":{"dtype":"BF16","shape":[%u,%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("window.biases":{"dtype":"BF16","shape":[%u,%u,%u],"data_offsets":[%llu,%llu]},)"
      R"("norm":{"dtype":"BF16","shape":[64],"data_offsets":[%llu,%llu]}})",
      kPartRows, kColumns / 8, kPartRows * kColumns / 2, kPartRows, kColumns / 64,
      (unsigned long long)parts[0].scales, (unsigned long long)parts[0].biases, kPartRows,
      kColumns / 64, (unsigned long long)parts[0].biases, (unsigned long long)parts[1].weight,
      kPartRows, kColumns / 8, (unsigned long long)parts[1].weight, (unsigned long long)parts[1].scales,
      kPartRows, kColumns / 64, (unsigned long long)parts[1].scales, (unsigned long long)parts[1].biases,
      kPartRows, kColumns / 64, (unsigned long long)parts[1].biases, (unsigned long long)windowWeightOffset,
      kWindowExperts, kWindowStride, kColumns / 8, (unsigned long long)windowWeightOffset,
      (unsigned long long)windowScalesOffset,
      kWindowExperts, kWindowStride, kColumns / 64, (unsigned long long)windowScalesOffset,
      (unsigned long long)windowBiasesOffset,
      kWindowExperts, kWindowStride, kColumns / 64, (unsigned long long)windowBiasesOffset,
      (unsigned long long)norm, (unsigned long long)(norm), (unsigned long long)(norm + 128));
  std::vector<uint8_t> file(sizeof(uint64_t));
  const uint64_t headerBytes = uint64_t(length);
  std::memcpy(file.data(), &headerBytes, sizeof headerBytes);
  file.insert(file.end(), header, header + length);
  file.insert(file.end(), data.begin(), data.end());
  richengine::test::writeFile(root / "model.safetensors", file);
}

// The fixture's image: a bf16 norm copy, then a fused projection of the two
// parts, as AffineTarget.cpp's projection() plans it.
affine::Image fixture() {
  affine::Image result = affine::image("layer-0.bin", "MDFN0001", 0, 1);
  affine::copy(result, "norm", {64});
  affine::Section section;
  section.kind = affine::SectionKind::Projection;
  section.rows = kRows;
  section.columns = kColumns;
  section.bits = 4;
  section.experts = 1;
  section.bytes = uint64_t(kRows) * kColumns * 4 / 8 + uint64_t(kRows) * kColumns / 16;
  for (const char *name : {"a", "b"}) {
    result.quantized.emplace_back(name, 4);
    affine::ProjectionPart part{kPartRows, {}};
    part.fields.push_back({std::string(name) + ".weight", {"U32"}, {kPartRows, kColumns / 8}});
    part.fields.push_back({std::string(name) + ".scales", {"BF16"}, {kPartRows, kColumns / 64}});
    part.fields.push_back({std::string(name) + ".biases", {"BF16"}, {kPartRows, kColumns / 64}});
    section.parts.push_back(std::move(part));
  }
  affine::append(result, std::move(section));
  return result;
}

// The fused-window projection, alone in its own image: `experts` windows of
// `window` rows over the fused source's per-expert stride, at its row offset
// (the up half of a fused gate_up block here).
affine::Image windowFixture() {
  affine::Image result = affine::image("layer-window.bin", "MDFN0001", 0, 1);
  result.quantized.emplace_back("window", 4);
  affine::Section window;
  window.kind = affine::SectionKind::Projection;
  window.rows = kWindowRows;
  window.columns = kColumns;
  window.bits = 4;
  window.experts = kWindowExperts;
  affine::ProjectionPart windowPart{kWindowRows, {}};
  windowPart.expertStride = kWindowStride;
  windowPart.expertOffset = kWindowOffset;
  windowPart.fields.push_back({"window.weight", {"U32"}, {kWindowExperts, kWindowStride, kColumns / 8}});
  windowPart.fields.push_back({"window.scales", {"BF16"}, {kWindowExperts, kWindowStride, kColumns / 64}});
  windowPart.fields.push_back({"window.biases", {"BF16"}, {kWindowExperts, kWindowStride, kColumns / 64}});
  window.parts.push_back(std::move(windowPart));
  window.bytes = uint64_t(kWindowRows) * kColumns * 4 / 8 * kWindowExperts +
                 uint64_t(kWindowRows) * kColumns / 16 * kWindowExperts;
  affine::append(result, std::move(window));
  return result;
}

const affine::Section &projectionOf(const affine::Image &image) {
  for (const affine::Section &section : image.sections)
    if (section.kind == affine::SectionKind::Projection || section.kind == affine::SectionKind::Block)
      return section;
  throw std::logic_error("the image has no projection");
}

// The fp32 weight the affine tiles hold at (row, column).
float affineValue(std::span<const uint8_t> image, const affine::Section &section, uint32_t row,
                  uint32_t column) {
  const uint32_t groups = section.columns / 64;
  const uint64_t at = (uint64_t(row / kQ4StorageN) * groups + column / 64) * kQ4StorageN + row % kQ4StorageN;
  const uint8_t *weights = image.data() + section.offset + at * 32;
  const auto *scales = reinterpret_cast<const uint16_t *>(image.data() + section.offset +
                                                        uint64_t(section.rows) * groups * 32);
  const auto *biases = scales + uint64_t(section.rows) * groups;
  const uint32_t code = (weights[(column % 64) / 2] >> (4 * (column & 1))) & 15;
  return fromBf16(scales[at]) * float(code) + fromBf16(biases[at]);
}

// The fp32 weight the block planes hold at (row, column).
float blockValue(std::span<const uint8_t> image, const affine::Section &section, uint32_t row,
                 uint32_t column) {
  const uint32_t groups = section.columns / 32, blocks = section.columns / 64;
  const uint64_t plane0 = section.offset + affine::blockSectionPlane0(section.rows, section.columns);
  const uint64_t meta = section.offset + affine::blockSectionMeta(section.rows, section.columns);
  const uint8_t *packed = image.data() + plane0 + quant_tile_index(row, column / 32, groups) * 16;
  const uint32_t slot = quant_slot(column % 32);
  const uint32_t word = slot / 8, i = slot % 8;
  uint32_t w;
  std::memcpy(&w, packed + 4 * word, 4);
  const uint32_t code = (w >> ((i & 1) * 16 + 4 * (i >> 1))) & 15;
  const uint8_t *unit = image.data() + meta + quant_tile_index(row, column / 64, blocks) * 4;
  uint16_t scale, bias;
  std::memcpy(&scale, unit, 2);
  std::memcpy(&bias, unit + 2, 2);
  return fromBf16(scale) * float(code) + fromBf16(bias);
}

} // namespace

int main() {
  try {
    const richengine::test::TemporaryDirectory directory("richengine-affine-block");
    const std::filesystem::path &root = directory.path();
    writeFixture(root);
    auto planned = std::make_shared<affine::PlannedCheckpoint>(root);
    planned->images = {fixture()};
    for (auto &image : planned->images) affine::bind(image, planned->source);

    const affine::Image &affineImage = planned->images.front();
    const affine::Image blockImage = affine::blockImages(affineImage);
    // The norm keeps its place; the projection became one block section.
    require(blockImage.sections.size() == affineImage.sections.size() &&
                blockImage.sections[0].kind == affine::SectionKind::Copy &&
                projectionOf(blockImage).kind == affine::SectionKind::Block,
            "the block plan's sections changed shape");
    std::vector<uint8_t> affineBytes(affineImage.bytes), blockBytes(blockImage.bytes);
    affine::writeAffineImage(affineBytes, affineImage);
    affine::writeAffineImage(blockBytes, blockImage);

    // The descriptor names an AF4 tensor of the projection's full tile.
    const affine::Section &block = projectionOf(blockImage);
    GgufTensorDescriptor descriptor;
    std::memcpy(&descriptor, blockBytes.data() + block.offset, sizeof descriptor);
    const QuantFormat &format = kQuantFormats[GGUF_FMT_AF4];
    require(descriptor.type == format.ggml_type && descriptor.outputSize == kRows &&
                descriptor.inputSize == kColumns && descriptor.p0 == format.plane0_bytes &&
                !descriptor.p1 && !descriptor.plane1Bytes && descriptor.metaBytes == format.meta_bytes &&
                descriptor.metaGroups == format.meta_groups &&
                descriptor.plane0Bytes == uint64_t(kRows) * (kColumns / 32) * 16 &&
                descriptor.metaTotalBytes == uint64_t(kRows) * (kColumns / 64) * 4,
            "the AF4 descriptor is inconsistent");

    // Every element decodes to the same weight in both layouts, so every dot
    // product is the same.
    const affine::Section &affine = projectionOf(affineImage);
    double sumA = 0, sumB = 0;
    for (uint32_t row = 0; row < kRows; ++row)
      for (uint32_t column = 0; column < kColumns; ++column) {
        const float a = affineValue(affineBytes, affine, row, column);
        const float b = blockValue(blockBytes, block, row, column);
        require(a == b, "the block planes hold a different weight");
        const double x = double(int8_t(mix(row * kColumns + column))) / 64;
        sumA += x * a;
        sumB += x * b;
      }
    require(std::fabs(sumA - sumB) < 1e-9 * std::fabs(sumA) + 1e-12,
            "the block GEMM result differs from the affine one");

    // The fused-window section, planned and written on its own image: flat
    // row e*256+r must decode the source block's row e*512 + 256 + r — the
    // values are the fixture's generators evaluated at that source row.
    auto windowImage = windowFixture();
    affine::bind(windowImage, planned->source);
    std::vector<uint8_t> windowBytes(windowImage.bytes);
    affine::writeAffineImage(windowBytes, windowImage);
    const affine::Section &window = windowImage.sections.back();
    require(window.kind == affine::SectionKind::Projection && window.experts == kWindowExperts &&
                window.parts.size() == 1 && window.parts.front().expertStride == kWindowStride,
            "the fused window section changed shape");
    // The planes keep each expert's fields together: expert e's rows pack
    // from e * (bytes/experts), weights then scales then biases.
    for (uint32_t flat = 0; flat < kWindowExperts * kWindowRows; ++flat) {
      const uint32_t expert = flat / kWindowRows, row = flat % kWindowRows;
      const uint32_t source = expert * kWindowStride + kWindowOffset + row;
      const uint32_t groups = kColumns / 64;
      const uint64_t base = window.offset + uint64_t(expert) * (window.bytes / window.experts);
      for (uint32_t column = 0; column < kColumns; ++column) {
        const uint8_t byte = uint8_t(mix(source * (kColumns / 2) + column / 2 + 97));
        const uint32_t group = source * (kColumns / 64) + column / 64;
        const float scale =
            fromBf16(bf16(0.005F + float(mix(group + 43) % 13) * 0.0002F));
        const float expected =
            scale * float((byte >> (4 * (column & 1))) & 15) +
            fromBf16(bf16(-3.25F * (0.005F + float(mix(group + 43) % 13) * 0.0002F)));
        const uint64_t slot = (uint64_t(row / kQ4StorageN) * groups + column / 64) * kQ4StorageN +
                              row % kQ4StorageN;
        const uint8_t *weights = windowBytes.data() + base + slot * 32;
        const auto *scales = reinterpret_cast<const uint16_t *>(windowBytes.data() + base +
                                                                uint64_t(kWindowRows) * groups * 32);
        const auto *biases = scales + uint64_t(kWindowRows) * groups;
        const uint32_t code = (weights[(column % 64) / 2] >> (4 * (column & 1))) & 15;
        const float actual = fromBf16(scales[slot]) * float(code) + fromBf16(biases[slot]);
        require(actual == expected,
                "the window's row came from the wrong source rows");
      }
    }

    // Non-Q4 and expert sections have no block form.
    auto quantized = fixture();
    for (auto &section : quantized.sections)
      if (section.kind == affine::SectionKind::Projection) section.bits = 8;
    rejects([&] { static_cast<void>(affine::blockImages(quantized)); },
            "block weight images take 4-bit single-expert projections of whole blocks: layer-0.bin",
            "an 8-bit section took the block layout");
    // The three checkpoint draft kinds' projection sections all convert.
    for (const auto &images :
         {draftCheckpointImages(DFlashDraftLayout{}),
          dflashV1DraftCheckpointImages(DFlashDraftLayout{}),
          dsparkDraftCheckpointImages(DFlashDraftLayout{})}) {
      uint32_t projections = 0;
      for (const affine::Image &image : affine::blockImages(images))
        for (const affine::Section &section : image.sections)
          if (section.kind == affine::SectionKind::Block) ++projections;
      require(projections > 0, "a draft checkpoint's projections did not take the block layout");
    }
    std::cout << "affine block: descriptor, repacked planes equal to the affine tiles, fused parts, "
                 "section rejection, draft checkpoints PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
