#include "model/AffineToBlock.hpp"

#include "Checked.hpp"
#include "model/AffinePlan.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace richengine::model::affine {
namespace {

// The section is one extent: the descriptor's block, the plane0 block and
// the meta plane, the first two 16 KiB-aligned.
uint64_t plane0Bytes(uint32_t rows, uint32_t columns) {
  return uint64_t(rows) * (columns / 32) * kQuantFormats[GGUF_FMT_AF4].plane0_bytes;
}
uint64_t metaBytes(uint32_t rows, uint32_t columns) {
  return uint64_t(rows) * (columns / 64) * kQuantFormats[GGUF_FMT_AF4].meta_bytes;
}

// Rows [firstRow, firstRow + rows) of the parts, gathered into staging:
// every row's packed codes, then its scales, then its biases. Rows past the
// parts are zero, as the Projection writer leaves them.
void gatherRows(const Section &section, uint32_t firstRow, uint32_t rows, uint8_t *staging) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint64_t codeBytes = uint64_t(rows) * (section.columns / 2);
  const uint64_t metaRowBytes = uint64_t(groups) * 2;
  uint8_t *const scales = staging + codeBytes;
  uint8_t *const biases = scales + uint64_t(rows) * metaRowBytes;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + rows, partStart + part.rows);
    for (uint32_t row = begin; row < end; ++row) {
      const uint64_t local = row - partStart, out = row - firstRow;
      part.fields[0].tensor->read(local * (section.columns / 2),
                                  {staging + out * (section.columns / 2), section.columns / 2});
      part.fields[1].tensor->read(local * metaRowBytes, {scales + out * metaRowBytes, metaRowBytes});
      part.fields[2].tensor->read(local * metaRowBytes, {biases + out * metaRowBytes, metaRowBytes});
    }
    partStart += part.rows;
  }
  if (firstRow + rows > partStart) {
    const uint32_t gathered = std::max(firstRow, partStart) - firstRow;
    std::memset(staging + gathered * (section.columns / 2), 0, (rows - gathered) * (section.columns / 2));
    std::memset(scales + gathered * metaRowBytes, 0, (rows - gathered) * metaRowBytes);
    std::memset(biases + gathered * metaRowBytes, 0, (rows - gathered) * metaRowBytes);
  }
}

// The 16 plane0 bytes of group j of the 64-group whose codes are `qs`:
// element e's nibble of qs[16 j + e / 2] at the slot order's place
// (metal/abi/QuantFormat.h's pair words).
void writeGroup(const uint8_t *qs, uint8_t *to) {
  uint8_t slots[32];
  for (uint32_t e = 0; e < 32; ++e) slots[quant_slot(e)] = (qs[e / 2] >> (4 * (e & 1))) & 15;
  for (uint32_t c = 0; c < 4; ++c) {
    uint32_t word = 0;
    for (uint32_t i = 0; i < 8; ++i) word |= uint32_t(slots[8 * c + i]) << ((i & 1) * 16 + 4 * (i >> 1));
    std::memcpy(to + 4 * c, &word, 4);
  }
}

// Quantizes the step's BF16 rows (a Quantize section's parts) straight into
// the section's plane0 and meta tiles: the scale and bias a 64-group's
// quantization computes describe both its 32-groups' codes.
void quantizeRows(const Section &section, uint32_t firstRow, uint32_t rows, uint8_t *staging,
                  uint8_t *plane0, uint8_t *meta) {
  const uint32_t groups = section.columns / 32, blocks = section.columns / 64;
  const uint64_t rowBytes = uint64_t(section.columns) * kBFloat16Bytes;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + rows, partStart + part.rows);
    for (uint32_t row = begin; row < end; ++row)
      part.fields[0].tensor->read(uint64_t(row - partStart) * rowBytes,
                                  {staging + uint64_t(row - firstRow) * rowBytes, rowBytes});
    partStart += part.rows;
  }
  if (firstRow + rows > partStart)
    std::memset(staging + uint64_t(std::max(firstRow, partStart) - firstRow) * rowBytes, 0,
                uint64_t(firstRow + rows - std::max(firstRow, partStart)) * rowBytes);
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t r = firstRow + row;
    const uint8_t *source = staging + uint64_t(row) * rowBytes;
    for (uint32_t block = 0; block < blocks; ++block) {
      uint8_t codes[kQ4GroupElements / 2];
      uint16_t scale, bias;
      quantizeGroup(source + uint64_t(block) * kQ4GroupElements * kBFloat16Bytes, codes, scale, bias);
      writeGroup(codes, plane0 + quant_tile_index(r, 2 * block, groups) * 16);
      writeGroup(codes + 16, plane0 + quant_tile_index(r, 2 * block + 1, groups) * 16);
      uint8_t *unit = meta + quant_tile_index(r, block, blocks) * 4;
      std::memcpy(unit, &scale, 2);
      std::memcpy(unit + 2, &bias, 2);
    }
  }
}

// Repacks the step's rows into the section's plane0 and meta tiles.
void repackRows(const Section &section, uint32_t firstRow, uint32_t rows, uint8_t *staging,
                uint8_t *plane0, uint8_t *meta) {
  const uint32_t groups = section.columns / 32, blocks = section.columns / 64;
  const uint64_t codeBytes = uint64_t(rows) * (section.columns / 2);
  const uint64_t metaRowBytes = uint64_t(blocks) * 2;
  const uint8_t *const scales = staging + codeBytes;
  const uint8_t *const biases = scales + uint64_t(rows) * metaRowBytes;
  for (uint32_t row = 0; row < rows; ++row) {
    const uint32_t r = firstRow + row;
    const uint8_t *codes = staging + uint64_t(row) * (section.columns / 2);
    const uint8_t *scale = scales + uint64_t(row) * metaRowBytes;
    const uint8_t *bias = biases + uint64_t(row) * metaRowBytes;
    for (uint32_t block = 0; block < blocks; ++block) {
      writeGroup(codes + 32 * block,
                 plane0 + quant_tile_index(r, 2 * block, groups) * 16);
      writeGroup(codes + 32 * block + 16,
                 plane0 + quant_tile_index(r, 2 * block + 1, groups) * 16);
      uint8_t *unit = meta + quant_tile_index(r, block, blocks) * 4;
      std::memcpy(unit, scale + 2 * block, 2);
      std::memcpy(unit + 2, bias + 2 * block, 2);
    }
  }
}

} // namespace

uint64_t blockSectionPlane0(uint32_t, uint32_t) { return kWeightFileAlignment; }
uint64_t blockSectionMeta(uint32_t rows, uint32_t columns) {
  return blockSectionPlane0(rows, columns) + alignWeightOffset(plane0Bytes(rows, columns));
}
uint64_t blockSectionBytes(uint32_t rows, uint32_t columns) {
  return checkedAdd<WeightStoreError>(blockSectionMeta(rows, columns), metaBytes(rows, columns),
                                      "block projection section");
}

Image blockImages(Image image) {
  Image result = affine::image(std::string(image.name), image.magic, image.layer,
                               image.type | kBlockWeightsImageType);
  result.quantized = image.quantized;
  for (const Section &section : image.sections) {
    // Quantize sections hold the same 4-bit whole-block projections, their
    // parts just store the BF16 source rather than the MLX tiles.
    const bool projected = section.kind == SectionKind::Projection ||
                           section.kind == SectionKind::Quantize;
    if (!projected) {
      append(result, section);
      continue;
    }
    if (section.bits != 4 || section.experts != 1 || section.columns % kGgufBlockColumns)
      throw WeightStoreError("block weight images take 4-bit single-expert projections of whole blocks: " +
                             image.name);
    Section block = section;
    block.kind = SectionKind::Block;
    block.bytes = blockSectionBytes(section.rows, section.columns);
    append(result, std::move(block));
  }
  return result;
}

std::vector<Image> blockImages(std::vector<Image> images) {
  for (Image &image : images) image = blockImages(std::move(image));
  return images;
}

void addBlockTasks(const Section &section, uint8_t *at,
                   std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  const uint32_t rows = section.rows, columns = section.columns;
  const uint64_t plane0 = blockSectionPlane0(rows, columns), meta = blockSectionMeta(rows, columns);
  const uint64_t codes = plane0Bytes(rows, columns);
  // The descriptor, and the section's internal alignment gaps: the extent is
  // the whole section, so zeroUnwritten covers none of them.
  tasks.push_back([at, rows, columns, plane0, meta, codes](std::vector<uint8_t> &) {
    GgufTensorDescriptor descriptor{};
    descriptor.type = kQuantFormats[GGUF_FMT_AF4].ggml_type;
    descriptor.outputSize = rows;
    descriptor.inputSize = columns;
    const QuantFormat &format = kQuantFormats[GGUF_FMT_AF4];
    descriptor.p0 = format.plane0_bytes;
    descriptor.p1 = format.plane1_bytes;
    descriptor.metaBytes = format.meta_bytes;
    descriptor.metaGroups = format.meta_groups;
    descriptor.plane0Bytes = codes;
    descriptor.metaTotalBytes = uint64_t(rows) * (columns / 64) * format.meta_bytes;
    std::memcpy(at, &descriptor, sizeof descriptor);
    std::memset(at + sizeof descriptor, 0, plane0 - sizeof descriptor);
    std::memset(at + plane0 + codes, 0, meta - (plane0 + codes));
  });
  // Steps of whole rows within the staging bound, as the affine writer's.
  // A converted Quantize section's parts hold one BF16 weight field and
  // quantize on write; a Projection's hold the stored codes, scale and bias.
  const bool quantize = section.parts.front().fields.size() == 1;
  const uint64_t rowBytes = quantize ? uint64_t(columns) * kBFloat16Bytes
                                     : uint64_t(columns) / 2 + 2 * (uint64_t(columns) / 32);
  const uint32_t stepRows = std::max<uint32_t>(1, static_cast<uint32_t>(kLoadStepBytes / rowBytes));
  for (uint32_t firstRow = 0; firstRow < rows; firstRow += stepRows) {
    const uint32_t count = std::min(stepRows, rows - firstRow);
    tasks.push_back([&section, firstRow, count, at, plane0, meta, quantize, rowBytes](std::vector<uint8_t> &staging) {
      if (staging.size() < uint64_t(count) * rowBytes) staging.resize(uint64_t(count) * rowBytes);
      if (quantize)
        quantizeRows(section, firstRow, count, staging.data(), at + plane0, at + meta);
      else {
        gatherRows(section, firstRow, count, staging.data());
        repackRows(section, firstRow, count, staging.data(), at + plane0, at + meta);
      }
    });
  }
}

} // namespace richengine::model::affine
