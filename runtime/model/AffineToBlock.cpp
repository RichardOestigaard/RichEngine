#include "model/AffineToBlock.hpp"

#include "Checked.hpp"
#include "metal/abi/GgufRepack.h"
#include "model/AffinePlan.hpp"
#include "model/GgufImageLayout.hpp"
#include "model/GgufPreparation.hpp"
#include "model/WeightImages.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
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

// The task writing the descriptor and the section's internal alignment
// gaps: the extent is the whole section, so zeroUnwritten covers none of
// them. The CPU and the stored-planes GPU writes of a section share it.
void addBlockDescriptorTask(const Section &section, uint8_t *at,
                            std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks) {
  const uint32_t rows = section.rows, columns = section.columns;
  const uint64_t plane0 = blockSectionPlane0(rows, columns), meta = blockSectionMeta(rows, columns);
  const uint64_t codes = plane0Bytes(rows, columns);
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
}

// The bytes of one row of the codes plane (4-bit codes, two a byte) and of
// the scales and biases planes (a bf16 a 64-column group) of a section's
// parts, staged as the checkpoint stores them.
constexpr uint32_t codeRowBytes(uint32_t columns) { return columns / 2; }
constexpr uint32_t metaRowBytes(uint32_t columns) { return (columns / kQ4GroupElements) * 2; }

// The reads a chunk's planes stage as, as GgufPreparation splits its
// chunks' rows into kChunkReadBytes: each part's run is a contiguous piece
// of each of its three tensors, split so the load threads keep reads in
// flight.
constexpr uint64_t kBlockReadBytes = 1 << 20;
struct PlaneRead {
  const SourceTensor *tensor;
  uint64_t at;
  uint8_t *to;
  uint64_t bytes;
};

// Rows [firstRow, firstRow + rows) of the parts staged as stored planes:
// every part's run of the codes, scales and biases tensors, each a
// contiguous read. Rows past the parts are zero, as gatherRows leaves them.
void stagePlanes(const Section &section, uint32_t firstRow, uint32_t rows, uint8_t *staging) {
  const uint64_t codes = codeRowBytes(section.columns), meta = metaRowBytes(section.columns);
  uint8_t *const scales = staging + uint64_t(rows) * codes;
  uint8_t *const biases = scales + uint64_t(rows) * meta;
  std::vector<PlaneRead> reads;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + rows, partStart + part.rows);
    if (begin < end) {
      const uint64_t out = begin - firstRow, run = end - begin, source = begin - partStart;
      reads.push_back({part.fields[0].tensor, source * codes, staging + out * codes, run * codes});
      reads.push_back({part.fields[1].tensor, source * meta, scales + out * meta, run * meta});
      reads.push_back({part.fields[2].tensor, source * meta, biases + out * meta, run * meta});
    }
    partStart += part.rows;
  }
  std::vector<PlaneRead> pieces;
  for (const PlaneRead &read : reads)
    for (uint64_t at = 0; at < read.bytes; at += kBlockReadBytes)
      pieces.push_back({read.tensor, read.at + at, read.to + at, std::min(kBlockReadBytes, read.bytes - at)});
  parallelFor(pieces.size(), [&](size_t index, unsigned) {
    const PlaneRead &piece = pieces[index];
    piece.tensor->read(piece.at, {piece.to, piece.bytes});
  });
  if (firstRow + rows > partStart) {
    const uint32_t gathered = std::max(firstRow, partStart) - firstRow;
    std::memset(staging + gathered * codes, 0, uint64_t(rows - gathered) * codes);
    std::memset(scales + gathered * meta, 0, uint64_t(rows - gathered) * meta);
    std::memset(biases + gathered * meta, 0, uint64_t(rows - gathered) * meta);
  }
}

template <class Params>
std::vector<uint8_t> paramBytes(const Params &params) {
  std::vector<uint8_t> bytes(sizeof(params));
  std::memcpy(bytes.data(), &params, sizeof(params));
  return bytes;
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
  addBlockDescriptorTask(section, at, tasks);
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

bool blockStoredPlanes(const Section &section) {
  if (section.bits != 4 || section.experts != 1 || section.columns % kGgufBlockColumns) return false;
  // The kernel's offsets into the dst view are 32-bit.
  if (blockSectionBytes(section.rows, section.columns) - blockSectionPlane0(section.rows, section.columns) >
      std::numeric_limits<uint32_t>::max())
    return false;
  for (const ProjectionPart &part : section.parts)
    if (part.fields.size() != 3 || part.unquantized || part.dequantize || part.nativeRows ||
        part.expertStride || part.expertOffset || !part.fields[0].tensor || !part.fields[1].tensor ||
        !part.fields[2].tensor)
      return false;
  return true;
}

void addBlockGpuTasks(const Section &section, uint8_t *at,
                      std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks,
                      std::vector<BlockGpuChunk> &chunks) {
  addBlockDescriptorTask(section, at, tasks);
  const uint32_t rows = section.rows, columns = section.columns;
  const uint64_t plane0 = blockSectionPlane0(rows, columns), meta = blockSectionMeta(rows, columns);
  const uint64_t codes = codeRowBytes(columns), metaRow = metaRowBytes(columns);
  const uint64_t rowBytes = codes + 2 * metaRow;
  // Chunks of whole rows within the repack staging bound (the GGUF path's):
  // a chunk's staged planes are its codes, then its scales, then its biases.
  const uint32_t stepRows = std::max<uint32_t>(1, static_cast<uint32_t>(kGgufRepackStagingBytes / rowBytes));
  for (uint32_t firstRow = 0; firstRow < rows; firstRow += stepRows) {
    const uint32_t count = std::min(stepRows, rows - firstRow);
    const uint64_t scales = uint64_t(count) * codes, biases = scales + uint64_t(count) * metaRow;
    GgufRepackStoredParams params{};
    params.rows = count;
    params.input_size = columns;
    params.first_row = firstRow;
    params.fmt = GGUF_FMT_AF4;
    params.codes_row_bytes = static_cast<uint32_t>(codes);
    params.meta_row_bytes = static_cast<uint32_t>(metaRow);
    params.src_scales = static_cast<uint32_t>(scales);
    params.src_biases = static_cast<uint32_t>(biases);
    params.dst_meta = static_cast<uint32_t>(meta - plane0);
    chunks.push_back({[&section, firstRow, count](uint8_t *host) {
                        stagePlanes(section, firstRow, count, host);
                      },
                      section.offset + plane0,
                      meta + metaBytes(rows, columns) - plane0,
                      paramBytes(params),
                      uint64_t(count) * (columns / 32),
                      uint64_t(count) * rowBytes});
  }
}

void writeBlockGpuChunks(metal::MetalBackend &backend, const metal::MetalBuffer &image,
                         const std::vector<BlockGpuChunk> &chunks) {
  if (chunks.empty()) return;
  uint64_t staging = 0;
  for (const BlockGpuChunk &chunk : chunks) staging = std::max(staging, chunk.stagingBytes);
  // Two staging buffers serve every chunk, one while a single chunk has
  // nothing to overlap — writeGgufImage's scheme (model/GgufPreparation.cpp).
  std::array<metal::MetalBuffer, 2> inputs;
  for (size_t i = 0; i < std::min<size_t>(2, chunks.size()); ++i)
    inputs[i] = backend.allocateBuffer(staging, metal::BufferStorage::Shared, "load/affine-planes");
  pipelineChunks(
      chunks.size(),
      [&](size_t index, unsigned buffer) {
        chunks[index].stage(static_cast<uint8_t *>(inputs[buffer].contents()));
      },
      [&](size_t index, unsigned buffer) {
        const BlockGpuChunk &chunk = chunks[index];
        const metal::ComputeDispatch dispatch{
            "gguf_repack_stored",
            {{0, inputs[buffer]}, {1, backend.view(image, chunk.viewOffset, chunk.viewBytes)}},
            {{2, chunk.params.data(), chunk.params.size()}},
            {(chunk.items + 255) / 256, 1, 1},
            {256, 1, 1}};
        return backend.submitCommandAsync(std::span<const metal::ComputeDispatch>(&dispatch, 1));
      });
}

} // namespace richengine::model::affine
