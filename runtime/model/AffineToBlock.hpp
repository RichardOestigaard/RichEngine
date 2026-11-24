#pragma once

// The block (MDGG0001) layout of an affine image's 4-bit projections: the
// same weights the Projection sections write as 256-row affine tiles,
// emitted instead as one AF4 tensor each (metal/abi/QuantFormat.h) — a
// GgufTensorDescriptor, the plane0 code tiles and the meta plane, so a dense
// target loaded with preferBlockWeights reads its projections through the
// GGUF block readers and kernels. Sections that are not projections (the
// norms, the token table's rows) keep their bytes exactly. blockImages
// re-plans an image at load time; writeAffineImage's SectionKind::Block case
// writes it through addBlockTasks.
//
// Quantize sections (the drafts' BF16-sourced projections) can convert — the
// block writer quantizes them on write — but nothing plans draft images in
// the block layout yet: draft attention reads the fused QKV's affine tile
// directly, so draft block loading needs block-aware draft kernels first.
// What a follow-up needs for the MoE targets' images: their non-Q4 sections
// (8-bit routers, expert slabs) have no block form here — blockImages
// rejects them rather than mixing layouts in one file.

#include "model/AffinePreparation.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace richengine::metal {
class MetalBackend;
class MetalBuffer;
} // namespace richengine::metal

namespace richengine::model::affine {

// The section bytes of a rows x columns projection in the block layout, and
// the plane0 and meta offsets within it.
[[nodiscard]] uint64_t blockSectionBytes(uint32_t rows, uint32_t columns);
[[nodiscard]] uint64_t blockSectionPlane0(uint32_t rows, uint32_t columns);
[[nodiscard]] uint64_t blockSectionMeta(uint32_t rows, uint32_t columns);

// A copy of `image` with every projection section re-planned as a Block
// section. Throws WeightStoreError on a projection that is not 4-bit or has
// experts, and on a width that is not a whole block (the block readers take
// a multiple of 256 columns).
[[nodiscard]] Image blockImages(Image image);
[[nodiscard]] std::vector<Image> blockImages(std::vector<Image> images);

// The tasks writeAffineImage runs for a Block section at `at` in its image:
// the descriptor and the repack of every (row, 64-column group) of the
// parts' weights, scales and biases into the plane0 and meta tiles.
void addBlockTasks(const Section &section, uint8_t *at,
                   std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks);

// Whether a Block section's planes can be staged as the checkpoint stores
// them and repacked on the GPU (gguf_repack_stored): every part a bound
// three-field quantized source read by whole rows — the codes, scales and
// biases planes contiguous in it (upstream splash's storedPlanes, ported to
// the affine path). Everything else keeps addBlockTasks' CPU repack.
[[nodiscard]] bool blockStoredPlanes(const Section &section);

// A chunk of a Block section's GPU writes: `stage` reads the chunk's stored
// planes (codes, then scales, then biases) into a staging buffer's host
// bytes, and a gguf_repack_stored dispatch writes their plane0 and meta
// tiles into [viewOffset, viewOffset + viewBytes) of the image buffer.
struct BlockGpuChunk final {
  std::function<void(uint8_t *host)> stage;
  uint64_t viewOffset = 0, viewBytes = 0;
  std::vector<uint8_t> params;   // a GgufRepackStoredParams
  uint64_t items = 0;            // (row, 32-column group) pairs
  uint64_t stagingBytes = 0;
};

// The descriptor task of a Block section whose planes the GPU writes
// (blockStoredPlanes), and its chunks.
void addBlockGpuTasks(const Section &section, uint8_t *at,
                      std::vector<std::function<void(std::vector<uint8_t> &)>> &tasks,
                      std::vector<BlockGpuChunk> &chunks);

// Runs the chunks through two shared staging buffers, staging chunk i while
// the GPU writes chunk i - 1 into image (pipelineChunks,
// model/GgufPreparation.hpp).
void writeBlockGpuChunks(metal::MetalBackend &backend, const metal::MetalBuffer &image,
                         const std::vector<BlockGpuChunk> &chunks);

} // namespace richengine::model::affine
