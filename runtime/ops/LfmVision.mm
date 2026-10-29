#include "LfmVision.hpp"

#include "Checked.hpp"
#include "metal/abi/Vision.h"
#include "ops/KernelNames.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace richengine::ops {
namespace {

using metal::CommandGraph;
using metal::MetalBuffer;

constexpr uint32_t kGemmRowTile = 64;
constexpr uint32_t kGemmColumnTile = 128;
constexpr uint32_t kMergerRowTile = 32;
constexpr uint32_t kMergerColumnTile = 256;
constexpr uint32_t kKeyTile = 128;
constexpr uint32_t kQueryTile = 64;
constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kFloatBytes = 4;

// The LFM3-VL tower the kernels are specialized for (LfmVisionWeights'
// declared layout must match it except outputHiddenSize, the text model's
// hidden width).
constexpr LfmVisionLayout kTower{.depth = 14,
                                 .hiddenSize = 1024,
                                 .patchDimension = 1536,
                                 .intermediateSize = 2560,
                                 .mergedHiddenSize = 4096,
                                 .outputHiddenSize = 2048,
                                 .heads = 16,
                                 .kvHeads = 8,
                                 .headDimension = 64,
                                 .patchSize = 16,
                                 .spatialMerge = 2};
static_assert(kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.intermediateSize % kGemmColumnTile == 0 &&
                  3 * kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.mergedHiddenSize % kMergerColumnTile == 0 &&
                  kTower.patchDimension % kGemmColumnTile == 0,
              "lfm vision kernels' tile assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

// Scratch tensor byte sizes for one encoder sized to maximumPatches, in
// Scratch order. GEMM tiles read whole 64-row tiles, attention whole
// 128-key tiles, and the merger reads (patches / 4, 4096) gathered rows.
std::array<uint64_t, 16> scratchLayout(const LfmVisionLayout &layout,
                                       uint32_t maximumPatches) {
  const uint64_t rows = roundUp(maximumPatches, kGemmRowTile);
  const uint64_t padded = roundUp(maximumPatches, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t kv = uint64_t{layout.kvHeads} * layout.headDimension;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.patchDimension * kBf16Bytes,                     // Patches
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeCos
      uint64_t{maximumPatches} * layout.headDimension * kFloatBytes, // RopeSin
      rows * hidden * kFloatBytes,                                   // Hidden
      rows * hidden * kBf16Bytes,                                    // Normalized
      std::max(rows * 3 * hidden, mergedRows * layout.mergedHiddenSize) *
          kBf16Bytes,                                                // Wide
      rows * layout.intermediateSize * kBf16Bytes,                   // Gate
      rows * layout.intermediateSize * kBf16Bytes,                   // Up
      headRows * layout.headDimension * kBf16Bytes,                  // Queries
      headRows * layout.headDimension * kBf16Bytes,                  // Keys
      headRows * layout.headDimension * kBf16Bytes,                  // Values
      headRows * layout.headDimension * kBf16Bytes,                  // Attention
      rows * hidden * kBf16Bytes,                                    // Query
      rows * kv * kBf16Bytes,                                        // Key
      rows * kv * kBf16Bytes,                                        // Value
      rows * hidden * kBf16Bytes,                                    // Context
  };
}

void requireLayout(const LfmVisionLayout &layout) {
  LfmVisionLayout tower = layout;
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize ||
      layout.outputHiddenSize % kMergerColumnTile) {
    throw std::invalid_argument(
        "vision kernels are specialized for the LFM3-VL 14-block tower");
  }
}

} // namespace

uint64_t LfmVision::scratchBytes(const LfmVisionLayout &layout,
                                 uint32_t maximumPatches) {
  requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vision scratch byte count overflows");
    }
    total += aligned;
  }
  return total;
}

LfmVision::LfmVision(metal::MetalBackend &backend,
                     const LfmVisionWeights &model, uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "lfm-vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (cursor != total)
    throw std::logic_error("lfm vision scratch arena mismatch");
}

void LfmVision::addGemm(CommandGraph &graph, const char *pipeline,
                        const MetalBuffer &input, const MetalBuffer &weight,
                        const MetalBuffer &bias, const MetalBuffer &output,
                        const MetalBuffer &residual, uint32_t outputSize,
                        uint32_t inputSize, uint32_t rows, uint32_t tileRows,
                        uint32_t tileColumns) const {
  graph.add(pipeline, {input, weight, bias, output, residual},
            VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void LfmVision::addNorm(CommandGraph &graph, const MetalBuffer &input,
                        const MetalBuffer &weight, const MetalBuffer &output,
                        uint32_t rows) const {
  graph.add(std::string(kLfmVisionRmsnorm), {input, weight, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

void LfmVision::encode(CommandGraph &graph, ImageGrid grid,
                       const MetalBuffer &pixels,
                       const MetalBuffer &embeddings) const {
  const LfmVisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_) {
    throw std::invalid_argument("image grid exceeds the vision encoder");
  }
  // The patch embed has two temporal kernels: a still image fills both with
  // the same pixels, a video pair carries frame 0 and frame 1.
  if (!grid.frames || grid.frames > 2) {
    throw std::invalid_argument(
        "the LFM3-VL encoder takes one still frame or one frame pair");
  }
  const auto tokens = static_cast<uint32_t>(grid.patches());
  if (!pixels || pixels.sizeBytes() < grid.pixelBytes()) {
    throw std::invalid_argument("image pixels do not cover the grid");
  }
  const uint64_t embeddingBytes =
      uint64_t{Vision::embeddingRows(grid)} * layout.outputHiddenSize *
      kBf16Bytes;
  if (!embeddings || embeddings.sizeBytes() < embeddingBytes) {
    throw std::invalid_argument("embedding buffer is too small for the grid");
  }

  const uint32_t padded = roundUp(tokens, kKeyTile);
  const uint32_t merged = grid.mergedTokens();
  const VisionGridParams gridParams{grid.height, grid.width};
  const VisionPatchifyParams patchifyParams{grid.height, grid.width,
                                            grid.frames};
  const VisionQkvParams qkvParams{tokens, padded, /*fused=*/1};
  const VisionAttentionParams attentionParams{
      tokens, padded, 1.0F / std::sqrt(static_cast<float>(layout.headDimension))};
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &wide = scratch(Scratch::Wide);
  const MetalBuffer &context = scratch(Scratch::Context);

  graph.add(std::string(kLfmVisionPatchify),
            {pixels, scratch(Scratch::Patches)}, patchifyParams,
            {tokens, 1, 1});
  // Pad rows start zeroed so the residual GEMMs and key tiles read finite
  // values; the same pass builds the axial rope tables.
  graph.add(std::string(kLfmVisionPrepare),
            {hidden, scratch(Scratch::RopeCos), scratch(Scratch::RopeSin)},
            gridParams, {roundUp(tokens, kGemmRowTile), 1, 1});
  addGemm(graph, kVisionGemmM64n128.data(), scratch(Scratch::Patches),
          model_.patchEmbedding, model_.zeroBias, context, context,
          layout.hiddenSize, layout.patchDimension, tokens, kGemmRowTile,
          kGemmColumnTile);
  // The residual stream starts from the embedding norm (fp32 rows).
  graph.add(std::string(kLfmVisionRmsnormEmbed),
            {context, model_.embeddingNorm, hidden},
            VisionNormParams{layout.hiddenSize},
            {roundUp(tokens, kGemmRowTile), 1, 1});

  for (const LfmVisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1.weight, normalized, tokens);
    if (block.conv) {
      // in_proj -> [B|C|X] in the wide buffer; the conv kernel writes
      // C * conv(B * X) rows, out_proj folds them into the residual.
      addGemm(graph, kVisionGemmM64n128.data(), normalized, block.inProjection,
              model_.zeroBias, wide, context, 3 * layout.hiddenSize,
              layout.hiddenSize, tokens, kGemmRowTile, kGemmColumnTile);
      graph.add(std::string(kLfmVisionConv2d),
                {wide, block.convKernel, normalized}, gridParams,
                {tokens, 1, 1});
      addGemm(graph, kVisionGemmM64n128Residual.data(), normalized,
              block.outProjection, model_.zeroBias, hidden, hidden,
              layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
              kGemmColumnTile);
    } else {
      // One fused GEMM writes the whole [Q | K | V] row set across the
      // contiguous Query/Key/Value scratch slots — the narrow K and V
      // projections no longer pay a per-GEMM fixed cost.
      const uint32_t kvWidth = layout.kvHeads * layout.headDimension;
      const MetalBuffer &fusedQkv = scratch(Scratch::Query);
      graph.add(std::string(kLfmVisionGemmQkv),
                {normalized, block.query, block.key, block.value, fusedQkv},
                VisionGemmParams{layout.hiddenSize + 2 * kvWidth,
                                 layout.hiddenSize},
                {(tokens + kGemmRowTile - 1) / kGemmRowTile,
                 (layout.hiddenSize + 2 * kvWidth) / kGemmColumnTile, 1});
      // Per-head q/k RMSNorm, axial rope and the 8->16 head GQA scatter in
      // one 32-lane group per (token, head); padded rows write zeros.
      graph.add(std::string(kLfmVisionQkvPrepare),
                {fusedQkv, fusedQkv, fusedQkv, block.queryNorm,
                 block.keyNorm, scratch(Scratch::RopeCos),
                 scratch(Scratch::RopeSin), scratch(Scratch::Queries),
                 scratch(Scratch::Keys), scratch(Scratch::Values)},
                qkvParams, {padded * layout.heads, 1, 1}, {32, 1, 1});
      graph.add(std::string(kLfmVisionAttention),
                {scratch(Scratch::Queries), scratch(Scratch::Keys),
                 scratch(Scratch::Values), scratch(Scratch::Attention)},
                attentionParams,
                {(tokens + kQueryTile - 1) / kQueryTile, layout.heads, 1});
      graph.add(std::string(kLfmVisionAttentionPack),
                {scratch(Scratch::Attention), context}, qkvParams,
                {tokens, 1, 1});
      addGemm(graph, kVisionGemmM64n128Residual.data(), context,
              block.outProjection, model_.zeroBias, hidden, hidden,
              layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
              kGemmColumnTile);
    }

    addNorm(graph, hidden, block.norm2.weight, normalized, tokens);
    addGemm(graph, kVisionGemmM64n128.data(), normalized, block.ffnGate,
            model_.zeroBias, scratch(Scratch::Gate), context,
            layout.intermediateSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    // The up projection's GEMM epilogue is the gated multiply: it reads
    // the gate rows as its fifth binding and writes silu(gate) * up back
    // over them — the standalone silu_mul pass is gone.
    addGemm(graph, kLfmVisionGemmM64n128SiluUp.data(), normalized,
            block.ffnUp, model_.zeroBias, scratch(Scratch::Gate),
            scratch(Scratch::Gate), layout.intermediateSize,
            layout.hiddenSize, tokens, kGemmRowTile, kGemmColumnTile);
    addGemm(graph, kVisionGemmM64n128Residual.data(), scratch(Scratch::Gate),
            block.ffnDown, model_.zeroBias, hidden, hidden, layout.hiddenSize,
            layout.intermediateSize, tokens, kGemmRowTile, kGemmColumnTile);
  }

  // The projector's input LayerNorm (eps 1e-6, with bias) is the shared
  // vision kernel; the merge gather interleaves raster order into the
  // 2x2-block 4096 rows mm.0 reads.
  graph.add(std::string(kVisionLayerNorm),
            {hidden, model_.mergerNorm.weight, model_.mergerNorm.bias,
             normalized},
            VisionNormParams{layout.hiddenSize}, {tokens, 1, 1});
  graph.add(std::string(kLfmVisionMerge), {normalized, wide}, gridParams,
            {merged, 1, 1});
  addGemm(graph, kVisionGemmM32n256GeluErf.data(), wide,
          model_.mergerUpProjection.weight, model_.mergerUpProjection.bias,
          scratch(Scratch::Gate), context, layout.mergedHiddenSize,
          layout.mergedHiddenSize, merged, kMergerRowTile, kMergerColumnTile);
  addGemm(graph, kVisionGemmM32n256.data(), scratch(Scratch::Gate),
          model_.mergerDownProjection.weight, model_.mergerDownProjection.bias,
          embeddings, context, layout.outputHiddenSize,
          layout.mergedHiddenSize, merged, kMergerRowTile, kMergerColumnTile);
}

} // namespace richengine::ops
