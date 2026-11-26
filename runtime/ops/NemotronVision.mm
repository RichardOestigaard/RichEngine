#include "NemotronVision.hpp"

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

// The one nemotron_omni tower these kernels implement: Bolt's 32-block
// C-RADIO ViT. outputHiddenSize is the text model's width and is checked
// separately, as for the other towers.
constexpr NemotronVisionLayout kTower{.depth = 32,
                                      .hiddenSize = 1280,
                                      .patchDimension = 768,
                                      .intermediateSize = 5120,
                                      .mergedHiddenSize = 5120,
                                      .projectorHiddenSize = 20480,
                                      .outputHiddenSize = 4096,
                                      .heads = 16,
                                      .headDimension = 80,
                                      .positionGridSide = 128,
                                      .patchSize = 16,
                                      .spatialMerge = 2,
                                      .registers = 10};
static_assert(kTower.hiddenSize % kGemmColumnTile == 0 &&
                  kTower.intermediateSize % kGemmColumnTile == 0 &&
                  kTower.patchDimension % kGemmColumnTile == 0 &&
                  kTower.mergedHiddenSize % kMergerColumnTile == 0 &&
                  kTower.projectorHiddenSize % kMergerColumnTile == 0 &&
                  kTower.headDimension % 16 == 0,
              "nemotron vision kernels' tile assumptions");

uint32_t roundUp(uint32_t value, uint32_t multiple) noexcept {
  return (value + multiple - 1) / multiple * multiple;
}

std::array<uint64_t, 15>
scratchLayout(const NemotronVisionLayout &layout, uint32_t maximumPatches) {
  const uint32_t vitTokens = maximumPatches + layout.registers;
  const uint64_t rows = roundUp(vitTokens, kGemmRowTile);
  const uint64_t padded = roundUp(vitTokens, kKeyTile);
  const uint64_t mergedRows = roundUp(maximumPatches / 4, kMergerRowTile);
  const uint64_t hidden = layout.hiddenSize;
  const uint64_t headRows = uint64_t{layout.heads} * padded;
  return {
      rows * layout.patchDimension * kBf16Bytes,       // Patches
      rows * hidden * kFloatBytes,                     // Hidden
      rows * hidden * kBf16Bytes,                      // Normalized
      rows * hidden * kBf16Bytes,                      // Query
      rows * hidden * kBf16Bytes,                      // Key
      rows * hidden * kBf16Bytes,                      // Value
      headRows * layout.headDimension * kBf16Bytes,    // Queries
      headRows * layout.headDimension * kBf16Bytes,    // Keys
      headRows * layout.headDimension * kBf16Bytes,    // Values
      headRows * layout.headDimension * kBf16Bytes,    // Attention
      rows * hidden * kBf16Bytes,                      // Context
      rows * layout.intermediateSize * kBf16Bytes,     // Ffn
      mergedRows * layout.mergedHiddenSize * kFloatBytes, // Merged
      mergedRows * layout.mergedHiddenSize * kBf16Bytes,  // ProjectorInput
      mergedRows * layout.projectorHiddenSize * kBf16Bytes, // ProjectorHidden
  };
}

void requireLayout(const NemotronVisionLayout &layout) {
  NemotronVisionLayout tower = layout;
  tower.outputHiddenSize = kTower.outputHiddenSize;
  if (tower != kTower || !layout.outputHiddenSize ||
      layout.outputHiddenSize % kMergerColumnTile) {
    throw std::invalid_argument(
        "vision kernels are specialized for the Nemotron Omni 32-block tower");
  }
}

} // namespace

uint64_t NemotronVision::scratchBytes(const NemotronVisionLayout &layout,
                                      uint32_t maximumPatches) {
  requireLayout(layout);
  if (!maximumPatches || maximumPatches % 4) {
    throw std::invalid_argument(
        "vision scratch must cover a positive multiple of four patches");
  }
  uint64_t total = 0;
  for (uint64_t bytes : scratchLayout(layout, maximumPatches)) {
    const uint64_t aligned = alignUp(bytes);
    if (aligned > std::numeric_limits<uint64_t>::max() - total)
      throw std::overflow_error("vision scratch byte count overflows");
    total += aligned;
  }
  return total;
}

NemotronVision::NemotronVision(metal::MetalBackend &backend,
                               const NemotronVisionWeights &model,
                               uint32_t maximumPatches)
    : model_(model), maximumPatches_(maximumPatches) {
  const uint64_t total = scratchBytes(model.layout, maximumPatches);
  arena_ = backend.allocateBuffer(total, metal::BufferStorage::Shared,
                                  "nemotron-vision-scratch");
  uint64_t cursor = 0;
  const auto layout = scratchLayout(model.layout, maximumPatches);
  for (uint32_t index = 0; index < layout.size(); ++index) {
    scratch_[index] = backend.view(arena_, cursor, layout[index]);
    cursor += alignUp(layout[index]);
  }
  if (cursor != total)
    throw std::logic_error("nemotron vision scratch arena mismatch");
}

void NemotronVision::addGemm(CommandGraph &graph, const char *pipeline,
                             const MetalBuffer &input, const MetalBuffer &weight,
                             const MetalBuffer &bias, const MetalBuffer &output,
                             const MetalBuffer &residual, uint32_t outputSize,
                             uint32_t inputSize, uint32_t rows,
                             uint32_t tileRows, uint32_t tileColumns) const {
  graph.add(pipeline, {input, weight, bias, output, residual},
            VisionGemmParams{outputSize, inputSize},
            {(rows + tileRows - 1) / tileRows, outputSize / tileColumns, 1});
}

void NemotronVision::addNorm(CommandGraph &graph, const MetalBuffer &input,
                             const VisionNorm &weights,
                             const MetalBuffer &output, uint32_t rows) const {
  graph.add(std::string(kVisionLayerNorm),
            {input, weights.weight, weights.bias, output},
            VisionNormParams{model_.layout.hiddenSize}, {rows, 1, 1});
}

void NemotronVision::encode(CommandGraph &graph, ImageGrid grid,
                            const MetalBuffer &pixels,
                            const MetalBuffer &embeddings) const {
  const NemotronVisionLayout &layout = model_.layout;
  if (!grid.valid() || grid.patches() > maximumPatches_ || grid.frames != 1)
    throw std::invalid_argument("image grid exceeds the Nemotron vision encoder");
  const auto patches = static_cast<uint32_t>(grid.patches());
  const uint32_t tokens = patches + layout.registers;
  if (!pixels || pixels.sizeBytes() < grid.pixelBytes())
    throw std::invalid_argument("image pixels do not cover the grid");
  const uint64_t embeddingBytes =
      uint64_t{Vision::embeddingRows(grid)} * layout.outputHiddenSize * kBf16Bytes;
  if (!embeddings || embeddings.sizeBytes() < embeddingBytes)
    throw std::invalid_argument("embedding buffer is too small for the grid");

  const uint32_t rowsGemm = roundUp(tokens, kGemmRowTile);
  const uint32_t padded = roundUp(tokens, kKeyTile);
  const uint32_t merged = grid.mergedTokens();
  const VisionGridParams gridParams{grid.height, grid.width};
  const NemotronVisionPrepareParams prepareParams{grid.height, grid.width,
                                                  tokens, layout.registers};
  const VisionQkvParams qkvParams{tokens, padded, 0};
  const VisionAttentionParams attentionParams{
      tokens, padded,
      1.0F / std::sqrt(static_cast<float>(layout.headDimension))};
  const MetalBuffer &hidden = scratch(Scratch::Hidden);
  const MetalBuffer &normalized = scratch(Scratch::Normalized);
  const MetalBuffer &context = scratch(Scratch::Context);
  const uint64_t registerBytes = uint64_t{layout.registers} * layout.hiddenSize * kFloatBytes;
  const MetalBuffer hiddenPatch =
      hidden.subview(registerBytes,
                     uint64_t{rowsGemm} * layout.hiddenSize * kFloatBytes - registerBytes);

  graph.add(std::string(kNemotronVisionPatchify),
            {pixels, scratch(Scratch::Patches)}, gridParams, {patches, 1, 1});
  graph.add(std::string(kNemotronVisionPrepare),
            {model_.positionTable, model_.classEmbedding, hidden}, prepareParams,
            {rowsGemm, 1, 1});
  addGemm(graph, kVisionGemmM64n128Residual.data(), scratch(Scratch::Patches),
          model_.patchEmbedding, model_.zeroBias, hiddenPatch, hiddenPatch,
          layout.hiddenSize, layout.patchDimension, patches, kGemmRowTile,
          kGemmColumnTile);

  for (const NemotronVisionBlock &block : model_.blocks) {
    addNorm(graph, hidden, block.norm1, normalized, tokens);
    addGemm(graph, kVisionGemmM64n128.data(), normalized, block.query.weight,
            block.query.bias, scratch(Scratch::Query), context,
            layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    addGemm(graph, kVisionGemmM64n128.data(), normalized, block.key.weight,
            block.key.bias, scratch(Scratch::Key), context,
            layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    addGemm(graph, kVisionGemmM64n128.data(), normalized, block.value.weight,
            block.value.bias, scratch(Scratch::Value), context,
            layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);
    graph.add(std::string(kNemotronVisionQkvPrepare),
              {scratch(Scratch::Query), scratch(Scratch::Key),
               scratch(Scratch::Value), scratch(Scratch::Queries),
               scratch(Scratch::Keys), scratch(Scratch::Values)},
              qkvParams, {padded, 1, 1});
    graph.add(std::string(kNemotronVisionAttention),
              {scratch(Scratch::Queries), scratch(Scratch::Keys),
               scratch(Scratch::Values), scratch(Scratch::Attention)},
              attentionParams,
              {(tokens + kQueryTile - 1) / kQueryTile, layout.heads, 1});
    graph.add(std::string(kNemotronVisionAttentionPack),
              {scratch(Scratch::Attention), context}, qkvParams,
              {tokens, 1, 1});
    addGemm(graph, kVisionGemmM64n128Residual.data(), context,
            block.projection.weight, block.projection.bias, hidden, hidden,
            layout.hiddenSize, layout.hiddenSize, tokens, kGemmRowTile,
            kGemmColumnTile);

    addNorm(graph, hidden, block.norm2, normalized, tokens);
    addGemm(graph, kVisionGemmM64n128GeluErf.data(), normalized,
            block.upProjection.weight, block.upProjection.bias,
            scratch(Scratch::Ffn), context, layout.intermediateSize,
            layout.hiddenSize, tokens, kGemmRowTile, kGemmColumnTile);
    addGemm(graph, kVisionGemmM64n128Residual.data(), scratch(Scratch::Ffn),
            block.downProjection.weight, block.downProjection.bias, hidden,
            hidden, layout.hiddenSize, layout.intermediateSize, tokens,
            kGemmRowTile, kGemmColumnTile);
  }

  // The register rows precede the patch rows in the ViT. The merger receives
  // a view beginning after them, gathers the 2x2 blocks in Nemotron's
  // channel/horizontal/vertical feature order, then the projector applies
  // RMSNorm(1e-5) -> up -> relu(x)^2 -> down.
  graph.add(std::string(kNemotronVisionMerge),
            {hiddenPatch, scratch(Scratch::Merged)}, gridParams,
            {merged, 1, 1});
  graph.add(std::string(kLfmVisionRmsnorm),
            {scratch(Scratch::Merged), model_.projectorNorm,
             scratch(Scratch::ProjectorInput)},
            VisionNormParams{layout.mergedHiddenSize}, {merged, 1, 1});
  addGemm(graph, kVisionGemmM32n256ReluSquared.data(),
          scratch(Scratch::ProjectorInput), model_.projectorUpProjection,
          model_.zeroBias, scratch(Scratch::ProjectorHidden), context,
          layout.projectorHiddenSize, layout.mergedHiddenSize, merged,
          kMergerRowTile, kMergerColumnTile);
  addGemm(graph, kVisionGemmM32n256.data(), scratch(Scratch::ProjectorHidden),
          model_.projectorDownProjection, model_.zeroBias, embeddings, context,
          layout.outputHiddenSize, layout.projectorHiddenSize, merged,
          kMergerRowTile, kMergerColumnTile);
}

} // namespace richengine::ops
