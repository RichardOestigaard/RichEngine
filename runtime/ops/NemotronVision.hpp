#pragma once

#include "ops/Vision.hpp"

#include <cstdint>
#include <vector>

namespace richengine::ops {

// The Nemotron Omni vision tower's geometry (the nemotron_omni clip
// projector of Nemotron 3.5 Bolt's mmproj): a 32-block C-RADIO ViT at
// hidden 1280, bidirectional attention of sixteen 80-wide heads, GELU(erf)
// FFNs of 5120, patch 16 with one still-image slice, ten register tokens,
// a learned 128x128 position table resampled bilinearly, a 2x2 pixel
// unshuffle and an RMSNorm + relu-squared MLP projector into the text
// model's hidden width.
struct NemotronVisionLayout final {
  uint32_t depth = 0;
  uint32_t hiddenSize = 0;
  uint32_t patchDimension = 0;
  uint32_t intermediateSize = 0;
  uint32_t mergedHiddenSize = 0;
  uint32_t projectorHiddenSize = 0;
  uint32_t outputHiddenSize = 0;
  uint32_t heads = 0;
  uint32_t headDimension = 0;
  uint32_t positionGridSide = 0;
  uint32_t patchSize = 0;
  uint32_t spatialMerge = 0;
  uint32_t registers = 0;

  bool operator==(const NemotronVisionLayout &) const = default;
};

struct NemotronVisionBlock final {
  VisionNorm norm1;
  VisionAffine query;
  VisionAffine key;
  VisionAffine value;
  VisionAffine projection;
  VisionNorm norm2;
  VisionAffine upProjection;
  VisionAffine downProjection;
};

struct NemotronVisionWeights final {
  NemotronVisionLayout layout;
  metal::MetalBuffer patchEmbedding;
  metal::MetalBuffer positionTable;
  metal::MetalBuffer classEmbedding;
  std::vector<NemotronVisionBlock> blocks;
  metal::MetalBuffer projectorNorm;
  metal::MetalBuffer projectorUpProjection;
  metal::MetalBuffer projectorDownProjection;
  // The patch embedding and projector carry no bias; the shared GEMM
  // kernels read a bias row regardless, so they bind this zero row.
  metal::MetalBuffer zeroBias;
};

// The Nemotron Omni encoder, parallel to ops::LfmVision: pixels in raster
// order, a bf16 tower over an fp32 residual stream, embeddings as one bf16
// row per merged 2x2 block. The register tokens precede the patch sequence
// inside the ViT and never reach the merger.
class NemotronVision final : public VisionEncoder {
public:
  [[nodiscard]] static uint64_t scratchBytes(const NemotronVisionLayout &layout,
                                             uint32_t maximumPatches);

  NemotronVision(metal::MetalBackend &backend, const NemotronVisionWeights &model,
                 uint32_t maximumPatches);
  NemotronVision(const NemotronVision &) = delete;
  NemotronVision &operator=(const NemotronVision &) = delete;

  [[nodiscard]] uint64_t arenaBytes() const noexcept override {
    return arena_.sizeBytes();
  }
  [[nodiscard]] uint32_t maximumPatches() const noexcept override {
    return maximumPatches_;
  }

  void encode(metal::CommandGraph &graph, ImageGrid grid,
              const metal::MetalBuffer &pixels,
              const metal::MetalBuffer &embeddings) const override;

private:
  enum class Scratch : uint32_t {
    Patches,
    Hidden,
    Normalized,
    Query,
    Key,
    Value,
    Queries,
    Keys,
    Values,
    Attention,
    Context,
    Ffn,
    Merged,
    ProjectorInput,
    ProjectorHidden,
    Count,
  };

  [[nodiscard]] const metal::MetalBuffer &scratch(Scratch tensor) const noexcept {
    return scratch_[static_cast<uint32_t>(tensor)];
  }
  void addGemm(metal::CommandGraph &graph, const char *pipeline,
               const metal::MetalBuffer &input, const metal::MetalBuffer &weight,
               const metal::MetalBuffer &bias, const metal::MetalBuffer &output,
               const metal::MetalBuffer &residual, uint32_t outputSize,
               uint32_t inputSize, uint32_t rows, uint32_t tileRows,
               uint32_t tileColumns) const;
  void addNorm(metal::CommandGraph &graph, const metal::MetalBuffer &input,
               const VisionNorm &weights, const metal::MetalBuffer &output,
               uint32_t rows) const;

  const NemotronVisionWeights &model_;
  uint32_t maximumPatches_ = 0;
  metal::MetalBuffer arena_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace richengine::ops
