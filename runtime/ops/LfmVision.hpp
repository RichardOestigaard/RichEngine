#pragma once

#include "ops/Vision.hpp"

#include <cstdint>
#include <vector>

namespace richengine::ops {

// The LFM3-VL vision tower's geometry (the lfm3vl clip projector of an
// LFM3-Omni mmproj): a fourteen-block hybrid at hidden 1024 — depthwise-3x3
// conv blocks at 0,1,3,5,7,9,11,13 and bidirectional GQA attention blocks at
// 2,4,6,8,10,12 — silu-gated FFNs of 2560, patch 16 with two temporal slices,
// a 2x2 spatial merge and a LayerNorm + GELU(erf) projector into the text
// model's hidden width. A block kind follows its tensors: a conv block
// carries conv2d_in/out projections and a 3x3 kernel, an attention block the
// bias-free q/k/v/out set and per-head q/k norms.
struct LfmVisionLayout final {
  uint32_t depth = 0;
  uint32_t hiddenSize = 0;
  uint32_t patchDimension = 0;
  uint32_t intermediateSize = 0;
  uint32_t mergedHiddenSize = 0;
  uint32_t outputHiddenSize = 0;
  uint32_t heads = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  uint32_t patchSize = 0;
  uint32_t spatialMerge = 0;

  bool operator==(const LfmVisionLayout &) const = default;
};

struct LfmVisionBlock final {
  VisionNorm norm1; // RMSNorm: the weight is used, the bias stays zero.
  // Conv block: in_proj hidden->3*hidden ([B|C|X]), the depthwise 3x3 kernel
  // (hidden x 9 rows), out_proj. Attention block: bias-free q/k/v/out with
  // per-head q/k RMSNorms.
  bool conv = false;
  metal::MetalBuffer inProjection;
  metal::MetalBuffer convKernel;
  metal::MetalBuffer outProjection;
  metal::MetalBuffer query;
  metal::MetalBuffer key;
  metal::MetalBuffer value;
  metal::MetalBuffer queryNorm;
  metal::MetalBuffer keyNorm;
  VisionNorm norm2;
  metal::MetalBuffer ffnGate;
  metal::MetalBuffer ffnUp;
  metal::MetalBuffer ffnDown;
};

struct LfmVisionWeights final {
  LfmVisionLayout layout;
  // The two temporal conv kernels packed side by side: rows x patchDimension.
  metal::MetalBuffer patchEmbedding;
  metal::MetalBuffer embeddingNorm;
  std::vector<LfmVisionBlock> blocks;
  VisionAffine mergerNorm;   // mm.input_norm: LayerNorm with bias, eps 1e-6.
  VisionAffine mergerUpProjection;   // mm.0: mergedHidden -> mergedHidden, GELU(erf).
  VisionAffine mergerDownProjection; // mm.2: mergedHidden -> outputHidden.
  // The tower's projections are bias-free; the shared GEMM kernels read a
  // bias row regardless, so they bind this zero row.
  metal::MetalBuffer zeroBias;
};

// The LFM3-VL encoder, parallel to ops::Vision: pixels in raster order (the
// merge interleave happens at the projector, not at patchify), a bf16 tower
// over an fp32 residual stream, embeddings as one bf16 row per merged 2x2
// block.
class LfmVision final : public VisionEncoder {
public:
  [[nodiscard]] static uint64_t scratchBytes(const LfmVisionLayout &layout,
                                             uint32_t maximumPatches);

  LfmVision(metal::MetalBackend &backend, const LfmVisionWeights &model,
            uint32_t maximumPatches);
  LfmVision(const LfmVision &) = delete;
  LfmVision &operator=(const LfmVision &) = delete;

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
    RopeCos,
    RopeSin,
    Hidden,
    Normalized,
    Wide,
    Gate,
    Up,
    Queries,
    Keys,
    Values,
    Attention,
    Query,
    Key,
    Value,
    Context,
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
               const metal::MetalBuffer &weight, const metal::MetalBuffer &output,
               uint32_t rows) const;

  const LfmVisionWeights &model_;
  uint32_t maximumPatches_ = 0;
  metal::MetalBuffer arena_;
  metal::MetalBuffer scratch_[static_cast<uint32_t>(Scratch::Count)];
};

} // namespace richengine::ops
