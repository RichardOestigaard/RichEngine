#pragma once

#include "Gemma4Moe.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace richengine::model {

// DiffusionGemma 26B-A4B: the Gemma 4 MoE trunk run in denoising mode. The
// encoder is ordinary causal prefill; the decoder replays a 256-token canvas
// per step with bidirectional canvas attention over a read-only prefix, a
// self-conditioning MLP fed by the previous step's soft embeddings, and an
// entropy-bounded accept sampler (runtime/model/DiffusionSampler.hpp).
struct DiffusionSchedule final {
  uint32_t canvasLength = 256;
  uint32_t maxDenoisingSteps = 48;
  // The per-step temperature interpolates t = tMin + (tMax - tMin) * step /
  // maxDenoisingSteps (0.4 -> 0.8 over 48 steps).
  float tMin = 0.4F;
  float tMax = 0.8F;
  // A position is accepted when the exclusive prefix sum of the ascendingly
  // sorted entropies at its rank stays within this budget.
  float entropyBound = 0.1F;
  // Early exit: the canvas argmax unchanged for stabilityThreshold
  // consecutive steps and mean entropy below confidenceThreshold.
  float confidenceThreshold = 0.005F;
  uint32_t stabilityThreshold = 1;
  // The token that fills canvas positions past the first stop token of a
  // committed canvas.
  uint32_t padToken = 0;
  // The previous-step soft embeddings' scale (sqrt(hidden)).
  float softEmbedScale = 0.0F; // derived: sqrt(hiddenSize)

  [[nodiscard]] constexpr float temperature(uint32_t step) const noexcept {
    return tMin + (tMax - tMin) * static_cast<float>(step) /
                      static_cast<float>(maxDenoisingSteps);
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return canvasLength && !(canvasLength % kv::kPageTokens) &&
           maxDenoisingSteps && tMin > 0.0F && tMax >= tMin &&
           entropyBound >= 0.0F && confidenceThreshold >= 0.0F &&
           stabilityThreshold;
  }

  bool operator==(const DiffusionSchedule &) const = default;
};

struct DiffusionGemmaLayout final : Gemma4MoeLayout {
  DiffusionSchedule diffusion{};

  bool operator==(const DiffusionGemmaLayout &) const = default;
};

// The self-conditioning block's weights (target/self_conditioning.bin,
// GEMM0002 type 3): pre_norm over the soft embeddings, then a GeGLU triple
// (gate, up at the packed shared width, down back to hidden).
struct DiffusionSelfConditioningWeights final {
  ops::NormWeights preNorm;
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;
};

using DiffusionGemmaTrunkWeights =
    TargetModelWeights<DiffusionGemmaLayout, Gemma4MoeLayerWeights>;

// The trunk layers carry the DECODER's layer_scalar values; the encoder's
// come from target/encoder_scalars.bin (GEMM0002 type 4), one fp32 per layer.
struct DiffusionGemmaWeights final : DiffusionGemmaTrunkWeights {
  DiffusionSelfConditioningWeights selfConditioning;
  std::vector<float> encoderLayerScalars;
};

[[nodiscard]] DiffusionGemmaWeights
loadDiffusionGemmaWeights(metal::MetalBackend &backend,
                          DiffusionGemmaLayout layout,
                          const TargetFiles<DiffusionGemmaLayout> &files);

} // namespace richengine::model
