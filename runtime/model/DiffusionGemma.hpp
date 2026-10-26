#pragma once

#include "Gemma4Moe.hpp"
#include "TargetModel.hpp"
#include "TargetFiles.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/Weights.hpp"

#include <cstdint>
#include <vector>

namespace richengine::model {

// DiffusionGemma 26B-A4B: the Gemma 4 MoE trunk run in denoising mode. The
// encoder is ordinary causal prefill; the decoder replays a 256-token canvas
// per step with bidirectional canvas attention over a read-only prefix, a
// self-conditioning MLP fed by the previous step's soft embeddings, and an
// entropy-bounded accept sampler (runtime/model/DiffusionSampler.hpp).
// The denoising schedule is DiffusionSchedule (TargetLayout.hpp, where it
// moved from this header); a target's layout carries it in `diffusion`.
[[nodiscard]] constexpr TargetLayout diffusionGemmaTargetLayout() noexcept {
  // The Gemma 4 trunk preset under the family's own registry discriminator.
  // The schedule keeps DiffusionSchedule's defaults, which the packed
  // manifest's `diffusion` block is validated against.
  TargetLayout layout = gemma4TargetLayout();
  layout.arch = "diffusion_gemma";
  return layout;
}

// The self-conditioning block's weights (target/self_conditioning.bin,
// GEMM0002 type 3): pre_norm over the soft embeddings, then a GeGLU triple
// (gate, up at the packed shared width, down back to hidden).
struct DiffusionSelfConditioningWeights final {
  ops::NormWeights preNorm;
  ops::Projection gate;
  ops::Projection up;
  ops::Projection down;
};

using DiffusionGemmaTrunkWeights = TargetModelWeights;

// The trunk layers carry the DECODER's layer_scalar values; the encoder's
// come from target/encoder_scalars.bin (GEMM0002 type 4), one fp32 per layer.
struct DiffusionGemmaWeights final : TargetModelWeights {
  DiffusionSelfConditioningWeights selfConditioning;
  std::vector<float> encoderLayerScalars;
};

[[nodiscard]] DiffusionGemmaWeights
loadDiffusionGemmaWeights(metal::MetalBackend &backend,
                          const TargetLayout &layout,
                          const TargetFiles &files);

} // namespace richengine::model
