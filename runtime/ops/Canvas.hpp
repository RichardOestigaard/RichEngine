#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>

namespace richengine::ops {

// The DiffusionGemma canvas-mode kernels (metal/kernels/decode/canvas.metal;
// the contract is documented in metal/kernels/common/GemmaKernels.h). The
// live row count is runtime-parameterized up to kRows = 256 positions of a
// 262144-token vocabulary and a 2816-element hidden row — a shorter canvas
// skips denoising a request's dead token budget.
class Canvas final {
public:
  static constexpr uint32_t kRows = 256;
  static constexpr uint32_t kThreads = 256;

  // Parameters shared with the kernels (decode/canvas.metal); kept as
  // host-side structs of identical layout.
  struct NoiseParams final {
    uint32_t vocabulary;
    uint32_t seed;
    uint32_t count;
  };
  struct RowStatsParams final {
    uint32_t vocabulary;
    uint32_t seed;
  };
  struct AcceptParams final {
    float entropyBound;
    uint32_t vocabulary;
    uint32_t seed;
    uint32_t rows;
  };
  struct SoftEmbedParams final {
    uint32_t vocabulary;
    uint32_t topK;
  };
  // The fused variants' {cap, invT}: cap > 0 applies cap*tanh(l/cap) on load,
  // invT != 0 multiplies; {0, 0} is the identity. Passing the production
  // softcap and 1/temperature lets the caller skip decode_logit_softcap and
  // canvas_logits_scale entirely.
  struct LogitTransform final {
    float cap;
    float scale;
  };
  struct RowStatsFusedParams final {
    uint32_t vocabulary;
    uint32_t seed;
    LogitTransform transform;
  };
  struct SoftEmbedHistParams final {
    uint32_t vocabulary;
    uint32_t topK;
    LogitTransform transform;
  };
  static_assert(sizeof(NoiseParams) == 12);
  static_assert(sizeof(RowStatsParams) == 8);
  static_assert(sizeof(AcceptParams) == 16);
  static_assert(sizeof(SoftEmbedParams) == 8);
  static_assert(sizeof(RowStatsFusedParams) == 16);
  static_assert(sizeof(SoftEmbedHistParams) == 16);

  // tokens[i] = Uniform(0, vocabulary) over `count` positions (init /
  // renoise fill); `seed` distinguishes steps.
  static void addUniformNoise(metal::CommandGraph &graph,
                              metal::MetalBuffer tokens, uint32_t vocabulary,
                              uint32_t seed, uint32_t count);
  // logits[i] *= inverseTemperature over `count` fp32 elements, in place.
  static void addLogitsScale(metal::CommandGraph &graph,
                             metal::MetalBuffer logits,
                             float inverseTemperature, uint64_t count);
  // cap * tanh(logits / cap), in place — the shared decode softcap kernel.
  static void addLogitSoftcap(metal::CommandGraph &graph,
                              metal::MetalBuffer logits, float cap,
                              uint64_t count);
  // Per canvas row of [rows, vocabulary] fp32 logits: the multinomial draw
  // of softmax(logits), the argmax (lowest index on ties) and the entropy.
  static void addRowStats(metal::CommandGraph &graph,
                          metal::MetalBuffer logits, metal::MetalBuffer sampled,
                          metal::MetalBuffer argmax,
                          metal::MetalBuffer entropy, uint32_t vocabulary,
                          uint32_t seed, uint32_t rows,
                          float logitSoftcap = 0.0F,
                          float inverseTemperature = 0.0F);
  // The entropy-bounded accept of one step over `rows` live canvas rows:
  // canvasOut[i] = sampled[i] when position i is accepted else a fresh
  // uniform token; argmaxOut is this step's argmax canvas (argmaxCur is
  // read, argmaxPrev was the previous step's); stats = {mean entropy over
  // the live rows, all argmax equal to previous}.
  static void addEntropyAccept(metal::CommandGraph &graph,
                               metal::MetalBuffer entropy,
                               metal::MetalBuffer sampled,
                               metal::MetalBuffer argmaxPrev,
                               metal::MetalBuffer canvasOut,
                               metal::MetalBuffer argmaxOut,
                               metal::MetalBuffer stats,
                               metal::MetalBuffer argmaxCur,
                               float entropyBound, uint32_t vocabulary,
                               uint32_t seed, uint32_t rows);
  // softmax(logits) @ embedding Q4 table * embeddingScale -> out bf16
  // [rows, hidden]: the production top-k path and the exact reference.
  // Whether both fused logit-tail kernels are active
  // (RICHENGINE_CANVAS_EMBED_HIST and RICHENGINE_CANVAS_STATS_FUSED, both
  // default on). When true, callers may pass logitSoftcap and
  // inverseTemperature into addRowStats/addSoftEmbed and skip the
  // standalone transform passes; when false, the logits buffer must be
  // transformed in place beforehand and the transform args must be 0.
  static bool fusedLogitTail();

  static void addSoftEmbed(metal::CommandGraph &graph,
                           metal::MetalBuffer logits,
                           const EmbeddingWeights &table,
                           metal::MetalBuffer output, uint32_t vocabulary,
                           uint32_t rows, uint32_t topK, float embeddingScale,
                           bool exact = false, float logitSoftcap = 0.0F,
                           float inverseTemperature = 0.0F);
  // out = scaleless_rms(inputsEmbeds + sc): the fused residual tail of the
  // self-conditioning block.
  static void addSelfCondition(metal::CommandGraph &graph,
                               metal::MetalBuffer inputsEmbeds,
                               metal::MetalBuffer sc, metal::MetalBuffer output,
                               uint32_t width, uint32_t rows = kRows);
};

} // namespace richengine::ops
