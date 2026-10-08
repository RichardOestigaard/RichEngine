#include "ops/Canvas.hpp"

#include "Tuning.hpp"
#include "ops/KernelNames.hpp"
#include "ops/Weights.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace richengine::ops {

namespace {

[[nodiscard]] metal::DispatchSize elementwiseGroups(uint64_t elements) {
  return {static_cast<uint32_t>((elements + Canvas::kThreads - 1) /
                                Canvas::kThreads),
          1, 1};
}

// A/B gates: the new paths are the default (parity-tested by the
// canvas-kernels metal test); setting the flag to "0" selects the old
// kernels. RICHENGINE_CANVAS_EMBED_HIST=0 restores the 32-iteration
// bisection, RICHENGINE_CANVAS_STATS_FUSED=0 the two-pass row stats.
[[nodiscard]] bool histogramEmbedEnabled() {
  return tuning().canvasEmbedHist;
}

[[nodiscard]] bool fusedRowStatsEnabled() {
  return tuning().canvasStatsFused;
}

} // namespace

bool Canvas::fusedLogitTail() {
  return histogramEmbedEnabled() && fusedRowStatsEnabled();
}

void Canvas::addUniformNoise(metal::CommandGraph &graph,
                             metal::MetalBuffer tokens, uint32_t vocabulary,
                             uint32_t seed, uint32_t count) {
  // canvas_uniform_noise(tokens, {vocabulary, seed, count}): one thread per
  // live position.
  const NoiseParams params{vocabulary, seed, count};
  graph.add(std::string(kCanvasUniformNoise), {std::move(tokens)}, params, {1, 1, 1},
            {kThreads, 1, 1});
}

void Canvas::addLogitsScale(metal::CommandGraph &graph,
                            metal::MetalBuffer logits,
                            float inverseTemperature, uint64_t count) {
  if (!count || count > UINT32_MAX)
    throw std::invalid_argument("canvas logits scale count out of range");
  graph.addTail(std::string(kCanvasLogitsScale), {std::move(logits)},
                inverseTemperature, static_cast<uint32_t>(count),
                elementwiseGroups(count));
}

void Canvas::addLogitSoftcap(metal::CommandGraph &graph,
                             metal::MetalBuffer logits, float cap,
                             uint64_t count) {
  if (!count || count > UINT32_MAX)
    throw std::invalid_argument("canvas logit softcap count out of range");
  graph.addTail(std::string(kCanvasLogitSoftcap), {std::move(logits)}, cap,
                static_cast<uint32_t>(count), elementwiseGroups(count));
}

void Canvas::addRowStats(metal::CommandGraph &graph,
                         metal::MetalBuffer logits,
                         metal::MetalBuffer sampled,
                         metal::MetalBuffer argmax,
                         metal::MetalBuffer entropy, uint32_t vocabulary,
                         uint32_t seed, uint32_t rows, float logitSoftcap,
                         float inverseTemperature) {
  if (fusedRowStatsEnabled()) {
    const RowStatsFusedParams params{
        vocabulary, seed, {logitSoftcap, inverseTemperature}};
    graph.add(std::string(kCanvasRowStatsFused),
              {std::move(logits), std::move(sampled), std::move(argmax),
               std::move(entropy)},
              params, {rows, 1, 1}, {kThreads, 1, 1});
    return;
  }
  const RowStatsParams params{vocabulary, seed};
  graph.add(std::string(kCanvasRowStats),
            {std::move(logits), std::move(sampled), std::move(argmax),
             std::move(entropy)},
            params, {rows, 1, 1}, {kThreads, 1, 1});
}

void Canvas::addEntropyAccept(metal::CommandGraph &graph,
                              metal::MetalBuffer entropy,
                              metal::MetalBuffer sampled,
                              metal::MetalBuffer argmaxPrev,
                              metal::MetalBuffer canvasOut,
                              metal::MetalBuffer argmaxOut,
                              metal::MetalBuffer stats,
                              metal::MetalBuffer argmaxCur,
                              float entropyBound, uint32_t vocabulary,
                              uint32_t seed, uint32_t rows) {
  const AcceptParams params{entropyBound, vocabulary, seed, rows};
  // Buffer 6 is the constant params; argmax_cur rides at buffer 7 behind it.
  graph.addParamsAt(std::string(kCanvasEntropyAccept),
                    {std::move(entropy), std::move(sampled),
                     std::move(argmaxPrev), std::move(canvasOut),
                     std::move(argmaxOut), std::move(stats),
                     std::move(argmaxCur)},
                    params, 6, {1, 1, 1}, {kThreads, 1, 1});
}

void Canvas::addSoftEmbed(metal::CommandGraph &graph,
                          metal::MetalBuffer logits,
                          const EmbeddingWeights &table,
                          metal::MetalBuffer output, uint32_t vocabulary,
                          uint32_t rows, uint32_t topK, float embeddingScale,
                          bool exact, float logitSoftcap,
                          float inverseTemperature) {
  const AffineWeights &affine = table.affine();
  if (!exact && histogramEmbedEnabled()) {
    const SoftEmbedHistParams params{
        vocabulary, topK, {logitSoftcap, inverseTemperature}};
    graph.addTail(std::string(kCanvasSoftEmbedHistogram),
                  {std::move(logits), affine.weights, affine.scales,
                   affine.biases, std::move(output)},
                  params, embeddingScale, {rows, 1, 1}, {kThreads, 1, 1});
    return;
  }
  const SoftEmbedParams params{vocabulary, exact ? 0u : topK};
  graph.addTail(std::string(exact ? kCanvasSoftEmbedExact : kCanvasSoftEmbedTopk),
                {std::move(logits), affine.weights, affine.scales,
                 affine.biases, std::move(output)},
                params, embeddingScale, {rows, 1, 1}, {kThreads, 1, 1});
}

void Canvas::addSelfCondition(metal::CommandGraph &graph,
                              metal::MetalBuffer inputsEmbeds,
                              metal::MetalBuffer sc,
                              metal::MetalBuffer output, uint32_t width,
                              uint32_t rows) {
  graph.add(std::string(kCanvasSelfCondition),
            {std::move(inputsEmbeds), std::move(sc), std::move(output)},
            width, {rows, 1, 1}, {kThreads, 1, 1});
}

} // namespace richengine::ops
