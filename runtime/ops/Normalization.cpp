#include "Normalization.hpp"

#include "ops/BufferExtent.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <utility>
#include <stdexcept>

namespace splash::ops {

std::string normKernel(std::string_view name, const NormWeights &weights, uint32_t width) {
  if (!weights.buffer || weights.buffer.sizeBytes() < weights.bytes(width))
    throw std::invalid_argument("norm weights are below the width");
  std::string result(name);
  if (weights.rmsEpsilon != 1e-6F) result += "_e5";
  if (weights.float32) result += "_f32";
  return result;
}

PreparedInput Normalization::addRms(metal::CommandGraph &graph,
                                    metal::MetalBuffer input,
                                    const NormWeights &weight,
                                    metal::MetalBuffer output, uint32_t width,
                                    uint32_t rows, LinearScratch scratch,
                                    LinearInput layout) {
  if (layout != LinearInput::Plain) {
    requireTableScratch(scratch, layout, width, rows);
    // Packed takes the same two scratch slots as a table: the fp16 plane and
    // the per-(row, 32) exponent bytes the mxfp4p decode kernels read.
    graph.add(normKernel(std::string("norm_rms") + tableSuffix(layout) + "_decode", weight, width),
              {input, weight.buffer, output, scratch.input, scratch.sums}, width, {rows, 1, 1});
    return {std::move(output), layout};
  }
  if (rows <= SPLASH_STAGED_NORM_ROWS && width <= SPLASH_STAGED_NORM_WIDTH && width % 4 == 0)
    graph.add(normKernel("norm_rms_staged", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1}, {SPLASH_STAGED_NORM_THREADS, 1, 1});
  else
    graph.add(normKernel("norm_rms", weight, width), {std::move(input), weight.buffer, output},
              width, {rows, 1, 1});
  return {};
}

void Normalization::addRmsWithQ4Sums(
    metal::CommandGraph &graph, metal::MetalBuffer input,
    const NormWeights &weight, metal::MetalBuffer output,
    metal::MetalBuffer sums, uint32_t width, uint32_t rows) {
  if (weight.float32 || !rows || !width || width % 64)
    throw std::invalid_argument("the Q4-sum norm takes bf16 weights and whole 64-input groups");
  const uint64_t bytes = uint64_t{rows} * width * 2;
  requireBytes(input, bytes, "norm input");
  requireBytes(output, bytes, "norm output");
  // The sums are [32-row tile][64-input group][row of the tile]: the last
  // row's sum of the last group ends them.
  const uint64_t groups = width / 64, last = rows - 1;
  requireBytes(sums, ((last / 32 * groups + groups - 1) * 32 + last % 32 + 1) * sizeof(float), "norm sums");
  graph.add(normKernel("prefill_norm_rms_sums32", weight, width),
            {std::move(input), weight.buffer, std::move(output),
             std::move(sums)},
            width, {rows, 1, 1});
}

} // namespace splash::ops
