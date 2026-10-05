#include "ops/LfmConv.hpp"

#include "metal/abi/LfmConv.h"

#include <stdexcept>

namespace splash::ops {
namespace {

[[nodiscard]] SplashLfmConvParams params(uint32_t rows, const LfmConvShape &shape,
                                         bool tapsMajor, uint32_t layer,
                                         uint64_t stateLayerBytes, uint32_t lanes) {
  SplashLfmConvParams result{};
  result.rows = rows;
  result.dimension = shape.dimension;
  result.taps = shape.taps;
  result.taps_major = tapsMajor ? 1u : 0u;
  result.state_layer_bytes = stateLayerBytes;
  result.layer = layer;
  result.lanes = lanes;
  return result;
}

void requireStateSet(std::span<const metal::MetalBuffer> states, uint32_t lanes) {
  if (states.size() < lanes)
    throw std::invalid_argument("conv state lanes are below the batch");
}

} // namespace

void LfmConv::addPrefill(metal::CommandGraph &graph, const LfmConvPrefillBuffers &buffers,
                         const LfmConvShape &shape, uint32_t rows, bool tapsMajor) {
  if (!shape.valid() || !rows)
    throw std::invalid_argument("invalid conv shape");
  const auto p = params(rows, shape, tapsMajor, 0, 0, 1);
  graph.add("prefill_lfm_conv",
            {buffers.packed, buffers.weights, buffers.stateIn, buffers.stateOut,
             buffers.output},
            p, {static_cast<uint32_t>((uint64_t{rows} + shape.taps - 1) * shape.dimension), 1, 1});
}

void LfmConv::addVerify(metal::CommandGraph &graph, const LfmConvVerifyBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes, uint32_t rows,
                        uint32_t layer, uint64_t stateLayerBytes, bool tapsMajor) {
  if (!shape.valid() || !rows)
    throw std::invalid_argument("invalid conv shape");
  requireStateSet(buffers.currentStates, lanes);
  const auto p = params(rows, shape, tapsMajor, layer, stateLayerBytes, lanes);
  graph.addPatchable("verify_lfm_conv",
            {buffers.packed, buffers.weights, buffers.currentStates[0],
             buffers.currentStates[1], buffers.currentStates[2],
             buffers.currentStates[3], buffers.mixed, buffers.output},
            p, {static_cast<uint32_t>(uint64_t{lanes} * rows * shape.dimension), 1, 1});
}

void LfmConv::addCommit(metal::CommandGraph &graph, const LfmConvCommitBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes, uint32_t rows,
                        uint32_t layer, uint64_t stateLayerBytes, bool tapsMajor) {
  if (!shape.valid() || !rows)
    throw std::invalid_argument("invalid conv shape");
  requireStateSet(buffers.currentStates, lanes);
  requireStateSet(buffers.nextStates, lanes);
  const auto p = params(rows, shape, tapsMajor, layer, stateLayerBytes, lanes);
  graph.addPatchable("commit_lfm_conv",
            {buffers.mixed, buffers.retainedCounts, buffers.currentStates[0],
             buffers.currentStates[1], buffers.currentStates[2],
             buffers.currentStates[3], buffers.nextStates[0], buffers.nextStates[1],
             buffers.nextStates[2], buffers.nextStates[3]},
            p, {static_cast<uint32_t>(uint64_t{lanes} * (shape.taps - 1) * shape.dimension), 1, 1});
}

} // namespace splash::ops
