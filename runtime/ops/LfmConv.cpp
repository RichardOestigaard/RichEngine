#include "ops/LfmConv.hpp"

#include "metal/abi/LfmConv.h"

#include <stdexcept>

namespace richengine::ops {
namespace {

constexpr uint32_t kThreads = metal::CommandGraph::kDefaultThreads;
// verify_lfm_conv_v4 stages one lane's rows per threadgroup: rows*32 threads
// cover a 128-channel block, so it needs dimension % 128 == 0 and
// rows*32 <= 1024.
constexpr uint32_t kVerifyVecColumns = 32;
constexpr uint32_t kVerifyChannelBlock = kVerifyVecColumns * 4;

[[nodiscard]] RichLfmConvParams params(uint32_t rows, const LfmConvShape &shape,
                                         bool tapsMajor, uint32_t layer,
                                         uint64_t stateLayerBytes, uint32_t lanes) {
  RichLfmConvParams result{};
  result.rows = rows;
  result.dimension = shape.dimension;
  result.taps = shape.taps;
  result.taps_major = tapsMajor ? 1u : 0u;
  result.state_layer_bytes = stateLayerBytes;
  result.layer = layer;
  result.lanes = lanes;
  result.layers = 1;
  result.mixed_layer_stride = 0;
  return result;
}

void requireStateSet(std::span<const metal::MetalBuffer> states, uint32_t lanes) {
  if (states.size() < lanes)
    throw std::invalid_argument("conv state lanes are below the batch");
}

[[nodiscard]] metal::DispatchSize groups(uint64_t tasks) {
  return {static_cast<uint32_t>((tasks + kThreads - 1) / kThreads), 1, 1};
}

} // namespace

void LfmConv::addPrefill(metal::CommandGraph &graph, const LfmConvPrefillBuffers &buffers,
                         const LfmConvShape &shape, uint32_t rows, bool tapsMajor) {
  if (!shape.valid() || !rows)
    throw std::invalid_argument("invalid conv shape");
  const auto p = params(rows, shape, tapsMajor, 0, 0, 1);
  const uint64_t tasks = uint64_t{rows} + shape.taps - 1;
  if (shape.dimension % 4 == 0) {
    graph.add("prefill_lfm_conv_v4",
              {buffers.packed, buffers.weights, buffers.stateIn, buffers.stateOut,
               buffers.output},
              p, groups(tasks * (shape.dimension / 4)));
  } else {
    graph.add("prefill_lfm_conv",
              {buffers.packed, buffers.weights, buffers.stateIn, buffers.stateOut,
               buffers.output},
              p, groups(tasks * shape.dimension));
  }
}

void LfmConv::addVerify(metal::CommandGraph &graph, const LfmConvVerifyBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes, uint32_t rows,
                        uint32_t layer, uint64_t stateLayerBytes, bool tapsMajor) {
  if (!shape.valid() || !rows)
    throw std::invalid_argument("invalid conv shape");
  requireStateSet(buffers.currentStates, lanes);
  const auto p = params(rows, shape, tapsMajor, layer, stateLayerBytes, lanes);
  const std::vector<metal::MetalBuffer> bindings =
      {buffers.packed, buffers.weights, buffers.currentStates[0],
       buffers.currentStates[1], buffers.currentStates[2],
       buffers.currentStates[3], buffers.mixed, buffers.output};
  if (shape.dimension % kVerifyChannelBlock == 0 &&
      uint64_t{rows} * kVerifyVecColumns <= 1024) {
    graph.addPatchable(
        "verify_lfm_conv_v4", bindings, p,
        {static_cast<uint32_t>(uint64_t{lanes} * shape.dimension / kVerifyChannelBlock), 1, 1},
        {static_cast<uint32_t>(rows * kVerifyVecColumns), 1, 1});
    return;
  }
  graph.addPatchable("verify_lfm_conv", bindings, p,
                     groups(uint64_t{lanes} * rows * shape.dimension));
}

void LfmConv::addCommit(metal::CommandGraph &graph, const LfmConvCommitBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes, uint32_t rows,
                        uint32_t layer, uint64_t stateLayerBytes, uint32_t layers,
                        uint64_t mixedLayerStride, bool tapsMajor) {
  if (!shape.valid() || !rows || !layers)
    throw std::invalid_argument("invalid conv shape");
  requireStateSet(buffers.currentStates, lanes);
  requireStateSet(buffers.nextStates, lanes);
  auto p = params(rows, shape, tapsMajor, layer, stateLayerBytes, lanes);
  p.layers = layers;
  p.mixed_layer_stride = mixedLayerStride;
  const std::vector<metal::MetalBuffer> bindings =
      {buffers.mixed, buffers.retainedCounts, buffers.currentStates[0],
       buffers.currentStates[1], buffers.currentStates[2],
       buffers.currentStates[3], buffers.nextStates[0], buffers.nextStates[1],
       buffers.nextStates[2], buffers.nextStates[3]};
  const uint64_t tasks =
      uint64_t{layers} * lanes * (shape.taps - 1) * shape.dimension;
  if (shape.dimension % 4 == 0) {
    graph.addPatchable("commit_lfm_conv_v4", bindings, p, groups(tasks / 4));
  } else {
    graph.addPatchable("commit_lfm_conv", bindings, p, groups(tasks));
  }
}

} // namespace richengine::ops
