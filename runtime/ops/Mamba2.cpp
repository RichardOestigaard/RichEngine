#include "ops/Mamba2.hpp"
#include "ops/KernelNames.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Mamba2.h"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace richengine::ops {
namespace {

static_assert(offsetof(Mamba2DecodeBatchParams, conv_layer_bytes) == 24);

// The compiled Mamba-2 geometry; Nemotron64 is Nemotron 3.5's 64 x 64 heads,
// 8 groups of 128 states, 6144 conv channels, 10496 packed (10304 logical
// rows padded to the storage tile). Only this shape has compiled variants.
void kernelShape(const Mamba2Shape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid Mamba2 shape");
  const Mamba2Shape nemotron{64, 64, 8, 128, 4, 4096, 6144, 10496,
                             shape.dtMinimum, shape.dtMaximum, shape.core};
  if (!(shape.core == Mamba2Core::Nemotron64 && shape == nemotron))
    throw std::invalid_argument(
        "declared Mamba2 core does not match the shape");
}

// The kernel variant loading the norm weights in their stored type: bf16, or
// F32 as a GGUF stores them (_f32). The grouped RMSNorm's epsilon rides the
// dispatch parameters instead of an _e5 name tag, unlike normKernel.
[[nodiscard]] std::string mamba2Kernel(std::string_view name,
                                       const NormWeights &weights,
                                       uint32_t width) {
  if (!weights.buffer || weights.buffer.sizeBytes() < weights.bytes(width))
    throw std::invalid_argument("norm weights are below the width");
  std::string result(name);
  if (weights.float32)
    result += "_f32";
  return result;
}

} // namespace

void Mamba2::addPrefill(metal::CommandGraph &graph,
                        Mamba2PrefillBuffers buffers, Mamba2Shape shape,
                        uint32_t tokens, metal::MetalBuffer sums) {
  if (!tokens)
    throw std::invalid_argument("invalid Mamba2 prefill geometry");
  (void)kernelShape(shape);
  const Mamba2PrefillParams params{tokens, shape.dtMinimum, shape.dtMaximum};
  graph.add(kPrefillMamba2Prepare.data(),
            {buffers.packed, buffers.convolutionWeights, buffers.convolutionIn,
             buffers.convolutionOut, buffers.postConvolution,
             buffers.timeBias, buffers.stepSizes, buffers.convolutionBias},
            params, {uint64_t{tokens} * shape.groups, 1, 1}, {128, 1, 1});
  graph.add(kPrefillMamba2Scan.data(),
            {buffers.postConvolution, buffers.stepSizes, buffers.aWeights,
             buffers.skipWeights, buffers.recurrentIn, buffers.recurrentOut,
             buffers.recurrentRows},
            params,
            {uint64_t{shape.heads} * shape.headDimension /
                 RICHENGINE_GDN_SCAN_STATE_ROWS,
             1, 1},
            {RICHENGINE_GDN_SCAN_THREADS, 1, 1});
  const std::string gate = mamba2Kernel(
      (sums ? kPrefillMamba2GateSums : kPrefillMamba2Gate).data(),
      buffers.mixerNorm, shape.innerSize);
  std::vector<metal::MetalBuffer> gateBindings{
      buffers.recurrentRows, buffers.packed, buffers.mixerNorm.buffer,
      buffers.hidden};
  if (sums)
    gateBindings.push_back(sums);
  graph.add(gate, gateBindings,
            Mamba2GateParams{0, buffers.mixerNorm.rmsEpsilon},
            {uint64_t{tokens} * shape.groups, 1, 1}, {256, 1, 1});
}

PreparedInput Mamba2::addDecode(metal::CommandGraph &graph,
                                Mamba2DecodeBuffers buffers, Mamba2Shape shape,
                                uint32_t lanes, uint32_t layer,
                                Mamba2StateStrides state, LinearInput input,
                                std::span<const uint32_t> liveRows,
                                uint32_t verifyStride,
                                const Mamba2TapeLanes *tapeLanes) {
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH || !state.valid() ||
      !verifyStride || verifyStride > RICHENGINE_TARGET_VERIFY_ROWS ||
      (!liveRows.empty() && liveRows.size() != lanes))
    throw std::invalid_argument("invalid Mamba2 decode geometry");
  (void)kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.innerSize,
                        lanes * verifyStride);
  const bool folded = tapeLanes && buffers.tape && tapeLanes->layerBytes;
  bindings.reserve(prepare ? 21 : 19);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.aWeights, buffers.timeBias,
                   buffers.stepSizes, buffers.skipWeights,
                   buffers.mixerNorm.buffer, buffers.hidden});
  if (prepare)
    bindings.insert(bindings.end(),
                    {buffers.linearScratch.input, buffers.linearScratch.sums});
  if (folded) {
    // The tape binds at index 20 for every variant: the plain kernel's
    // params sit at 17, so two unused but real buffers fill 18 and 19.
    if (!prepare)
      bindings.insert(bindings.end(), {buffers.hidden, buffers.hidden});
    bindings.push_back(buffers.tape);
  }
  // The conv bias binds at 21 past the params slot (17 plain, 19 with a
  // linear-input table) and the tape at 20; fillers keep it in place.
  while (bindings.size() < 20)
    bindings.push_back(buffers.hidden);
  bindings.push_back(buffers.convolutionBias);
  Mamba2DecodeBatchParams params{folded ? MAMBA2_TAPE_FOLD : 0u,
                                 layer,
                                 shape.dtMinimum,
                                 shape.dtMaximum,
                                 buffers.mixerNorm.rmsEpsilon,
                                 state.convolutionLayerBytes,
                                 state.recurrentLayerBytes,
                                 state.convolutionStateBytes,
                                 {},
                                 verifyStride,
                                 {},
                                 {},
                                 {}};
  for (uint32_t lane = 0; lane < RICHENGINE_MAXIMUM_BATCH_WIDTH; ++lane)
    params.live_rows[lane] =
        lane < liveRows.size()
            ? std::clamp(liveRows[lane], uint32_t{1}, verifyStride)
            : verifyStride;
  if (folded) {
    for (uint32_t lane = 0; lane < RICHENGINE_MAXIMUM_BATCH_WIDTH; ++lane) {
      params.pending_tape[lane] =
          tapeLanes->pending[lane] + uint64_t{layer} * tapeLanes->layerBytes;
      params.step_tape[lane] =
          tapeLanes->step[lane] + uint64_t{layer} * tapeLanes->layerBytes;
      params.pending_rows[lane] =
          lane < lanes ? tapeLanes->rows[lane] : 0;
    }
  }
  const std::string name =
      std::string(kVerifyMamba2Fused) + tableSuffix(input);
  const std::string kernel_name =
      mamba2Kernel(name, buffers.mixerNorm, shape.innerSize);
  // Every variant takes the params-through-binding-list form: the conv
  // bias's slot at 21 sits past params either way.
  graph.addParamsAt(kernel_name, std::move(bindings), params,
                    prepare ? 19u : 17u, {shape.groups, lanes, 1});
  if (!prepare)
    return {};
  return {buffers.hidden, input};
}

void Mamba2::addCommit(metal::CommandGraph &graph, Mamba2CommitBuffers buffers,
                       Mamba2Shape shape, uint32_t layers, uint32_t lanes,
                       Mamba2StateStrides state, uint32_t verifyStride) {
  if (!layers || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH ||
      !state.valid() || !verifyStride)
    throw std::invalid_argument("invalid Mamba2 commit geometry");
  (void)kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.stepSizes, buffers.aWeights};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  const Mamba2BatchCommitParams params{state.convolutionLayerBytes,
                                       state.recurrentLayerBytes,
                                       state.convolutionStateBytes,
                                       verifyStride};
  // Static parameters and arena/state buffers: replayable. The current/next
  // state swap alternates between the span cache's two slots.
  graph.beginBakedSpan();
  graph.add(kVerifyMamba2Commit.data(), std::move(bindings), params,
            {shape.groups, layers, lanes});
  graph.endBakedSpan();
}

} // namespace richengine::ops
