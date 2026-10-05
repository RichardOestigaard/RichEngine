#include "ops/GDN.hpp"
#include "Env.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/GDN.h"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace richengine::ops {
namespace {

static_assert(offsetof(GDNDecodeBatchParams, conv_layer_bytes) == 8);

enum class KernelLayout : uint8_t { Value48, Value32 };

[[nodiscard]] KernelLayout kernelShape(const GdnShape &shape) {
  if (!shape.valid())
    throw std::invalid_argument("invalid GDN shape");
  if (shape == GdnShape{16, 48, 128, 10240, 16640})
    return KernelLayout::Value48;
  if (shape == GdnShape{16, 32, 128, 8192, 12544})
    return KernelLayout::Value32;
  throw std::invalid_argument("unsupported compiled GDN shape");
}

[[nodiscard]] const char *kernelName(KernelLayout shape,
                                     const char *value48,
                                     const char *value32) noexcept {
  return shape == KernelLayout::Value48 ? value48 : value32;
}

// RICHENGINE_GDN_CHUNKED selects the WY/UT scan's chunk factor (32/64/128).
[[nodiscard]] uint32_t gdnChunkFactor() {
  const uint32_t parsed = envUint("RICHENGINE_GDN_CHUNKED", 0);
  return parsed == 32 || parsed == 64 || parsed == 128 ? parsed : 0;
}

// Scratch floats per (value head, chunk) slot; must match
// gdn_chunk_stride<C>() in prefill/gdn_chunked.metal.
[[nodiscard]] constexpr uint64_t gdnChunkStride(uint32_t factor) {
  return 4 * uint64_t{factor} * factor + uint64_t{factor} * factor / 2 +
         386 * uint64_t{factor} + 8;
}

} // namespace

uint64_t GDN::chunkScratchFloats(const GdnShape &shape, uint32_t tokens,
                                 uint32_t factor) {
  if (!factor)
    return 0;
  const uint64_t chunks = (tokens + factor - 1) / factor;
  return uint64_t{shape.valueHeads} * chunks * gdnChunkStride(factor);
}

void GDN::addPrefill(metal::CommandGraph &graph, GdnPrefillBuffers buffers,
                     GdnShape shape, uint32_t tokens, GdnHeadOrder order,
                     metal::MetalBuffer sums) {
  if (!tokens)
    throw std::invalid_argument("invalid GDN prefill geometry");
  const KernelLayout kernel = kernelShape(shape);
  const std::string gate =
      normKernel(kernelName(kernel,
                            sums ? "prefill_gdn_gate_sums" : "prefill_gdn_gate",
                            sums ? "prefill_gdn_gate_sums_vh32" : "prefill_gdn_gate_vh32"),
                 buffers.mixerNorm, shape.headDimension);
  const GDNPrefillParams params{tokens};
  graph.add(kernelName(kernel, "prefill_gdn_prepare",
                       "prefill_gdn_prepare_vh32"),
            {buffers.packed, buffers.convolutionWeights,
             buffers.convolutionIn, buffers.convolutionOut, buffers.queries,
             buffers.keys, buffers.values, buffers.decayWeights,
             buffers.timeBias, buffers.decay, buffers.beta},
            params, {uint64_t{tokens} * shape.keyHeads, 1, 1},
            {shape.headDimension, 1, 1});
  const uint32_t chunk = buffers.chunkScratch ? gdnChunkFactor() : 0;
  if (chunk) {
    const GDNChunkedParams chunked{tokens, shape.keyHeads, shape.valueHeads,
                                   (tokens + chunk - 1) / chunk};
    const std::vector<metal::MetalBuffer> bindings{
        buffers.queries,      buffers.keys,       buffers.values,
        buffers.decay,        buffers.beta,       buffers.recurrentIn,
        buffers.recurrentOut, buffers.recurrentRows, buffers.chunkScratch};
    const std::string suffix = "_c" + std::to_string(chunk);
    graph.add("gdn_chunked_prep" + suffix, bindings, chunked,
              {uint64_t{shape.valueHeads} * chunked.chunks, 1, 1},
              {128, 1, 1});
    graph.add("gdn_chunked_scan" + suffix, bindings, chunked,
              {uint64_t{shape.valueHeads} * (shape.headDimension / 16), 1, 1},
              {128, 1, 1});
  } else {
    graph.add(kernelName(kernel, "prefill_gdn_scan",
                         "prefill_gdn_scan_vh32"),
              {buffers.queries, buffers.keys, buffers.values, buffers.decay,
               buffers.beta, buffers.recurrentIn, buffers.recurrentOut,
               buffers.recurrentRows},
              params,
              {uint64_t{shape.valueHeads} * shape.headDimension /
                   RICHENGINE_GDN_SCAN_STATE_ROWS,
               1, 1},
              {RICHENGINE_GDN_SCAN_THREADS, 1, 1});
  }
  std::vector<metal::MetalBuffer> gateBindings{buffers.recurrentRows, buffers.packed,
                                                buffers.mixerNorm.buffer, buffers.hidden};
  if (sums) gateBindings.push_back(sums);
  graph.add(gate, gateBindings,
            GDNGatePrefillParams{order == GdnHeadOrder::Tiled},
            {uint64_t{tokens} * shape.valueHeads, 1, 1}, {128, 1, 1});
}

PreparedInput GDN::addDecode(metal::CommandGraph &graph, GdnDecodeBuffers buffers,
                             GdnShape shape, uint32_t lanes, uint32_t layer,
                             GdnStateStrides state, GdnHeadOrder order, LinearInput input,
                             std::span<const uint32_t> liveRows) {
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH || !state.valid() ||
      (!liveRows.empty() && liveRows.size() != lanes))
    throw std::invalid_argument("invalid GDN decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.valueHeads * shape.headDimension,
                        lanes * RICHENGINE_TARGET_VERIFY_ROWS);
  bindings.reserve(prepare ? 19 : 17);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.mixerNorm.buffer,
                   buffers.hidden});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                              layer,
                              state.convolutionLayerBytes,
                              state.recurrentLayerBytes,
                              state.convolutionStateBytes,
                              {}};
  for (uint32_t lane = 0; lane < RICHENGINE_MAXIMUM_BATCH_WIDTH; ++lane)
    params.live_rows[lane] =
        lane < liveRows.size()
            ? std::clamp(liveRows[lane], uint32_t{1},
                         uint32_t{RICHENGINE_TARGET_VERIFY_ROWS})
            : RICHENGINE_TARGET_VERIFY_ROWS;
  const std::string name = std::string("verify_gdn_fused") + tableSuffix(input) + kernelName(kernel, "", "_vh32");
  graph.add(normKernel(name, buffers.mixerNorm, shape.headDimension), std::move(bindings), params,
            {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

PreparedInput GDN::addDecodeTree(metal::CommandGraph &graph,
                                 GdnDecodeBuffers buffers,
                                 metal::MetalBuffer treeNodes,
                                 metal::MetalBuffer treeCounts, GdnShape shape,
                                 uint32_t lanes, uint32_t layer,
                                 GdnStateStrides state, GdnHeadOrder order,
                                 LinearInput input) {
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH || !state.valid() ||
      !treeNodes || !treeCounts)
    throw std::invalid_argument("invalid GDN tree decode geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{buffers.packed,
                                           buffers.convolutionWeights};
  const bool prepare = input != LinearInput::Plain;
  if (prepare)
    requireTableScratch(buffers.linearScratch, input, shape.valueHeads * shape.headDimension,
                        lanes * RICHENGINE_TREE_VERIFY_NODES);
  bindings.reserve(prepare ? 21 : 19);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(),
                  {buffers.mixed, buffers.decayWeights, buffers.timeBias,
                   buffers.decay, buffers.beta, buffers.mixerNorm.buffer,
                   buffers.hidden, treeNodes, treeCounts});
  if (prepare)
    bindings.insert(bindings.end(), {buffers.linearScratch.input, buffers.linearScratch.sums});
  GDNDecodeBatchParams params{order == GdnHeadOrder::Tiled,
                              layer,
                              state.convolutionLayerBytes,
                              state.recurrentLayerBytes,
                              state.convolutionStateBytes,
                              {}};
  for (uint32_t lane = 0; lane < RICHENGINE_MAXIMUM_BATCH_WIDTH; ++lane)
    params.live_rows[lane] = RICHENGINE_TARGET_VERIFY_ROWS;
  const std::string name = std::string("verify_tree_gdn_fused") + tableSuffix(input) + kernelName(kernel, "", "_vh32");
  graph.add(normKernel(name, buffers.mixerNorm, shape.headDimension), std::move(bindings), params,
            {shape.valueHeads, lanes, 1});
  if (!prepare) return {};
  return {buffers.hidden, input};
}

void GDN::addCommit(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                    GdnShape shape, uint32_t layers, uint32_t lanes,
                    GdnStateStrides state) {
  if (!layers || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH ||
      !state.valid())
    throw std::invalid_argument("invalid GDN commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(13);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.push_back(buffers.retainedCounts);
  const GDNBatchCommitParams params{state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  // Static parameters and arena/state buffers: replayable. The current/next
  // state swap alternates between the span cache's two slots.
  graph.beginBakedSpan();
  graph.add(kernelName(kernel, "verify_gdn_commit",
                       "verify_gdn_commit_vh32"),
            std::move(bindings), params,
            {shape.valueHeads, layers, lanes});
  graph.endBakedSpan();
}

void GDN::addCommitTree(metal::CommandGraph &graph, GdnCommitBuffers buffers,
                        metal::MetalBuffer retainedPath, GdnShape shape,
                        uint32_t layers, uint32_t lanes,
                        GdnStateStrides state) {
  if (!layers || !lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH ||
      !state.valid() || !retainedPath)
    throw std::invalid_argument("invalid GDN tree commit geometry");
  const KernelLayout kernel = kernelShape(shape);
  std::vector<metal::MetalBuffer> bindings{
      buffers.packed, buffers.mixed, buffers.decay, buffers.beta};
  bindings.reserve(14);
  appendLaneBindings(bindings, buffers.currentStates, buffers.nextStates);
  bindings.insert(bindings.end(), {buffers.retainedCounts, retainedPath});
  const GDNBatchCommitParams params{state.convolutionLayerBytes,
                                    state.recurrentLayerBytes,
                                    state.convolutionStateBytes};
  graph.beginBakedSpan();
  graph.add(kernelName(kernel, "verify_gdn_commit_tree",
                       "verify_gdn_commit_tree_vh32"),
            std::move(bindings), params,
            {shape.valueHeads, layers, lanes});
  graph.endBakedSpan();
}

} // namespace richengine::ops
