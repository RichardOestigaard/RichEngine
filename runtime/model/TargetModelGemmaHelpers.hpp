#pragma once

// The internals TargetModel.cpp and TargetModelGemma.cpp share: the prefill
// and verify step records both targets' layer encoders walk, and the row
// and input-sum buffer views they slice per sequence. Kept out of
// TargetModel.hpp so only the two target translation units see them.
#include "model/TargetModel.hpp"

namespace richengine::model {

// The state a prefill command's layers share: its inputs and the next GDN
// and attention layer of the step.
struct TargetModel::PrefillStep {
  metal::CommandGraph &graph;
  const TargetModelPrefillBuffers &buffers;
  std::span<const TargetModelPrefillSequence> sequences;
  uint32_t rows;
  std::span<const RichKvLayer> kvLayers;
  // Each sequence's attention plan, which every attention layer runs.
  std::vector<ops::PrefillAttentionPlan> attention{};
  // The same sequences' plans in the alternate (Gemma global) geometry;
  // empty on single-geometry targets.
  std::vector<ops::PrefillAttentionPlan> altAttention{};
  std::optional<ops::MoePlan> moe{};
  ops::AneFfn *aneFfn = nullptr;
  // A DiffusionGemma encoder pass's per-layer scalars (empty on every other
  // pass: the packed layer_scalar applies).
  std::span<const float> layerScalars{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  // A DiffusionGemma trunk pass: every sequence is a canvas row span, and
  // plan selection takes the canvas family (MoePhase::Canvas, canvas-marked
  // linear workloads).
  bool canvas = false;
};

struct TargetModel::VerifyStep {
  metal::CommandGraph &graph;
  const TargetModelVerifyBuffers &buffers;
  std::span<const RichKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  // The alternate-geometry (Gemma global) plan; empty on single-geometry
  // targets.
  std::optional<ops::VerifyAttentionPlan> altAttention{};
  std::optional<ops::MoePlan> moe{};
  // A tree batch keeps `lanes` real lanes of `rowCapacity` node rows, but the
  // row-space operators (linear plans, MoE, the head) see planLanes virtual
  // lanes of targetVerifyRows rows — the same tiles as a wider chain batch.
  uint32_t planLanes = 0;
  uint32_t rowCapacity = 0;
  bool tree = false;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  std::span<const uint32_t> liveRows{};
};

namespace {

// Rows [begin, begin + count) of a row-major buffer of `width` values of T.
template <class T>
metal::MetalBuffer rowsOf(metal::MetalBackend &backend, const metal::MetalBuffer &buffer, uint32_t begin,
                          uint32_t count, uint32_t width) {
  return backend.view(buffer, uint64_t{begin} * width * sizeof(T), uint64_t{count} * width * sizeof(T));
}

// The out-projection's input sums of the rows [begin, begin + count): one
// float per 64-wide group of the `width`-element rows, in the flat
// [row][group] layout of prefill_linear_q4_sums32.
metal::MetalBuffer prefillSums(metal::MetalBackend &backend, const metal::MetalBuffer &sums, uint32_t begin,
                               uint32_t count, uint32_t width) {
  return backend.view(sums, uint64_t{begin} * (width / 64) * sizeof(float),
                      uint64_t{count} * (width / 64) * sizeof(float));
}

} // namespace

} // namespace richengine::model
