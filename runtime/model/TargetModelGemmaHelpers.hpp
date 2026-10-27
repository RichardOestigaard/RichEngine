#pragma once

// The internals the target translation units and their layer encoders
// share: the prefill and verify step records the encoders walk, and the
// row and input-sum buffer views they slice per sequence. Kept out of
// TargetModel.hpp so only the encode translation units see them.
#include "model/TargetModel.hpp"

#include <any>

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
  std::optional<ops::MoePlan> moe{};
  ops::FfnSplit *aneFfn = nullptr;
  // The active layer encoder's per-pass state, owned by it: addPrefill
  // seeds it with the encoder-routed inputs (TargetLayerEncoder.hpp's
  // TargetPrefillInputs) before the encoder's beginPrefill installs its
  // own record — Gemma's holds the alternate-geometry plans, the canvas
  // flag and the DiffusionGemma encoder pass's layer scalars.
  std::any extension{};
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
};

struct TargetModel::VerifyStep {
  metal::CommandGraph &graph;
  const TargetModelVerifyBuffers &buffers;
  std::span<const RichKvLayer> kvLayers;
  std::span<const kv::ChunkedPrefillParams> chunks;
  uint32_t lanes;
  uint32_t rows;
  ops::VerifyAttentionPlan attention;
  std::optional<ops::MoePlan> moe{};
  // Bounds a tree batch's live node count where the caller already knows
  // the emitted counts — the attention splits and verify stores then size
  // for the real rows; the encoder's alternate-geometry plan reads it too.
  uint32_t liveNodes = 0;
  // The active layer encoder's per-pass state, owned by it (Gemma's holds
  // the alternate-geometry plan).
  std::any extension{};
  // A tree batch keeps `lanes` real lanes of `rowCapacity` node rows, but the
  // row-space operators (linear plans, MoE, the head) see planLanes virtual
  // lanes of targetVerifyRows rows — the same tiles as a wider chain batch.
  uint32_t planLanes = 0;
  uint32_t rowCapacity = 0;
  bool tree = false;
  uint32_t gdnLayer = 0;
  uint32_t attentionLayer = 0;
  std::span<const uint32_t> liveRows{};
  // The lanes' GDN decode tape (folded commit): the previous step's retained
  // rows the scan folds in. Null runs the standalone commit pass instead.
  const ops::GdnTapeLanes *tapeLanes = nullptr;
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
[[maybe_unused]] metal::MetalBuffer prefillSums(metal::MetalBackend &backend, const metal::MetalBuffer &sums, uint32_t begin,
                               uint32_t count, uint32_t width) {
  return backend.view(sums, uint64_t{begin} * (width / 64) * sizeof(float),
                      uint64_t{count} * (width / 64) * sizeof(float));
}

} // namespace

} // namespace richengine::model
