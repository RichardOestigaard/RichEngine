#pragma once

// The prefill chunk's shared scratch layout and its allocation owner.

#include "model/RuntimeGeometry.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace richengine::model {

enum class PrefillTensor : uint32_t {
  Hidden0,
  Hidden1,
  InputTokens,
  Normalized,
  Captured,
  GdnPacked,
  GdnQueries,
  GdnKeys,
  GdnValues,
  GdnDecay,
  GdnBeta,
  Recurrent,
  GdnHidden,
  GdnOutput,
  GateIntermediate,
  Intermediate,
  FullPacked,
  FullQueries,
  FullAttention,
  AttentionPartials,
  AttentionStatistics,
  AttentionHidden,
  AttentionOutput,
  ProjectionSums,
  DownProjectionSums,
  TargetPositions,
  DraftPositions,
  TargetInverseFrequencies,
  DraftInverseFrequencies,
  RopeCos,
  RopeSin,
  ContextProjected,
  ContextHidden,
  ContextKv,
  DraftRopeCos,
  DraftRopeSin,
  // The alternate-geometry (Gemma global) layers' rope tables and inverse
  // frequencies; zero-sized on single-geometry targets.
  TargetInverseFrequencies2,
  RopeCosAlt,
  RopeSinAlt,
  // The zeroed hidden-width rows Gemma's routed MoE combine adds to;
  // memset once at arena creation.
  ZeroResidual,
  ChunkKeys,
  ChunkValues,
  // One tensor per ops::kMoeScratchFields entry, in its order (moeScratchTensor).
  MoeScratch,
  MoeScratchLast = MoeScratch + ops::kMoeScratchFields.size() - 1,
  LinearPartials,
  LinearCounters,
  // The rotated input of a rotated projection (ops::LinearScratch::rotated).
  LinearRotated,
  // The pack planes of a GGUF prefill chunk's MXFP4 segments
  // (ops::LinearScratch::input and ::sums, kernels/shared/gguf_mxfp4p.metal).
  LinearPacked,
  LinearExponents,
  // WY/UT scratch of the chunked GDN scan (RICHENGINE_GDN_CHUNKED); zero-sized
  // unless the flag selects a chunk factor.
  GdnChunkScratch,
  // RICHENGINE_PREFILL_FAST_INT8 operand buffers: the two split terms' uint8
  // codes (rows x inputSize) and their (scale, lo, Jx) float4 records
  // (rows x inputSize/64); zero-sized unless the flag is set.
  I8Codes,
  I8CodesLo,
  I8Params,
  I8ParamsLo,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// Sizes depend on the geometry and the device's operator plans.
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators);

class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators)
      : bytes_(plannedPrefillBytes(geometry, operators)) {
    const auto sizes = prefillTensorBytes(geometry, operators);
    base_ = backend.allocateBuffer(bytes_, metal::BufferStorage::Shared,
                                   "shared-prefill");
    uint64_t cursor = 0;
    for (uint32_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index])
        tensors_[index] = backend.view(base_, cursor, sizes[index]);
      cursor += alignUp(sizes[index]);
    }
    if (cursor != bytes_)
      throw std::logic_error("prefill arena mismatch");
    // The host writes these three tensors before every chunk; a submit-ahead
    // chunk needs a bank the in-flight chunk does not read. Bank 0 is the
    // arena view; bank 1 lives in its own small allocation.
    uint64_t bankedBytes = 0;
    for (const PrefillTensor tensor :
         {PrefillTensor::InputTokens, PrefillTensor::TargetPositions,
          PrefillTensor::DraftPositions})
      bankedBytes += alignUp(sizes[static_cast<uint32_t>(tensor)]);
    bankedBase_ = backend.allocateBuffer(bankedBytes, metal::BufferStorage::Shared,
                                         "shared-prefill-inputs");
    uint64_t bankedCursor = 0;
    for (const PrefillTensor tensor :
         {PrefillTensor::InputTokens, PrefillTensor::TargetPositions,
          PrefillTensor::DraftPositions}) {
      const uint32_t index = static_cast<uint32_t>(tensor);
      if (sizes[index])
        bankedTensors_[index] =
            backend.view(bankedBase_, bankedCursor, sizes[index]);
      bankedCursor += alignUp(sizes[index]);
    }
    auto *target = static_cast<float *>(
        get(PrefillTensor::TargetInverseFrequencies).contents());
    auto *draft = static_cast<float *>(
        get(PrefillTensor::DraftInverseFrequencies).contents());
    if (!target || !draft)
      throw std::logic_error("RoPE frequencies are not CPU-visible");
    for (uint32_t dim = 0; dim < geometry.target.rotaryPairs; ++dim) {
      target[dim] =
          std::pow(geometry.target.rotaryTheta,
                   -static_cast<float>(dim) / geometry.target.rotaryPairs);
    }
    for (uint32_t dim = 0; dim < geometry.draftRotaryPairs(); ++dim) {
      draft[dim] = std::pow(geometry.draft.rotaryTheta,
                            -static_cast<float>(dim) / geometry.draftRotaryPairs());
    }
    // The dual-geometry target's alternate (Gemma global) frequencies.
    // Proportional rope derives its inverse frequencies over the full head
    // dimension, not the rotated-pair count: base^(-2i/headDim).
    if (const metal::MetalBuffer alt =
            get(PrefillTensor::TargetInverseFrequencies2)) {
      auto *frequencies = static_cast<float *>(alt.contents());
      for (uint32_t dim = 0; dim < geometry.target.altRotaryPairs; ++dim)
        frequencies[dim] =
            std::pow(geometry.target.altRotaryTheta,
                     -2.0F * static_cast<float>(dim) /
                         geometry.target.altHeadDimension);
    }
    // Split projections return their counters to zero; they start there.
    if (const metal::MetalBuffer counters = get(PrefillTensor::LinearCounters))
      std::memset(counters.contents(), 0, counters.sizeBytes());
    if (const metal::MetalBuffer zero = get(PrefillTensor::ZeroResidual))
      std::memset(zero.contents(), 0, zero.sizeBytes());
  }

  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor) const {
    return tensors_[static_cast<uint32_t>(tensor)];
  }
  // Bank 1 exists only for the host-written input tensors; every other
  // tensor resolves to the same view at either bank.
  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor,
                                       uint32_t bank) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (bank && bankedTensors_[index]) return bankedTensors_[index];
    return tensors_[index];
  }
  [[nodiscard]] ops::MoeScratch moeScratch() const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          get(moeScratchTensor<PrefillTensor>(field));
    return scratch;
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  metal::MetalBuffer base_;
  metal::MetalBuffer bankedBase_;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  std::array<metal::MetalBuffer, prefillTensorCount> bankedTensors_{};
  uint64_t bytes_ = 0;
};

} // namespace richengine::model
