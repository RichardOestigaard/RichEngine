#pragma once

// The prefill chunk's shared scratch layout and its allocation owner.

#include "model/CompositeStateStorage.hpp"
#include "model/RuntimeGeometry.hpp"
#include "model/YarnRope.hpp"

#include "Checked.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/ExecutionPlans.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

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
  // nemotron's Mamba-2 prepare kernel's post-conv x|B|C rows
  // ([rows][convolutionDimension] bf16); zero-sized on other targets.
  GdnPostConv,
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
  ContextQkv,
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
  // Bolt's LatentMoE rows: the latent-down rows, the experts' latent
  // combine and the immutable zero residual it reads. Zero-sized elsewhere.
  MoeLatentRows,
  MoeLatentOutput,
  MoeZeroLatent,
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
  // The maxwell-moe MTP draft's prefill chunk scratch (DraftKind::Mtp
  // only): kMtpPrefillChunk rows of the fp32 streams, the eh operand and
  // projection, the mHC planes, the rolled tokens and their normed
  // embeddings, and the K/V rows mtp_kv_store_prefill writes.
  MtpStreams,
  MtpEh,
  MtpEhOut,
  MtpPlane,
  MtpMappings,
  MtpTokens,
  MtpEmbed,
  MtpEmbedNormed,
  MtpKeys,
  MtpValues,
  // A command's scratch GDN cell, which carries a straddled lane's state
  // from one step's part to the other's (PrefillArena::groupCell): one
  // layer's convolution and recurrent state, and the auxiliary.
  GroupCell,
  Count,
};

constexpr uint32_t prefillTensorCount =
    static_cast<uint32_t>(PrefillTensor::Count);

// Whether a tensor's rows outlive the interleave of a command's steps,
// which then holds each step's own from step s times a step's rows on: the
// hidden rows the layers carry between them, the normalized rows and mixer
// outputs the split's pending evaluation and its join read, the RoPE
// tables the steps' attention layers read, the host's per-step inputs
// (tokens, positions) and the rows the draft's context reads once the
// layers ran (the captured rows and their positions and rope tables). The
// rest of the scratch each step's dispatches use inside its own slice of a
// layer is shared, one step's after another's (TargetModel::addPrefill).
[[nodiscard]] constexpr bool prefillStepTensor(PrefillTensor tensor) noexcept {
  switch (tensor) {
  case PrefillTensor::Hidden0:
  case PrefillTensor::Hidden1:
  case PrefillTensor::InputTokens:
  case PrefillTensor::Normalized:
  case PrefillTensor::Captured:
  case PrefillTensor::GdnOutput:
  case PrefillTensor::AttentionOutput:
  case PrefillTensor::TargetPositions:
  case PrefillTensor::DraftPositions:
  case PrefillTensor::RopeCos:
  case PrefillTensor::RopeSin:
  case PrefillTensor::RopeCosAlt:
  case PrefillTensor::RopeSinAlt:
  case PrefillTensor::DraftRopeCos:
  case PrefillTensor::DraftRopeSin:
    return true;
  default:
    return false;
  }
}

// Sizes depend on the geometry and the device's operator plans.
[[nodiscard]] std::array<uint64_t, prefillTensorCount>
prefillTensorBytes(const RuntimeGeometry &geometry,
                   const ops::ExecutionPlans &operators);
[[nodiscard]] uint64_t plannedPrefillBytes(const RuntimeGeometry &geometry,
                                           const ops::ExecutionPlans &operators);

// The bytes of a command's scratch cell (PrefillTensor::GroupCell): every
// layer's views are one layer's convolution and recurrent state, then the
// auxiliary.
[[nodiscard]] uint64_t groupCellBytes(const GdnStateLayout &layout) noexcept;
[[nodiscard]] GdnParityBuffers
groupCellBuffers(metal::MetalBackend &backend, metal::MetalBuffer base,
                 const GdnStateLayout &layout);

class PrefillArena final {
public:
  PrefillArena(metal::MetalBackend &backend, const RuntimeGeometry &geometry,
                const ops::ExecutionPlans &operators)
      : chunks_(geometry.prefillChunks),
        bytes_(plannedPrefillBytes(geometry, operators)) {
    const auto sizes = prefillTensorBytes(geometry, operators);
    base_ = backend.allocateBuffer(bytes_, metal::BufferStorage::Shared,
                                   "shared-prefill");
    uint64_t cursor = 0;
    for (uint32_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index]) {
        tensors_[index] = backend.view(base_, cursor, sizes[index]);
        if (chunks_ > 1 &&
            prefillStepTensor(static_cast<PrefillTensor>(index)))
          chunkViews_[index] =
              stepViews(backend, base_, cursor, sizes[index], chunks_);
      }
      cursor += alignUp(sizes[index]);
    }
    if (chunks_ > 1)
      groupCell_ = groupCellBuffers(
          backend, get(PrefillTensor::GroupCell), geometry.target.stateLayout);
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
      if (sizes[index]) {
        bankedTensors_[index] =
            backend.view(bankedBase_, bankedCursor, sizes[index]);
        if (chunks_ > 1 &&
            prefillStepTensor(static_cast<PrefillTensor>(index)))
          bankedChunkViews_[index] =
              stepViews(backend, bankedBase_, bankedCursor, sizes[index],
                        chunks_);
      }
      bankedCursor += alignUp(sizes[index]);
    }
    auto *target = static_cast<float *>(
        get(PrefillTensor::TargetInverseFrequencies).contents());
    auto *draft = static_cast<float *>(
        get(PrefillTensor::DraftInverseFrequencies).contents());
    // A NoPE target's (or draft's) zero-pair tensor is legitimately empty.
    if ((geometry.target.rotaryPairs && !target) ||
        (geometry.draftRotaryPairs() && !draft))
      throw std::logic_error("RoPE frequencies are not CPU-visible");
    for (uint32_t dim = 0; dim < geometry.target.rotaryPairs; ++dim) {
      target[dim] =
          std::pow(geometry.target.rotaryTheta,
                   -static_cast<float>(dim) / geometry.target.rotaryPairs);
    }
    // YaRN-scaled where the draft declares it (ropeFactor over 1): the
    // helper fills the unscaled theta^(-dim/pairs) progression otherwise,
    // so a "default" rope's frequencies are bit-identical to before.
    yarnInverseRotaryFrequencies(
        geometry.draft.rotaryTheta, geometry.draftRotaryPairs(),
        geometry.draft.ropeFactor, geometry.draft.ropeOriginalMax,
        {draft, geometry.draftRotaryPairs()});
    // The dual-geometry target's alternate (Gemma global) frequencies.
    // Proportional rope derives its inverse frequencies over the full head
    // dimension, not the rotated-pair count: base^(-2i/headDim).
    if (const metal::MetalBuffer alt =
            get(PrefillTensor::TargetInverseFrequencies2)) {
      auto *frequencies = static_cast<float *>(alt.contents());
      // LFM3's zero-tail partial rotary: only the first altRotaryRealPairs
      // entries carry the progression; the rest read 0, leaving those dims
      // unrotated (cos 1, sin 0).
      const uint32_t real = geometry.target.altRotaryRealPairs
                                ? geometry.target.altRotaryRealPairs
                                : geometry.target.altRotaryPairs;
      for (uint32_t dim = 0; dim < geometry.target.altRotaryPairs; ++dim)
        frequencies[dim] =
            dim < real
                ? std::pow(geometry.target.altRotaryTheta,
                           -2.0F * static_cast<float>(dim) /
                               geometry.target.altHeadDimension)
                : 0.0F;
    }
    // Split projections return their counters to zero; they start there.
    if (const metal::MetalBuffer counters = get(PrefillTensor::LinearCounters))
      std::memset(counters.contents(), 0, counters.sizeBytes());
    if (const metal::MetalBuffer zero = get(PrefillTensor::ZeroResidual))
      std::memset(zero.contents(), 0, zero.sizeBytes());
    // Bolt's latent combine residual: zeroed once here and never mutated
    // (moe_combine reads residual only).
    if (const metal::MetalBuffer zero = get(PrefillTensor::MoeZeroLatent))
      std::memset(zero.contents(), 0, zero.sizeBytes());
  }

  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor) const {
    return get(tensor, 0);
  }
  // Step `step`'s view of a step tensor (prefillStepTensor), or the tensor
  // of any shared one.
  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor,
                                       uint32_t step) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (step >= chunks_)
      throw std::out_of_range("prefill arena step");
    return chunkViews_[index].empty() ? tensors_[index]
                                      : chunkViews_[index][step];
  }
  // Bank 1 exists only for the host-written input tensors; every other
  // tensor resolves to the same view at either bank.
  [[nodiscard]] metal::MetalBuffer get(PrefillTensor tensor, uint32_t step,
                                       uint32_t bank) const {
    const uint32_t index = static_cast<uint32_t>(tensor);
    if (step >= chunks_)
      throw std::out_of_range("prefill arena step");
    if (bank && !bankedChunkViews_[index].empty())
      return bankedChunkViews_[index][step];
    if (bank && bankedTensors_[index])
      return bankedTensors_[index];
    return get(tensor, step);
  }
  [[nodiscard]] ops::MoeScratch moeScratch() const {
    ops::MoeScratch scratch;
    for (size_t field = 0; field < ops::kMoeScratchFields.size(); ++field)
      scratch.*ops::kMoeScratchFields[field].buffer =
          get(moeScratchTensor<PrefillTensor>(field));
    return scratch;
  }
  // A command's scratch cell (PrefillTensor::GroupCell), which carries a
  // straddled lane's state between its two steps' parts: its layers share
  // one layer's state, which a part writes at a GDN layer and the other
  // step's part reads there before any part's next GDN layer writes it
  // again. Null for a command of one step.
  [[nodiscard]] const GdnParityBuffers *groupCell() const noexcept {
    return groupCell_ ? &*groupCell_ : nullptr;
  }
  [[nodiscard]] uint32_t chunks() const noexcept { return chunks_; }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }

private:
  // The views of a step tensor of `bytes` from `offset` of `base`, of
  // `steps` equal parts: step s's from part s to the tensor's end, so the
  // first step's covers every row a command of one step holds.
  [[nodiscard]] static std::vector<metal::MetalBuffer>
  stepViews(metal::MetalBackend &backend, const metal::MetalBuffer &base,
            uint64_t offset, uint64_t bytes, uint32_t steps) {
    std::vector<metal::MetalBuffer> views;
    const uint64_t part = bytes / steps;
    for (uint32_t step = 0; step < steps; ++step)
      views.push_back(backend.view(base, offset + step * part,
                                   bytes - step * part));
    return views;
  }

  metal::MetalBuffer base_;
  metal::MetalBuffer bankedBase_;
  uint32_t chunks_ = 1;
  std::array<metal::MetalBuffer, prefillTensorCount> tensors_{};
  std::array<metal::MetalBuffer, prefillTensorCount> bankedTensors_{};
  std::array<std::vector<metal::MetalBuffer>, prefillTensorCount>
      chunkViews_{};
  std::array<std::vector<metal::MetalBuffer>, prefillTensorCount>
      bankedChunkViews_{};
  std::optional<GdnParityBuffers> groupCell_;
  uint64_t bytes_ = 0;
};

} // namespace richengine::model
