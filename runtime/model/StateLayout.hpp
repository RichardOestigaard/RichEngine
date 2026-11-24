#pragma once

#include "Checked.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <cstdint>

namespace richengine::model {

// The GDN short convolution has four taps; the recurrent state keeps the
// three previous inputs.
inline constexpr uint32_t kGdnConvolutionTaps = 4;

// Physical state geometry is supplied by the paired target and draft models.
// The engine sees only opaque CompositeState handles and byte accounting.
struct GdnStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t convolutionHistory = 0;
  uint32_t convolutionChannels = 0;
  uint32_t recurrentGroups = 0;
  uint32_t recurrentRows = 0;
  uint32_t recurrentColumns = 0;
  // Bytes after the recurrent matrix that the cell keeps for a layer's other
  // state: qwen4exp's PLE convolution history and its QSA indexer's pending
  // keys (TargetLayout::pleHistoryBytes + indexerPendingBytes). Zero on
  // every recurrent target without them.
  uint64_t auxiliaryBytes = 0;

  // A pure dense target carries no recurrent layers (layers == 0); LFM2's
  // convolutions keep the FIFO but no recurrent matrix (zero rows and
  // columns). GDN layers keep both.
  [[nodiscard]] constexpr bool valid() const noexcept {
    if (!layers)
      return !convolutionHistory && !convolutionChannels && !recurrentGroups &&
             !recurrentRows && !recurrentColumns;
    if (!convolutionHistory || !convolutionChannels)
      return false;
    const bool recurrent = recurrentGroups && recurrentRows && recurrentColumns;
    const bool empty = !recurrentGroups && !recurrentRows && !recurrentColumns;
    return recurrent || empty;
  }
  // The layers keep a convolution FIFO only (LFM2), not the GDN matrix.
  [[nodiscard]] constexpr bool convolutionOnly() const noexcept {
    return layers && !recurrentGroups;
  }
  [[nodiscard]] constexpr uint64_t convolutionLayerBytes() const noexcept {
    return alignUp(uint64_t{convolutionHistory} * convolutionChannels *
                   bfloat16Bytes);
  }
  [[nodiscard]] constexpr uint64_t convolutionBytes() const noexcept {
    return uint64_t{layers} * convolutionLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t recurrentLayerBytes() const noexcept {
    return alignUp(uint64_t{recurrentGroups} * recurrentRows *
                   recurrentColumns * sizeof(float));
  }
  [[nodiscard]] constexpr uint64_t recurrentBytes() const noexcept {
    return uint64_t{layers} * recurrentLayerBytes();
  }
  [[nodiscard]] constexpr uint64_t cellBytes() const noexcept {
    return convolutionBytes() + recurrentBytes() + auxiliaryBytes;
  }

  // A cached state's recurrent matrix stores int8 codes with one bf16 scale
  // per block of this many values, the block's largest magnitude over 127
  // (StateCodec); the convolution FIFO and auxiliary bytes stay exact.
  static constexpr uint32_t codecBlock = 32;
  // Whether the recurrent matrix codes evenly into codecBlock blocks;
  // otherwise a cached state keeps the running layout (cachedCellBytes ==
  // cellBytes).
  [[nodiscard]] constexpr bool codable() const noexcept {
    return recurrentColumns % codecBlock == 0;
  }
  // Every value row of one layer: each group's recurrentRows of
  // recurrentColumns values.
  [[nodiscard]] constexpr uint64_t recurrentRowCount() const noexcept {
    return uint64_t{recurrentGroups} * recurrentRows;
  }
  // One layer's codes and scales, aligned like the running layer.
  [[nodiscard]] constexpr uint64_t codecLayerBytes() const noexcept {
    if (!recurrentColumns || !codable())
      return recurrentLayerBytes();
    const uint64_t rows = recurrentRowCount();
    return alignUp(rows * recurrentColumns +
                   rows * (recurrentColumns / codecBlock) * bfloat16Bytes);
  }
  // A cached state's cell: the convolution FIFO and auxiliary bytes, the
  // recurrent matrix as codes.
  [[nodiscard]] constexpr uint64_t cachedCellBytes() const noexcept {
    return convolutionBytes() + uint64_t{layers} * codecLayerBytes() +
           auxiliaryBytes;
  }

  bool operator==(const GdnStateLayout &) const = default;
};

// One ring of RICHENGINE_DRAFT_SLIDING_WINDOW slots per KV head for the keys and
// one for the values of every draft layer.
struct DraftStateLayout final {
  static constexpr uint32_t bfloat16Bytes = 2;

  uint32_t layers = 0;
  uint32_t kvHeads = 0;
  uint32_t headDimension = 0;
  // The draft's effective attention window: its declared sliding_window
  // clamped to the ring's RICHENGINE_DRAFT_SLIDING_WINDOW physical slots.
  uint32_t window = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    // A draft-less model's layout is entirely zero; a real draft's is
    // entirely set.
    return (layers && kvHeads && headDimension && window) ||
           (!layers && !kvHeads && !headDimension && !window);
  }
  [[nodiscard]] constexpr uint64_t tensorBytes() const noexcept {
    return uint64_t{kvHeads} * RICHENGINE_DRAFT_SLIDING_WINDOW * headDimension *
           bfloat16Bytes;
  }
  [[nodiscard]] constexpr uint64_t ringBytes() const noexcept {
    return uint64_t{layers} * 2 * tensorBytes();
  }

  bool operator==(const DraftStateLayout &) const = default;
};

struct CompositeStateLayout final {
  // The GDN cells a lane holds, with one draft ring.
  static constexpr uint32_t kLaneGdnCells = 2;

  GdnStateLayout target;
  DraftStateLayout draft;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return target.valid() && draft.valid();
  }
  [[nodiscard]] constexpr uint64_t laneBytes() const noexcept {
    return kLaneGdnCells * target.cellBytes() + draft.ringBytes();
  }
  [[nodiscard]] constexpr uint64_t cachedBytes() const noexcept {
    return target.cachedCellBytes() + draft.ringBytes();
  }
  // A cached state stored as it runs, not coded
  // (RICHENGINE_CACHED_STATES=exact): the running cell's bytes.
  [[nodiscard]] constexpr uint64_t exactBytes() const noexcept {
    return target.cellBytes() + draft.ringBytes();
  }

  bool operator==(const CompositeStateLayout &) const = default;
};

} // namespace richengine::model
