#pragma once

#include "metal/CommandGraph.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// The LFM2 short-convolution mixer's shape: the double-gated causal
// depthwise convolution of `dimension` channels over `taps` taps, fed the
// packed in_proj rows of width `packedWidth` (3 * dimension).
struct LfmConvShape final {
  uint32_t dimension = 0;
  uint32_t taps = 0;
  uint32_t packedWidth = 0;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return dimension && taps >= 2 && packedWidth == 3 * dimension;
  }
};

struct LfmConvPrefillBuffers final {
  metal::MetalBuffer packed;    // [rows][packedWidth] in_proj rows
  metal::MetalBuffer weights;   // conv taps (tapsMajor: [tap][dim], else [dim][tap])
  metal::MetalBuffer stateIn;   // [(taps-1)*dim] FIFO
  metal::MetalBuffer stateOut;
  metal::MetalBuffer output;    // [rows][dim] out-projection input
};

struct LfmConvVerifyBuffers final {
  metal::MetalBuffer packed;    // the layer's [lanes*rows][packedWidth]
  metal::MetalBuffer weights;
  // One conv-state storage per lane (all layers); the layer offset is
  // stateLayerBytes * layer inside it.
  std::span<const metal::MetalBuffer> currentStates;
  metal::MetalBuffer mixed;     // bx rows, lanes*rows*dim, for the commit
  metal::MetalBuffer output;    // [lanes*rows][dim]
};

struct LfmConvCommitBuffers final {
  metal::MetalBuffer mixed;         // the layer's bx rows, lanes*rows*dim
  metal::MetalBuffer retainedCounts;
  std::span<const metal::MetalBuffer> currentStates;
  std::span<const metal::MetalBuffer> nextStates;
};

class LfmConv final {
public:
  // `tapsMajor` records the conv weight's stored order; every current
  // source is [dim][tap] (the squeezed HF tensor's layout), so false.
  static void addPrefill(metal::CommandGraph &graph,
                         const LfmConvPrefillBuffers &buffers,
                         const LfmConvShape &shape, uint32_t rows,
                         bool tapsMajor);
  static void addVerify(metal::CommandGraph &graph,
                        const LfmConvVerifyBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes,
                        uint32_t rows, uint32_t layer,
                        uint64_t stateLayerBytes, bool tapsMajor);
  // `layers` layers' states committed from the step's retained rows in one
  // dispatch: layer l's state slot is layer+l and its `mixed` block sits
  // l*mixedLayerStride bytes into the buffer.
  static void addCommit(metal::CommandGraph &graph,
                        const LfmConvCommitBuffers &buffers,
                        const LfmConvShape &shape, uint32_t lanes,
                        uint32_t rows, uint32_t layer,
                        uint64_t stateLayerBytes, uint32_t layers,
                        uint64_t mixedLayerStride, bool tapsMajor);
};

} // namespace splash::ops
