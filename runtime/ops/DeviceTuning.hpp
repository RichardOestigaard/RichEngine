#pragma once

#include "ops/DevicePolicy.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <optional>
#include <span>

// The measured and fitted kernel choices of a DevicePolicy, in one place
// (DeviceTuning.cpp): the GGUF decode split tiers and rules, and the
// per-shape measured rows of the affine and GGUF decode paths. Each measured
// row carries the device scope it was measured on — families, an optional
// core count — so a row only steers the devices it names, and a newer
// family's own row out-scores a row it inherited.
namespace richengine::ops {

// One tier of a GGUF decode tile's K-split rule: the tile splits again while
// the grid holds fewer than `threadgroups` threadgroups per core and each
// partition would still keep `inputs` inputs.
struct SplitTier final {
  uint32_t threadgroups;
  uint32_t inputs;
};

// The K splits of a GGUF decode tile on `device`: the largest power of two
// some tier still asks for (LinearConfig::kMaximumSplits at most).
[[nodiscard]] uint32_t decodeSplits(const DevicePolicy &device, uint32_t n, uint32_t k,
                                    std::span<const SplitTier> tiers);
// The staged tile's split tiers on `device`: the register tile's fitted
// tiers on Apple9, the one fitted tier on family 10 and up.
[[nodiscard]] std::span<const SplitTier> stagedTiers(const DevicePolicy &device) noexcept;
// The native-format decode tiles' split tier (the MXFP4 multiplane tile and
// the packed-operand formats), which wants deeper splits than the staged
// tile's.
[[nodiscard]] std::span<const SplitTier> mxfp4Tiers() noexcept;

// The GGUF decode tile configurations over an n x k matrix on `device`: the
// exact register tile of Apple9, and the staged tile whose splits follow
// stagedTiers.
[[nodiscard]] LinearConfig registerDecode(const DevicePolicy &device, uint32_t n, uint32_t k);
[[nodiscard]] LinearConfig stagedDecode(const DevicePolicy &device, uint32_t n, uint32_t k);

// The measured per-shape affine plan of `workload` on `device` (both phase
// tables), or none when no row scopes to it.
[[nodiscard]] std::optional<LinearConfig> measuredLinearPlan(const DevicePolicy &device,
                                                             LinearWorkload workload) noexcept;
// The measured K splits of a GGUF decode tile of `tileRows` rows over an
// n x k matrix: the staged table's, or the native-format table's when the
// plan's segments decode natively. Zero when no row scopes to `device`.
[[nodiscard]] uint32_t measuredGgufSplits(const DevicePolicy &device, uint32_t n, uint32_t k,
                                          uint32_t tileRows, bool native) noexcept;
// The largest split either measured table scopes to the shape, for scratch
// sizing before the segments are known.
[[nodiscard]] uint32_t measuredGgufSplitBound(const DevicePolicy &device, uint32_t n, uint32_t k,
                                              uint32_t tileRows) noexcept;

} // namespace richengine::ops
