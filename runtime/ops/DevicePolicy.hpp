#pragma once

#include "metal/DeviceCapabilities.hpp"

#include <cstdint>

namespace richengine::ops {

// The GPU core count kernel policy plans for: the reported one, or
// kAssumedGpuCores when the device does not report it.
[[nodiscard]] constexpr uint32_t plannedGpuCores(const DeviceCapabilities &device) noexcept {
  return device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores;
}

// The chip-specific kernel policy every operator plans against, derived once
// from the probed capabilities. Operators read this object and the tuning
// tables scoped to it (DeviceTuning.cpp); they do not re-derive family or
// core-count decisions themselves.
struct DevicePolicy final {
  // The device generation the static policy constants were fitted to.
  // Family 9 is Apple9; family 10 and every newer, not-yet-re-tiered family
  // is Apple10 (they run its kernels and, until their own measurements land,
  // its tuning — the one decision this name covers, not a per-site
  // comparison). Anything older is Unknown and takes each policy's default:
  // the startup floor rejects those families before they plan.
  enum class Tier : uint8_t { Unknown, Apple9, Apple10 };

  // MTLGPUFamilyAppleN.
  uint32_t family = 0;
  // The planned core count: the reported one, or kAssumedGpuCores.
  uint32_t cores = kAssumedGpuCores;
  // Whether `cores` is a real count. A measured row scoped to an exact core
  // count matches only a reported one: an assumed count is a plan, not a
  // measurement the row was fitted to.
  bool coresReported = false;

  [[nodiscard]] static constexpr DevicePolicy of(const DeviceCapabilities &device) noexcept {
    return {device.appleGpuFamily, plannedGpuCores(device), device.gpuCoreCount != 0};
  }

  [[nodiscard]] constexpr Tier tier() const noexcept {
    return family == 9 ? Tier::Apple9 : family >= 10 ? Tier::Apple10 : Tier::Unknown;
  }
  // The Apple9 generation exactly: MTLGPUFamilyApple9. Older families are
  // Unknown, not Apple9 — the policy defaults apply to them.
  [[nodiscard]] constexpr bool isApple9() const noexcept { return family == 9; }
  // In the Apple10 tuning generation: family 10 and every newer family the
  // tuning tables have not been re-measured for.
  [[nodiscard]] constexpr bool apple10Plus() const noexcept { return tier() == Tier::Apple10; }
  // The Metal 4.1 packed-FP4 kernels the `_n`/`mxfp4n` formats decode on,
  // family 10 and up. Earlier families never create their pipelines, whose
  // AIR would not lower there.
  [[nodiscard]] constexpr bool nativeFormats() const noexcept { return family >= 10; }
  // The kernel name suffix of a native-format dispatch.
  [[nodiscard]] constexpr const char *nativeFormatSuffix() const noexcept {
    return nativeFormats() ? "_n" : "";
  }
};

} // namespace richengine::ops
