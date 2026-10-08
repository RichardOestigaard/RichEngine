#pragma once

#include "ops/DraftAttention.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"

#include <span>

namespace richengine::ops {

// The device's plans of every operator a model runs. One runtime owns this
// object and production models borrow it; it never changes after creation.
class ExecutionPlans final {
public:
  // moeUnionCap: --moe-union, the decode plan's routed-expert budget.
  explicit ExecutionPlans(const DeviceCapabilities &device,
                          uint32_t moeUnionCap = 0);
  [[nodiscard]] const Linear &linear() const noexcept { return linear_; }

  [[nodiscard]] PrefillAttentionPlan prefillAttention(
      uint32_t rows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] VerifyAttentionPlan verifyAttention(
      uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
      std::span<const uint32_t> historyTokens, bool tree = false,
      uint32_t liveNodes = 0) const;
  [[nodiscard]] DraftAttentionPlan draftAttention(
      DraftAttentionShape shape, uint32_t lanes) const;
  [[nodiscard]] MoePlan moePrefill(MoeShape shape, uint32_t rows) const;
  // The DiffusionGemma trunk's plan family: canvas-phase expert sweeps.
  [[nodiscard]] MoePlan moeCanvas(MoeShape shape, uint32_t rows) const;
  [[nodiscard]] MoePlan moeDecode(MoeShape shape, uint32_t lanes) const;

  // Bounds cover every row count up to the requested maximum, not just that
  // one. Packed decode arenas use a per-lane stride of
  // max_B ceil(requiredBytes(B)/B), independently for each scratch field.
  [[nodiscard]] AttentionWorkspace prefillAttentionWorkspace(
      uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] AttentionWorkspace verifyAttentionWorkspacePerLane(
      uint32_t queryHeads, kv::Layout layout) const;
  [[nodiscard]] DraftAttentionWorkspace draftAttentionWorkspacePerLane(
      DraftAttentionShape shape) const;
  [[nodiscard]] MoeWorkspace moePrefillWorkspace(
      MoeShape shape, uint32_t maximumRows) const;
  [[nodiscard]] MoeWorkspace moeDecodeWorkspacePerLane(MoeShape shape) const;
  // This scratch is one whole-command buffer, not a per-lane arena field.
  [[nodiscard]] uint64_t gateUpWorkspace(ProjectionShape shape) const;

private:
  // The device's configuration of a MoE plan of `rows` rows in `phase`.
  [[nodiscard]] MoeConfig moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const;

  // The device policy every operator's plans derive from (ops/DevicePolicy.hpp).
  DevicePolicy policy_{};
  Linear linear_;
  uint32_t moeUnionCap_;
  uint32_t moeRouteWideRows_;
  MoeExpertSimdgroups moeDecodeSimdgroups_ = MoeExpertSimdgroups::Eight;
};

} // namespace richengine::ops
