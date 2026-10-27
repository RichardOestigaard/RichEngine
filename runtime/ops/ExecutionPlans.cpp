#include "ops/ExecutionPlans.hpp"

#include "Tuning.hpp"
#include "model/ModelPolicy.hpp"

#include <algorithm>
#include <stdexcept>

namespace richengine::ops {
namespace {

constexpr uint32_t kMaximumLanes = RICHENGINE_MAXIMUM_BATCH_WIDTH;
constexpr uint32_t kDecodeRows = RICHENGINE_TARGET_VERIFY_ROWS;
static_assert(kMaximumLanes == 4);

template <typename Workspace, size_t N>
void include(Workspace &bound, const Workspace &required,
             const std::array<uint64_t Workspace::*, N> &fields,
             uint32_t lanes = 1) {
  for (auto field : fields) {
    const uint64_t bytes = required.*field;
    bound.*field = std::max(bound.*field,
                           bytes / lanes + uint64_t{bytes % lanes != 0});
  }
}

[[nodiscard]] constexpr DevicePolicy policyFor(const DeviceCapabilities &device,
                                               std::string_view model) noexcept {
  DevicePolicy policy = DevicePolicy::of(device);
  policy.model = model;
  return policy;
}

} // namespace

ExecutionPlans::ExecutionPlans(const DeviceCapabilities &device,
                               uint32_t moeUnionCap, std::string_view model)
    : policy_(policyFor(device, model)),
      linear_(policy_),
      // --moe-union wins; else the resolved policy (an explicit
      // RICHENGINE_MOE_UNION over the model's declared cap).
      moeUnionCap_(moeUnionCap ? moeUnionCap
                              : model::modelPolicy().moeUnionCap),
      moeRouteWideRows_(moeRouteWideRows(policy_.cores)),
      // RICHENGINE_MOE_SG4 overrides the family's simdgroups for A/Bs.
      moeDecodeSimdgroups_(tuning().moeSg4 ? MoeExpertSimdgroups::Four
                                          : moeDecodeSimdgroups(policy_)) {}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillPlan(rows, queryHeads, layout);
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, bool tree,
    uint32_t liveNodes, uint32_t rows) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens,
                                    tree, liveNodes, rows);
}

DraftAttentionPlan ExecutionPlans::draftAttention(DraftAttentionShape shape,
                                                 uint32_t lanes,
                                                 uint32_t queryRows) const {
  return DraftAttention::plan(shape, lanes, queryRows);
}

// The router threshold, the expert tile, the simdgroups of a decode plan's
// 8-row tiles and, for a GGUF plan, its tiles.
MoeConfig ExecutionPlans::moeConfig(MoeShape shape, uint32_t rows, MoePhase phase) const {
  const bool canvas = phase == MoePhase::Canvas;
  const bool prefill = phase == MoePhase::Prefill;
  MoeConfig config;
  config.routeWideRows = moeRouteWideRows_;
  if (!prefill && !canvas) {
    config.unionCap = moeUnionCap_;
    config.m8Simdgroups = moeDecodeSimdgroups_;
  }
  // Canvas plans take M16 tiles: a canvas step's union leaves experts well
  // under 32 rows each, and the doubled tile count buys parallelism the
  // weight re-reads cost (measured on the 25B MoE trunk at its 256-row
  // canvas: up pass 742 -> 523 us, ~314 GB/s vs ~148). Prefill keeps M32 at
  // any rows; the GGUF branch below overrides this for block weights.
  config.expertTile = prefill ? MoeExpertTile::M32 : MoeExpertTile::M8;
  if (canvas) config.expertTile = MoeExpertTile::M16;
  if (shape.weightLayout == WeightLayout::Block32) {
    const MoeGgufTile tile = moeGgufTile(policy_, shape);
    if (prefill) config.expertTile = moeGgufPrefillTile(shape, rows, tile);
    config.ggufTile = tile;
    config.ggufRouterTile = linear_.ggufFloatTile(rows, shape.experts);
    config.mxfp4Native = policy_.nativeFormats();
  }
  return config;
}

MoePlan ExecutionPlans::moePrefill(MoeShape shape, uint32_t rows) const {
  return MoE::prefillPlan(shape, rows, moeConfig(shape, rows, MoePhase::Prefill));
}

MoePlan ExecutionPlans::moeCanvas(MoeShape shape, uint32_t rows) const {
  return MoE::canvasPlan(shape, rows, moeConfig(shape, rows, MoePhase::Canvas));
}

MoePlan ExecutionPlans::moeDecode(MoeShape shape, uint32_t lanes,
                                  uint32_t verifyRows) const {
  // Validate before multiplying an untrusted width into the plan's rows.
  if (!lanes || lanes > kMaximumLanes || !verifyRows ||
      verifyRows > kDecodeRows)
    throw std::invalid_argument("invalid MoE decode width");
  return MoE::decodePlan(shape, lanes, verifyRows,
                         moeConfig(shape, lanes * verifyRows, MoePhase::Decode));
}

AttentionWorkspace ExecutionPlans::prefillAttentionWorkspace(
    uint32_t maximumRows, uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::prefillWorkspace(maximumRows, queryHeads, layout);
}

// The verify bound is linear in the lanes: one lane's is every width's share.
AttentionWorkspace ExecutionPlans::verifyAttentionWorkspacePerLane(
    uint32_t queryHeads, kv::Layout layout) const {
  return PagedAttention::verifyWorkspace(1, queryHeads, layout);
}

// The draft workspace is linear in the lanes: one lane's is every width's
// share.
DraftAttentionWorkspace ExecutionPlans::draftAttentionWorkspacePerLane(
    DraftAttentionShape shape) const {
  return DraftAttention::plan(shape, 1).workspace();
}

MoeWorkspace ExecutionPlans::moePrefillWorkspace(MoeShape shape,
                                               uint32_t maximumRows) const {
  // Validate the bound before iterating; every row is included even if a
  // future grouped layout's largest field is not monotone in row count.
  auto bound = moePrefill(shape, maximumRows).workspace();
  for (uint32_t rows = 1; rows <= maximumRows; ++rows)
    include(bound, moePrefill(shape, rows).workspace(), kMoeWorkspaceFields);
  // A canvas plan's M16 tiles need more tile descriptors than the same rows
  // on M32; include its bound over the canvas row span. Canvas plans are
  // affine-only, so the bound only exists for affine shapes.
  if (shape.weightLayout == WeightLayout::Affine64)
    for (uint32_t rows = 1; rows <= std::min(maximumRows, 256u); ++rows)
      include(bound, moeCanvas(shape, rows).workspace(), kMoeWorkspaceFields);
  return bound;
}

MoeWorkspace ExecutionPlans::moeDecodeWorkspacePerLane(MoeShape shape) const {
  MoeWorkspace bound;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes)
    include(bound, moeDecode(shape, lanes).workspace(), kMoeWorkspaceFields, lanes);
  return bound;
}

uint64_t ExecutionPlans::gateUpWorkspace(ProjectionShape shape) const {
  uint64_t bound = 0;
  for (uint32_t lanes = 1; lanes <= kMaximumLanes; ++lanes) {
    const LinearWorkload workload{{shape.outputSize, shape.inputSize}, lanes * kDecodeRows,
                                  LinearPhase::Decode, LinearEpilogue::GateUp, shape.layout};
    bound = std::max(bound, linear_.plan(workload).gateScratchBytes());
  }
  return bound;
}

} // namespace richengine::ops
