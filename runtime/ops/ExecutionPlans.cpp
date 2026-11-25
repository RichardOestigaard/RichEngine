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
                               uint32_t moeUnionCap, std::string_view model,
                               std::string_view checkpoint)
    : policy_(policyFor(device, model)),
      linear_(policy_),
      // --moe-union wins; else the resolved policy (an explicit
      // RICHENGINE_MOE_UNION over the model's declared cap).
      moeUnionCap_(moeUnionCap ? moeUnionCap
                              : model::modelPolicy().moeUnionCap),
      moeRouteWideRows_(moeRouteWideRows(policy_.cores)),
      // RICHENGINE_MOE_SG4 overrides the family's simdgroups for A/Bs.
      moeDecodeSimdgroups_(tuning().moeSg4 ? MoeExpertSimdgroups::Four
                                          : moeDecodeSimdgroups(policy_)),
      measuredMoeIndirectDispatch_(device.deviceName == "Apple M5 Pro" &&
                                   device.appleGpuFamily == 10 &&
                                   device.gpuCoreCount == 20 &&
                                   model == "Qwen3.6-35B-A3B" &&
                                   checkpoint == "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q2_K_XL") {}

uint32_t ExecutionPlans::int8QueriesFromPages() const noexcept {
  if (!tuning().q8qAttention || !policy_.apple10Plus()) return 0;
  // Upstream's calibrated threshold: eight pages a split on the neural
  // accelerators' generation; RICHENGINE_Q8Q_ATTENTION_PAGES retunes it.
  return tuning().q8qAttentionPages ? tuning().q8qAttentionPages : 8;
}

uint32_t ExecutionPlans::prefillPagesPerSplit() const noexcept {
  // RICHENGINE_PREFILL_PAGES_PER_SPLIT overrides on any tier — a value past
  // the history's pages in one split degenerates to the shared splits, so a
  // large one disables the history term. Unset, the neural accelerators'
  // generation takes the measured 128 pages a split (M5 Pro, int8 KV:
  // 7-34% less attention time at 32K-128K histories); Apple9 keeps the
  // shared splits until its generation is measured.
  if (tuning().prefillPagesPerSplit) return tuning().prefillPagesPerSplit;
  return policy_.apple10Plus() ? 128 : 0;
}

PrefillAttentionPlan ExecutionPlans::prefillAttention(
    uint32_t rows, uint32_t queryHeads, kv::Layout layout,
    uint32_t committedTokens) const {
  return PagedAttention::prefillPlan(rows, queryHeads, layout, false,
                                     committedTokens, int8QueriesFromPages(),
                                     prefillPagesPerSplit());
}

VerifyAttentionPlan ExecutionPlans::verifyAttention(
    uint32_t lanes, uint32_t queryHeads, kv::Layout layout,
    std::span<const uint32_t> historyTokens, bool tree,
    uint32_t liveNodes, uint32_t rows) const {
  return PagedAttention::verifyPlan(lanes, queryHeads, layout, historyTokens,
                                    tree, liveNodes, rows, false,
                                    int8QueriesFromPages());
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
    config.routeM16 = tuning().moeRouteM16;
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
    // RICHENGINE_MOE_DECODE_M32: wide decode batches take the 32-row tile so
    // an expert's weights stream once per 32 of its rows, not per 8.
    else if (!canvas && tile == MoeGgufTile::Staged && tuning().moeDecodeM32)
      config.expertTile = MoeExpertTile::M32;
    config.ggufTile = tile;
    // Live-tile indirect dispatch (RICHENGINE_MOE_INDIRECT_DISPATCH=auto):
    // measured on the allowlisted combination only, decode plans past one
    // row — a single-row step's grid has no live-tile slack to recover, and
    // the ungated Bolt path stays opt-in. Auto also excludes M32 plans;
    // MoePlan excludes register tiles in every mode.
    config.indirectDispatch = phase == MoePhase::Decode && rows > 1 &&
                              measuredMoeIndirectDispatch_ &&
                              !shape.ungatedExperts &&
                              config.expertTile == MoeExpertTile::M8;
    config.ggufRouterTile = linear_.ggufFloatTile(rows, shape.experts);
    config.mxfp4Native = policy_.nativeFormats();
    // A chunk's experts run the int8 passes where its dense projections run
    // the int8 prefill tile (prefill chunks past the decode tiles' rows,
    // RICHENGINE_PREFILL_FAST_INT8 on, the int8-matmul families) and the
    // block's experts take it (shape.int8Experts), once its routes average
    // 6 or more an expert — the quantity that drives the passes' cost, not
    // the chunk's rows. Below it the int8 tile runs both 16-row fragments
    // of an expert's tile whatever its live rows, where the staged passes
    // run 8-, 16- or 32-row matmuls by live rows (upstream 1.7.1's measured
    // crossover: from 6 routes an expert shorter in every run). A streamed
    // block never reaches them — the streamed passes have no int8 form.
    config.int8Passes =
        prefill && shape.int8Experts && policy_.apple10Plus() &&
        model::modelPolicy().prefillFastInt8 && rows > kMaximumDecodeTileRows &&
        uint64_t{rows} * shape.expertsPerToken >= 6 * uint64_t{shape.experts};
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
