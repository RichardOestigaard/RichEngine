#include "TestChecks.hpp"
#include "ops/DeviceTuning.hpp"
#include "ops/ExecutionPlans.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using namespace richengine;
using namespace richengine::ops;

using richengine::test::require;
template <typename Function> void rejects(Function function) {
  bool rejected = false;
  try { function(); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "invalid operator lookup was accepted");
}

// The target attention shapes: query heads over a KV layout.
struct AttentionShape final {
  uint32_t queryHeads;
  kv::Layout layout;
};
constexpr std::array attentionShapes{
    AttentionShape{24, {1, 4, 256}}, AttentionShape{16, {1, 2, 256}}};
constexpr std::array draftShapes{
    DraftAttentionShape{5120, 1280, 6144, 4096, 32, 8, 128},
    DraftAttentionShape{2048, 512, 6144, 4096, 32, 8, 128}};
constexpr MoeShape routedShape{2048, 256, 8, 512};
constexpr std::array moeShapes{
    routedShape, MoeShape{768, 7, 3, 256}, MoeShape{256, 1, 1, 256}};
constexpr std::array matrices{
    LinearMatrix{17408, 5120}, LinearMatrix{6144, 5120},
    LinearMatrix{6144, 2048}, LinearMatrix{512, 2048},
    LinearMatrix{768, 768}};
// The affine gate/up projection of `matrix`, which gateUpWorkspace sizes.
constexpr ProjectionShape affineGateUp(LinearMatrix matrix) {
  return {matrix.outputSize, matrix.inputSize, WeightLayout::Affine64};
}
constexpr std::array attentionFields{
    &AttentionWorkspace::partialsBytes, &AttentionWorkspace::statisticsBytes};
constexpr std::array draftFields{
    &DraftAttentionWorkspace::convolutionBytes,
    &DraftAttentionWorkspace::qkvBytes,
    &DraftAttentionWorkspace::groupedQueriesBytes,
    &DraftAttentionWorkspace::queryKeysBytes,
    &DraftAttentionWorkspace::queryValuesBytes};

DeviceCapabilities device(uint32_t family = 10) {
  DeviceCapabilities value;
  value.appleGpuFamily = family;
  return value;
}
template <typename Workspace, size_t N>
void covers(const Workspace &stride, const Workspace &needed, uint32_t lanes,
            const std::array<uint64_t Workspace::*, N> &fields) {
  for (auto field : fields)
    require((stride.*field) * lanes >= needed.*field,
            "workspace does not cover an installed plan");
}

void baselinePlans() {
  for (uint32_t family : {9U, 10U, 11U}) {
    const ExecutionPlans plans(device(family));
    const Linear baseline(device(family));
    for (auto matrix : matrices) {
      uint64_t gateBound = 0;
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::GateUp}) {
          const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' decode Linear departed from the device policy");
          if (epilogue == LinearEpilogue::GateUp)
            gateBound = std::max(gateBound, baseline.plan(w).gateScratchBytes());
        }
      }
      require(plans.gateUpWorkspace(affineGateUp(matrix)) == gateBound &&
                  gateBound == (family == 9 ? 0 : uint64_t{32} * matrix.outputSize * 2),
              "gate/up workspace disagrees with fused or decomposed baseline");
      for (uint32_t rows : {1U, 17U, 2048U})
        for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                              LinearEpilogue::UpWithGate}) {
          const LinearWorkload w{matrix, rows, LinearPhase::Prefill, epilogue};
          require(plans.linear().plan(w).configuration() ==
                      baseline.plan(w).configuration(),
                  "the plans' prefill Linear departed from the device policy");
        }
    }
    for (const auto &[queryHeads, kvLayout] : attentionShapes) {
      const auto memory = plans.prefillAttentionWorkspace(2048, queryHeads, kvLayout);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(memory, plans.prefillAttention(rows, queryHeads, kvLayout).workspace, 1,
               attentionFields);
      const auto stride = plans.verifyAttentionWorkspacePerLane(queryHeads, kvLayout);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const std::array<uint32_t, 4> histories{0, 31, 2048, 8192};
        const auto selected = plans.verifyAttention(lanes, queryHeads, kvLayout,
                                                    std::span(histories).first(lanes));
        require(selected.splits == 32, "verify baseline changed");
        covers(stride, selected.workspace, lanes, attentionFields);
      }
      {
        const std::array<uint32_t, 1> deep{131072};
        const auto scaled = plans.verifyAttention(1, queryHeads, kvLayout, deep);
        require(scaled.splits == kv::kVerifyMaximumSplits &&
                    scaled.laneSplits[0] == scaled.splits,
                "verify splits did not scale with history");
        covers(stride, scaled.workspace, 1, attentionFields);
      }
    }
    for (auto shape : draftShapes) {
      const auto stride = plans.draftAttentionWorkspacePerLane(shape);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        covers(stride, plans.draftAttention(shape, lanes).workspace(), lanes, draftFields);
    }
    for (auto shape : moeShapes) {
      const auto stride = plans.moeDecodeWorkspacePerLane(shape);
      require(stride == plans.moeDecode(shape, 1).workspace(), "workspace bound changed");
      const auto prefill = plans.moePrefillWorkspace(shape, 2048);
      for (uint32_t rows = 1; rows <= 2048; ++rows)
        covers(prefill, plans.moePrefill(shape, rows).workspace(), 1, kMoeWorkspaceFields);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const auto selected = plans.moeDecode(shape, lanes);
        require(selected.tileRows() == 8 &&
                    selected.configuration().m8Simdgroups == moeDecodeSimdgroups(DevicePolicy{family}),
                "MoE decode baseline changed");
        covers(stride, selected.workspace(), lanes, kMoeWorkspaceFields);
      }
    }
  }
}

// Affine decode plans run the fused 8-row expert tiles, four-simdgroup on
// Apple9 and the shipped N128 x 8 tile on every other family; affine prefill
// plans run the split 32-row passes on every family and keep the shipped
// simdgroups they do not run.
void moeDeviceTiles() {
  for (uint32_t family : {0U, 9U, 10U, 11U}) {
    const auto expected = family == 9 ? MoeExpertSimdgroups::Four
                                      : MoeExpertSimdgroups::Eight;
    require(moeDecodeSimdgroups(DevicePolicy{family}) == expected,
            "decode expert simdgroups are not gated on GPU family 9");
    const ExecutionPlans plans(device(family));
    for (auto shape : moeShapes) {
      for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
        const MoePlan plan = plans.moeDecode(shape, lanes);
        require(plan.configuration().m8Simdgroups == expected && plan.tileRows() == 8 &&
                    !plan.splitExperts(),
                "MoE decode plan departed from the device tile policy");
      }
      for (uint32_t rows : {1U, 8U, 17U, 2048U}) {
        const MoePlan plan = plans.moePrefill(shape, rows);
        require(plan.tileRows() == 32 && plan.splitExperts() &&
                    plan.configuration().m8Simdgroups == MoeExpertSimdgroups::Eight,
                "MoE prefill plan left the split 32-row passes");
      }
    }
  }
}

// GGUF MoE plans (Block32 weights) run the three expert passes: the
// exact register tile on Apple9, with its Table16 row sums in the workspace
// bounds, staged tiles everywhere else (32-row tiles for prefill chunks past
// one route per expert).
void ggufMoePlans() {
  MoeShape shape = routedShape;
  shape.weightLayout = WeightLayout::Block32;
  for (uint32_t family : {0U, 9U, 10U, 11U}) {
    ExecutionPlans plans(device(family));
    const MoeGgufTile expected = family == 9 ? MoeGgufTile::Register : MoeGgufTile::Staged;
    require(moeGgufTile(DevicePolicy{family}, shape) == expected, "GGUF expert tile is not gated on GPU family 9");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moeDecode(shape, lanes);
      require(plan.configuration().ggufTile == expected && plan.tileRows() == 8 && plan.splitExperts() &&
                  plan.configuration().ggufRouterTile == FloatTile::Simdgroup,
              "GGUF MoE decode plan left its device tile");
      // Register plans sum the widest input (hidden, 3 K / 4 fp32) per
      // 8-row tile; staged Apple10+ plans reserve the MXFP4 exponent bytes
      // of the packed decode plane in the same slot.
      require(plan.workspace().groupedSumsBytes ==
                  (expected == MoeGgufTile::Register
                       ? uint64_t{plan.maximumTiles()} * 2048 * 3
                       : plan.configuration().mxfp4Native
                             ? uint64_t{plan.maximumTiles()} * 8 * (2048 / 32)
                             : 0),
              "GGUF register plan sums its Table16 tiles");
      covers(plans.moeDecodeWorkspacePerLane(shape), plan.workspace(), lanes, kMoeWorkspaceFields);
      require(plans.moeDecode(routedShape, lanes).configuration().ggufTile == MoeGgufTile::Staged &&
                  plans.moeDecode(routedShape, lanes).workspace().groupedSumsBytes == 0,
              "affine MoE plan took the GGUF register tile");
    }
    // Apple9 stages experts mostly in a format it stages (IQ2_XS: UD-Q2_K_XL)
    // and keeps the register tile for the others (Q4_K: UD-Q4_K_M).
    MoeShape staged = shape, q4k = shape;
    staged.expertFormat = GGUF_FMT_IQ2XS;
    q4k.expertFormat = GGUF_FMT_Q4K;
    require(moeGgufTile(DevicePolicy{family}, staged) == MoeGgufTile::Staged &&
                moeGgufTile(DevicePolicy{family}, q4k) == expected,
            "GGUF expert tile does not follow the experts' format on GPU family 9");
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      const MoePlan plan = plans.moeDecode(staged, lanes);
      require(plan.configuration().ggufTile == MoeGgufTile::Staged &&
                  plan.workspace().groupedSumsBytes ==
                      (plan.configuration().mxfp4Native
                           ? uint64_t{plan.maximumTiles()} * 8 * (2048 / 32)
                           : 0),
              "GGUF MoE plan of staged experts took the register tile");
      covers(plans.moeDecodeWorkspacePerLane(staged), plan.workspace(), lanes, kMoeWorkspaceFields);
    }
    // Prefill: the register tile's 8 rows, or staged 8-row tiles while the
    // routes average at most one row per expert (32 rows of 8 of 256
    // experts). The router's float tile follows Linear::ggufFloatTile (32
    // assumed cores: the neural accelerator from 321 rows, never on Apple9).
    // The register plans' sums slot also holds the packed plane's exponent
    // bytes on Apple10+, as in the decode plans.
    for (uint32_t rows : {1U, 8U, 17U, 32U, 33U, 100U, 256U, 257U, 320U, 321U, 2048U}) {
      const MoePlan plan = plans.moePrefill(shape, rows);
      const uint32_t tileRows = expected == MoeGgufTile::Register || rows <= 32 ? 8 : 32;
      const FloatTile router = family != 9 && rows > 320 ? FloatTile::NeuralAccelerator : FloatTile::Simdgroup;
      const bool sums = expected == MoeGgufTile::Register || plan.configuration().mxfp4Native;
      require(plan.configuration().ggufTile == expected && plan.tileRows() == tileRows && plan.splitExperts() &&
                  (plan.workspace().groupedSumsBytes > 0) == sums &&
                  plan.configuration().ggufRouterTile == router,
              "GGUF MoE prefill plan left the device's tile");
      covers(plans.moePrefillWorkspace(shape, 2048), plan.workspace(), 1, kMoeWorkspaceFields);
    }
    // The prefill bound holds the device's plans and nothing else: on Apple9
    // the register tile's 8-row tiles (20480 grouped rows at 2048 rows), not
    // the staged 32-row tiles it never runs (26624).
    MoeWorkspace devicePlans;
    for (uint32_t rows = 1; rows <= 2048; ++rows) {
      const MoeWorkspace workspace = plans.moePrefill(shape, rows).workspace();
      for (const auto field : kMoeWorkspaceFields) devicePlans.*field = std::max(devicePlans.*field, workspace.*field);
    }
    require(plans.moePrefillWorkspace(shape, 2048) == devicePlans,
            "GGUF MoE prefill bound is not the bound of the device's plans");
  }
  // The register tile reads GGUF 8-row tiles only.
  rejects([&] { (void)MoE::decodePlan(routedShape, 1, {MoeExpertTile::M8, moeRouteWideRows(kAssumedGpuCores),
                                                       MoeExpertSimdgroups::Eight, MoeGgufTile::Register}); });
  rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M32, moeRouteWideRows(kAssumedGpuCores),
                                                 MoeExpertSimdgroups::Eight, MoeGgufTile::Register}); });
  // GGUF kernels exist for 8-row tiles and 32-row prefill tiles only, affine
  // ones for 32-row prefill and 8-row decode tiles.
  rejects([&] { (void)MoE::decodePlan(shape, 1, {MoeExpertTile::M32}); });
  rejects([&] { (void)MoE::decodePlan(routedShape, 1, {MoeExpertTile::M32}); });
  rejects([&] { (void)MoE::prefillPlan(routedShape, 9, {MoeExpertTile::M8}); });
}

// A family the tuning tables have not been re-measured for inherits the
// Apple10 policy and every Apple10-scoped measured row by design: isApple9
// stays exact, families at or above 10 form the Apple10 tier, and a
// measured row scoped {10, 0, ...} carries forward. A row still steers
// only the devices its scope names — its family range and a reported core
// count — which is how a newer family's own row out-scores the rows it
// inherited when it lands. These pins make a change to that carry-forward
// loud rather than silent.
void futureFamilies() {
  for (uint32_t family : {0U, 8U}) {
    const DevicePolicy older{family};
    require(older.tier() == DevicePolicy::Tier::Unknown && !older.isApple9() &&
                !older.apple10Plus() && !older.nativeFormats(),
            "a family below Apple9 took a measured tier");
  }
  const DevicePolicy future{99}, apple10{10};
  require(!future.isApple9() && future.apple10Plus() &&
              future.tier() == DevicePolicy::Tier::Apple10 &&
              future.nativeFormats() == apple10.nativeFormats() &&
              std::string_view{future.nativeFormatSuffix()} ==
                  std::string_view{apple10.nativeFormatSuffix()},
          "a family above the measured ones lost the Apple10 tier");
  require(stagedTiers(future).data() == stagedTiers(apple10).data() &&
              stagedTiers(future).data() != stagedTiers(DevicePolicy{9}).data(),
          "a newer family's GGUF decode left the Apple10 split tiers");

  // A probed newer family reports its cores, and the Apple10-scoped
  // measured rows — the {10, 0, 0} affine and staged-split rows and the
  // {10, 0, 20} core-scoped rows alike — carry forward to it.
  DeviceCapabilities capabilities = device(99);
  capabilities.gpuCoreCount = 20;
  const DevicePolicy measured = DevicePolicy::of(capabilities);
  require(measured.family == 99 && measured.cores == 20 && measured.coresReported,
          "probing a newer family lost its reported core count");
  const LinearWorkload measuredShape{{1280, 5120}, 32, LinearPhase::Decode,
                                     LinearEpilogue::None};
  require(measuredLinearPlan(measured, measuredShape) ==
                  LinearConfig{LinearTile::Split128, 0, LinearSimdgroups::Eight, 4} &&
              measuredGgufSplits(measured, 2048, 2048, 8, false) == 4 &&
              measuredGgufSplits(measured, 17408, 5120, 8, false) == 4 &&
              measuredGgufSplits(measured, 128000, 2048, 16, true) == 2 &&
              stagedDecode(measured, 2048, 2048) ==
                  stagedDecode(DevicePolicy{10, 20, true}, 2048, 2048),
          "an Apple10-scoped measured row stopped carrying forward to a newer family");
  // The same scope that carries a row forward keeps it exact: a core-scoped
  // row does not steer a newer family reporting other cores, and the rows'
  // family floor keeps Apple9 and older out of every Apple10 row.
  const DevicePolicy otherCores{99, 24, true}, apple9{9, 20, true};
  require(measuredGgufSplits(otherCores, 17408, 5120, 8, false) == 0 &&
              measuredGgufSplits(otherCores, 2048, 2048, 8, false) == 4 &&
              measuredGgufSplits(apple9, 2048, 2048, 8, false) == 0 &&
              !measuredLinearPlan(apple9, measuredShape),
          "a measured row steered a device outside its scope");
  // An assumed count is a plan, not a measurement: coresReported == false
  // never matches a core-scoped row, even at the row's own count.
  const DevicePolicy assumed{99, 20, false};
  require(measuredGgufSplits(assumed, 17408, 5120, 8, false) == 0 &&
              measuredGgufSplits(assumed, 2048, 2048, 8, false) == 4,
          "an assumed core count matched a core-scoped measured row");
}

// A device that reports no core count gets the plans of kAssumedGpuCores
// cores, Linear and MoE alike.
void unknownCoreCount() {
  for (uint32_t family : {9U, 10U}) {
    DeviceCapabilities assumed = device(family);
    assumed.gpuCoreCount = kAssumedGpuCores;
    const ExecutionPlans unknown(device(family)), planned(assumed);
    for (auto matrix : matrices)
      for (auto layout : {WeightLayout::Affine64, WeightLayout::Block32})
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
            const LinearWorkload w{matrix, lanes * 8, LinearPhase::Decode, epilogue, layout};
            require(unknown.linear().plan(w).configuration() == planned.linear().plan(w).configuration(),
                    "an unknown core count planned a decode projection for other than the assumed cores");
          }
    MoeShape block = routedShape;
    block.weightLayout = WeightLayout::Block32;
    for (auto shape : {routedShape, block})
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        require(unknown.moeDecode(shape, lanes).configuration() == planned.moeDecode(shape, lanes).configuration(),
                "an unknown core count planned a MoE decode step for other than the assumed cores");
  }
}

void workspaceBounds() {
  const ExecutionPlans plans(device());
  const std::array<uint32_t, 4> histories{31, 32, 2049, std::numeric_limits<uint32_t>::max()};
  rejects([&] { (void)plans.verifyAttention(3, 24, attentionShapes[0].layout, histories); });
  const auto exact =
      plans.verifyAttention(3, 24, attentionShapes[0].layout, std::span(histories).first(3));
  require(exact.laneSplits[3] == 0 && exact.splits == kv::verifyAttentionSplits(2049),
          "verify policy did not resolve one history per lane");
  const auto verify = plans.verifyAttentionWorkspacePerLane(24, attentionShapes[0].layout);
  require(verify.partialsBytes ==
                  uint64_t{RICHENGINE_TREE_VERIFY_NODES} *
                      kv::kVerifyMaximumSplits * 24 * 256 * 4 &&
              verify.statisticsBytes ==
                  uint64_t{RICHENGINE_TREE_VERIFY_NODES} *
                      kv::kVerifyMaximumSplits * 24 * 2 * 4,
          "verify workspace does not cover the maximum split count");
  // 65 tiles of 8 grouped rows per lane at every width.
  const auto moe = plans.moeDecodeWorkspacePerLane(routedShape);
  require(moe.groupedInputBytes == 2129920 && moe.expertOutputBytes == 2129920 &&
              moe.expertIntermediateBytes == 532480 && moe.groupedRoutesBytes == 2080 &&
              moe.tileDescriptorsBytes == 520 && moe.tileCountBytes == 4,
          "MoE decode workspace per lane changed");
  require(plans.gateUpWorkspace(affineGateUp(matrices[0])) == 1114112 &&
              plans.gateUpWorkspace(affineGateUp(matrices[1])) == 393216,
          "gate/up workspace omitted the B3/B4 gate pass");
  const auto draft = plans.draftAttentionWorkspacePerLane(draftShapes[0]);
  // Grouped queries per lane plus eight heads x four splits of 32 x 130 fp32
  // attention partials behind them.
  require(draft.convolutionBytes == 81920 && draft.qkvBytes == 98304 &&
              draft.groupedQueriesBytes == 65536 + 8 * 4 * 16640 &&
              draft.queryKeysBytes == 16384 && draft.queryValuesBytes == 16384,
          "draft workspace ABI changed");
}

void invalidLookupsAndContextEdges() {
  const ExecutionPlans plans(device());
  const auto kvLayout = attentionShapes[0].layout;
  const std::array<uint32_t, 4> histories{0, 1, 2, 3};
  rejects([&] { (void)plans.verifyAttention(0, 24, kvLayout, histories); });
  rejects([&] { (void)plans.verifyAttention(UINT32_MAX, 24, kvLayout, histories); });
  rejects([&] { (void)plans.verifyAttention(3, 24, kvLayout, std::span(histories).first(2)); });
  rejects([&] { (void)plans.verifyAttention(1, 24, {}, std::span(histories).first(1)); });
  rejects([&] { (void)plans.prefillAttention(1, 24, {}); });
  rejects([&] { (void)plans.prefillAttentionWorkspace(0, 24, kvLayout); });
  rejects([&] { (void)plans.prefillAttentionWorkspace(UINT32_MAX, 24, kvLayout); });
  rejects([&] { (void)plans.moeDecode(routedShape, UINT32_MAX); });
  rejects([&] { (void)plans.moeDecode(routedShape, 0); });
  rejects([&] { (void)plans.moePrefillWorkspace(routedShape, 0); });
  rejects([&] { (void)plans.gateUpWorkspace({256, 64}); });
  rejects([&] { (void)plans.draftAttentionWorkspacePerLane({}); });
  std::array<uint32_t, 1> edge{kv::kMaximumPhysicalTokens - 8};
  const auto finalVerify = plans.verifyAttention(1, 24, kvLayout, edge);
  require(finalVerify.splits == kv::kVerifyMaximumSplits,
          "valid final physical verify rows were rejected");
  ++edge[0];
  rejects([&] { (void)plans.verifyAttention(1, 24, kvLayout, edge); });
}
} // namespace

int main() {
  try {
    baselinePlans();
    moeDeviceTiles();
    ggufMoePlans();
    futureFamilies();
    unknownCoreCount();
    workspaceBounds();
    invalidLookupsAndContextEdges();
    std::cout << "PASS execution plans: device policies and family carry-forward, "
                 "device MoE tiles, B1-B4 and prefill workspace bounds (CPU only)\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL execution plans: " << error.what() << '\n';
    return 1;
  }
}
