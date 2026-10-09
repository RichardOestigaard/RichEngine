// Device tuning: every kernel choice fitted on a measured device class, and
// the scope that says which devices each choice names.
#include "ops/DeviceTuning.hpp"

#include "metal/abi/Gguf.h"

#include <algorithm>

namespace richengine::ops {
namespace {

// The device a measured row was fitted for: the GPU families it applies to
// ([minFamily, maxFamily], maxFamily 0 = unbounded, minFamily 0 = every
// family) and the reported core count it was measured at (0 = every count).
// A row carried forward to a newer family loses to a row naming that family
// (specificity below), so unbounded family scope stays the safe default for
// a measurement the new generation has not re-measured yet.
struct MeasuredScope final {
  uint32_t minFamily;
  uint32_t maxFamily;
  uint32_t cores;
  [[nodiscard]] constexpr bool matches(const DevicePolicy &device) const noexcept {
    return device.family >= minFamily && (!maxFamily || device.family <= maxFamily) &&
           (!cores || (device.coresReported && cores == device.cores));
  }
  // Ordering among matching rows: more constrained fields win (a
  // core-scoped row beats an unscoped one), then the higher minFamily — the
  // family's own measurement beats a carried-forward one.
  [[nodiscard]] constexpr unsigned specificity() const noexcept {
    return (minFamily ? 4u : 0u) + (maxFamily ? 2u : 0u) + (cores ? 1u : 0u);
  }
};

// The scopes the tables below use: the Apple10 tuning generation at any
// core count, and that generation's 20-core parts where a measured row
// names an exact core count.
constexpr MeasuredScope kApple10Plus{10, 0, 0};
constexpr MeasuredScope kApple10Plus20Cores{10, 0, 20};

// A measured affine plan: exact {matrix, rows, epilogue} rows so no
// unmeasured workload changes policy.
struct MeasuredLinear final {
  MeasuredScope scope;
  LinearMatrix matrix;
  uint32_t rows;
  LinearEpilogue epilogue;
  LinearConfig config;
};
// A measured GGUF decode split of one tile height over an n x k matrix.
struct MeasuredSplit final {
  MeasuredScope scope;
  uint32_t output;
  uint32_t input;
  uint32_t tileRows;
  uint32_t splits;
};

// The best-matching row of `fits` in `table` for `device`: the most specific
// scope, then the newest family's, then the first in table order.
template <class Row, class Fits>
const Row *measuredRow(std::span<const Row> table, const DevicePolicy &device, Fits &&fits) noexcept {
  const Row *best = nullptr;
  for (const Row &row : table) {
    if (!fits(row) || !row.scope.matches(device)) continue;
    if (!best || row.scope.specificity() > best->scope.specificity() ||
        (row.scope.specificity() == best->scope.specificity() &&
         row.scope.minFamily > best->scope.minFamily))
      best = &row;
  }
  return best;
}

// Measured prefill overrides (tune-kernels, 20-core M5 Pro, 2026-10-05),
// scoped to Apple10 and up like the table's old `family >= 10` gate.
constexpr MeasuredLinear kMeasuredPrefill[] = {
    // LFM2.5 shapes (the 2.6B target and both DSpark drafts).
    {kApple10Plus, {2048, 10240}, 64, LinearEpilogue::None,
     {LinearTile::N128, 0, LinearSimdgroups::Eight, 1}},       // +5.1%
    {kApple10Plus, {2048, 10752}, 64, LinearEpilogue::Residual,
     {LinearTile::N128, 0, LinearSimdgroups::Eight, 1}},       // +5.4%
};

// Measured decode overrides (tune-kernels, 20-core M5 Pro, 2026-10-05):
// these decode shapes beat their policy defaults — mostly deeper split-K
// than the wave-fit rule reaches, +5% to +24% GPU. Scoped to Apple10 and up;
// the measurements have not been re-run per SKU, so no row narrows its scope
// to a core count yet.
constexpr MeasuredLinear kMeasuredDecode[] = {
    // Qwen3.8-27B shapes.
    {kApple10Plus, {1280, 5120}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +18.7%
    {kApple10Plus, {6144, 5120}, 24, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +7.5%
    {kApple10Plus, {6144, 5120}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +10.4%
    {kApple10Plus, {14336, 5120}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +14.0%
    {kApple10Plus, {16640, 5120}, 16, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +6.3%
    {kApple10Plus, {16640, 5120}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +12.8%
    {kApple10Plus, {5120, 17408}, 24, LinearEpilogue::Residual,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +5.1%
    {kApple10Plus, {5120, 17408}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +5.5%
    {kApple10Plus, {5120, 25600}, 24, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +6.1%
    {kApple10Plus, {5120, 25600}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +6.1%
    {kApple10Plus, {17408, 5120}, 32, LinearEpilogue::GateUp,
     {LinearTile::N256, 68, LinearSimdgroups::Eight, 1}},      // +6.9%
    // Ornith-1.5-9B shapes.
    {kApple10Plus, {4096, 12288}, 24, LinearEpilogue::Residual,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +13.5%
    {kApple10Plus, {4096, 12288}, 32, LinearEpilogue::Residual,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +15.7%
    {kApple10Plus, {4096, 32768}, 24, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 8}},   // +23.6%
    {kApple10Plus, {4096, 32768}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 8}},   // +24.0%
    {kApple10Plus, {6144, 4096}, 32, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +10.1%
    {kApple10Plus, {10240, 4096}, 8, LinearEpilogue::None,
     {LinearTile::N128, 80, LinearSimdgroups::Eight, 1}},      // +14.2%
    {kApple10Plus, {10240, 4096}, 32, LinearEpilogue::None,
     {LinearTile::N128, 80, LinearSimdgroups::Eight, 1}},      // +12.5%
    {kApple10Plus, {12288, 4096}, 8, LinearEpilogue::GateUp,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +11.4%
    {kApple10Plus, {12288, 4096}, 16, LinearEpilogue::GateUp,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +7.5%
    {kApple10Plus, {12544, 4096}, 8, LinearEpilogue::None,
     {LinearTile::Paired256, 49, LinearSimdgroups::Four, 1}},  // +8.9%
    {kApple10Plus, {12544, 4096}, 16, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +6.7%
    // The 16-simdgroup N256 tile wins on the narrow-N FFN shapes at 64 rows
    // but wastes registers where the column count already fills the grid;
    // the plain N128 tile measured better on both wide-N drafts heads.
    {kApple10Plus, {12544, 4096}, 64, LinearEpilogue::None,
     {LinearTile::N128, 98, LinearSimdgroups::Eight, 1}},      // +9.6%
    {kApple10Plus, {248320, 4096}, 64, LinearEpilogue::None,
     {LinearTile::N128, 1940, LinearSimdgroups::Eight, 1}},    // +4.6%
    // LFM2.5 shapes (the 2.6B target and both DSpark drafts).
    {kApple10Plus, {6144, 2048}, 32, LinearEpilogue::GateUp,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +9.9%
    {kApple10Plus, {2048, 2048}, 32, LinearEpilogue::Residual,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 2}},   // +13.7%
    {kApple10Plus, {2048, 10240}, 8, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 8}},   // +10.2%
    {kApple10Plus, {2048, 10752}, 24, LinearEpilogue::Residual,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 8}},   // +6.4%
    // LFM2.5 DSpark draft k/v projections (affine): the wave-fit rule
    // over-splits this 512-wide shape at three lanes. MiniCPM5's draft
    // kv is 256-wide and never reaches it.
    {kApple10Plus, {512, 2048}, 24, LinearEpilogue::None,
     {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}},   // +3%
    // MiniCPM5-2B, 20-core M5 Pro: Paired128 fills the machine better than
    // the Paired256 low-lane pick on the 130560-row head (+28% isolated).
    // The {2560,2048}@32 and {2048,10240}@32 rows measured +15%/+8%
    // isolated too but are dropped: both sit on the draft's affine path and
    // their split reassociation collapsed Q4_K_M draft acceptance
    // end-to-end (0.152→0.007) — the same hazard the GGUF staged
    // overrides showed.
    {kApple10Plus, {130560, 2048}, 8, LinearEpilogue::None,
     {LinearTile::Paired128, 640, LinearSimdgroups::Eight, 1}}, // +28%
};

// Measured staged-tile splits, the shapes the tiers under-split: each beats
// the tier's pick by 6-45% on the cores it names (dev/benchmarks/
// gguf_decode_sweep.mm and gguf_projection_benchmark.mm; zero cores means
// every core count). The 2048-wide entries are MiniCPM5-2B's attention out,
// its fused QKV's small KV segments and the LFM drafts' KV on the M5 Pro;
// the PQ2_0 entries are Ternary-Bonsai-2-27B's gate/up, down and fused GDN
// inputs at 8 and 16 rows on the 20-core M5 Pro, where 2.3-bit weights want
// two to four times the splits the size-only tier picks.
constexpr MeasuredSplit kMeasuredStagedSplits[] = {
    {kApple10Plus, 2'048, 2'048, 8, 4},   {kApple10Plus, 2'048, 2'048, 16, 4},
    {kApple10Plus, 2'048, 2'048, 32, 4},
    {kApple10Plus, 512, 2'048, 8, 8},     {kApple10Plus, 512, 2'048, 16, 4},
    {kApple10Plus, 512, 2'048, 32, 4},
    {kApple10Plus, 1'024, 2'048, 8, 8},   {kApple10Plus, 1'024, 2'048, 16, 4},
    {kApple10Plus, 1'024, 2'048, 32, 4},
    {kApple10Plus20Cores, 17'408, 5'120, 8, 4}, {kApple10Plus20Cores, 17'408, 5'120, 16, 4},
    {kApple10Plus20Cores, 5'120, 17'408, 8, 8}, {kApple10Plus20Cores, 5'120, 17'408, 16, 8},
    {kApple10Plus20Cores, 14'336, 5'120, 8, 4}, {kApple10Plus20Cores, 14'336, 5'120, 16, 4},
    {kApple10Plus20Cores, 16'640, 5'120, 8, 4}, {kApple10Plus20Cores, 16'640, 5'120, 16, 4},
    // Granite-4.2-3B and -8B shapes on the 20-core M5 Pro: the fused QKV and
    // gate/up packs, the narrow KV segments at 8 rows, the 8B attention out,
    // gate/up and 32-row KV, and the LFM2.5 16-row down.
    {kApple10Plus20Cores, 3'584, 2'560, 8, 2},  {kApple10Plus20Cores, 3'584, 2'560, 16, 4},
    {kApple10Plus20Cores, 3'584, 2'560, 32, 2}, {kApple10Plus20Cores, 16'384, 2'560, 8, 2},
    {kApple10Plus20Cores, 16'384, 2'560, 16, 2}, {kApple10Plus20Cores, 4'096, 4'096, 16, 4},
    {kApple10Plus20Cores, 512, 2'560, 8, 8},    {kApple10Plus20Cores, 12'800, 4'096, 8, 2},
    {kApple10Plus20Cores, 1'024, 4'096, 32, 4}, {kApple10Plus20Cores, 2'048, 10'752, 16, 8},
    {kApple10Plus20Cores, 25'600, 4'096, 8, 2}, {kApple10Plus20Cores, 25'600, 4'096, 16, 2},
    // Isolated winners rejected on the 20-core M5 Pro for flipping emitted
    // tokens end-to-end (decode-profile --dump-tokens): LFM2.5-2.6B's
    // {6144,2048}@32 (conv in_proj, s1 over s2), {10752,2048} gate/up at
    // 8/16 rows (s2 over the unsplit tier, +15%/+5% isolated) and A1B's
    // {7168,2048}@32 dense up (s1 0.0435 vs s2 0.0565). Same reassociation
    // hazard as the entries below.
};

// Measured splits for the native MXFP4 tiles, same fields and source, kept
// separate because a projection's n x k does not say which format its
// segments hold. The LFM2.5 MXFP4 head at 16 rows splits once on the 20-core
// M5 Pro — the only tile over the 128K row grid where a second partition
// pays (mx_head sweep: 1.175 vs 1.236 ms).
constexpr MeasuredSplit kMeasuredMxfp4Splits[] = {
    {kApple10Plus20Cores, 128'000, 2'048, 16, 2},
    // MiniCPM5-2B measured split changes (staged {6144,2048}@32 and head@8,
    // MXFP4 fused QKV s8->s4) won up to 16% in isolation but shifted draft
    // acceptance 4-5 points end-to-end: any reassociation of these reductions
    // flips near-tie draft proposals. Left at tier defaults.
    // LFM2.5-2.6B MXFP4 shapes on the 20-core M5 Pro, measured through the
    // packed-activation decode (mxfp4p) the single-tensor plans dispatch:
    // the mxfp4 tiers split every one of these too deep (tier picks 8, 8 and
    // 4 respectively). {2048,2048} is the attention and conv out
    // projections; {6144,2048} the conv in_proj; {10752,2048} the gate and
    // up passes; {2048,10752} down. The fused {3072,2048} QKV's s8->s4/2
    // wins (+12%/+29%) stay at the tier like MiniCPM5's above — the same
    // end-to-end draft-acceptance hazard.
    // (The packed decode's split reassociation was checked against emitted
    // tokens on this install — identical streams, unlike the staged Q4_K
    // gate/up and down splits above, which flip them and stay out.)
    {kApple10Plus20Cores, 2'048, 2'048, 8, 4},   {kApple10Plus20Cores, 2'048, 2'048, 16, 4},
    {kApple10Plus20Cores, 2'048, 2'048, 32, 4},
    {kApple10Plus20Cores, 6'144, 2'048, 8, 4},   {kApple10Plus20Cores, 6'144, 2'048, 16, 2},
    {kApple10Plus20Cores, 6'144, 2'048, 32, 1},
    {kApple10Plus20Cores, 10'752, 2'048, 8, 2},  {kApple10Plus20Cores, 10'752, 2'048, 16, 1},
    {kApple10Plus20Cores, 10'752, 2'048, 32, 1},
    {kApple10Plus20Cores, 2'048, 10'752, 16, 4}, {kApple10Plus20Cores, 2'048, 10'752, 32, 4},
};

// Apple9 register tile (128 threads). Four of its threadgroups are resident
// on a core at once: on a 40-core M3 Max its time steps every four per core
// (Q4_K, K = 8192, one lane, ms: 3 per core 0.156, 4 0.157, 5 0.220, 7 0.281,
// 8 0.286; the same steps at two to four lanes and for Q8_0). Below one wave
// a core must fill it, down to one 256-input coefficient unit per partition;
// below eight waves more threadgroups shrink the last wave's tail while
// partitions of 1024 inputs amortize the partial sums (flat from eight to 32
// waves). Over every 27B and 35B projection kind at one to four lanes and
// 10-80 cores emulated by width, the decode step's projections run 0.95%
// slower than the fastest split of each shape on average and 2.3% at worst
// (sixteen threadgroups per core with two units per partition: 2.8%, 7.8%).
constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}};

// Staged tile (64 threads): one fitted tier, six threadgroups per core with
// 512 inputs per partition. Six is not a residency (12-17 of these
// threadgroups run at once per core on the M5 Pro): past it a core's memory
// and neural accelerator are busy and more partitions only add reduction.
// Over the 27B and 35B dense shapes, all formats, one to four lanes, on the
// 16- and 20-core M5 Pro and 10-, 30- and 40-core GPUs emulated by width:
// 3.6% over the fastest split of each shape in total and 36% at worst on a
// 15-us shape (the register tiers in threads per core: 6.6%; the previous 32
// per core with 1024 inputs and unsplit fused and gate/up kernels: 6.4%).
constexpr SplitTier kStagedTiers[] = {{6, 512}};

// The MXFP4 multiplane tile wants more partitions than the staged tile: at
// the staged tiers' two splits it ties or loses; from four to eight it pulls
// ahead (see decodeFormat). Applied to any decode projection that holds an
// MXFP4 segment.
constexpr SplitTier kMxfp4Tiers[] = {{24, 256}};

uint32_t measuredSplits(std::span<const MeasuredSplit> table, const DevicePolicy &device,
                        uint32_t n, uint32_t k, uint32_t tileRows) noexcept {
  const MeasuredSplit *row = measuredRow(table, device, [&](const MeasuredSplit &r) {
    return n == r.output && k == r.input && tileRows == r.tileRows;
  });
  return row ? row->splits : 0;
}

} // namespace

uint32_t decodeSplits(const DevicePolicy &device, uint32_t n, uint32_t k,
                      std::span<const SplitTier> tiers) {
  // K splits of a decode tile, one rule for both tiles. A tier asks for more
  // partitions while the grid holds fewer than `threadgroups` threadgroups
  // per core and each partition would still keep `inputs` inputs; the split
  // count doubles, up to the maximum, while some tier asks. Decode K is a
  // multiple of 256, so eight partitions always hold whole 32-input groups.
  // The rule ignores the batch width: bounds that depended on it did not pay
  // on either family.
  const uint64_t grid = n / GGUF_TILE_COLUMNS;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return grid * splits < uint64_t{t.threadgroups} * device.cores && k / (2 * splits) >= t.inputs;
  };
  while (splits < LinearConfig::kMaximumSplits && std::any_of(tiers.begin(), tiers.end(), asks))
    splits *= 2;
  return splits;
}

// Apple9 cores take as many of the staged tile's threadgroups as of the
// register tile's, and its tiers: on a 40-core M3 Max over the 27B and 35B
// dense shapes at one to four lanes they come within 0-9% of each shape's
// fastest split (20% at one lane on 2048 x 512, a 0.012 ms projection),
// where the fitted tier above is 13-27% slower on 17408 x 5120 and
// 12288 x 5120 and 50% on 2048 x 512.
std::span<const SplitTier> stagedTiers(const DevicePolicy &device) noexcept {
  return device.isApple9() ? std::span<const SplitTier>(kRegisterTiers)
                           : std::span<const SplitTier>(kStagedTiers);
}

std::span<const SplitTier> mxfp4Tiers() noexcept { return kMxfp4Tiers; }

LinearConfig registerDecode(const DevicePolicy &device, uint32_t n, uint32_t k) {
  return {.tile = LinearTile::GgufRegister, .splits = decodeSplits(device, n, k, kRegisterTiers)};
}
LinearConfig stagedDecode(const DevicePolicy &device, uint32_t n, uint32_t k) {
  return {.tile = LinearTile::GgufStaged, .splits = decodeSplits(device, n, k, stagedTiers(device))};
}

std::optional<LinearConfig> measuredLinearPlan(const DevicePolicy &device, LinearWorkload w) noexcept {
  const std::span<const MeasuredLinear> table =
      w.phase == LinearPhase::Prefill ? std::span<const MeasuredLinear>(kMeasuredPrefill)
                                      : std::span<const MeasuredLinear>(kMeasuredDecode);
  const MeasuredLinear *row = measuredRow(table, device, [&](const MeasuredLinear &r) {
    return r.matrix == w.matrix && r.rows == w.rows && r.epilogue == w.epilogue;
  });
  return row ? std::optional(row->config) : std::nullopt;
}

uint32_t measuredGgufSplits(const DevicePolicy &device, uint32_t n, uint32_t k,
                            uint32_t tileRows, bool native) noexcept {
  return measuredSplits(native ? std::span<const MeasuredSplit>(kMeasuredMxfp4Splits)
                                : std::span<const MeasuredSplit>(kMeasuredStagedSplits),
                        device, n, k, tileRows);
}

uint32_t measuredGgufSplitBound(const DevicePolicy &device, uint32_t n, uint32_t k,
                                uint32_t tileRows) noexcept {
  return std::max(measuredSplits(kMeasuredStagedSplits, device, n, k, tileRows),
                  measuredSplits(kMeasuredMxfp4Splits, device, n, k, tileRows));
}

} // namespace richengine::ops
