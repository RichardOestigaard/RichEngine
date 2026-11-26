#pragma once

#include "Tuning.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/MoE.h"
#include "metal/abi/QuantFormat.h"
#include "ops/Linear.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace richengine::ops {

// The shared expert is an expert every row visits, so it has the routed
// experts' intermediate width and runs through the same grouped tiles.
// A shape without one (sharedExpert false) is LFM2-MoE's: sigmoid gating
// with a per-expert selection bias (GGUF expert_gating_func 2), routing
// weights normalized over the top_k winners, and no shared-expert route.
struct MoeShape final {
  uint32_t hiddenSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  // How the weights are stored: affine Q4/Q8 slabs, or the tensors of a GGUF
  // (BlockMoeWeights), which run their own router and expert kernels.
  WeightLayout weightLayout = WeightLayout::Affine64;
  // A GGUF's format of most routed expert weights, which picks the expert
  // tile on Apple9 (moeGgufTile); GGUF_FMT_COUNT for affine weights.
  uint32_t expertFormat = GGUF_FMT_COUNT;
  // Whether every row carries a shared-expert route (id `experts`) weighted
  // by the sigmoid of its scalar gate, as the Qwen MoE does.
  bool sharedExpert = true;
  // NVIDIA Nemotron's ungated experts: the projections are up and down
  // alone (no gate slab, BlockMoeWeights::gate stays empty) and the up pass
  // stores relu(up)^2 instead of silu(gate) * up — the EpReluSquared
  // epilogue of the `_r` kernels. GGUF blocks only.
  bool ungatedExperts = false;
  // A shared expert whose route weight is a constant 1 — no scalar gate
  // (sharedScalarGate stays empty; the block still carries expertBias for
  // the routed select) — run on the dense shared passes
  // (moe_shared_gguf_*, grouping shared == 2), so its intermediate width
  // does not have to match the routed experts'.
  bool sharedExpertUnitWeight = false;
  // The shared expert's intermediate width when it differs from the routed
  // experts' (Nemotron's is twice). Zero reads expertIntermediateSize.
  // Only the unit-weight dense passes honor it.
  uint32_t sharedExpertIntermediateSize = 0;
  // The routed scaling factor the select multiplies the norm_topk_prob
  // weights by (GGUF expert_weights_scale; Nemotron's is 2.5).
  float expertWeightScale = 1.0f;
  // Whether every block's experts are resident and in formats the int8
  // prefill tile takes (TargetModelGeometry::moeInt8Experts): its prefill
  // plans then run the int8 passes and size their scratch
  // (MoeConfig::int8Passes).
  bool int8Experts = false;
  // Nemotron Bolt's latent expert width: the encoder projects the hidden
  // rows down to this width before the grouped experts and projects the
  // op's latent combine back up after it. Zero keeps the experts at
  // hiddenSize. Latent experts are ungated GGUF blocks without an in-op
  // shared expert.
  uint32_t latentSize = 0;

  // The shared expert's intermediate width (expertIntermediateSize unless
  // sharedExpertIntermediateSize overrides it).
  [[nodiscard]] constexpr uint32_t sharedIntermediate() const noexcept {
    return sharedExpertIntermediateSize ? sharedExpertIntermediateSize
                                        : expertIntermediateSize;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    if ((weightLayout != WeightLayout::Affine64 &&
         weightLayout != WeightLayout::Block32) ||
        !hiddenSize || !experts || !expertsPerToken ||
        expertsPerToken > experts || !expertIntermediateSize)
      return false;
    // Ungated experts are a GGUF-block-only feature (the affine slabs'
    // fused gate/up tiles have no relu^2 form), and an ungated block's
    // shared expert is either absent or unit-weight: the scalar-gated
    // shared route needs the gated select and passes. The unit-weight
    // route and the separate shared intermediate only exist on the
    // ungated dense passes.
    if ((ungatedExperts && weightLayout != WeightLayout::Block32) ||
        (ungatedExperts && sharedExpert && !sharedExpertUnitWeight) ||
        (sharedExpertUnitWeight && (!ungatedExperts || !sharedExpert)) ||
        (sharedExpertIntermediateSize &&
         (!sharedExpertUnitWeight || sharedExpertIntermediateSize % 64 != 0)))
      return false;
    // Latent experts are Bolt's ungated GGUF form: the encoder runs the
    // hidden<->latent projections and any shared expert outside the op, so
    // the op's routed slabs both operate at latentSize. The staged tiles'
    // 64-column K tiling sets the alignment.
    if (latentSize &&
        (weightLayout != WeightLayout::Block32 || !ungatedExperts ||
         latentSize % 64 != 0 || latentSize > hiddenSize || sharedExpert))
      return false;
    // The ungated block path bounds its 256-column gather and combine
    // blocks (moe_gather_rows, moe_combine) and stages 64-column tiles, so
    // it needs only the 64 alignment the tile math requires of K; every
    // other shape keeps the affine kernels' 256.
    const bool ungatedBlock = ungatedExperts;
    if (hiddenSize % (ungatedBlock ? 64u : 256u) != 0)
      return false;
    if (experts <= MOE_EXPERT_SLOTS)
      return expertIntermediateSize % (ungatedBlock ? 64u : 256u) == 0;
    // More experts than the narrow routing kernels' slots hold: the wide
    // (_e512) family routes a GGUF block on 512-slot scores and 512-slot
    // select and grouping instances. The staged expert tiles take any
    // 64-column intermediate; the affine slabs' N256 tiles keep the 256
    // expert bound.
    return weightLayout == WeightLayout::Block32 &&
           experts <= MOE_WIDE_EXPERT_SLOTS &&
           expertIntermediateSize % 64 == 0;
  }
  // Whether the block routes on the wide (_e512) kernel instances.
  [[nodiscard]] constexpr bool wideExperts() const noexcept {
    return experts > MOE_EXPERT_SLOTS;
  }
  // The router's score slots per row — the expert slots of the select and
  // grouping kernels that route the block (metal/abi/MoE.h), one thread
  // each.
  [[nodiscard]] constexpr uint32_t routerWidth() const noexcept {
    return wideExperts() ? MOE_WIDE_EXPERT_SLOTS : MOE_EXPERT_SLOTS;
  }
  // Routed experts followed by the shared expert, if there is one.
  [[nodiscard]] constexpr uint32_t routesPerToken() const noexcept {
    return expertsPerToken + (sharedExpert ? 1 : 0);
  }
  bool operator==(const MoeShape &) const = default;
};

// A GGUF expert projection: every routed expert in one segment of experts *
// N rows, expert e's planes from tile e * N / 256 on (moe_gguf_segment in
// kernels/common/moe_expert_slab.h), and the shared expert's segment, whose
// format may differ.
struct BlockExpertProjection final {
  QuantizedSegment routed;
  QuantizedSegment shared;
};

// The sparse MoE block of a GGUF target. The router and the shared expert's
// scalar gate are float tensors llama.cpp keeps unquantized, and they run in
// fp32. A target without a shared expert (sharedExpert false) leaves the
// shared segments empty and carries the F32 per-expert selection bias
// (exp_probs_b) instead. An ungated block (MoeShape::ungatedExperts) leaves
// `gate` empty, and a unit-weight shared expert (sharedExpertUnitWeight)
// leaves sharedScalarGate empty but keeps expertBias, whose segments the
// shared up and down segments size by MoeShape::sharedIntermediate().
struct BlockMoeWeights final {
  QuantizedSegment router;           // [experts][hidden]
  QuantizedSegment sharedScalarGate; // [1][hidden]
  BlockExpertProjection gate;
  BlockExpertProjection up;
  BlockExpertProjection down;
  QuantizedSegment expertBias{};     // [1][experts] F32, sigmoid-gated MoE only
  // Bolt's LatentMoE projections, [latent][hidden] and [hidden][latent];
  // empty everywhere else. The encoder runs them outside the MoE op.
  QuantizedSegment latentDown{};
  QuantizedSegment latentUp{};
};

// The affine weights of a sparse MoE block: Q8 router and shared-expert
// scalar gate, and Q4 expert slabs. The shared expert is a one-expert slab.
// Without one, the shared fields stay empty and expertBias holds the F32
// per-expert selection bias.
struct AffineMoeWeights final {
  Q8Projection router;
  ExpertProjection expertGate;
  ExpertProjection expertUp;
  ExpertProjection expertDown;
  ExpertProjection sharedGate;
  ExpertProjection sharedUp;
  ExpertProjection sharedDown;
  Q8Projection sharedScalarGate;
  metal::MetalBuffer expertBias{}; // experts F32 values, sigmoid-gated MoE only
};

// All weights for one sparse MoE block. The model package owns the buffers;
// this value only exposes semantic projections to the operator.
using MoeWeights = LayoutWeights<AffineMoeWeights, BlockMoeWeights>;

// Gemma 4's MoE weights: a bias-free bf16 router ([experts][hidden],
// expert-major), the F32 per-expert routing scale and Q4 expert slabs whose
// per-expert stride holds the logical width (704) padded to the expert
// tiles' 128-column granularity (768) — the padding reads and writes zeros.
// The dense shared expert is not a routed slot: it runs as ordinary
// projections beside this block.
struct GemmaMoeWeights final {
  metal::MetalBuffer routerWeights;
  metal::MetalBuffer perExpertScale;
  ExpertProjection expertGate;
  ExpertProjection expertUp;
  ExpertProjection expertDown;
};

// Router score tiles: 8 x 32 for short chunks, 32 x 128 for longer chunks.
// The measured Apple10 crossover is about 26 rows per GPU core; an unknown
// core count plans for kAssumedGpuCores (DeviceCapabilities.hpp). Both tiles
// preserve scores.
struct MoeRouteTile final {
  uint32_t rows;
  uint32_t experts;
};
inline constexpr uint32_t kMoeRouteRowsPerCore = 26;

[[nodiscard]] constexpr uint32_t moeRouteWideRows(uint32_t plannedCores) noexcept {
  return plannedCores * kMoeRouteRowsPerCore;
}

[[nodiscard]] constexpr MoeRouteTile
moeRouteTile(uint32_t rows, uint32_t wideRows) noexcept {
  return rows >= wideRows ? MoeRouteTile{32, 128} : MoeRouteTile{8, 32};
}

// Grouped-row scratch. Routes are sorted by expert into tiles of tileRows
// rows; every routed expert may leave one partially filled tile and no tile
// is empty, and the shared expert fills one tile per tileRows rows. A split
// prefill plan also parks the gate projection in expertOutput before the
// down pass overwrites it, so that field spans the wider of the two widths.
[[nodiscard]] constexpr uint32_t moeMaximumTiles(uint32_t rows, MoeShape shape,
                                                 uint32_t tileRows) noexcept {
  const uint32_t routed = rows * shape.expertsPerToken;
  return std::min(routed / tileRows + shape.experts, routed) +
         (shape.sharedExpert ? (rows + tileRows - 1) / tileRows : 0);
}

struct MoeWorkspace final {
  uint64_t selectedExpertsBytes = 0;
  uint64_t routingWeightsBytes = 0;
  uint64_t tileDescriptorsBytes = 0;
  uint64_t tileCountBytes = 0;
  uint64_t groupedRoutesBytes = 0;
  uint64_t routeRowsBytes = 0;
  uint64_t groupedInputBytes = 0;
  uint64_t expertIntermediateBytes = 0;
  uint64_t expertOutputBytes = 0;
  // The row sums of the Table16 tiles the GGUF register expert tile reads.
  // Packed decode plans (the MXFP4 expert tiles) hold the gathered rows'
  // per-(row, group) exponent bytes there instead.
  uint64_t groupedSumsBytes = 0;
  // Packed decode plans only: the slot-permuted fp16 image of the grouped
  // rows (moe_gather_packed) and of the down pass's intermediates
  // (gguf_pack_half), the packed MXFP4 expert tiles' A operand.
  uint64_t groupedPackedBytes = 0;
  // Packed decode plans only: the fp32 split partials and per-tile arrival
  // counters the K-split MXFP4 expert tiles publish and reduce through
  // (kernels/common/split_reduce.h); counters self-reset, so zero init once.
  uint64_t expertPartialsBytes = 0;
  uint64_t expertCountersBytes = 0;
  // The int8 passes' (MoeConfig::int8Passes): the chunk's input rows as
  // int8 codes and their scales per 64 inputs, and the grouped rows' (the
  // input's, then the intermediate's), a row per grouped slot.
  uint64_t inputCodesBytes = 0;
  uint64_t inputScalesBytes = 0;
  uint64_t groupedCodesBytes = 0;
  uint64_t groupedScalesBytes = 0;
  // One tile of zero-padded dense shared-expert input rows, used only by
  // composition-invariant GGUF prefill's ragged row block.
  uint64_t sharedInputBytes = 0;
  bool operator==(const MoeWorkspace &) const = default;
};

// The scratch buffers of one MoE dispatch, sized by a plan's MoeWorkspace.
struct MoeScratch final {
  // rows * routesPerToken() routes: expert ids and fp32 routing weights.
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  // Sized by moeMaximumTiles(): tile descriptors, one tile count, the route
  // at each grouped row, each route's grouped row, and the grouped rows'
  // inputs, intermediates and outputs. The router parks its fp32 scores in
  // groupedInput until the gather claims it.
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  // GGUF register plans: the row sums of the Table16 tiles in groupedInput.
  // Packed decode plans: the gathered rows' and packed intermediates'
  // exponent bytes.
  metal::MetalBuffer groupedSums;
  // Packed decode plans: the slot-permuted fp16 activation plane the MXFP4
  // expert segments read (kernels/common/gguf_mxfp4p_tile.h's layout).
  metal::MetalBuffer groupedPacked;
  // Packed decode plans: the K split's fp32 partial sums, tiled per expert
  // tile, and the splits' arrival counters of split_reduce.h.
  metal::MetalBuffer expertPartials;
  metal::MetalBuffer expertCounters;
  // The int8 passes' (MoeConfig::int8Passes): the chunk's input rows as
  // int8 codes and their scales per 64 inputs, and the grouped rows' codes
  // and scales — the input's first, then the intermediate's for the down
  // pass. Empty for plans without them.
  metal::MetalBuffer inputCodes;
  metal::MetalBuffer inputScales;
  metal::MetalBuffer groupedCodes;
  metal::MetalBuffer groupedScales;
  // One tile of padded dense shared-expert input rows (moe_pad_rows).
  metal::MetalBuffer sharedInput;
};

// Each scratch buffer with the workspace field that sizes it, in field order.
struct MoeScratchField final {
  metal::MetalBuffer MoeScratch::*buffer;
  uint64_t MoeWorkspace::*bytes;
};
inline constexpr std::array<MoeScratchField, 18> kMoeScratchFields{{
    {&MoeScratch::selectedExperts, &MoeWorkspace::selectedExpertsBytes},
    {&MoeScratch::routingWeights, &MoeWorkspace::routingWeightsBytes},
    {&MoeScratch::tileDescriptors, &MoeWorkspace::tileDescriptorsBytes},
    {&MoeScratch::tileCount, &MoeWorkspace::tileCountBytes},
    {&MoeScratch::groupedRoutes, &MoeWorkspace::groupedRoutesBytes},
    {&MoeScratch::routeRows, &MoeWorkspace::routeRowsBytes},
    {&MoeScratch::groupedInput, &MoeWorkspace::groupedInputBytes},
    {&MoeScratch::expertIntermediate, &MoeWorkspace::expertIntermediateBytes},
    {&MoeScratch::expertOutput, &MoeWorkspace::expertOutputBytes},
    {&MoeScratch::groupedSums, &MoeWorkspace::groupedSumsBytes},
    {&MoeScratch::groupedPacked, &MoeWorkspace::groupedPackedBytes},
    {&MoeScratch::expertPartials, &MoeWorkspace::expertPartialsBytes},
    {&MoeScratch::expertCounters, &MoeWorkspace::expertCountersBytes},
    {&MoeScratch::inputCodes, &MoeWorkspace::inputCodesBytes},
    {&MoeScratch::inputScales, &MoeWorkspace::inputScalesBytes},
    {&MoeScratch::groupedCodes, &MoeWorkspace::groupedCodesBytes},
    {&MoeScratch::groupedScales, &MoeWorkspace::groupedScalesBytes},
    {&MoeScratch::sharedInput, &MoeWorkspace::sharedInputBytes},
}};
static_assert(sizeof(MoeWorkspace) == kMoeScratchFields.size() * sizeof(uint64_t) &&
              sizeof(MoeScratch) == kMoeScratchFields.size() * sizeof(metal::MetalBuffer),
              "every scratch buffer is in kMoeScratchFields");
// The workspace fields of kMoeScratchFields, in its order.
inline constexpr auto kMoeWorkspaceFields = [] {
  std::array<uint64_t MoeWorkspace::*, kMoeScratchFields.size()> fields{};
  for (size_t i = 0; i < fields.size(); ++i) fields[i] = kMoeScratchFields[i].bytes;
  return fields;
}();

// The tile applies to grouping, gather and both expert projections together;
// changing it never changes the physical rows in a command. Affine plans
// consume Q4 expert slabs in StorageN=256 order: decode plans run the fused
// gate/up tile on M8 tiles, prefill plans the experts on M32 tiles as three
// N256 passes (gate, up with the silu gate, down) whose tiles shrink to the
// descriptor's live rows; prefill plans at <= 256 rows use M16 tiles: the
// union leaves experts well under 32 rows each, and the extra tiles' weight
// reads are repaid by the more parallel grid (~314 GB/s vs ~148 GB/s on the
// DiffusionGemma up pass at its 256-row canvas shape). GGUF plans run three passes of M8 tiles, or of M32
// tiles to prefill (moeGgufPrefillTile).
enum class MoeExpertTile : uint8_t { M8 = 8, M16 = 16, M32 = 32 };

// Simdgroups per affine 8-row expert tile, a device policy the execution
// plans set. Eight is the shipped N128 tile for both projections.
// Four halves the threadgroup to 128 threads and runs gate/up at N128 and
// down at N256, for Apple9 decode plans: family 9 has no per-core matrix
// unit, and a decode expert grid leaves it latency-bound at low occupancy.
// Measured on a 40-core Apple9 GPU at the 35B shape (H=2048, E=256,
// top_k=8, I=512), ms per layer at rows 8/16/24/32: gate/up
// 0.332/0.551/0.728/0.859 -> 0.314/0.503/0.618/0.699 (1.06x-1.23x), down
// 0.157/0.274/0.363/0.419 -> 0.137/0.226/0.297/0.335 (1.15x-1.25x). Apple10
// variants had mixed results across shapes, so family 10 keeps eight.
// Smaller Apple9 core counts still need performance validation; this family
// gate does not establish their optimum. Either choice writes bit-identical
// outputs and needs the same workspace; only the down pass's column grid
// changes.
enum class MoeExpertSimdgroups : uint8_t { Eight = 8, Four = 4 };

// Families below 9 are rejected at startup; 10 and later keep the shipped
// tile, as does an unknown family.
[[nodiscard]] constexpr MoeExpertSimdgroups
moeDecodeSimdgroups(const DevicePolicy &device) noexcept {
  return device.isApple9() ? MoeExpertSimdgroups::Four
                           : MoeExpertSimdgroups::Eight;
}

// The expert tile of GGUF plans, which run three grouped passes (gate, up
// with silu(gate), down) over the GGUF image: the half-staged tiles of
// kernels/shared/moe_gguf.metal, or Register, the exact register tile of
// kernels/decode/linear_gguf_sgmatrix.metal over Table16 tiles of the
// grouped rows (8-row tiles only). Apple9 runs Register in both phases but
// for experts mostly in a format it stages. In decode, as its dense GGUF
// projections do (LinearGguf.cpp): its matrix operations share the FP32 pipe,
// where the register tile beats staging. In prefill it equals the decode
// numerics; against the staged 32-row tiles, on the 35B's real routes on the
// 40-core M3 Max (ms per layer), it is faster at 512 rows (3.52 vs 3.72) and
// slower at 2048 (12.6-13.1 vs 11.0-11.7).
enum class MoeGgufTile : uint8_t { Staged, Register };

// Apple9 stages experts mostly in a format apple9StagesFormat names: one
// 35B-shaped layer on a 40-core M3 Max decodes UD-Q2_K_XL's IQ2_XS and
// IQ3_XXS experts, and the IQ2, IQ3_XXS and IQ1 formats alone, 4-21% faster
// staged at B1-B4, where Q4_K/Q5_K, Q2_K and IQ4_XS experts take 3-34% longer.
[[nodiscard]] inline MoeGgufTile moeGgufTile(const DevicePolicy &device, MoeShape shape) noexcept {
  // Ungated experts have no register tile (the Table16 gathers' 256-column
  // blocks do not bound their ragged last block); staged it is.
  if (shape.ungatedExperts)
    return MoeGgufTile::Staged;
  // RICHENGINE_MOE_GGUF_TILE=register|staged forces the tile for a family
  // the policy has not measured (Apple10's staged K-quants).
  const std::string_view env = richengine::tuning().moeGgufTile;
  if (env == "staged")
    return MoeGgufTile::Staged;
  if (env == "register" &&
      !apple9StagesFormat(shape.expertFormat))
    return MoeGgufTile::Register;
  return device.isApple9() && !apple9StagesFormat(shape.expertFormat) ? MoeGgufTile::Register
                                                                      : MoeGgufTile::Staged;
}

// The rows of a GGUF prefill plan's tiles on the device's `tile`: 8 on the
// register tile. Staged: 8-row tiles while the chunk's routes average at most
// one row per expert (rows * topK <= experts), 32-row tiles beyond, which
// stream an expert's weights once for up to 32 of its rows (its tiles run 8-,
// 16- or 32-row matmuls by their live rows, moe_live_rows). On the 35B's
// real prefill routes (wikitext, 16-core M5 Pro, the three expert passes of a
// layer, ms, with 16-row matmuls for up to 16 live rows) 8- vs 32-row tiles:
// 32 rows 0.72 / 0.71, 64 rows 1.03 / 0.97, 128 rows 1.59 / 1.29, 256 rows
// 2.60 / 1.71.
[[nodiscard]] constexpr MoeExpertTile moeGgufPrefillTile(MoeShape shape, uint32_t rows,
                                                         MoeGgufTile tile) noexcept {
  return tile == MoeGgufTile::Register ||
                 uint64_t{rows} * shape.expertsPerToken <= shape.experts
             ? MoeExpertTile::M8
             : MoeExpertTile::M32;
}

enum class MoePhase : uint8_t { Prefill, Decode,
  // The DiffusionGemma trunk's grouped-expert sweep over a canvas's rows: a
  // prefill-shaped pass, but its own plan family so canvas tuning (the M16
  // tile) never changes a normal chunk's plan.
  Canvas };

// The device's policy for a MoE plan, which the execution plans derive.
struct MoeConfig final {
  // M32 for affine prefill plans, M8 for decode plans; a GGUF prefill plan
  // takes moeGgufPrefillTile's.
  MoeExpertTile expertTile = MoeExpertTile::M32;
  // Rows from which the router uses the 32-row scores tile; the execution
  // plans derive it from the GPU core count.
  uint32_t routeWideRows = moeRouteWideRows(kAssumedGpuCores);
  // Simdgroups of the affine 8-row expert tiles, which decode plans run; the
  // execution plans derive it from the GPU family for decode plans and keep
  // eight for prefill plans.
  MoeExpertSimdgroups m8Simdgroups = MoeExpertSimdgroups::Eight;
  // GGUF plans only; the execution plans derive it from the GPU family.
  MoeGgufTile ggufTile = MoeGgufTile::Staged;
  // The tile of a GGUF plan's F32 router; the execution plans derive it from
  // the device and the plan's rows (Linear::ggufFloatTile).
  FloatTile ggufRouterTile = FloatTile::Simdgroup;
  // GGUF plans only: the `_n` expert kernels, whose MXFP4 segments decode on
  // the Metal 4.1 packed FP4-E2M1 path. Set on Apple GPU family 10 and up.
  bool mxfp4Native = false;
  // Decode plans only: the union cap moe_route_cap applies (serve-native's
  // --moe-union); 0 keeps every selected route.
  uint32_t unionCap = 0;
  // Decode plans only: route through the 16x64 score tile instead of the
  // 8x32 tile when the 32-row tile would not already be selected.
  bool routeM16 = false;
  // A resident block's prefill passes on int8 activations where its expert
  // projections are formats the int8 prefill tile takes (MoE::add's
  // addInt8Experts, kernels/shared/moe_int8.metal).
  bool int8Passes = false;
  // Live-tile indirect dispatch of the staged GGUF decode expert passes
  // (MoE::add): the plan derives it — RICHENGINE_MOE_INDIRECT_DISPATCH
  // forces it, the execution plans' auto mode arms it on measured
  // device/model/checkpoint combinations. Callers plan through
  // ExecutionPlans or set it directly.
  bool indirectDispatch = false;
  bool operator==(const MoeConfig &) const = default;
};

// Constructed by the operator so workspace sizing and encoding use the same plan.
class MoePlan final {
public:
  [[nodiscard]] MoeShape shape() const noexcept { return shape_; }
  [[nodiscard]] uint32_t rows() const noexcept { return rows_; }
  [[nodiscard]] MoeConfig configuration() const noexcept { return config_; }
  [[nodiscard]] uint32_t tileRows() const noexcept {
    return static_cast<uint32_t>(config_.expertTile);
  }
  [[nodiscard]] MoePhase phase() const noexcept { return phase_; }
  // A decode plan's packed verify row stride (a trained-block draft's verify
  // depth, at most RICHENGINE_TARGET_VERIFY_ROWS); zero for prefill and
  // canvas plans.
  [[nodiscard]] uint32_t laneRows() const noexcept { return laneRows_; }
  [[nodiscard]] bool splitExperts() const noexcept { return splitExperts_; }
  [[nodiscard]] uint32_t maximumTiles() const noexcept { return maximumTiles_; }
  [[nodiscard]] const MoeWorkspace &workspace() const noexcept {
    return workspace_;
  }

private:
  friend struct MoE;
  MoePlan(MoeShape shape, uint32_t rows, MoeConfig config, MoePhase phase);

  MoeShape shape_;
  uint32_t rows_;
  MoeConfig config_;
  MoePhase phase_;
  uint32_t laneRows_ = 0;
  bool splitExperts_;
  uint32_t maximumTiles_;
  MoeWorkspace workspace_;
};

struct MoeBuffers final {
  // Rows of hiddenSize bf16 values. The router always reads these rows.
  metal::MetalBuffer input;
  // Rows of hiddenSize bf16 normally. On a latent shape they are rows of
  // latentSize zeros: the encoder projects the op's latent result back to
  // hidden width and applies the true residual and shared expert outside.
  metal::MetalBuffer residual;
  // Rows of hiddenSize bf16 normally, rows of latentSize bf16 on a latent
  // shape (the routed experts' weighted sum, including residual above).
  metal::MetalBuffer output;
  MoeScratch scratch;
  // Gemma 4 routes from the post-attention residual, not the FFN input: the
  // rows the router reads, when different from `input`. Empty binds `input`.
  metal::MetalBuffer routerInput;
  // A verify step's lanes' verify rows, a uint32 per lane
  // (ops::DraftSelector::addVerifyRows): its grouping routes each lane's
  // verify rows alone (moe_group_routes_verified). Empty, every row routes.
  metal::MetalBuffer verifyRows;
  // Latent-MoE only (MoeShape::latentSize): rows of latentSize bf16 values,
  // the encoder's latent-down projection of `input`. The routed-expert
  // gather reads these rows instead of `input`; empty at latentSize 0. The
  // trailing position keeps existing aggregate initializers intact.
  metal::MetalBuffer expertInput{};
};

// The streamed routed experts' device state a streamed GGUF expert pass
// binds (model::ExpertArena's published tables): slotTable holds each
// routed expert's global arena slot (~0u while it holds none — the pass
// skips that expert's tiles), chunkAddresses the chunks' nine region GPU
// addresses each (gate, up, down plane0/plane1/meta), and the chunk
// geometry the kernels map a slot through.
struct MoeStreamBinding final {
  metal::MetalBuffer slotTable;
  metal::MetalBuffer chunkAddresses;
  uint32_t floorSlots = 0;
  uint32_t chunkSlots = 0;
};

// Routes and executes grouped experts from immutable weight views.
struct MoE final {
  [[nodiscard]] static MoePlan prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config);
  // The DiffusionGemma trunk's expert sweep: prefill-shaped dispatches on a
  // canvas plan family.
  [[nodiscard]] static MoePlan canvasPlan(MoeShape shape, uint32_t rows, MoeConfig config);
  // verifyRows is the lanes' packed verify row stride: a trained-block
  // draft's verify depth, at most RICHENGINE_TARGET_VERIFY_ROWS.
  [[nodiscard]] static MoePlan decodePlan(MoeShape shape, uint32_t lanes,
                                          uint32_t verifyRows,
                                          MoeConfig config);
  static void add(metal::CommandGraph &graph, const MoeBuffers &buffers,
                  const MoeWeights &weights, const MoePlan &plan);
  // The route-first split of add() for streamed routed experts (a GGUF
  // block, Block32 weights): addRouting encodes the router, select, cap
  // and grouping dispatches; the host then learns the step's expert union,
  // stages it into the arena and calls addGgufGather once and
  // addGgufPasses once per staged batch — each pass skips the tiles of
  // experts the binding's slot table leaves empty — and addCombine once
  // after the last batch. The split methods encode the same dispatches
  // add() would for a block plan; they throw for affine weights and for a
  // streamed binding on a plan that is not the staged GGUF tile. add()
  // itself is unchanged and remains the path of every other caller.
  static void addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                         const MoeWeights &weights, const MoePlan &plan);
  static void addGgufGather(metal::CommandGraph &graph, const MoeBuffers &buffers,
                            const MoeWeights &weights, const MoePlan &plan);
  static void addGgufPasses(metal::CommandGraph &graph, const MoeBuffers &buffers,
                            const MoeWeights &weights, const MoePlan &plan,
                            const MoeStreamBinding *streamed = nullptr);
  static void addCombine(metal::CommandGraph &graph, const MoeBuffers &buffers,
                         const MoeWeights &weights, const MoePlan &plan);
  // Gemma 4's block: moe_route_scores_gemma over `routerInput` (or `input`),
  // moe_route_select_gemma's softmax-top_k-renormalized routes scaled by
  // perExpertScale, the grouped GeGLU expert tiles and moe_combine into
  // `output` on `residual` (the caller's zero residual keeps the routed sum
  // unmixed). The plan's shape carries the padded expert width.
  static void addGemma(metal::CommandGraph &graph, const MoeBuffers &buffers,
                       const GemmaMoeWeights &weights, const MoePlan &plan);
  // RICHENGINE_MOE_STATS: prints "moe-stats rows=N union=E tiles=T" for
  // every dispatch record written since the last call. The tile-count
  // scratch is Shared storage; call only after the command completed — a
  // readback mid-command stalls. A no-op unless the flag is "1".
  static void logStats(const metal::MetalBuffer &tileCount);
};

} // namespace richengine::ops
