#include "ops/MoE.hpp"

#include "Tuning.hpp"
#include "ops/KernelNames.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace richengine::ops {
namespace {

static_assert(offsetof(MoeExpertParams, expert_stride_bytes_0) == 24);

// A kill switch for the packed expert paths, read once (the decode plan
// calls this per step).
bool packedDisabled() {
  return tuning().moePackedOff;
}

// RICHENGINE_MOE_STATS=1 logs each MoE dispatch's routed-expert union: the
// grouping kernel appends a MoeStatsRecord to a ring in the tile-count
// scratch, which logStats prints after the command completes.
bool statsEnabled() {
  return tuning().moeStats;
}

std::atomic<uint32_t> &statsSequence() {
  static std::atomic<uint32_t> sequence{0};
  return sequence;
}

// RICHENGINE_MOE_UNION caps a decode dispatch's routed-expert union at the
// given count (0 disables). Prefill is never capped: pruning routes there
// would corrupt the KV a long prompt shares with decode.
uint32_t unionCap() {
  return tuning().moeUnionCap;
}

bool matches(const Q8Projection &projection, uint32_t output,
             uint32_t input) noexcept {
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t parameterBytes = elements / 32;
  const AffineWeights &planes = projection.planes;
  return planes.weights && planes.scales && planes.biases &&
         projection.outputSize == output && projection.inputSize == input &&
         planes.weights.sizeBytes() >= elements &&
         planes.scales.sizeBytes() >= parameterBytes &&
         planes.biases.sizeBytes() >= parameterBytes;
}

bool matches(const ExpertProjection &projection, uint32_t experts,
             uint32_t output, uint32_t input) noexcept {
  if (!projection.packed || projection.experts != experts || !experts ||
      projection.outputSize != output || projection.inputSize != input)
    return false;
  const uint64_t elements = uint64_t{output} * input;
  const uint64_t payloadBytes = elements / 2 + elements / 16;
  const uint64_t stride = projection.expertStrideBytes;
  const uint64_t available = projection.packed.sizeBytes();
  if (!stride || stride < payloadBytes || stride % sizeof(uint16_t) ||
      available < payloadBytes)
    return false;
  // The shader reads [weights][BF16 scales][BF16 biases] at each stride.
  // Allow padding between experts, without requiring it after the last one.
  // Division proves the last payload fits without overflowing expert*stride.
  return uint64_t{experts - 1} <= (available - payloadBytes) / stride;
}

// A float segment holds [output][input] floats in plane0; a quantized one
// its planes in a GGUF_FMT_* format.
bool matches(const QuantizedSegment &segment, uint32_t output, uint32_t input,
             bool floatWeights) noexcept {
  if (!segment.plane0 || segment.isFloat() != floatWeights ||
      segment.outputSize != output || segment.inputSize != input)
    return false;
  return floatWeights
             ? segment.plane0.sizeBytes() >= uint64_t{output} * input * sizeof(float)
             : segment.formatId < GGUF_FMT_COUNT && segment.meta;
}

bool matches(const BlockExpertProjection &projection, uint32_t experts,
             uint32_t output, uint32_t input, bool sharedExpert) noexcept {
  return matches(projection.routed, experts * output, input, false) &&
         (!sharedExpert || matches(projection.shared, output, input, false));
}

// The sigmoid-gated MoE's per-expert selection bias: F32 values, as a float
// segment (a GGUF) or a raw buffer (affine files).
bool matches(const QuantizedSegment &bias, uint32_t experts) noexcept {
  return bias.isFloat() && bias.plane0 &&
         bias.plane0.sizeBytes() >= uint64_t{experts} * sizeof(float);
}
bool matches(const metal::MetalBuffer &bias, uint32_t experts) noexcept {
  return bias && bias.sizeBytes() >= uint64_t{experts} * sizeof(float);
}

void validate(const MoeWeights &weights, MoeShape shape) {
  if (shape.weightLayout != weights.layout())
    throw std::invalid_argument("MoE weight layout does not match plan");
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  const bool shared = shape.sharedExpert;
  if (shape.weightLayout == WeightLayout::Block32) {
    const BlockMoeWeights &blocks = weights.blocks();
    if (!shape.valid() ||
        !matches(blocks.router, shape.experts, hidden, true) ||
        (shared && !matches(blocks.sharedScalarGate, 1, hidden, true)) ||
        (!shared && !matches(blocks.expertBias, shape.experts)) ||
        !matches(blocks.gate, shape.experts, intermediate, hidden, shared) ||
        !matches(blocks.up, shape.experts, intermediate, hidden, shared) ||
        !matches(blocks.down, shape.experts, hidden, intermediate, shared))
      throw std::invalid_argument("block MoE weights do not match execution shape");
    return;
  }
  const AffineMoeWeights &affine = weights.affine();
  if (!shape.valid() || !matches(affine.router, 256, hidden) ||
      (shared && !matches(affine.sharedScalarGate, 256, hidden)) ||
      (!shared && !matches(affine.expertBias, shape.experts)) ||
      !matches(affine.expertGate, shape.experts, intermediate, hidden) ||
      !matches(affine.expertUp, shape.experts, intermediate, hidden) ||
      !matches(affine.expertDown, shape.experts, hidden, intermediate) ||
      (shared && (!matches(affine.sharedGate, 1, intermediate, hidden) ||
                  !matches(affine.sharedUp, 1, intermediate, hidden) ||
                  !matches(affine.sharedDown, 1, hidden, intermediate)))) {
    throw std::invalid_argument("MoE weights do not match execution shape");
  }
}

// K partitions of a packed MXFP4 expert pass at decode (moe_gguf.metal's
// `_p` tiles). Measured on the LFM2.5-8B-A1B shape (gguf-moe-benchmark,
// 20-core M5 Pro): splits cost — the partials write and arrive-last reduce
// outweigh the shorter serial K loop, +13% pass time at 4 partitions and
// +2% at 2 — because the expert passes already run near DRAM bandwidth
// (~200 GB/s of ~5.8 MB per expert per pass). 1 keeps them unsplit.
constexpr uint32_t kMoeExpertSplits = 1;

MoeWorkspace workspaceFor(MoeShape shape, uint32_t rows, uint32_t tileRows,
                          bool splitExperts, MoeGgufTile ggufTile,
                          bool packedPlane) {
  if (!shape.valid())
    throw std::invalid_argument("invalid MoE workspace shape");
  const uint64_t routes = uint64_t{rows} * shape.routesPerToken();
  const uint32_t tiles = moeMaximumTiles(rows, shape, tileRows);
  const uint64_t groupedRows = uint64_t{tiles} * tileRows;
  const uint32_t widest = std::max(shape.hiddenSize, shape.expertIntermediateSize);
  const uint32_t outputWidth = splitExperts ? widest : shape.hiddenSize;
  // The router's rows x 256 fp32 scores live in the grouped input until the
  // gather overwrites them. Register plans also hold the down pass's Table16
  // tiles there.
  // The affine scores tiles park the shared gate's logit in a 257th column.
  const uint64_t scoreBytes = uint64_t{rows} * (256 + 1) * sizeof(float);
  const bool table16 = ggufTile == MoeGgufTile::Register;
  // A packed decode plan reserves the slot-permuted fp16 plane of every
  // grouped row — the packed MXFP4 expert tiles' A operand — and the
  // per-(row, group) exponent bytes in the register plans' sums slot; one
  // plane serves gate/up (the gather's output) and down (gguf_pack_half's
  // of the intermediates), each indexed by its own row stride.
  const uint64_t packedBytes = packedPlane ? groupedRows * widest * sizeof(uint16_t) : 0;
  const uint64_t exponentBytes = packedPlane ? groupedRows * (widest / 32) : 0;
  const uint64_t sumsBytes =
      std::max(table16 ? tableSumsBytes(LinearInput::Table16, widest, groupedRows) : 0, exponentBytes);
  // K-split packed expert tiles, decode plans only (kMoeExpertSplits; the
  // prefill tiles' grid already saturates): fp32 partials per tile and
  // split, one arrival counter per 64-column segment of a tile.
  const uint64_t partialBytes = packedPlane && tileRows == 8 && kMoeExpertSplits > 1
      ? uint64_t{tiles} * kMoeExpertSplits * tileRows * widest * sizeof(float)
      : 0;
  const uint64_t counterBytes = packedPlane
      ? uint64_t{tiles} * (widest / GGUF_TILE_COLUMNS) * sizeof(uint32_t)
      : 0;
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor),
          statsEnabled() ? kMoeStatsLogOffset +
                               uint64_t{kMoeStatsLogSlots} * sizeof(MoeStatsRecord)
                         : sizeof(uint32_t),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * outputWidth * sizeof(uint16_t), sumsBytes, packedBytes,
          partialBytes, counterBytes};
}

// Pipelines, column tiles and threadgroup width of a decode plan's fused
// gate/up and down passes; see MoeExpertSimdgroups for the four-simdgroup
// form's geometry and measurements.
struct ExpertPasses final {
  const char *gateUp;
  const char *down;
  uint32_t gateUpColumns;
  uint32_t downColumns;
  uint32_t threads;
};

ExpertPasses fusedExpertPasses(const MoeConfig &config) noexcept {
  const uint32_t threads = static_cast<uint32_t>(config.m8Simdgroups) * 32;
  if (config.m8Simdgroups == MoeExpertSimdgroups::Four)
    return {kMoeExpertGateUpQ4M8N128Sg4.data(),
            kMoeExpertDownQ4M8N256Sg4.data(), 128, 256, threads};
  return {kMoeExpertGateUpQ4M8.data(), kMoeExpertDownQ4M8.data(), 128, 128,
          threads};
}

void addAffineExperts(metal::CommandGraph &graph, const MoeBuffers &buffers,
                      const MoeScratch &scratch, const AffineMoeWeights &weights,
                      const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  // A shared-expert-free block's kernels never see the shared expert's id;
  // its unused slab slots bind the routed ones.
  const metal::MetalBuffer &sharedGate =
      shape.sharedExpert ? weights.sharedGate.packed : weights.expertGate.packed;
  const metal::MetalBuffer &sharedUp =
      shape.sharedExpert ? weights.sharedUp.packed : weights.expertUp.packed;
  const metal::MetalBuffer &sharedDown =
      shape.sharedExpert ? weights.sharedDown.packed : weights.expertDown.packed;
  // The two expert strides are the gate and up slabs of the fused tile; a
  // single-matrix pass reads only the first, so its params repeat one stride.
  // routes_per_row rides in reserved0 so the indirect tiles can map a
  // grouped row's route back to its input row.
  const uint32_t routesPerRow = shape.routesPerToken();
  // RICHENGINE_MOE_TILE_MAJOR (decode only): the fused expert grids swap
  // their axes so threadgroups sweep a tile's column blocks for every
  // grouped tile at one column — same-expert row-tiles read identical
  // weight slices back-to-back. The kernels remap group through
  // params.tile_major.
  const uint32_t expertTileMajor =
      tuning().moeTileMajor && plan.phase() == MoePhase::Decode ? 1u : 0u;
  const MoeExpertParams gateUp{shape.hiddenSize, shape.expertIntermediateSize,
                               shape.experts, routesPerRow, expertTileMajor,
                               weights.expertGate.expertStrideBytes,
                               weights.expertUp.expertStrideBytes};
  const MoeExpertParams gate{shape.hiddenSize, shape.expertIntermediateSize,
                             shape.experts, routesPerRow, 0,
                             weights.expertGate.expertStrideBytes,
                             weights.expertGate.expertStrideBytes};
  const MoeExpertParams up{shape.hiddenSize, shape.expertIntermediateSize,
                           shape.experts, routesPerRow, 0,
                           weights.expertUp.expertStrideBytes,
                           weights.expertUp.expertStrideBytes};
  const MoeExpertParams down{shape.expertIntermediateSize, shape.hiddenSize,
                             shape.experts, routesPerRow, expertTileMajor,
                             weights.expertDown.expertStrideBytes,
                             weights.expertDown.expertStrideBytes};
  if (plan.splitExperts()) {
    // The gate and up passes read each grouped row's input through
    // grouped_routes — the gather does not run for affine plans. The gate
    // lands in expertOutput, which the down pass overwrites only after the
    // up pass has consumed it.
    const bool m16 = plan.tileRows() == 16;
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256IndirectM16
                              : kPrefillMoeExpertQ4N256IndirectM32),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               sharedGate, scratch.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256UpSiluIndirectM16
                              : kPrefillMoeExpertQ4N256UpSiluIndirectM32),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, weights.expertUp.packed,
               sharedUp, scratch.expertOutput,
               scratch.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256M16
                              : kPrefillMoeExpertQ4N256M32),
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               sharedDown, scratch.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    // The workspace holds the same grouped rows whatever the column tile;
    // only the grid's column count and the threadgroup width follow it.
    const ExpertPasses passes = fusedExpertPasses(plan.configuration());
    const auto expertGrid = [&](uint32_t columns) {
      return expertTileMajor ? metal::DispatchSize{tiles, columns, 1}
                             : metal::DispatchSize{columns, tiles, 1};
    };
    // RICHENGINE_MOE_INDIRECT (decode, 8-row tiles, hidden % 256 == 0): the
    // gate/up pass resolves each grouped row through grouped_routes in its
    // staging block — the gather's groupedInput write never happens.
    const bool indirect =
        tuning().moeIndirectGather && plan.tileRows() == 8 &&
        shape.hiddenSize % 256 == 0;
    if (indirect) {
      graph.add(std::string(plan.configuration().m8Simdgroups ==
                                    MoeExpertSimdgroups::Four
                                ? kMoeExpertGateUpQ4M8N128Sg4Indirect
                                : kMoeExpertGateUpQ4M8Indirect),
                {buffers.input, scratch.groupedRoutes,
                 scratch.tileDescriptors, scratch.tileCount,
                 weights.expertGate.packed, weights.expertUp.packed,
                 sharedGate, sharedUp, scratch.expertIntermediate},
                gateUp,
                expertGrid(shape.expertIntermediateSize / passes.gateUpColumns),
                {passes.threads, 1, 1});
    } else {
      graph.add(passes.gateUp,
                {scratch.groupedInput, scratch.tileDescriptors,
                 scratch.tileCount, weights.expertGate.packed,
                 weights.expertUp.packed, sharedGate,
                 sharedUp, scratch.expertIntermediate},
                gateUp,
                expertGrid(shape.expertIntermediateSize / passes.gateUpColumns),
                {passes.threads, 1, 1});
    }
    graph.add(passes.down,
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               sharedDown, scratch.expertOutput},
              down, expertGrid(shape.hiddenSize / passes.downColumns),
              {passes.threads, 1, 1});
  }
}

// The three GGUF expert passes over grouped tiles: gate into expertOutput
// (the down pass overwrites it after the up pass consumed it), up with
// silu(gate) into expertIntermediate, down into expertOutput. Register plans
// read Table16 tiles from groupedInput: the gather writes the gate/up input's
// and a prepare dispatch the down input's.
void addGgufExperts(metal::CommandGraph &graph, const MoeScratch &scratch,
                    const BlockMoeWeights &weights, const MoePlan &plan,
                    bool packed) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  // RICHENGINE_MOE_TILE_MAJOR (decode only): expert grids swap axes —
  // threadgroups sweep a tile's column blocks for every grouped tile at one
  // column, so same-expert row-tiles re-read identical weight slices while
  // they are still cache-resident.
  const uint32_t ggufTileMajor =
      tuning().moeTileMajor && plan.phase() == MoePhase::Decode ? 1u : 0u;
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &packedInput,
                        const metal::MetalBuffer &output, uint32_t n, uint32_t k) {
    // A shared-expert-free block's tiles never carry its id; the unused
    // shared-segment slots bind the routed segment.
    const QuantizedSegment &shared =
        shape.sharedExpert ? projection.shared : projection.routed;
    std::vector<metal::MetalBuffer> bindings{input};
    if (table16) bindings.push_back(scratch.groupedSums);
    // The packed kernels take the fp16 plane and the exponent bytes ahead of
    // the tile descriptors; their MXFP4 segments read them, the rest stage
    // `input`.
    else if (packed)
      bindings.insert(bindings.end(), {packedInput, scratch.groupedSums});
    bindings.insert(bindings.end(),
                    {scratch.tileDescriptors, scratch.tileCount,
                     projection.routed.plane0, projection.routed.plane1Slot(),
                     projection.routed.meta, shared.plane0,
                     shared.plane1Slot(), shared.meta, output,
                     scratch.expertOutput});
    if (packed)
      // Unsplit plans hold no partials; the kernel never touches either
      // buffer then, but the signature still binds them.
      bindings.insert(bindings.end(),
                      {scratch.expertPartials ? scratch.expertPartials : scratch.expertCounters,
                       scratch.expertCounters});
    const std::string kernel = packed || !table16
                                      ? std::string(kMoeExpertGgufM) + std::to_string(plan.tileRows())
                                      : std::string(kMoeExpertGgufSg);
    const std::string suffix =
        packed ? "_p" : plan.configuration().mxfp4Native ? "_n" : "";
    // The split needs every segment on the packed tile: a segment it does
    // not serve runs whole-K on one partition while the rest exit early.
    const uint32_t splits =
        packed && plan.tileRows() == 8 &&
                projection.routed.formatId == GGUF_FMT_MXFP4 &&
                shared.formatId == GGUF_FMT_MXFP4
            ? kMoeExpertSplits
            : 1;
    const uint32_t columns = n / GGUF_TILE_COLUMNS;
    graph.add(kernel + (up ? "_g" : "_a") + suffix, std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  shared.formatId, splits, ggufTileMajor},
              ggufTileMajor ? metal::DispatchSize{tiles * splits, columns, 1}
                            : metal::DispatchSize{columns, tiles * splits, 1},
              {table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  // The `_gu` tile runs gate and up in one pass, storing silu(gate) * up to
  // the intermediate — the split passes' EpUpWithGate association without the
  // expertOutput round trip or a second grouped-input read. Staged, unpacked
  // plans only (no register or `_p` variant); a plan whose gate or up holds
  // an MXFP4 segment keeps the split `_n` kernels, which decode it natively.
  // RICHENGINE_MOE_GGUF_FUSED_OFF restores the split passes for A/Bs.
  const auto mxfp4 = [](const QuantizedSegment &a, const QuantizedSegment &b) {
    return a.formatId == GGUF_FMT_MXFP4 || b.formatId == GGUF_FMT_MXFP4;
  };
  const bool nativeSegments =
      plan.configuration().mxfp4Native &&
      (mxfp4(weights.gate.routed, weights.gate.shared) ||
       mxfp4(weights.up.routed, weights.up.shared));
  if (!table16 && !packed && !nativeSegments && tuning().moeGgufFused) {
    const QuantizedSegment &gateShared =
        shape.sharedExpert ? weights.gate.shared : weights.gate.routed;
    const QuantizedSegment &upShared =
        shape.sharedExpert ? weights.up.shared : weights.up.routed;
    graph.add(std::string(kMoeExpertGgufM) + std::to_string(plan.tileRows()) + "_gu",
              {scratch.groupedInput, scratch.tileDescriptors, scratch.tileCount,
               weights.gate.routed.plane0, weights.gate.routed.plane1Slot(),
               weights.gate.routed.meta, gateShared.plane0,
               gateShared.plane1Slot(), gateShared.meta,
               weights.up.routed.plane0, weights.up.routed.plane1Slot(),
               weights.up.routed.meta, upShared.plane0,
               upShared.plane1Slot(), upShared.meta,
               scratch.expertIntermediate},
              MoeGgufFusedParams{hidden, intermediate, shape.experts,
                                 weights.gate.routed.formatId, gateShared.formatId,
                                 weights.up.routed.formatId, upShared.formatId,
                                 ggufTileMajor},
              ggufTileMajor ? metal::DispatchSize{tiles, intermediate / GGUF_TILE_COLUMNS, 1}
                            : metal::DispatchSize{intermediate / GGUF_TILE_COLUMNS, tiles, 1},
              {GGUF_STAGED_THREADS, 1, 1});
  } else {
    pass(weights.gate, false, scratch.groupedInput, scratch.groupedPacked,
         scratch.expertOutput, intermediate, hidden);
    pass(weights.up, true, scratch.groupedInput, scratch.groupedPacked,
         scratch.expertIntermediate, intermediate, hidden);
  }
  if (table16)
    graph.add(std::string(kMoePrepareTable16),
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  // The down pass's packed input: gguf_pack_half over the bf16
  // intermediates (the dense kernels' pack, one thread per packed element),
  // into the same planes the gather filled for gate/up — only when a down
  // segment reads it.
  if (packed && (weights.down.routed.formatId == GGUF_FMT_MXFP4 ||
                 weights.down.shared.formatId == GGUF_FMT_MXFP4)) {
    graph.add(std::string(kGgufPackHalf),
              {scratch.expertIntermediate, scratch.groupedPacked,
               scratch.groupedSums, scratch.expertIntermediate,
               scratch.expertIntermediate, scratch.expertIntermediate,
               scratch.expertIntermediate, scratch.expertIntermediate},
              GgufDecodeParams{intermediate, 1, hidden, 0, 0},
              {uint64_t{tiles} * plan.tileRows() * intermediate / 32, 1, 1},
              {32, 1, 1});
  }
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.groupedPacked, scratch.expertOutput, hidden, intermediate);
}

// The streamed GGUF expert passes (MoE::addGgufPasses with a binding): the
// same three split passes of addGgufExperts — gate into expertOutput, up
// with silu(gate) into expertIntermediate, down into expertOutput — but
// every tile's routed expert resolves through the arena's slot and chunk
// tables (moe_gguf_stream_segment) instead of the resident planes, and a
// tile whose expert the current stage left out writes nothing. Staged
// tiles only: the register, packed and fused variants have no streamed
// form, so the caller plans ggufTile Staged and the weights must hold no
// MXFP4 segment (packed) — checked by the caller before it binds.
void addStreamedGgufExperts(metal::CommandGraph &graph,
                            const MoeScratch &scratch,
                            const BlockMoeWeights &weights,
                            const MoePlan &plan,
                            const MoeStreamBinding &streamed) {
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const auto pass = [&](const BlockExpertProjection &projection, bool up,
                        const metal::MetalBuffer &input,
                        const metal::MetalBuffer &output, uint32_t n,
                        uint32_t k, uint32_t projectionIndex) {
    // A shared-expert-free block's tiles never carry its id; the unused
    // shared-segment slots bind the routed segment's planes, which stay
    // resident and valid even though the streamed kernels never read them
    // for routed experts.
    const QuantizedSegment &shared =
        shape.sharedExpert ? projection.shared : projection.routed;
    graph.add(std::string("moe_expert_gguf_streamed_m") +
                  std::to_string(plan.tileRows()) + (up ? "_g" : "_a"),
              {input, scratch.tileDescriptors, scratch.tileCount,
               streamed.slotTable, streamed.chunkAddresses, shared.plane0,
               shared.plane1Slot(), shared.meta, output,
               scratch.expertOutput},
              MoeGgufStreamParams{k, n, shape.experts,
                                  projection.routed.formatId, shared.formatId,
                                  streamed.floorSlots, streamed.chunkSlots,
                                  projectionIndex, 0},
              {n / GGUF_TILE_COLUMNS, tiles, 1},
              {GGUF_STAGED_THREADS, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  pass(weights.gate, false, scratch.groupedInput, scratch.expertOutput,
       intermediate, hidden, 0);
  pass(weights.up, true, scratch.groupedInput, scratch.expertIntermediate,
       intermediate, hidden, 1);
  pass(weights.down, false, scratch.expertIntermediate,
       scratch.expertOutput, hidden, intermediate, 2);
}

} // namespace

// Prefill plans and GGUF plans run the three expert passes of the split
// plan, affine decode plans the fused gate/up tile.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config), phase_(phase),
      splitExperts_(shape.weightLayout == WeightLayout::Block32 || phase != MoePhase::Decode) {
  // RICHENGINE_MOE_TOPK shrinks top_k for routing-cost experiments, clamped
  // to the model's k so every buffer bound sized from the shape still
  // holds; routes_per_row, select and combine all follow this field.
  static const uint32_t topK = tuning().moeTopK;
  if (topK)
    shape_.expertsPerToken = std::clamp(topK, 1u, shape_.expertsPerToken);
  // Affine plans have 32- or 16-row prefill/canvas and 8-row decode
  // kernels, GGUF plans 8-row kernels in both phases and 32-row prefill
  // kernels; wide GGUF decode batches may take the 32-row kernel so a union
  // expert's weights stream once instead of per 8-row tile. Canvas is
  // affine-only.
  const bool gguf = shape.weightLayout == WeightLayout::Block32;
  const bool canvas = phase == MoePhase::Canvas;
  const bool prefill = phase == MoePhase::Prefill;
  const bool affineRowsTile =
      config.expertTile == MoeExpertTile::M32 ||
      config.expertTile == MoeExpertTile::M16;
  const bool wideGgufDecode =
      gguf && !prefill && !canvas && config.expertTile == MoeExpertTile::M32;
  if ((!prefill && !canvas && !wideGgufDecode &&
       config.expertTile != MoeExpertTile::M8) ||
      (canvas && (gguf || !affineRowsTile)) ||
      (!gguf && prefill && config.expertTile != MoeExpertTile::M32) ||
      (gguf && prefill && config.expertTile == MoeExpertTile::M16))
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register &&
      (shape.weightLayout != WeightLayout::Block32 || config.expertTile != MoeExpertTile::M8))
    throw std::invalid_argument("the register expert tile takes block 8-row tiles");
  // Plans that may pack their activations (the MXFP4 expert tiles) reserve
  // the fp16 plane and exponent bytes up front; whether a dispatch actually
  // packs is decided per weight set in add().
  workspace_ = workspaceFor(shape_, rows, tileRows(), splitExperts_, config.ggufTile,
                            gguf && config.mxfp4Native);
  maximumTiles_ = moeMaximumTiles(rows, shape_, tileRows());
}

void MoE::add(metal::CommandGraph &graph, const MoeBuffers &buffers,
              const MoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  const uint32_t tileRows = plan.tileRows();
  validate(weights, shape);
  const uint32_t tiles = plan.maximumTiles();
  const MoeWorkspace &required = plan.workspace();
  const uint64_t rowBytes = uint64_t{rows} * shape.hiddenSize * sizeof(uint16_t);
  if (buffers.input.sizeBytes() < rowBytes ||
      buffers.residual.sizeBytes() < rowBytes ||
      buffers.output.sizeBytes() < rowBytes)
    throw std::invalid_argument("MoE row buffers are smaller than execution shape");
  const MoeScratch &scratch = buffers.scratch;
  for (const MoeScratchField &field : kMoeScratchFields)
    if ((scratch.*field.buffer).sizeBytes() < required.*field.bytes)
      throw std::invalid_argument("MoE grouped scratch is smaller than its bound");
  // A decode plan's routing, grouping, gather, expert and combine dispatches
  // are identical on every step of one batch width: mark them replayable.
  bool bakeable = plan.phase() == MoePhase::Decode;
  if (bakeable) graph.beginBakedSpan();
  MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                             shape.expertsPerToken, 0};
  // The stats generation makes the dispatch's record slot unique across the
  // process; the baked span replays it through addPatchable below.
  const uint32_t stats = statsEnabled() ? ++statsSequence() : 0;
  // RICHENGINE_MOE_SHARED_DENSE (affine decode only): the shared expert runs
  // as one dense tile per lane block (moe_shared_*_q4) instead of a tile per
  // tile_rows rows. The grouping keeps its routes aimed at the grouped rows
  // past the routed ones (shared == 2), where the dense passes write.
  const bool denseShared =
      shape.sharedExpert && weights.layout() == WeightLayout::Affine64 &&
      plan.phase() == MoePhase::Decode && tuning().moeSharedDense &&
      (plan.laneRows() == 8 || plan.laneRows() == 16) &&
      rows % plan.laneRows() == 0;
  const MoeGroupParams groupParams{
      rows, shape.expertsPerToken, tileRows, shape.experts,
      shape.sharedExpert ? (denseShared ? 2u : 1u) : 0u, stats,
      plan.laneRows()};
  // The union cap runs between select and grouping; it needs the separate
  // select even where the fused route-group would apply. The plan's value
  // (serve-native --moe-union) wins; RICHENGINE_MOE_UNION covers callers
  // that plan without an ExecutionPlans.
  uint32_t cap = plan.configuration().unionCap;
  if (!cap)
    cap = unionCap();
  cap = plan.phase() == MoePhase::Decode ? std::min(cap, shape.experts) : 0;
  // A sigmoid-gated block (no shared expert) at a single lane's rows folds
  // the select into the grouping dispatch: one threadgroup selects a row
  // per simdgroup, then sorts the routes by expert. Wider batches keep the
  // separate select: its row-per-group grid parallelizes where the fused
  // kernel's row loop serializes.
  const bool fusedRoute = !shape.sharedExpert && rows <= 8 && !cap;
  metal::MetalBuffer expertBias;
  const bool block = weights.layout() == WeightLayout::Block32;
  if (block) {
    // fp32 scores of the F32 router in rows of 256, as the select kernel reads.
    addGgufFloat(graph, buffers.input, weights.blocks().router, scratch.groupedInput, rows,
                 256, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
    if (shape.sharedExpert) {
      graph.add(std::string(kMoeRouteSelectF32),
                {scratch.groupedInput, buffers.input,
                 weights.blocks().sharedScalarGate.plane0, scratch.selectedExperts,
                 scratch.routingWeights},
                routeParams, {rows, 1, 1});
    } else if (fusedRoute) {
      expertBias = weights.blocks().expertBias.plane0;
    } else {
      // Sigmoid gating with the per-expert selection bias; no shared expert.
      graph.add(std::string(kMoeRouteSelectSigmoid),
                {scratch.groupedInput, weights.blocks().expertBias.plane0,
                 scratch.selectedExperts, scratch.routingWeights},
                routeParams, {rows, 1, 1});
    }
  } else {
    const AffineMoeWeights &affine = weights.affine();
    // shared == 1 folds the scalar gate into the scores tile (the select
    // reads the parked logit); shared == 2 re-dots the input in the select,
    // the pre-fold path RICHENGINE_MOE_FOLDED_GATE_OFF A/Bs against.
    routeParams.shared =
        shape.sharedExpert ? (tuning().moeFoldedGate ? 1u : 2u) : 0u;
    MoeRouteTile route = moeRouteTile(rows, plan.configuration().routeWideRows);
    if (plan.configuration().routeM16 && route.rows == 8)
      route = MoeRouteTile{16, 64};
    // The scores tile folds the shared gate in as a 257th column (bound but
    // unread when the shape has no shared expert, so the router planes stand
    // in); the select then reads the parked logit instead of re-dotting.
    const AffineWeights &gatePlanes = shape.sharedExpert
        ? affine.sharedScalarGate.planes
        : affine.router.planes;
    graph.add(std::string(route.rows == 8   ? kMoeRouteScoresQ8M8
                              : route.rows == 16 ? kMoeRouteScoresQ8M16
                                                 : kMoeRouteScoresQ8M32),
              {buffers.input, affine.router.planes.weights,
               affine.router.planes.scales, affine.router.planes.biases,
               scratch.groupedInput, gatePlanes.weights, gatePlanes.scales,
               gatePlanes.biases},
              routeParams,
              {(rows + route.rows - 1) / route.rows,
               (shape.experts + route.experts - 1) / route.experts, 1});
    if (shape.sharedExpert) {
      graph.add(std::string(kMoeRouteSelectQ8),
                {scratch.groupedInput, scratch.selectedExperts,
                 scratch.routingWeights, buffers.input, gatePlanes.weights,
                 gatePlanes.scales, gatePlanes.biases},
                routeParams, {rows, 1, 1});
    } else if (fusedRoute) {
      expertBias = affine.expertBias;
    } else {
      graph.add(std::string(kMoeRouteSelectSigmoid),
                {scratch.groupedInput, affine.expertBias,
                 scratch.selectedExperts, scratch.routingWeights},
                routeParams, {rows, 1, 1});
    }
  }
  // The cap dispatch rewrites the select's routes in place: experts whose
  // summed routing weight misses the budget lose their routes to ~0u.
  if (cap)
    graph.add(std::string(kMoeRouteCap),
              {scratch.selectedExperts, scratch.routingWeights},
              MoeCapParams{rows, shape.routesPerToken(), shape.experts, cap},
              {1, 1, 1}, {256, 1, 1});
  if (fusedRoute) {
    // The scores kernel's output feeds the fused select-and-group.
    if (stats)
      graph.addPatchable(std::string(kMoeRouteGroupSigmoid),
                         {scratch.groupedInput, expertBias, scratch.selectedExperts,
                          scratch.routingWeights, scratch.tileDescriptors,
                          scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
                         MoeRouteGroupParams{routeParams, groupParams}, {1, 1, 1});
    else
      graph.add(std::string(kMoeRouteGroupSigmoid),
                {scratch.groupedInput, expertBias, scratch.selectedExperts,
                 scratch.routingWeights, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
                MoeRouteGroupParams{routeParams, groupParams}, {1, 1, 1});
  } else {
    // The counting and scatter loops stride by the threadgroup width; a
    // 1024-thread group quadruples both while threads past the 256 experts
    // skip the per-expert bookkeeping between them.
    // RICHENGINE_MOE_VERIFIED_ROWS: the verify step's grouping routes each
    // lane's live rows alone — padding rows (anchor repeats) keep their
    // routes out of the expert union.
    const bool verified = buffers.verifyRows && plan.laneRows() &&
                          tuning().moeVerifiedRows;
    std::vector<metal::MetalBuffer> groupBindings{scratch.selectedExperts,
                                                scratch.tileDescriptors,
                                                scratch.tileCount,
                                                scratch.groupedRoutes,
                                                scratch.routeRows};
    if (verified)
      groupBindings.push_back(buffers.verifyRows);
    if (stats)
      graph.addPatchable(std::string(verified ? kMoeGroupRoutesVerified
                                              : kMoeGroupRoutes),
                         std::move(groupBindings), groupParams, {1, 1, 1},
                         {1024, 1, 1});
    else
      graph.add(std::string(verified ? kMoeGroupRoutesVerified
                                     : kMoeGroupRoutes),
                std::move(groupBindings), groupParams, {1, 1, 1},
                {1024, 1, 1});
  }
  if (denseShared) {
    const AffineMoeWeights &affine = weights.affine();
    const bool m16 = plan.laneRows() == 16;
    const bool four =
        plan.configuration().m8Simdgroups == MoeExpertSimdgroups::Four;
    const std::string_view gateUpKernel =
        m16 ? (four ? kMoeSharedGateUpQ4M16Sg4 : kMoeSharedGateUpQ4M16)
            : (four ? kMoeSharedGateUpQ4M8Sg4 : kMoeSharedGateUpQ4M8);
    const std::string_view downKernel =
        m16 ? (four ? kMoeSharedDownQ4M16Sg4 : kMoeSharedDownQ4M16)
            : (four ? kMoeSharedDownQ4M8Sg4 : kMoeSharedDownQ4M8);
    const uint32_t blocks = rows / plan.laneRows();
    graph.add(std::string(gateUpKernel),
              {buffers.input, scratch.tileCount, affine.sharedGate.packed,
               affine.sharedUp.packed, scratch.expertIntermediate},
              MoeSharedParams{rows, shape.hiddenSize,
                              shape.expertIntermediateSize, tileRows},
              {shape.expertIntermediateSize / 128, blocks, 1},
              {four ? 128u : 256u, 1, 1});
    graph.add(std::string(downKernel),
              {scratch.expertIntermediate, scratch.tileCount,
               affine.sharedDown.packed, scratch.expertOutput},
              MoeSharedParams{rows, shape.expertIntermediateSize,
                              shape.hiddenSize, tileRows},
              {shape.hiddenSize / (four ? 256u : 128u), blocks, 1},
              {four ? 128u : 256u, 1, 1});
  }
  // The packed path (packsDecode's MoE analog): a plan on the `_n` kernels
  // whose weights hold at least one MXFP4 expert segment gathers the packed
  // fp16 plane and exponent bytes alongside the bf16 rows, and its expert
  // passes run the `_p` kernels — MXFP4 segments on the packed multiplane
  // tile, the rest staged as before. Decode and prefill tiles both take it.
  const auto mxfp4Expert = [](const BlockExpertProjection &p) {
    return p.routed.formatId == GGUF_FMT_MXFP4 || p.shared.formatId == GGUF_FMT_MXFP4;
  };
  const bool packed =
      block && plan.tileRows() == 8 && plan.configuration().mxfp4Native &&
      plan.configuration().ggufTile != MoeGgufTile::Register &&
      !packedDisabled() &&
      (mxfp4Expert(weights.blocks().gate) || mxfp4Expert(weights.blocks().up) ||
       mxfp4Expert(weights.blocks().down));
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  // The gather materializes the grouped rows for GGUF plans and for the
  // affine decode tiles; affine prefill's indirect gate and up passes read
  // the rows' inputs through grouped_routes instead — as does the affine
  // decode gate/up under RICHENGINE_MOE_INDIRECT.
  const bool affineIndirect =
      !block && !plan.splitExperts() && tuning().moeIndirectGather &&
      plan.tileRows() == 8 && shape.hiddenSize % 256 == 0;
  if (block || (!plan.splitExperts() && !affineIndirect)) {
    if (plan.configuration().ggufTile == MoeGgufTile::Register)
      graph.add(std::string(kMoeGatherTable16),
                {buffers.input, scratch.groupedRoutes, scratch.tileCount,
                 scratch.groupedInput, scratch.groupedSums},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else if (packed)
      graph.add(std::string(kMoeGatherPacked),
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput, scratch.groupedPacked,
                 scratch.groupedSums},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else if (!block && !plan.splitExperts() && plan.tileRows() == 8 &&
             tuning().moeSmallRows)
      graph.add(std::string(kMoeGatherRowsSmall),
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / 256, 1});
    else
      graph.add(std::string(kMoeGatherRows),
                {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
                 scratch.tileCount, scratch.groupedInput},
                gather, {tiles, shape.hiddenSize / 256, 1});
  }
  // The GGUF expert kernels produce no output when they replay from an
  // indirect command buffer on this driver (gguf-moe's staged and register
  // decodes both): suspend their dispatches — a suspension also lifts them
  // out of an enclosing span (the verify FFN's), which an inner end could
  // not. Affine experts replay fine.
  if (bakeable && block) graph.suspendBakedSpan();
  if (block)
    addGgufExperts(graph, scratch, weights.blocks(), plan, packed);
  else
    addAffineExperts(graph, buffers, scratch, weights.affine(), plan);
  if (bakeable && block) graph.resumeBakedSpan();
  graph.add(std::string(kMoeCombine),
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
  if (bakeable) graph.endBakedSpan();
}

// The route-first split of add() (MoE.hpp): GGUF block plans only, for a
// host step between the routing dispatches and the expert passes. The
// pieces encode the same dispatches add() encodes for such a plan —
// add() itself is untouched for every other caller.
namespace {

void requireBlockPlan(const MoeWeights &weights, const MoePlan &plan) {
  if (weights.layout() != WeightLayout::Block32)
    throw std::invalid_argument("the route-first MoE split takes GGUF block weights");
  validate(weights, plan.shape());
}

// The packed and register plan facts add() derives for its gather and
// passes, recomputed here for the split pieces.
struct MoeGatherChoice final {
  bool table16;
  bool packed;
};

MoeGatherChoice gatherChoice(const MoeWeights &weights, const MoePlan &plan) {
  const bool table16 = plan.configuration().ggufTile == MoeGgufTile::Register;
  const bool block = true;
  const auto mxfp4Expert = [](const BlockExpertProjection &p) {
    return p.routed.formatId == GGUF_FMT_MXFP4 || p.shared.formatId == GGUF_FMT_MXFP4;
  };
  const bool packed =
      block && plan.tileRows() == 8 && plan.configuration().mxfp4Native &&
      !table16 && !packedDisabled() &&
      (mxfp4Expert(weights.blocks().gate) || mxfp4Expert(weights.blocks().up) ||
       mxfp4Expert(weights.blocks().down));
  return {table16, packed};
}

} // namespace

void MoE::addRouting(metal::CommandGraph &graph, const MoeBuffers &buffers,
                     const MoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  const uint32_t rows = plan.rows();
  requireBlockPlan(weights, plan);
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken, 0};
  const uint32_t stats = statsEnabled() ? ++statsSequence() : 0;
  const MoeGroupParams groupParams{
      rows, shape.expertsPerToken, plan.tileRows(), shape.experts,
      shape.sharedExpert ? 1u : 0u, stats, plan.laneRows()};
  uint32_t cap = plan.configuration().unionCap;
  if (!cap)
    cap = unionCap();
  cap = plan.phase() == MoePhase::Decode ? std::min(cap, shape.experts) : 0;
  const bool fusedRoute = !shape.sharedExpert && rows <= 8 && !cap;
  metal::MetalBuffer expertBias;
  const BlockMoeWeights &blocks = weights.blocks();
  // fp32 scores of the F32 router in rows of 256, as the select kernel reads.
  addGgufFloat(graph, buffers.input, blocks.router, buffers.scratch.groupedInput,
               rows, 256, 0, FloatOutput::Float32,
               plan.configuration().ggufRouterTile);
  if (shape.sharedExpert) {
    graph.add(std::string(kMoeRouteSelectF32),
              {buffers.scratch.groupedInput, buffers.input,
               blocks.sharedScalarGate.plane0, buffers.scratch.selectedExperts,
               buffers.scratch.routingWeights},
              routeParams, {rows, 1, 1});
  } else if (fusedRoute) {
    expertBias = blocks.expertBias.plane0;
  } else {
    graph.add(std::string(kMoeRouteSelectSigmoid),
              {buffers.scratch.groupedInput, blocks.expertBias.plane0,
               buffers.scratch.selectedExperts, buffers.scratch.routingWeights},
              routeParams, {rows, 1, 1});
  }
  if (cap)
    graph.add(std::string(kMoeRouteCap),
              {buffers.scratch.selectedExperts, buffers.scratch.routingWeights},
              MoeCapParams{rows, shape.routesPerToken(), shape.experts, cap},
              {1, 1, 1}, {256, 1, 1});
  if (fusedRoute) {
    if (stats)
      graph.addPatchable(std::string(kMoeRouteGroupSigmoid),
                         {buffers.scratch.groupedInput, expertBias,
                          buffers.scratch.selectedExperts,
                          buffers.scratch.routingWeights,
                          buffers.scratch.tileDescriptors,
                          buffers.scratch.tileCount,
                          buffers.scratch.groupedRoutes,
                          buffers.scratch.routeRows},
                         MoeRouteGroupParams{routeParams, groupParams}, {1, 1, 1});
    else
      graph.add(std::string(kMoeRouteGroupSigmoid),
                {buffers.scratch.groupedInput, expertBias,
                 buffers.scratch.selectedExperts,
                 buffers.scratch.routingWeights,
                 buffers.scratch.tileDescriptors,
                 buffers.scratch.tileCount,
                 buffers.scratch.groupedRoutes,
                 buffers.scratch.routeRows},
                MoeRouteGroupParams{routeParams, groupParams}, {1, 1, 1});
  } else {
    const bool verified = buffers.verifyRows && plan.laneRows() &&
                          tuning().moeVerifiedRows;
    std::vector<metal::MetalBuffer> groupBindings{
        buffers.scratch.selectedExperts, buffers.scratch.tileDescriptors,
        buffers.scratch.tileCount, buffers.scratch.groupedRoutes,
        buffers.scratch.routeRows};
    if (verified)
      groupBindings.push_back(buffers.verifyRows);
    if (stats)
      graph.addPatchable(std::string(verified ? kMoeGroupRoutesVerified
                                              : kMoeGroupRoutes),
                         std::move(groupBindings), groupParams, {1, 1, 1},
                         {1024, 1, 1});
    else
      graph.add(std::string(verified ? kMoeGroupRoutesVerified
                                     : kMoeGroupRoutes),
                std::move(groupBindings), groupParams, {1, 1, 1},
                {1024, 1, 1});
  }
}

void MoE::addGgufGather(metal::CommandGraph &graph, const MoeBuffers &buffers,
                        const MoeWeights &weights, const MoePlan &plan) {
  requireBlockPlan(weights, plan);
  const MoeShape shape = plan.shape();
  const uint32_t tiles = plan.maximumTiles();
  const MoeGatherChoice choice = gatherChoice(weights, plan);
  const MoeGatherParams gather{plan.tileRows(), shape.hiddenSize,
                               shape.routesPerToken()};
  const MoeScratch &scratch = buffers.scratch;
  if (choice.table16)
    graph.add(std::string(kMoeGatherTable16),
              {buffers.input, scratch.groupedRoutes, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else if (choice.packed)
    graph.add(std::string(kMoeGatherPacked),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput, scratch.groupedPacked,
               scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else
    graph.add(std::string(kMoeGatherRows),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
}

void MoE::addGgufPasses(metal::CommandGraph &graph, const MoeBuffers &buffers,
                        const MoeWeights &weights, const MoePlan &plan,
                        const MoeStreamBinding *streamed) {
  requireBlockPlan(weights, plan);
  if (!streamed) {
    const MoeGatherChoice choice = gatherChoice(weights, plan);
    addGgufExperts(graph, buffers.scratch, weights.blocks(), plan,
                   choice.packed);
    return;
  }
  // The streamed kernels are the staged tiles only: no register tile, no
  // packed MXFP4 plane, no fused gate/up dispatch.
  if (plan.configuration().ggufTile != MoeGgufTile::Staged ||
      (plan.tileRows() == 8 && gatherChoice(weights, plan).packed))
    throw std::invalid_argument(
        "streamed MoE expert passes take the staged GGUF tile");
  addStreamedGgufExperts(graph, buffers.scratch, weights.blocks(), plan,
                         *streamed);
}

void MoE::addCombine(metal::CommandGraph &graph, const MoeBuffers &buffers,
                     const MoeWeights &weights, const MoePlan &plan) {
  requireBlockPlan(weights, plan);
  const MoeShape shape = plan.shape();
  graph.add(std::string(kMoeCombine),
            {buffers.scratch.expertOutput, buffers.scratch.routeRows,
             buffers.scratch.routingWeights, buffers.residual, buffers.output},
            MoeCombineParams{plan.rows(), shape.hiddenSize,
                             shape.routesPerToken()},
            {plan.rows(), shape.hiddenSize / 256, 1});
}

// Gemma 4's routed block: a bias-free bf16 router over the router-input rows
// (the post-attention residual, normed scale-free with the learned scale and
// hidden^-0.5 by the caller), softmax over all experts, the top-k winners
// renormalized and scaled by per_expert_scale, then the grouped GeGLU expert
// tiles — the shared expert runs beside this as dense projections, not a
// route. `residual` adds into the combine; the caller binds its zero rows so
// `output` holds the routed sum alone (it is post-normed before it joins the
// residual).
void MoE::addGemma(metal::CommandGraph &graph, const MoeBuffers &buffers,
                   const GemmaMoeWeights &weights, const MoePlan &plan) {
  const MoeShape shape = plan.shape();
  if (!shape.valid() || shape.sharedExpert ||
      shape.weightLayout != WeightLayout::Affine64)
    throw std::invalid_argument("invalid Gemma MoE shape");
  const uint32_t rows = plan.rows();
  const uint32_t tiles = plan.maximumTiles();
  const uint32_t tileRows = plan.tileRows();
  const MoeScratch &scratch = buffers.scratch;
  const metal::MetalBuffer routerInput =
      buffers.routerInput ? buffers.routerInput : buffers.input;
  if (!routerInput || !buffers.input || !buffers.residual || !buffers.output ||
      !weights.routerWeights || !weights.perExpertScale ||
      weights.routerWeights.sizeBytes() <
          uint64_t{shape.experts} * shape.hiddenSize * sizeof(uint16_t) ||
      weights.perExpertScale.sizeBytes() <
          uint64_t{shape.experts} * sizeof(float) ||
      !matches(weights.expertGate, shape.experts,
               shape.expertIntermediateSize, shape.hiddenSize) ||
      !matches(weights.expertUp, shape.experts,
               shape.expertIntermediateSize, shape.hiddenSize) ||
      !matches(weights.expertDown, shape.experts, shape.hiddenSize,
               shape.expertIntermediateSize))
    throw std::invalid_argument("Gemma MoE weights do not match execution shape");
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken, 0};
  // fp32 scores of every expert, parked in groupedInput until the select
  // reads them.
  // Canvas plans take the staged+vectorized score tile (measured 262 ->
  // 142 us on the 256x2816x128 router; the four-row variant's staging
  // costed more occupancy than its weight reuse saved).
  // moe_route_scores_gemma_vec stages its row (8 KB covers hidden 4096);
  // larger hidden falls back to the strided kernel.
  const bool canvasRoute =
      plan.phase() == MoePhase::Canvas && shape.hiddenSize <= 4096;
  graph.add(std::string(canvasRoute ? kMoeRouteScoresGemmaVec
                        : kMoeRouteScoresGemma),
            {routerInput, weights.routerWeights, scratch.groupedInput},
            routeParams, {rows, shape.experts / 8, 1});
  graph.add(std::string(kMoeRouteSelectGemma),
            {scratch.groupedInput, weights.perExpertScale,
             scratch.selectedExperts, scratch.routingWeights},
            routeParams, {rows, 1, 1}, {256, 1, 1});
  const uint32_t stats = statsEnabled() ? ++statsSequence() : 0;
  const MoeGroupParams groupParams{rows, shape.expertsPerToken, tileRows,
                                   shape.experts, 0, stats, 0};
  graph.add(std::string(kMoeGroupRoutes),
            {scratch.selectedExperts, scratch.tileDescriptors,
             scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
            groupParams, {1, 1, 1}, {1024, 1, 1});
  const uint32_t routesPerRow = shape.routesPerToken();
  const MoeExpertParams gate{shape.hiddenSize, shape.expertIntermediateSize,
                             shape.experts, routesPerRow, 0,
                             weights.expertGate.expertStrideBytes,
                             weights.expertGate.expertStrideBytes};
  const MoeExpertParams up{shape.hiddenSize, shape.expertIntermediateSize,
                           shape.experts, routesPerRow, 0,
                           weights.expertUp.expertStrideBytes,
                           weights.expertUp.expertStrideBytes};
  const MoeExpertParams down{shape.expertIntermediateSize, shape.hiddenSize,
                             shape.experts, routesPerRow, 0,
                             weights.expertDown.expertStrideBytes,
                             weights.expertDown.expertStrideBytes};
  if (plan.splitExperts()) {
    // The indirect tiles read each grouped row's input through
    // grouped_routes; the gelu up pass folds the GeGLU into its stores. No
    // tile ever carries the shared expert id, so its slab slots bind the
    // routed slabs.
    const bool m16 = plan.tileRows() == 16;
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256IndirectM16
                              : kPrefillMoeExpertQ4N256IndirectM32),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.expertGate.packed, scratch.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256UpGeluIndirectM16
                              : kPrefillMoeExpertQ4N256UpGeluIndirectM32),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, weights.expertUp.packed,
               weights.expertUp.packed, scratch.expertOutput,
               scratch.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add(std::string(m16 ? kPrefillMoeExpertQ4N256M16
                              : kPrefillMoeExpertQ4N256M32),
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.expertDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    graph.add(std::string(kMoeGatherRows),
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              MoeGatherParams{tileRows, shape.hiddenSize, routesPerRow},
              {tiles, shape.hiddenSize / 256, 1});
    const bool four =
        plan.configuration().m8Simdgroups == MoeExpertSimdgroups::Four;
    const MoeExpertParams gateUp{shape.hiddenSize, shape.expertIntermediateSize,
                                 shape.experts, routesPerRow, 0,
                                 weights.expertGate.expertStrideBytes,
                                 weights.expertUp.expertStrideBytes};
    graph.add(std::string(four ? kMoeExpertGateUpQ4M8GeluN128Sg4
                   : kMoeExpertGateUpQ4M8Gelu),
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.expertUp.packed, weights.expertGate.packed,
               weights.expertUp.packed, scratch.expertIntermediate},
              gateUp, {shape.expertIntermediateSize / 128, tiles, 1},
              {four ? 128u : 256u, 1, 1});
    graph.add(std::string(four ? kMoeExpertDownQ4M8N256Sg4 : kMoeExpertDownQ4M8),
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               weights.expertDown.packed, scratch.expertOutput},
              down, {shape.hiddenSize / (four ? 256u : 128u), tiles, 1},
              {four ? 128u : 256u, 1, 1});
  }
  graph.add(std::string(kMoeCombine),
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, routesPerRow},
            {rows, shape.hiddenSize / 256, 1});
}

// Prints the stats records since the last call, oldest first. Only records
// a live generation could have written are trusted: the scratch ring is
// never zeroed, so stale bytes must fail the generation bound.
void MoE::logStats(const metal::MetalBuffer &tileCount) {
  static const bool on = tuning().moeStats;
  if (!on || !tileCount ||
      tileCount.sizeBytes() <
          kMoeStatsLogOffset + sizeof(MoeStatsRecord) ||
      !tileCount.contents())
    return;
  const uint32_t slots = static_cast<uint32_t>(
      (tileCount.sizeBytes() - kMoeStatsLogOffset) / sizeof(MoeStatsRecord));
  const auto *records = reinterpret_cast<const MoeStatsRecord *>(
      static_cast<const std::byte *>(tileCount.contents()) +
      kMoeStatsLogOffset);
  static uint32_t printed = 0;
  const uint32_t newest = statsSequence().load(std::memory_order_relaxed);
  std::vector<const MoeStatsRecord *> pending;
  for (uint32_t slot = 0; slot < slots; ++slot) {
    const MoeStatsRecord &record = records[slot];
    if (record.generation > printed && record.generation <= newest &&
        record.rows && record.routed_experts <= 256 && record.tiles)
      pending.push_back(&record);
  }
  std::sort(pending.begin(), pending.end(), [](const auto *a, const auto *b) {
    return a->generation < b->generation;
  });
  for (const MoeStatsRecord *record : pending) {
    fprintf(stderr, "moe-stats rows=%u union=%u tiles=%u\n", record->rows,
            record->routed_experts, record->tiles);
    printed = std::max(printed, record->generation);
  }
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > RICHENGINE_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, MoePhase::Prefill);
}

MoePlan MoE::canvasPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > RICHENGINE_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE canvas rows");
  return MoePlan(shape, rows, config, MoePhase::Canvas);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, uint32_t verifyRows,
                        MoeConfig config) {
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  if (!verifyRows || verifyRows > RICHENGINE_TARGET_VERIFY_ROWS)
    throw std::invalid_argument("invalid MoE decode verify rows");
  MoePlan plan(shape, lanes * verifyRows, config, MoePhase::Decode);
  plan.laneRows_ = verifyRows;
  return plan;
}

} // namespace richengine::ops
