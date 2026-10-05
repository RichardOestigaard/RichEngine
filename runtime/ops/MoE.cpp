#include "ops/MoE.hpp"

#include "Env.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/MoE.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace splash::ops {
namespace {

static_assert(offsetof(MoeExpertParams, expert_stride_bytes_0) == 16);

// A kill switch for the packed expert paths, read once (the decode plan
// calls this per step).
bool packedDisabled() {
  static const bool off = envFlag("SPLASH_MOE_PACKED_OFF");
  return off;
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
  const uint64_t scoreBytes = uint64_t{rows} * 256 * sizeof(float);
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
  return {routes * sizeof(uint32_t), routes * sizeof(float),
          uint64_t{tiles} * sizeof(MoeTileDescriptor), sizeof(uint32_t),
          groupedRows * sizeof(uint32_t), routes * sizeof(uint32_t),
          std::max(tableBytes(table16 ? widest : shape.hiddenSize, groupedRows), scoreBytes),
          groupedRows * shape.expertIntermediateSize * sizeof(uint16_t),
          groupedRows * outputWidth * sizeof(uint16_t), sumsBytes, packedBytes};
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
    return {"moe_expert_gate_up_q4_m8_n128_sg4",
            "moe_expert_down_q4_m8_n256_sg4", 128, 256, threads};
  return {"moe_expert_gate_up_q4_m8", "moe_expert_down_q4_m8", 128, 128,
          threads};
}

void addAffineExperts(metal::CommandGraph &graph, const MoeScratch &scratch,
                      const AffineMoeWeights &weights, const MoePlan &plan) {
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
  const MoeExpertParams gateUp{shape.hiddenSize, shape.expertIntermediateSize,
                               shape.experts, 0,
                               weights.expertGate.expertStrideBytes,
                               weights.expertUp.expertStrideBytes};
  const MoeExpertParams gate{shape.hiddenSize, shape.expertIntermediateSize,
                             shape.experts, 0,
                             weights.expertGate.expertStrideBytes,
                             weights.expertGate.expertStrideBytes};
  const MoeExpertParams up{shape.hiddenSize, shape.expertIntermediateSize,
                           shape.experts, 0, weights.expertUp.expertStrideBytes,
                           weights.expertUp.expertStrideBytes};
  const MoeExpertParams down{shape.expertIntermediateSize, shape.hiddenSize,
                             shape.experts, 0,
                             weights.expertDown.expertStrideBytes,
                             weights.expertDown.expertStrideBytes};
  if (plan.splitExperts()) {
    // The gate lands in expertOutput, which the down pass overwrites only
    // after the up pass has consumed it.
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               sharedGate, scratch.expertOutput},
              gate, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_up_silu_m32",
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertUp.packed,
               sharedUp, scratch.expertOutput,
               scratch.expertIntermediate},
              up, {shape.expertIntermediateSize / 256, tiles, 1});
    graph.add("prefill_moe_expert_q4_n256_m32",
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               sharedDown, scratch.expertOutput},
              down, {shape.hiddenSize / 256, tiles, 1});
  } else {
    // The workspace holds the same grouped rows whatever the column tile;
    // only the grid's column count and the threadgroup width follow it.
    const ExpertPasses passes = fusedExpertPasses(plan.configuration());
    graph.add(passes.gateUp,
              {scratch.groupedInput, scratch.tileDescriptors,
               scratch.tileCount, weights.expertGate.packed,
               weights.expertUp.packed, sharedGate,
               sharedUp, scratch.expertIntermediate},
              gateUp,
              {shape.expertIntermediateSize / passes.gateUpColumns, tiles, 1},
              {passes.threads, 1, 1});
    graph.add(passes.down,
              {scratch.expertIntermediate, scratch.tileDescriptors,
               scratch.tileCount, weights.expertDown.packed,
               sharedDown, scratch.expertOutput},
              down, {shape.hiddenSize / passes.downColumns, tiles, 1},
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
    const std::string kernel = packed ? "moe_expert_gguf_m8"
                                      : table16 ? "moe_expert_gguf_sg"
                                                : "moe_expert_gguf_m" + std::to_string(plan.tileRows());
    const std::string suffix =
        packed ? "_p" : plan.configuration().mxfp4Native ? "_n" : "";
    graph.add(kernel + (up ? "_g" : "_a") + suffix, std::move(bindings),
              MoeGgufExpertParams{k, n, shape.experts, projection.routed.formatId,
                                  shared.formatId},
              {n / GGUF_TILE_COLUMNS, tiles, 1}, {table16 ? GGUF_REGISTER_THREADS : GGUF_STAGED_THREADS, 1, 1});
  };
  const uint32_t hidden = shape.hiddenSize;
  const uint32_t intermediate = shape.expertIntermediateSize;
  pass(weights.gate, false, scratch.groupedInput, scratch.groupedPacked,
       scratch.expertOutput, intermediate, hidden);
  pass(weights.up, true, scratch.groupedInput, scratch.groupedPacked,
       scratch.expertIntermediate, intermediate, hidden);
  if (table16)
    graph.add("moe_prepare_table16",
              {scratch.expertIntermediate, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              intermediate, {tiles, intermediate / 256, 1});
  // The down pass's packed input: gguf_pack_half over the bf16
  // intermediates (the dense kernels' pack, one thread per packed element),
  // into the same planes the gather filled for gate/up — only when a down
  // segment reads it.
  if (packed && (weights.down.routed.formatId == GGUF_FMT_MXFP4 ||
                 weights.down.shared.formatId == GGUF_FMT_MXFP4)) {
    graph.add("gguf_pack_half",
              {scratch.expertIntermediate, scratch.groupedPacked,
               scratch.groupedSums, scratch.expertIntermediate,
               scratch.expertIntermediate, scratch.expertIntermediate,
               scratch.expertIntermediate, scratch.expertIntermediate},
              GgufDecodeParams{intermediate, 1, hidden, 0},
              {uint64_t{tiles} * plan.tileRows() * intermediate / 32, 1, 1},
              {32, 1, 1});
  }
  pass(weights.down, false,
       table16 ? scratch.groupedInput : scratch.expertIntermediate,
       scratch.groupedPacked, scratch.expertOutput, hidden, intermediate);
}

} // namespace

// Prefill plans and GGUF plans run the three expert passes of the split
// plan, affine decode plans the fused gate/up tile.
MoePlan::MoePlan(MoeShape shape, uint32_t rows, MoeConfig config,
                 MoePhase phase)
    : shape_(shape), rows_(rows), config_(config), phase_(phase),
      splitExperts_(shape.weightLayout == WeightLayout::Block32 || phase == MoePhase::Prefill) {
  // Affine plans have 32-row prefill and 8-row decode kernels, GGUF plans
  // 8-row kernels in both phases and 32-row prefill kernels.
  const bool gguf = shape.weightLayout == WeightLayout::Block32;
  const bool prefill = phase == MoePhase::Prefill;
  if ((!prefill && config.expertTile != MoeExpertTile::M8) ||
      (!gguf && prefill && config.expertTile != MoeExpertTile::M32))
    throw std::invalid_argument("invalid MoE expert tile configuration");
  if (config.ggufTile == MoeGgufTile::Register &&
      (shape.weightLayout != WeightLayout::Block32 || config.expertTile != MoeExpertTile::M8))
    throw std::invalid_argument("the register expert tile takes block 8-row tiles");
  // Decode plans that may pack their activations (the MXFP4 expert tiles)
  // reserve the fp16 plane and exponent bytes up front; whether a dispatch
  // actually packs is decided per weight set in add().
  workspace_ = workspaceFor(shape, rows, tileRows(), splitExperts_, config.ggufTile,
                            gguf && !prefill && config.mxfp4Native);
  maximumTiles_ = moeMaximumTiles(rows, shape, tileRows());
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
  const MoeRouteParams routeParams{rows, shape.hiddenSize, shape.experts,
                                   shape.expertsPerToken};
  const MoeGroupParams groupParams{rows, shape.expertsPerToken, tileRows,
                                   shape.experts, shape.sharedExpert};
  // A sigmoid-gated block (no shared expert) at a single lane's rows folds
  // the select into the grouping dispatch: one threadgroup selects a row
  // per simdgroup, then sorts the routes by expert. Wider batches keep the
  // separate select: its row-per-group grid parallelizes where the fused
  // kernel's row loop serializes.
  const bool fusedRoute = !shape.sharedExpert && rows <= 8;
  metal::MetalBuffer expertBias;
  const bool block = weights.layout() == WeightLayout::Block32;
  if (block) {
    // fp32 scores of the F32 router in rows of 256, as the select kernel reads.
    addGgufFloat(graph, buffers.input, weights.blocks().router, scratch.groupedInput, rows,
                 256, 0, FloatOutput::Float32, plan.configuration().ggufRouterTile);
    if (shape.sharedExpert) {
      graph.add("moe_route_select_f32",
                {scratch.groupedInput, buffers.input,
                 weights.blocks().sharedScalarGate.plane0, scratch.selectedExperts,
                 scratch.routingWeights},
                routeParams, {rows, 1, 1});
    } else if (fusedRoute) {
      expertBias = weights.blocks().expertBias.plane0;
    } else {
      // Sigmoid gating with the per-expert selection bias; no shared expert.
      graph.add("moe_route_select_sigmoid",
                {scratch.groupedInput, weights.blocks().expertBias.plane0,
                 scratch.selectedExperts, scratch.routingWeights},
                routeParams, {rows, 1, 1});
    }
  } else {
    const AffineMoeWeights &affine = weights.affine();
    const MoeRouteTile route = moeRouteTile(rows, plan.configuration().routeWideRows);
    graph.add(route.rows == 8 ? "moe_route_scores_q8_m8"
                              : "moe_route_scores_q8_m32",
              {buffers.input, affine.router.planes.weights, affine.router.planes.scales,
               affine.router.planes.biases, scratch.groupedInput},
              routeParams,
              {(rows + route.rows - 1) / route.rows, 256 / route.experts, 1});
    if (shape.sharedExpert) {
      graph.add("moe_route_select_q8",
                {scratch.groupedInput, buffers.input,
                 affine.sharedScalarGate.planes.weights,
                 affine.sharedScalarGate.planes.scales,
                 affine.sharedScalarGate.planes.biases, scratch.selectedExperts,
                 scratch.routingWeights},
                routeParams, {rows, 1, 1});
    } else if (fusedRoute) {
      expertBias = affine.expertBias;
    } else {
      graph.add("moe_route_select_sigmoid",
                {scratch.groupedInput, affine.expertBias,
                 scratch.selectedExperts, scratch.routingWeights},
                routeParams, {rows, 1, 1});
    }
  }
  if (fusedRoute) {
    // The scores kernel's output feeds the fused select-and-group.
    graph.add("moe_route_group_sigmoid",
              {scratch.groupedInput, expertBias, scratch.selectedExperts,
               scratch.routingWeights, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
              MoeRouteGroupParams{routeParams, groupParams}, {1, 1, 1});
  } else {
    graph.add("moe_group_routes",
              {scratch.selectedExperts, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedRoutes, scratch.routeRows},
              groupParams, {1, 1, 1});
  }
  // The packed decode (packsDecode's MoE analog): a decode plan on the
  // `_n` kernels whose weights hold at least one MXFP4 expert segment
  // gathers the packed fp16 plane and exponent bytes alongside the bf16
  // rows, and its expert passes run the `_p` kernels — MXFP4 segments on
  // the packed multiplane tile, the rest staged as before.
  const auto mxfp4Expert = [](const BlockExpertProjection &p) {
    return p.routed.formatId == GGUF_FMT_MXFP4 || p.shared.formatId == GGUF_FMT_MXFP4;
  };
  const bool packed =
      block && plan.phase() == MoePhase::Decode &&
      plan.configuration().mxfp4Native &&
      plan.configuration().ggufTile != MoeGgufTile::Register &&
      !packedDisabled() &&
      (mxfp4Expert(weights.blocks().gate) || mxfp4Expert(weights.blocks().up) ||
       mxfp4Expert(weights.blocks().down));
  const MoeGatherParams gather{tileRows, shape.hiddenSize, shape.routesPerToken()};
  if (plan.configuration().ggufTile == MoeGgufTile::Register)
    graph.add("moe_gather_table16",
              {buffers.input, scratch.groupedRoutes, scratch.tileCount,
               scratch.groupedInput, scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else if (packed)
    graph.add("moe_gather_packed",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput, scratch.groupedPacked,
               scratch.groupedSums},
              gather, {tiles, shape.hiddenSize / 256, 1});
  else
    graph.add("moe_gather_rows",
              {buffers.input, scratch.groupedRoutes, scratch.tileDescriptors,
               scratch.tileCount, scratch.groupedInput},
              gather, {tiles, shape.hiddenSize / 256, 1});
  // The GGUF expert kernels produce no output when they replay from an
  // indirect command buffer on this driver (gguf-moe's staged and register
  // decodes both): suspend their dispatches — a suspension also lifts them
  // out of an enclosing span (the verify FFN's), which an inner end could
  // not. Affine experts replay fine.
  if (bakeable && block) graph.suspendBakedSpan();
  if (block)
    addGgufExperts(graph, scratch, weights.blocks(), plan, packed);
  else
    addAffineExperts(graph, scratch, weights.affine(), plan);
  if (bakeable && block) graph.resumeBakedSpan();
  graph.add("moe_combine",
            {scratch.expertOutput, scratch.routeRows, scratch.routingWeights,
             buffers.residual, buffers.output},
            MoeCombineParams{rows, shape.hiddenSize, shape.routesPerToken()},
            {rows, shape.hiddenSize / 256, 1});
  if (bakeable) graph.endBakedSpan();
}

MoePlan MoE::prefillPlan(MoeShape shape, uint32_t rows, MoeConfig config) {
  if (!rows || rows > SPLASH_PREFILL_TOKEN_BUDGET)
    throw std::invalid_argument("invalid MoE prefill rows");
  return MoePlan(shape, rows, config, MoePhase::Prefill);
}

MoePlan MoE::decodePlan(MoeShape shape, uint32_t lanes, MoeConfig config) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid MoE decode batch width");
  return MoePlan(shape, lanes * SPLASH_TARGET_VERIFY_ROWS, config, MoePhase::Decode);
}

} // namespace splash::ops
