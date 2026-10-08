#include "Linear.hpp"
#include "Tuning.hpp"
#include "ops/DeviceTuning.hpp"
#include "ops/KernelNames.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace richengine::ops {
namespace {

constexpr uint32_t kAffinePrefillTileRows = 32;
constexpr uint32_t kQuantGroup = 64;
// The decode activation table's fixed eight-row tile (the q4sg table tile and
// the GGUF Table16 span), independent of a lane's verify row count.
constexpr uint32_t kDecodeActivationRows = 8;
static_assert(!(RICHENGINE_TARGET_VERIFY_ROWS % kDecodeActivationRows),
              "a lane's verify rows must tile the eight-row decode tiles");
// The kernels sum their input per block of four quant groups (256 inputs);
// Split128 partitions K in whole blocks.
constexpr uint32_t kInputSumBlock = 4 * kQuantGroup;

bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256;
}
// The decode tiles whose threadgroups stream several column tiles.
bool persistentTile(LinearTile tile) noexcept {
  return tile == LinearTile::N128 || tile == LinearTile::N256 || tile == LinearTile::Paired128 ||
         tile == LinearTile::Paired256;
}
// The tiles of block (GGUF) projections.
bool blockTile(LinearTile tile) noexcept {
  return tile == LinearTile::GgufStaged || tile == LinearTile::GgufPrefill || tile == LinearTile::GgufRegister;
}
// Simdgroups fixed by the kernel instance: the paired N256 tile runs four and
// Split128 the N128 tile's eight.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Simdgroup:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split128: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > RICHENGINE_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % kDecodeActivationRows ||
        w.rows > RICHENGINE_TARGET_VERIFY_ROWS * RICHENGINE_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid linear decode workload");
  }
}

// A buffer a plan does not use needs no bytes and may be absent.
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument(std::string("projection ") + what + " buffer holds " +
                                std::to_string(buffer.sizeBytes()) + " bytes, needs " + std::to_string(bytes));
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > RICHENGINE_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * RICHENGINE_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// LinearScratch::rotated bytes of `rows` bf16 rows of `width` inputs.
constexpr uint64_t rotatedBytes(uint32_t width, uint64_t rows) noexcept { return uint64_t{width} * rows * 2; }

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 N128
// plain and residual projections, all matrix row tiles, and the one-lane
// Paired256 (plain) tile.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Simdgroup) return w.phase == LinearPhase::Decode;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  if (tile == LinearTile::Paired256)
    return w.phase == LinearPhase::Decode && w.rows == kDecodeActivationRows &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      (w.rows == 24 && (w.epilogue == LinearEpilogue::None ||
                        w.epilogue == LinearEpilogue::Residual));
}

// Throws if `p` is a view of the leading inputs of wider weight rows
// (Projection::leadingInputs) that `plan` has no kernel instance for: only the
// prefill residual tiles of affine Q4 weights (N128, N256) and of quantized
// GGUF segments (GgufPrefill), unrotated, read one (leadingInputsInstance).
void requireLeadingInputs(const LinearPlan &plan, const Projection &p) {
  if (!p.planeInputs()) return;
  const LinearWorkload w = plan.workload();
  const LinearTile tile = plan.configuration().tile;
  if (w.phase != LinearPhase::Prefill || w.epilogue != LinearEpilogue::Residual ||
      (tile != LinearTile::N128 && tile != LinearTile::N256 && tile != LinearTile::GgufPrefill) || p.rotation)
    throw std::invalid_argument("a view of leading inputs runs only the quantized prefill residual tiles");
}

// Throws unless views of `p`'s planes can stand for it (takesPlaneViews).
void requirePlaneViews(const Projection &p) {
  if (p.takesPlaneViews()) return;
  throw std::invalid_argument(
      "views of a projection's planes take affine Q4 weights or one unrotated quantized GGUF tensor, not a view");
}

} // namespace

// Every plane of either layout holds its rows in tiles of QUANT_TILE_ROWS
// rows, each tile's units in order, so the leading rows' tiles lead it.
static_assert(RICHENGINE_AFFINE_TILE_ROWS == QUANT_TILE_ROWS, "affine Q4 and GGUF planes share their tiles");

bool Projection::takesPlaneViews() const noexcept {
  if (planeInputs_) return false;
  if (layout() == WeightLayout::Affine64) return true;
  const std::vector<QuantizedSegment> &segments = blocks().segments;
  return segments.size() == 1 && !segments.front().isFloat() && !rotation;
}

Projection Projection::leadingRows(const metal::MetalBackend &backend, uint32_t rows) const {
  requirePlaneViews(*this);
  if (!rows || rows % QUANT_TILE_ROWS || rows > outputSize)
    throw std::invalid_argument("a view of leading rows takes whole plane tiles of the projection's rows");
  // The leading rows of a plane of `rowBytes` bytes per row.
  const auto view = [&](const metal::MetalBuffer &plane, uint64_t rowBytes) {
    return rowBytes ? backend.view(plane, 0, rows * rowBytes) : metal::MetalBuffer{};
  };
  if (layout() == WeightLayout::Affine64) {
    const uint64_t groups = inputSize / kQuantGroup;
    const AffineWeights &planes = affine();
    return Projection(rows, inputSize,
                      AffineWeights{view(planes.weights, groups * kQuantGroup / 2), view(planes.scales, groups * 2),
                                    view(planes.biases, groups * 2)});
  }
  const QuantizedSegment &segment = blocks().segments.front();
  const QuantFormat &format = segment.format();
  const uint64_t groups = inputSize / 32;
  return Projection(rows, inputSize,
                    BlockWeights{{QuantizedSegment::planes(segment.formatId, rows, inputSize,
                                                           view(segment.plane0, groups * format.plane0_bytes),
                                                           view(segment.plane1, groups * format.plane1_bytes),
                                                           view(segment.meta,
                                                                groups / format.meta_groups * format.meta_bytes))}});
}

Projection Projection::leadingInputs(uint32_t inputs) const {
  requirePlaneViews(*this);
  const bool affineWeights = layout() == WeightLayout::Affine64;
  const uint32_t unit = affineWeights ? kQuantGroup : 32 * blocks().segments.front().format().meta_groups;
  if (!inputs || inputs > inputSize || inputs % unit)
    throw std::invalid_argument("a view of leading inputs takes whole quant groups and meta units of the projection's");
  const auto view = [&] {
    if (affineWeights) return Projection(outputSize, inputs, affine());
    const QuantizedSegment &segment = blocks().segments.front();
    return Projection(outputSize, inputs,
                      BlockWeights{{QuantizedSegment::planes(segment.formatId, outputSize, inputs, segment.plane0,
                                                             segment.plane1, segment.meta)}});
  };
  Projection result = view();
  result.planeInputs_ = inputSize;
  return result;
}

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / kDecodeActivationRows * table16_sums_per_tile(width) * sizeof(float)
       : layout == LinearInput::Table64 ? uint64_t{width} * rows / 16
       : layout == LinearInput::Packed ? rows * (width / 32) : 0;
}

void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows) {
  if (layout == LinearInput::Plain || !rows || rows % kDecodeActivationRows || width % 64 ||
      scratch.input.sizeBytes() < tableBytes(width, rows) ||
      scratch.sums.sizeBytes() < tableSumsBytes(layout, width, rows))
    throw std::invalid_argument("linear table scratch is below requirement");
}

const char *tableSuffix(LinearInput layout) noexcept {
  return layout == LinearInput::Table16 ? "_table16"
       : layout == LinearInput::Table64 ? "_table64"
       : layout == LinearInput::Packed  ? "_packed" : "";
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  // Each plane holds a scale, a bias or 64 weights per row and group, in
  // tiles of RICHENGINE_AFFINE_TILE_ROWS rows, each tile's groups in order. It
  // ends at the last group the projection reads of its last tile: a view of
  // leading inputs leaves the tile's later groups unread.
  const uint64_t groups = matrix.inputSize / kQuantGroup, rowGroups = p.planeInputSize() / kQuantGroup;
  const uint64_t parameters =
      uint64_t{matrix.outputSize} * rowGroups - RICHENGINE_AFFINE_TILE_ROWS * (rowGroups - groups);
  requireBytes(p.affine().weights, parameters * kQuantGroup / 2, "projection weight");
  requireBytes(p.affine().scales, parameters * 2, "projection scale");
  requireBytes(p.affine().biases, parameters * 2, "projection bias");
}

uint32_t LinearPlan::storageRows() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockStorageRows();
  if (workload_.phase != LinearPhase::Prefill) return workload_.rows;
  return ((workload_.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows) * kAffinePrefillTileRows;
}
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::GgufStaged:
  case LinearTile::GgufPrefill:
  case LinearTile::GgufRegister: return GGUF_TILE_COLUMNS;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::groups() const noexcept {
  return config_.groups ? config_.groups : workload_.matrix.outputSize / tileColumns();
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  switch (config_.tile) {
  case LinearTile::GgufStaged: return GGUF_STAGED_THREADS;
  case LinearTile::GgufPrefill: return GGUF_PREFILL_THREADS;
  case LinearTile::GgufRegister: return GGUF_REGISTER_THREADS;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split128:
  case LinearTile::Paired256:
  case LinearTile::Simdgroup: return static_cast<uint32_t>(config_.simdgroups) * 32;
  }
  return 0;
}
bool LinearPlan::usesSimdgroup() const noexcept { return config_.tile == LinearTile::Simdgroup; }
LinearInput LinearPlan::input() const noexcept {
  if (rotated_) return LinearInput::Plain;
  if (packedInput_) return LinearInput::Packed;
  if (config_.tile == LinearTile::GgufRegister) return LinearInput::Table16;
  return usesSimdgroup() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockScratchSize();
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  if (!usesSimdgroup()) return {};
  const uint64_t rows = workload_.rows;
  const uint64_t lanes = rows / kDecodeActivationRows;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile. Single-partition kernels use neither.
  return {tableBytes(k, workload_.rows), tableSumsBytes(LinearInput::Table64, k, workload_.rows),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : sizeof(float),
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : sizeof(uint32_t)};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // GGUF tiles run gate/up as a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp &&
       (!secondPipeline_.empty() || workload_.weightLayout == WeightLayout::Block32));
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate && workload_.weightLayout == WeightLayout::Affine64
      ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  const bool ggufTile = blockTile(config.tile);
  if (ggufTile != (w.weightLayout == WeightLayout::Block32))
    throw std::invalid_argument("block projections run the GGUF tiles, affine ones the Q4 tiles");
  if (w.phase == LinearPhase::Decode && persistentTile(config.tile)
          ? !config.groups || config.groups > w.matrix.outputSize / tileColumns()
          : config.groups != 0)
    throw std::invalid_argument("a persistent decode tile takes 1 to its column tiles in groups, every other plan 0");
  if (ggufTile) {
    // Kernel names follow the segment formats (LinearGguf.cpp).
    requireBlockConfiguration();
    return;
  }
  const bool splitsK = config.tile == LinearTile::Simdgroup || config.tile == LinearTile::Split128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the simdgroup or Split128 Q4 tile");
  if (config.simdgroups == LinearSimdgroups::Four && !supportsFourSimdgroups(w, config.tile))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  if (w.phase == LinearPhase::Prefill) {
    if (oneLaneTile(config.tile) || splitsK)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? kPrefillLinearQ4N128UpSiluSumsSg4
          : residual ? kPrefillLinearQ4N128ResidualSg4 : kPrefillLinearQ4N128Sg4;
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = kPrefillLinearQ4N256UpSiluSums;
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? kPrefillLinearQ4N128Residual : kPrefillLinearQ4N256Residual;
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? kPrefillLinearQ4N128 : kPrefillLinearQ4N256;
    }
    return;
  }
  // The row-tile index picks the fused-M kernel variant; verify rows are
  // always a multiple of sixteen, so only even tile counts dispatch, but the
  // tables carry a nearest entry at the unreachable odd indexes.
  const uint32_t tiles = w.rows / kDecodeActivationRows;
  const uint32_t lane = tiles - 1;
  if (lane > 7 && !usesSimdgroup())
    throw std::invalid_argument("decode row tiles exceed the fused variants");
  if (oneLaneTile(config.tile) && w.rows != kDecodeActivationRows)
    throw std::invalid_argument("paired Q4 tile requires one lane");
  if (usesSimdgroup()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (!config.validSplits() || groups % config.splits)
      throw std::invalid_argument("simdgroup Q4 requires whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? kDecodeLinearQ4SgGateUp :
        residual ? kDecodeLinearQ4SgResidual : kDecodeLinearQ4Sg;
    return;
  }
  if (config.tile == LinearTile::Split128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (!config.validSplits() || config.splits < 2 ||
        w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("Split128 requires 2, 4 or 8 K partitions of 256-input blocks");
    constexpr std::array plainNames{kDecodeLinearQ4N128Split, kDecodeLinearQ4N128SplitM16,
        kDecodeLinearQ4N128SplitM24, kDecodeLinearQ4N128SplitM32,
        kDecodeLinearQ4N128SplitM48, kDecodeLinearQ4N128SplitM48,
        kDecodeLinearQ4N128SplitM64, kDecodeLinearQ4N128SplitM64};
    constexpr std::array residualNames{kDecodeLinearQ4N128SplitResidual,
        kDecodeLinearQ4N128SplitResidualM16, kDecodeLinearQ4N128SplitResidualM24,
        kDecodeLinearQ4N128SplitResidualM32, kDecodeLinearQ4N128SplitResidualM48,
        kDecodeLinearQ4N128SplitResidualM48, kDecodeLinearQ4N128SplitResidualM64,
        kDecodeLinearQ4N128SplitResidualM64};
    constexpr std::array upSiluNames{kDecodeLinearQ4N128SplitUpSilu,
        kDecodeLinearQ4N128SplitUpSiluM16, kDecodeLinearQ4N128SplitUpSiluM24,
        kDecodeLinearQ4N128SplitUpSiluM32, kDecodeLinearQ4N128SplitUpSiluM48,
        kDecodeLinearQ4N128SplitUpSiluM48, kDecodeLinearQ4N128SplitUpSiluM64,
        kDecodeLinearQ4N128SplitUpSiluM64};
    // Gate/up at every lane count: a plain gate pass into the gate scratch,
    // then the up pass whose epilogue applies the SiLU gate.
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    if (w.epilogue == LinearEpilogue::GateUp) secondPipeline_ = upSiluNames[lane];
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = kDecodeLinearQ4N256PairedSg4;
    return;
  }
  if (four) {
    pipeline_ = residual ? kDecodeLinearQ4N128ResidualM24Sg4
                         : kDecodeLinearQ4N128M24Sg4;
    return;
  }
  // Verify rows are a multiple of sixteen, so only the even indexes of the
  // fused-M tables dispatch; the odd ones hold their nearest kernel.
  const bool sixteen = config.simdgroups == LinearSimdgroups::Sixteen;
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    if (sixteen && lane != 7)
      throw std::invalid_argument("the 16-simdgroup decode tile exists only at 64 rows");
    constexpr std::array names{kDecodeLinearQ4N256GateUp, kDecodeLinearQ4N256GateUpM16,
        kDecodeLinearQ4N256M24, kDecodeLinearQ4N256M32, kDecodeLinearQ4N256M48,
        kDecodeLinearQ4N256M48, kDecodeLinearQ4N256M64, kDecodeLinearQ4N256M64};
    constexpr std::array upSiluNames{kDecodeLinearQ4N256, kDecodeLinearQ4N256M16,
        kDecodeLinearQ4N256UpSiluM24, kDecodeLinearQ4N256UpSiluM32,
        kDecodeLinearQ4N256UpSiluM48, kDecodeLinearQ4N256UpSiluM48,
        kDecodeLinearQ4N256UpSiluM64, kDecodeLinearQ4N256UpSiluM64};
    pipeline_ = sixteen ? std::string_view{kDecodeLinearQ4N256M64Sg16}
                        : names[lane];
    if (lane >= 2)
      secondPipeline_ = sixteen ? std::string_view{kDecodeLinearQ4N256UpSiluM64Sg16}
                                : upSiluNames[lane];
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{kDecodeLinearQ4N128Residual, kDecodeLinearQ4N128ResidualM16,
        kDecodeLinearQ4N128ResidualM24, kDecodeLinearQ4N128ResidualM32,
        kDecodeLinearQ4N128ResidualM48, kDecodeLinearQ4N128ResidualM48,
        kDecodeLinearQ4N128ResidualM64, kDecodeLinearQ4N128ResidualM64};
    pipeline_ = config.tile == LinearTile::Paired128
        ? kDecodeLinearQ4N128ResidualPaired : names[lane];
  } else if (config.tile == LinearTile::N256) {
    if (sixteen && lane != 7)
      throw std::invalid_argument("the 16-simdgroup decode tile exists only at 64 rows");
    constexpr std::array names{kDecodeLinearQ4N256, kDecodeLinearQ4N256M16,
        kDecodeLinearQ4N256M24, kDecodeLinearQ4N256M32, kDecodeLinearQ4N256M48,
        kDecodeLinearQ4N256M48, kDecodeLinearQ4N256M64, kDecodeLinearQ4N256M64};
    pipeline_ = sixteen ? std::string_view{kDecodeLinearQ4N256M64Sg16}
                        : names[lane];
  } else {
    constexpr std::array names{kDecodeLinearQ4N128, kDecodeLinearQ4N128M16,
        kDecodeLinearQ4N128M24, kDecodeLinearQ4N128M32, kDecodeLinearQ4N128M48,
        kDecodeLinearQ4N128M48, kDecodeLinearQ4N128M64, kDecodeLinearQ4N128M64};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? kDecodeLinearQ4N128Paired : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one resident wave for
// longer chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Resident groups per core: one wave for this kernel's register footprint.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// Resident-wave and full-grid thresholds measured on 16/20-core Apple10 GPUs.
// Gate/up uses the conservative limit shared by both devices. Its many-wave
// threshold follows N256; the four-simdgroup threshold scales from N128. Those
// two extrapolations remain unmeasured.
constexpr DecodeGroupPolicy kN128Groups{4, 4, 12}, kN128M16Groups{5, 4, 12},
    kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};
// Apple9 retains its measured gate/up clamp. The round-robin policy above was
// measured on Apple10; applying it to Apple9 requires separate calibration.
constexpr double kApple9GateUpGroupsPerCore = 2.25;

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`;
  // the bound fails a policy that breaks that instead of walking past the
  // tile count, where `maxCoreTiles` no longer applies.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (groups <= tiles && maxCoreTiles(tiles, groups, cores) != balanced)
    ++groups;
  if (groups > tiles)
    throw std::invalid_argument(
        "decode group policy found no balanced group count within the tile count");
  return groups;
}
// A multi-row N256 decode tile halves the input re-reads of N128 but also
// halves the grid; it pays only while the N256 grid keeps two tiles per core.
constexpr uint32_t kWideDecodeTilesPerCore = 2;
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Paired-A/B tuning (tune-kernels) and the per-shape microprofile
// (benchmark-prefill) on a 32-core Apple9 GPU (M4 Max) measured the
// four-simdgroup N128 tile ahead of N256 on every prefill shape and probed
// row count: +6..10% GPU wherever the margin cleared the tuning threshold,
// never behind. Apple9 GPUs at or below that measured core count therefore
// share the Apple10 prefill rule. Larger Apple9 GPUs (40-core class) keep the
// wide-tile rule below; it was sized for them and remains unremeasured there.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;

// Apple10 and later decode split K across the threadgroups of the Split128
// tile (256 threads) by one rule at every batch width: the largest power of
// two up to LinearConfig::kMaximumSplits whose split grid still fits four
// threadgroups per core (1024 threads, twice the 512-thread occupancy knee),
// with at least one 256-input block per partition. A grid of more than two
// tiles per core keeps one split, the sequential tiles. Measured DRAM-cold on
// a 20-core M5 Pro over the 27B and 35B MLX 4-bit decode projections and their
// drafts at one to four lanes, with 10 to 80 cores emulated by width, and on a
// 12-core M6 (Apple11): grids that only reach the knee leave time (the 20
// tiles of 2560 x 4096 take 0.48-0.60 of the sequential time with four
// splits, 0.60-0.71 with two), and a grid past four per core loses to its
// second wave (6144 x 5120 in two splits is 9% slower at one lane). The rule is
// never slower than the sequential tiles on the M5 Pro or on the M6's own 12
// cores (20 cores emulated on the M6 ran 5120 x 4096 4% slower at one lane).
constexpr uint32_t kSplitGroupsPerCore = 4;

uint32_t apple10Splits(LinearMatrix matrix, uint32_t cores) noexcept {
  const uint64_t grid = matrix.outputSize / 128;
  uint32_t splits = 1;
  while (splits < LinearConfig::kMaximumSplits && grid * 2 * splits <= uint64_t{kSplitGroupsPerCore} * cores &&
         2 * splits <= matrix.inputSize / kInputSumBlock)
    splits *= 2;
  return splits;
}

// Apple10 wide plain projections reduce input re-reads with paired N256
// tiles at one resident wave. The crossover was measured at three tiles per
// core on the 20-core M5 Pro (tune-kernels, 2026-09-23). Split-K remains
// an offline candidate: its reassociation reduced speculative acceptance
// on some measured prompts. Apple9's simdgroup policy is independent.
constexpr uint32_t kPaired256TilesPerCore = 3;
constexpr uint32_t kPaired256WaveGroupsPerCore = 4;

std::optional<LinearConfig> apple10OneLaneConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t n = w.matrix.outputSize;
  const uint32_t tiles256 = n / 256;
  if (w.epilogue == LinearEpilogue::None && tiles256 >= kPaired256TilesPerCore * cores)
    return LinearConfig{LinearTile::Paired256,
                        std::min(tiles256, kPaired256WaveGroupsPerCore * cores),
                        LinearSimdgroups::Four};
  return std::nullopt;
}

} // namespace

Linear::Linear(const DeviceCapabilities &device) noexcept : Linear(DevicePolicy::of(device)) {}
Linear::Linear(const DevicePolicy &policy) noexcept : policy_(policy) {}

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  return plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, LinearEpilogue::None, shape.layout})
      .storageRows();
}

// GPU family selects variants; core count and workload tile counts determine
// parallelism.
LinearConfig Linear::baseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  validate(w);
  if (w.weightLayout == WeightLayout::Block32) return ggufBaseline(w, projections);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  if (w.phase == LinearPhase::Prefill) {
    // w.canvas marks a DiffusionGemma trunk pass: it shares this selection
    // until a canvas-only tile measures better (see moe-prefill-bench).
    // Measured per-shape plans (DeviceTuning.cpp) scope themselves to the
    // devices they name.
    if (const auto measured = measuredLinearPlan(policy_, w)) return *measured;
    if (policy_.apple10Plus() || policy_.cores <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * policy_.cores;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  const uint32_t lanes = w.rows / kDecodeActivationRows;
  // Keep the existing broad-column plain projection path for wider batches:
  // independent row tiles repeat its weight stream. Reuse the existing
  // two-N256-tiles-per-core boundary rather than model-specific dimensions.
  const bool widePlain = lanes >= 3 && w.epilogue == LinearEpilogue::None &&
      tiles256 >= kWideDecodeTilesPerCore * policy_.cores;
  if (policy_.isApple9() && !widePlain) {
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    const uint32_t grid = w.matrix.outputSize / columns, groups = w.matrix.inputSize / 64;
    uint32_t splits = 1;
    // Aim for sixteen independent column/K groups per core, retaining at
    // least twelve quant groups per partition to amortize the reduction.
    while (splits < LinearConfig::kMaximumSplits && uint64_t(grid) * splits < 16ULL * policy_.cores &&
           groups % (2 * splits) == 0 && groups / (2 * splits) >= 12)
      splits *= 2;
    return {LinearTile::Simdgroup, 0, LinearSimdgroups::Four, splits};
  }
  // The fused-M kernels top out at eight eight-row tiles; wider decode
  // batches run the z-tiled simdgroup kernel, which covers any tile count.
  if (lanes > 8)
    return {LinearTile::Simdgroup, 0, LinearSimdgroups::Four, 1};
  if (const auto measured = measuredLinearPlan(policy_, w)) return *measured;
  if (policy_.apple10Plus()) {
    if (const uint32_t splits = apple10Splits(w.matrix, policy_.cores); splits > 1)
      return {LinearTile::Split128, 0, LinearSimdgroups::Eight, splits};
    if (lanes == 1)
      if (const auto config = apple10OneLaneConfig(w, policy_.cores)) return *config;
  }
  // Apple9 reaches here only for wide plain projections of three or four
  // lanes, which keep their one-tile grids: the round-robin policy above was
  // measured on Apple10.
  const auto groups = [&](uint32_t tiles, DecodeGroupPolicy policy) {
    return policy_.apple10Plus() ? decodeGroups(tiles, policy_.cores, policy)
                                 : tiles;
  };
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (!policy_.apple10Plus()) {
      const auto resident = static_cast<uint32_t>(
          std::max(1L, std::lround(kApple9GateUpGroupsPerCore * policy_.cores)));
      return {LinearTile::N256, std::min(tiles256, resident)};
    }
    // One- and two-lane gate/up keeps the whole N256 grid resident while it
    // fits four tiles per core; the balanced wave measured slower on the
    // 20-core M5 Pro.
    if (lanes <= 2 && tiles256 <= 4 * policy_.cores)
      return {LinearTile::N256, tiles256};
    // The 64-row N256 tile spills its accumulators at eight simdgroups; the
    // 512-thread variant halves them back to the m32 tile's footprint.
    if (lanes == 8)
      return {LinearTile::N256, groups(tiles256, kGateUpGroups),
              LinearSimdgroups::Sixteen};
    return {LinearTile::N256, groups(tiles256, kGateUpGroups)};
  }
  // Pipelined N128 hides the latency of a single lane's weight stream.
  if (lanes == 1) return {LinearTile::Paired128, groups(tiles128, kN128Groups)};
  // Every M24 projection that gets here runs four SIMD groups.
  if (lanes == 3)
    return {LinearTile::N128, groups(tiles128, kFourSimdgroupGroups),
            LinearSimdgroups::Four};
  if (widePlain)
    return lanes == 8 ? LinearConfig{LinearTile::N256, groups(tiles256, kN256Groups),
                                     LinearSimdgroups::Sixteen}
                      : LinearConfig{LinearTile::N256, groups(tiles256, kN256Groups)};
  return {LinearTile::N128,
          groups(tiles128, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

namespace {

// The pre-packed MXFP4 prefill tiles (gguf_prefill_mxfp4p_* /
// gguf_decode_mxfp4p_*, LinearGguf.cpp) measured behind the staged tiles on
// the compute-bound prefill GEMM (mxfp4 5120 x 8192, 20-core M5 Pro, ms
// at rows 128/512/2048: 0.52/2.05/7.88 packed vs 0.51/1.86/6.96 staged), so
// they stay for benchmarks: RICHENGINE_GGUF_PACKED_ON selects them, the same
// opt-in the packed-operand decode kernels take (LinearGguf.cpp). The flag
// covers every chunk size — and must: the gguf_projection test compares
// every chunk's output bitwise against the 128-row tile's, and the fp4
// multiplane matmul of the packed chunk tiles is NOT bitwise equal to the
// staged mxfp4n prefill tile (a deterministic one-ulp divergence on rare
// elements: 1 of 32768 at 32 rows, observed at ~4e-5 magnitudes where the
// pack's exponent path engages). Equal within fp64, not bitwise.
bool packedPrefillEnabled() {
  static const bool on = tuning().ggufPackedOn;
  return on;
}
// A kill switch for the packed paths, for benchmarks and triage.
bool packedTilesDisabled() {
  static const bool off = tuning().ggufPackedOff;
  return off;
}
// A GGUF prefill plan may dispatch gguf_pack_half and the pre-packed MXFP4
// tiles (LinearGguf.cpp), which read LinearScratch::input and ::sums. The
// flag is set conservatively wherever the segments are unknown (the
// projection-free plan and the explicit-config plan): a non-MXFP4 plan then
// reserves the pack scratch without using it.
// RICHENGINE_PREFILL_FAST_INT8: opt-in two-term uint8 activation split driving
// the integer NA path (prefill_linear_i8_*). NOT bit-identical to the bf16
// tensor-op path — greedy output may diverge on near-ties.
bool fastInt8Prefill() {
  static const bool on = tuning().prefillFastInt8;
  return on;
}

bool packsPrefill(const DevicePolicy &device, LinearWorkload w) noexcept {
  return packedPrefillEnabled() && !packedTilesDisabled() &&
         device.nativeFormats() && w.phase == LinearPhase::Prefill &&
         w.weightLayout == WeightLayout::Block32;
}
bool hasMxfp4(const Projection *p) noexcept {
  return p && std::any_of(p->blocks().segments.begin(), p->blocks().segments.end(),
                          [](const QuantizedSegment &s) { return s.formatId == GGUF_FMT_MXFP4; });
}

// A GGUF decode plan packs its activations the same way when any of its
// projections is MXFP4: the pre-packed tile (gguf_decode_mxfp4p_*) wins at
// every tile height, so decode dispatch selects it unconditionally, unlike
// the env-gated prefill variant (LinearGguf.cpp's addGgufStaged).
bool packsDecode(const DevicePolicy &device, LinearWorkload w, const Projection *p,
                 const Projection *gate) noexcept {
  return device.nativeFormats() && !packedTilesDisabled() &&
         w.phase == LinearPhase::Decode &&
         w.weightLayout == WeightLayout::Block32 && (hasMxfp4(p) || hasMxfp4(gate));
}

} // namespace

LinearPlan Linear::plan(LinearWorkload workload) const {
  LinearPlan plan(workload, baseline(workload));
  plan.packs_ = packsPrefill(policy_, workload);
  return plan;
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  LinearPlan plan(workload, config, destination);
  // Any native-formats family, so forced plans size the pack scratch.
  plan.packs_ = packsPrefill(DevicePolicy{.family = 10}, workload);
  return plan;
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.weightLayout = p.layout();
  const std::array<const Projection *, 2> projections{&p, gate};
  LinearPlan plan(w, baseline(w, projections), p.destination);
  plan.rotated_ = static_cast<bool>(p.rotation);
  plan.packs_ = (packsPrefill(policy_, w) && (hasMxfp4(&p) || hasMxfp4(gate))) ||
                packsDecode(policy_, w, &p, gate);
  // A fused projection stages its input inside the decode kernel; only the
  // single-tensor path (a gate/up pair is two single-tensor dispatches on one
  // input, and a prefill chunk one dispatch per segment) dispatches or reads
  // a packed operand, so only it can take a packed input. A prefill chunk of
  // up to a decode batch's rows runs the same mxfp4p decode tiles as decode,
  // so its producer can emit the packed operand the same way; the 128-row
  // prefill tile still packs its own input.
  const auto singleTensor = [](const Projection *q) {
    return !q || q->blocks().segments.size() == 1;
  };
  plan.packedInput_ = plan.packs_ &&
                      (w.phase == LinearPhase::Decode || w.rows <= kMaximumDecodeTileRows) &&
                      singleTensor(&p) && singleTensor(gate);
  return plan;
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue,
                               bool canvas) const {
  return plan({{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue,
               p.layout(), canvas}, p);
}

LinearScratchSize Linear::decodeScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t lanes = 1; lanes <= RICHENGINE_MAXIMUM_BATCH_WIDTH; ++lanes)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
      LinearWorkload w = decode({shape.outputSize, shape.inputSize}, lanes, epilogue);
      w.weightLayout = shape.layout;
      bound.include(shape.layout == WeightLayout::Block32 ? ggufDecodeScratchSize(w) : plan(w).scratchSize());
    }
  // Decode plans store at most every lane's rows.
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, kMaximumDecodeTileRows);
  return bound;
}

LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  // Prefill plans take split scratch only in chunks of up to a decode batch,
  // which a GGUF projection runs on the staged tile (LinearGguf.cpp).
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout})
                        .scratchSize());
  // A full chunk's plan, whose GGUF prefill tile packs the activations its
  // MXFP4 segments multiply (the scratch LinearScratch::input and ::sums).
  bound.include(plan({{shape.outputSize, shape.inputSize}, RICHENGINE_PREFILL_TOKEN_BUDGET,
                      LinearPhase::Prefill, LinearEpilogue::None, shape.layout})
                    .scratchSize());
  // Every prefill plan stores at most the token budget.
  static_assert(RICHENGINE_PREFILL_TOKEN_BUDGET % GGUF_PREFILL_ROWS == 0, "the prefill tiles cover the budget exactly");
  if (shape.rotated) bound.rotated = rotatedBytes(shape.inputSize, RICHENGINE_PREFILL_TOKEN_BUDGET);
  return bound;
}


PreparedInput Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Projection &p, const LinearPlan &selected, const Projection *gate) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if (p.layout() != w.weightLayout || (gate && gate->layout() != w.weightLayout))
    throw std::invalid_argument("projection layout does not match execution plan");
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  requireLeadingInputs(selected, p);
  if (gate) requireLeadingInputs(selected, *gate);
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "residual");
  requireBytes(b.sums, selected.sumsBytes(), "sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "partials");
  requireBytes(b.scratch.counters, scratch.counters, "counters");
  if (p.layout() == WeightLayout::Block32) {
    if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "rotated input");
    // The GGUF decode kernels produce no output when they replay from an
    // indirect command buffer on this driver (the gguf MoE expert and
    // embedding kernels fail the same way): a suspension keeps them out of
    // every enclosing baked span.
    graph.suspendBakedSpan();
    addGguf(graph, b, p, selected, gate);
    graph.resumeBakedSpan();
    // Only quantized segments run the plan's tile: float segments alone
    // leave the scratch table as it was.
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    const bool tiled =
        std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
    // A rotated projection's plan prepares its table, if any, from the
    // rotated rows, which no other plan reads — but scratch.rotated still
    // holds H (D input), which another projection of the same input and
    // signs reuses.
    if (p.rotation) return tiled ? PreparedInput{b.input, LinearInput::Rotated, p.rotation.signs} : PreparedInput{};
    return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
  }
  requireAffineProjection(p, w.matrix);
  if (gate) requireAffineProjection(*gate, w.matrix);
  const AffineWeights &weights = p.affine();
  if (fastInt8Prefill() && w.phase == LinearPhase::Prefill &&
      w.epilogue != LinearEpilogue::GateUp && k % 64 == 0 && n % 256 == 0) {
    // Integer NA path: quantize the chunk's input into the two-term uint8
    // split, then run the int32-accumulating tile of its epilogue.
    const uint64_t groups = k / 64;
    requireBytes(b.scratch.i8codes, rows * k, "i8 codes");
    requireBytes(b.scratch.i8codesLo, rows * k, "i8 lo codes");
    requireBytes(b.scratch.i8params, rows * groups * 16, "i8 params");

    const metal::DispatchSize rowTiles{rows / kAffinePrefillTileRows, 1, 1};
    graph.add(std::string(kPrefillLinearI8Quant),
              {b.input, b.scratch.i8codes, b.scratch.i8codesLo,
               b.scratch.i8params},
              k, rowTiles, {256, 1, 1});
    const std::string_view kernel =
        w.epilogue == LinearEpilogue::UpWithGate
            ? kPrefillLinearI8N256UpSiluSums
        : w.epilogue == LinearEpilogue::Residual
            ? kPrefillLinearI8N256Residual
            : kPrefillLinearI8N256;
    const metal::MetalBuffer auxiliary =
        w.epilogue == LinearEpilogue::Residual ? b.residual
        : w.epilogue == LinearEpilogue::UpWithGate ? b.gateScratch
                                                 : b.output;
    const metal::MetalBuffer outSums =
        w.epilogue == LinearEpilogue::UpWithGate ? b.downSums : b.sums;
    graph.add(std::string(kernel),
              {b.scratch.i8codes, b.scratch.i8codesLo, b.scratch.i8params,
               weights.weights, weights.scales,
               weights.biases, auxiliary, b.output, outSums},
              Q4Params{n, k}, {rows / kAffinePrefillTileRows, n / 256, 1},
              {256, 1, 1});
    return b.prepared;
  }
  if (selected.usesSimdgroup()) {
    if (b.prepared.layout != LinearInput::Table64 || !b.prepared.source.sameView(b.input))
      graph.add(std::string(kDecodeLinearQ4Prepare), {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, w.rows / kDecodeActivationRows, 1}, {128, 1, 1});
    const AffineWeights &first = gate ? gate->affine() : weights;
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights, first.scales, first.biases,
                                             b.output, b.scratch.sums, b.scratch.partials, b.scratch.counters};
    if (gate) bindings.insert(bindings.end(), {weights.weights, weights.scales, weights.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings),
        Q4Params{n, k},
        {selected.groups(), selected.configuration().splits, w.rows / kDecodeActivationRows},
        {128, 1, 1});
    return {b.input, LinearInput::Table64};
  }
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill) {
      const metal::DispatchSize groups{selected.storageRows() / kAffinePrefillTileRows,
                                       n / selected.tileColumns(), 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      if (p.planeInputs())
        graph.add(leadingInputsInstance(name), bindings, Q4PrefillLeadingParams{{n, k}, p.planeInputs()}, groups,
                  threads);
      else
        graph.add(std::string(name), bindings, Q4Params{n, k}, groups, threads);
    } else {
      // Split128 binds its partials and counters after the sequential
      // kernel's buffers; every other decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      std::vector<metal::MetalBuffer> buffers(bindings);
      if (config.tile == LinearTile::Split128)
        buffers.insert(buffers.end(), {b.scratch.partials, b.scratch.counters});
      const std::string kernel = kernelInstance(name, selected.destination());
      const metal::DispatchSize groups{selected.groups(), config.splits, 1};
      const metal::DispatchSize threads{selected.threadsPerThreadgroup(), 1, 1};
      // The persistent tiles stride over the column tiles by their groups.
      if (persistentTile(config.tile))
        graph.add(kernel, std::move(buffers), Q4PersistentParams{n, k, selected.groups()}, groups, threads);
      else
        graph.add(kernel, std::move(buffers), Q4Params{n, k}, groups, threads);
    }
  };
  const bool prefill = w.phase == LinearPhase::Prefill;
  if (w.epilogue == LinearEpilogue::GateUp) {
    const AffineWeights &g = gate->affine();
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases,
                                     b.output, weights.weights, weights.scales, weights.biases});
    else {
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases, b.gateScratch});
      dispatch(selected.secondPipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                   b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (prefill)
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output, b.sums});
    else
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output});
  } else if (prefill)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output});
  return b.prepared;
}

void Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                            const Projection &consumer, uint32_t rows) const {
  validate({{consumer.outputSize, consumer.inputSize}, rows, LinearPhase::Prefill});
  const uint32_t tiles = (rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
  const uint64_t storageRows = uint64_t{tiles} * kAffinePrefillTileRows;
  requireBytes(input, storageRows * consumer.inputSize * 2, "input");
  requireBytes(sums, storageRows * (consumer.inputSize / kQuantGroup) * 4, "sums");
  graph.add(std::string(kPrefillLinearQ4Sums32), {input, sums}, consumer.inputSize, {tiles, 1, 1});
}
PreparedInput Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                 metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                                 LinearScratch scratch, PreparedInput prepared, bool canvas) const {
  return add(graph, {.input = input, .output = output, .sums = sums, .scratch = scratch, .prepared = prepared}, p,
             prefillPlan(p, rows, LinearEpilogue::None, canvas));
}
PreparedInput Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
                                         const Projection &p, metal::MetalBuffer residual,
                                         metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                                         LinearScratch scratch, PreparedInput prepared, bool canvas) const {
  return add(graph,
             {.input = input, .output = output, .sums = sums, .residual = residual, .scratch = scratch,
              .prepared = prepared},
             p, prefillPlan(p, rows, LinearEpilogue::Residual, canvas));
}
void Linear::addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn,
                              const PrefillFfnBuffers &b, metal::MetalBuffer residual, metal::MetalBuffer output,
                              uint32_t rows, bool canvas) const {
  // The gate's rotated rows stand in the scratch for the up pass, whose
  // projection reads the same input and rotation signs.
  const PreparedInput prepared =
      addPrefill(graph, b.normalized, *ffn.gate, b.gateScratch, b.sums, rows, b.scratch, {}, canvas);
  addPrefillUpWithGate(graph, b.normalized, *ffn.up, b.gateScratch, b.intermediate, b.sums, b.downSums, rows,
                       b.scratch, prepared, canvas);
  addPrefillResidual(graph, b.intermediate, *ffn.down, residual, output, b.downSums, rows, b.scratch,
                     {}, canvas);
}
PreparedInput Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input,
                                           const Projection &up, metal::MetalBuffer gateScratch,
                                           metal::MetalBuffer output, metal::MetalBuffer sums,
                                           metal::MetalBuffer downSums, uint32_t rows, LinearScratch scratch,
                                           PreparedInput prepared, bool canvas) const {
  return add(graph,
             {.input = input, .output = output, .sums = sums, .gateScratch = gateScratch, .downSums = downSums,
              .scratch = scratch, .prepared = prepared},
             up, prefillPlan(up, rows, LinearEpilogue::UpWithGate, canvas));
}

} // namespace richengine::ops
