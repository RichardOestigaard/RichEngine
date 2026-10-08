#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/CommandGraph.hpp"
#include "ops/DevicePolicy.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Sampling.h"
#include "ops/Weights.hpp"

#include <algorithm>
#include <compare>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace richengine::ops {

// The instance of kernel `name` that writes `destination`: a plain decode
// kernel's fp32 instance is "<name>_f32".
[[nodiscard]] inline std::string kernelInstance(std::string_view name, FloatOutput destination) {
  return std::string(name) + (destination == FloatOutput::Float32 ? "_f32" : "");
}
// The instance of prefill residual kernel `name` that reads a view of the
// leading inputs of wider weight rows (Projection::leadingInputs):
// "<name>_leading_inputs".
[[nodiscard]] inline std::string leadingInputsInstance(std::string_view name) {
  return std::string(name) + "_leading_inputs";
}
// The tile of a float projection (kernels/shared/gguf_float.metal): fp32
// simdgroup MMA on the weights as stored, or the neural accelerator's bf16
// matmul on each weight's three bf16 parts, which sum to it exactly. Both
// round only in fp32 accumulation; Linear::ggufFloatTile picks one.
enum class FloatTile : uint8_t { Simdgroup, NeuralAccelerator };
// out[r][outOffset + n] = sum_k input[r][k] W[n][k] for rows r < `rows` of a
// float segment, into a destination of `outStride` columns (LinearGguf.cpp).
void addGgufFloat(metal::CommandGraph &graph, metal::MetalBuffer input, const QuantizedSegment &weights,
                  metal::MetalBuffer output, uint32_t rows, uint32_t outStride, uint32_t outOffset,
                  FloatOutput type, FloatTile tile);

struct LinearMatrix final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  auto operator<=>(const LinearMatrix &) const = default;
};

// Throws unless `projection` is an affine projection of `matrix` whose planes
// hold all of its Q4 weights, scales and biases, in rows of its
// planeInputSize() inputs.
void requireAffineProjection(const Projection &projection, LinearMatrix matrix);
// Throws unless the planes of the quantized `segment` hold every tile of its
// outputSize x inputSize weights (metal/abi/QuantFormat.h), naming them
// "<what> plane0", "<what> plane1" and "<what> meta": in rows of
// `planeInputs` inputs, of which it is a view of the leading ones
// (Projection::leadingInputs), or of its inputSize when that is 0.
void requireSegmentPlanes(const QuantizedSegment &segment, std::string_view what, uint32_t planeInputs = 0);

enum class LinearPhase : uint8_t { Prefill, Decode };
enum class LinearEpilogue : uint8_t { None, Residual, GateUp, UpWithGate };
// Compute tiles over the StorageN=256 packing. Paired tiles pipeline two
// quant groups of one lane. Split128 is the N128 tile with K split across
// `splits` threadgroups, grid (column tiles, splits), every lane's rows in each
// tile; the last threadgroup of a tile to finish reduces the fp32 partial sums
// before the bf16 rounding. Paired256 is the four-simdgroup N256 paired tile.
// Simdgroup uses bf16 8x8 matrix operations and an explicit activation/split
// workspace.
// The GGUF tiles run 64 columns per threadgroup. The staged tiles, GgufStaged
// and GgufPrefill, dequantize GGUF weights into threadgroup memory for
// matmul2d. GgufStaged: the two-simdgroup staged tile of 8, 16 or 32 rows
// (decode, and prefill chunks of up to 32 rows), each simdgroup staging its
// own columns, with optional K splits. GgufPrefill: the 128-row shared-stage
// prefill tile. GgufRegister is the exact register tile on bf16 8x8 matrix
// operations (Apple9): every request lane in one threadgroup, optional K
// splits.
enum class LinearTile : uint8_t {
  N128, N256, Paired128, Split128, Paired256, Simdgroup, GgufStaged, GgufPrefill, GgufRegister
};
// The decode tiles hold at most a full decode batch; GGUF prefill chunks of up
// to this many rows run the staged tile (Linear::ggufBaseline).
inline constexpr uint32_t kMaximumDecodeTileRows = RICHENGINE_MAXIMUM_BATCH_WIDTH * RICHENGINE_TARGET_VERIFY_ROWS;
// The GGUF formats Apple9's staged tiles decode faster than its register
// tiles, dense and MoE: IQ3_XXS, the IQ2 formats and IQ1, whose operands the
// register tiles build from grid lookups beside their matrix operations
// (LinearGguf.cpp, MoE.hpp).
[[nodiscard]] bool apple9StagesFormat(uint32_t format) noexcept;
enum class LinearSimdgroups : uint8_t { Four = 4, Eight = 8, Sixteen = 16 };

struct LinearWorkload final {
  LinearMatrix matrix;
  uint32_t rows = 0;
  LinearPhase phase = LinearPhase::Decode;
  LinearEpilogue epilogue = LinearEpilogue::None;
  WeightLayout weightLayout = WeightLayout::Affine64;
  // A DiffusionGemma canvas step's projection: prefill-shaped, but on the
  // canvas plan family so canvas-only tiles never alter a normal chunk's
  // plan. Part of the plan key (operator<=>).
  bool canvas = false;
  auto operator<=>(const LinearWorkload &) const = default;
};

struct LinearConfig final {
  LinearTile tile = LinearTile::N128;
  // Persistent threadgroups of the N128, N256, Paired128 and Paired256 decode
  // tiles (1 to their column tiles); 0 for every other plan, whose grid
  // covers the matrix.
  uint32_t groups = 0;
  // Simdgroups of an affine Q4 tile's threadgroup, independent of the
  // persistent grid size: the cooperative scope of one tile (Paired256 runs
  // four). The GGUF tiles fix their threadgroups in their kernels
  // (GGUF_*_THREADS) and leave this at its default.
  LinearSimdgroups simdgroups = LinearSimdgroups::Eight;
  // Cross-threadgroup K partitions for Split128, Simdgroup, GgufStaged and
  // GgufRegister, a power of two up to kMaximumSplits (Split128 takes at
  // least two); all other tiles use one.
  uint32_t splits = 1;
  static constexpr uint32_t kMaximumSplits = 8;
  [[nodiscard]] constexpr bool validSplits() const noexcept {
    return splits && splits <= kMaximumSplits && !(splits & (splits - 1));
  }
  bool operator==(const LinearConfig &) const = default;
};

// Reused serially within one decode command stream. Counters are zeroed at
// allocation and restored by each completed split dispatch. Never share this
// workspace between concurrent command streams. Within a batched dispatch of
// the Simdgroup tile, each eight-row tile owns disjoint input, sums, partials
// and counters; Split128 holds every row of the step in each tile.
struct LinearScratch final {
  metal::MetalBuffer input;
  metal::MetalBuffer sums;
  metal::MetalBuffer partials;
  metal::MetalBuffer counters;
  // The bf16 input rows a rotated projection's quantized segments read
  // (ProjectionShape::rotated), sized by decode/prefillScratchSize(shape).rotated.
  metal::MetalBuffer rotated{};
  // RICHENGINE_PREFILL_FAST_INT8 operand buffers: two-term uint8 codes and
  // (scale, lo, Jx) float4 records of the projection input. Empty when the
  // flag is off.
  metal::MetalBuffer i8codes{};
  metal::MetalBuffer i8codesLo{};
  metal::MetalBuffer i8params{};
  metal::MetalBuffer i8paramsLo{};
};
// One layer's SwiGLU projections, affine Q4 or quantized GGUF tensors:
// down(silu(gate x) * up x).
struct SwiGluProjections final {
  const Projection *gate = nullptr;
  const Projection *up = nullptr;
  const Projection *down = nullptr;
};
// A prefill chunk's buffers of its dense FFN: the normalized rows and their
// Q4 sums, gate's output, the intermediate rows and their Q4 sums, and the
// linear scratch.
struct PrefillFfnBuffers final {
  metal::MetalBuffer normalized, sums, gateScratch, intermediate, downSums;
  LinearScratch scratch;
};
struct LinearScratchSize final {
  uint64_t input = 0, sums = 0, partials = 0, counters = 0, rotated = 0;
  [[nodiscard]] uint64_t bytes() const noexcept { return input + sums + partials + counters + rotated; }
  // Grows each field to hold `other`'s too.
  LinearScratchSize &include(const LinearScratchSize &other) noexcept {
    input = std::max(input, other.input);
    sums = std::max(sums, other.sums);
    partials = std::max(partials, other.partials);
    counters = std::max(counters, other.counters);
    rotated = std::max(rotated, other.rotated);
    return *this;
  }
};

// The activation layout a decode plan reads: the producer's bf16 rows, or an
// X^T table with fp32 row sums in LinearScratch that a producer can emit
// alongside its ordinary output.
enum class LinearInput : uint8_t {
  Plain,    // bf16 [rows][K]
  Table64,  // affine simdgroup table, one sum per 64 inputs (kernels/common/q4_sgmatrix.h)
  Table16,  // GGUF simdgroup table, sums per 16 and 32 inputs (kernels/common/gguf_sgmatrix.h)
  // The mxfp4p A-operand (kernels/common/gguf_mxfp4p_tile.h): the
  // slot-permuted fp16 plane in LinearScratch::input and the per-(row, 32)
  // exponent bytes in ::sums, which a producer can emit beside its bf16 rows
  // instead of a gguf_pack_half dispatch.
  Packed,
  // LinearScratch::rotated holds H (D source): no producer emits it — a
  // rotated projection's own gguf_rotate dispatch writes it, and a second
  // rotated projection of the same source and signs reuses it.
  Rotated,
};
// Scratch bytes a producer writes for `rows` rows of `width` inputs.
[[nodiscard]] constexpr uint64_t tableBytes(uint32_t width, uint64_t rows) noexcept {
  return uint64_t{width} * rows * 2;
}
[[nodiscard]] uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept;
// Throws unless `scratch` holds the `layout` table a producer writes for `rows`
// rows of `width` inputs: a table layout, whole lanes of rows and whole
// 64-input spans.
void requireTableScratch(const LinearScratch &scratch, LinearInput layout, uint32_t width, uint32_t rows);
// The kernel name suffix of a producer that writes the `layout` table:
// "_table16", "_table64", or none for Plain.
[[nodiscard]] const char *tableSuffix(LinearInput layout) noexcept;
// The scratch table currently holds `source` in `layout`. Plain means the
// scratch describes nothing. Producers return it, consumers accept it and
// return what the scratch describes after their dispatch.
struct PreparedInput final {
  metal::MetalBuffer source;
  LinearInput layout = LinearInput::Plain;
  // For layout == Rotated only: the sign vector the scratch rotation used.
  metal::MetalBuffer signs{};
};

class LinearPlan final {
public:
  [[nodiscard]] LinearWorkload workload() const noexcept { return workload_; }
  [[nodiscard]] LinearConfig configuration() const noexcept { return config_; }
  [[nodiscard]] FloatOutput destination() const noexcept { return destination_; }
  [[nodiscard]] uint32_t storageRows() const noexcept;
  [[nodiscard]] uint32_t tileColumns() const noexcept;
  // Threadgroups over the column tiles: the configured groups of a
  // persistent decode tile, every column tile otherwise.
  [[nodiscard]] uint32_t groups() const noexcept;
  [[nodiscard]] uint32_t threadsPerThreadgroup() const noexcept;
  [[nodiscard]] bool usesSimdgroup() const noexcept;
  // The layout the producer of this plan's input writes. A rotated
  // projection prepares its table from the rotated rows itself
  // (LinearGguf.cpp), so its producer writes plain rows. Packed only asks
  // producers that know it (norms, the GDN and attention gates); a producer
  // that stays Plain leaves the pack dispatch to the consumer.
  [[nodiscard]] LinearInput input() const noexcept;
  // A GGUF prefill plan whose MXFP4 segments run the pre-packed multiplane
  // tiles: one gguf_pack_half dispatch per chunk packs the bf16 rows into
  // LinearScratch::input (the slot-permuted fp16 plane) and ::sums (the
  // exponent bytes), which those kernels then read (LinearGguf.cpp).
  [[nodiscard]] bool packsActivations() const noexcept { return packs_; }
  [[nodiscard]] LinearScratchSize scratchSize() const noexcept;
  [[nodiscard]] uint64_t sumsBytes() const noexcept;
  [[nodiscard]] uint64_t gateScratchBytes() const noexcept;
  [[nodiscard]] uint64_t downSumsBytes() const noexcept;
  // The affine tile's kernel; the plan runs its kernelInstance for destination().
  [[nodiscard]] std::string_view pipeline() const noexcept { return pipeline_; }
  [[nodiscard]] std::string_view secondPipeline() const noexcept {
    return secondPipeline_;
  }

private:
  friend class Linear;
  LinearPlan(LinearWorkload workload, LinearConfig config, FloatOutput destination = FloatOutput::BFloat16);
  // Block plans (LinearGguf.cpp).
  void requireBlockConfiguration() const;
  [[nodiscard]] uint32_t blockStorageRows() const noexcept;
  [[nodiscard]] LinearScratchSize blockScratchSize() const noexcept;
  LinearWorkload workload_;
  LinearConfig config_;
  FloatOutput destination_;
  // The plan's projection multiplies the rotated input (InputRotation).
  bool rotated_ = false;
  // packsActivations(): set by Linear's plan methods.
  bool packs_ = false;
  // A decode plan — or a prefill chunk plan of up to a decode batch's rows —
  // whose single-tensor projections dispatch the pre-packed mxfp4p kernels:
  // input() then asks the producer for the packed operand
  // (LinearInput::Packed), which addGgufStaged takes instead of packing.
  bool packedInput_ = false;
  std::string_view pipeline_;
  std::string_view secondPipeline_;
};

// The plan defines which fields are used and how much scratch they require.
struct LinearBuffers final {
  metal::MetalBuffer input;
  metal::MetalBuffer output;
  metal::MetalBuffer sums;
  metal::MetalBuffer residual;
  metal::MetalBuffer gateScratch;
  metal::MetalBuffer downSums;
  LinearScratch scratch{};
  // What the scratch table holds (for example after fused RMSNorm). A plan
  // that reads a table prepares one unless this describes its input.
  PreparedInput prepared{};
};

// Owns projection pipeline selection and dispatch for both weight layouts.
// Device policy comes from one DevicePolicy (ops/DevicePolicy.hpp); the
// measured choices it scopes live in ops/DeviceTuning.cpp.
class Linear final {
public:
  explicit Linear(const DeviceCapabilities &device) noexcept;
  explicit Linear(const DevicePolicy &policy) noexcept;

  [[nodiscard]] LinearPlan plan(LinearWorkload workload) const;
  // The plan of `workload` in the projection's weight layout, into its destination type; a gate/up plan also runs
  // `gate`.
  [[nodiscard]] LinearPlan plan(LinearWorkload workload, const Projection &projection,
                                const Projection *gate = nullptr) const;
  // Rows of storage a decode step of `rows` rows binds for a projection of
  // `shape`: the storageRows of its decode plans, which every epilogue shares.
  [[nodiscard]] uint32_t decodeStorageRows(uint32_t rows, ProjectionShape shape) const;
  // The plans of this projection's matrix in its layout. A decode plan's
  // input() is the layout its producer writes.
  // `canvas` marks a DiffusionGemma trunk pass's projection.
  [[nodiscard]] LinearPlan prefillPlan(const Projection &projection, uint32_t rows,
                                       LinearEpilogue epilogue,
                                       bool canvas = false) const;
  [[nodiscard]] LinearPlan decodePlan(const Projection &projection, uint32_t lanes,
                                      LinearEpilogue epilogue = LinearEpilogue::None,
                                      const Projection *gate = nullptr) const;
  // The scratch of every decode plan of a projection of `shape`: each lane
  // count, the None, Residual and GateUp epilogues and every tile the device
  // may run them on, and the rotated rows of a full decode batch.
  [[nodiscard]] LinearScratchSize decodeScratchSize(ProjectionShape shape) const;
  // The scratch of every prefill chunk and epilogue of a projection of
  // `shape`: the split partials and counters of the chunks that run the GGUF
  // staged tile (LinearGguf.cpp), and the rotated rows of a full chunk.
  [[nodiscard]] LinearScratchSize prefillScratchSize(ProjectionShape shape) const;
  // The tile of a float projection of `rows` rows into `outputSize` columns
  // on this device (LinearGguf.cpp).
  [[nodiscard]] FloatTile ggufFloatTile(uint32_t rows, uint32_t outputSize) const noexcept;
  // The plan of `config` for `workload`, whether or not the device's policy
  // picks it.
  [[nodiscard]] static LinearPlan plan(LinearWorkload workload, LinearConfig config,
                                       FloatOutput destination);
  // Returns what the scratch table describes after the dispatch.
  PreparedInput add(metal::CommandGraph &graph, LinearBuffers buffers,
                    const Projection &projection, const LinearPlan &plan,
                    const Projection *gate = nullptr) const;

  // The Q4 input sums of `rows` rows an affine prefill projection reads.
  void addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                      const Projection &consumer, uint32_t rows) const;
  // The projections of `rows` rows through their own matrix. `scratch` holds
  // the partials and counters of split plans (GGUF chunks of up to 32 rows);
  // reused serially within one command stream, as in decode. `prepared`
  // describes the scratch as a producer left it (the rotated rows of a
  // rotated projection just run on `input`), and each returns what it
  // describes after its dispatch.
  // `canvas` marks a DiffusionGemma trunk pass's projection.
  PreparedInput addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &projection,
                           metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                           LinearScratch scratch = {}, PreparedInput prepared = {},
                           bool canvas = false) const;
  PreparedInput addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                                     metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                     metal::MetalBuffer sums, metal::MetalBuffer downSums, uint32_t rows,
                                     LinearScratch scratch, PreparedInput prepared = {},
                                     bool canvas = false) const;
  PreparedInput addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input,
                                   const Projection &projection, metal::MetalBuffer residual,
                                   metal::MetalBuffer output, metal::MetalBuffer sums,
                                   uint32_t rows, LinearScratch scratch, PreparedInput prepared = {},
                                   bool canvas = false) const;
  // output = residual + down(silu(gate x) * up x) of `rows` normalized rows,
  // whose Q4 sums the norm wrote.
  void addPrefillSwiGlu(metal::CommandGraph &graph, const SwiGluProjections &ffn, const PrefillFfnBuffers &buffers,
                        metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows,
                        bool canvas = false) const;

  // The fused greedy head of a block projection (gguf_decode_*_m*_amax in
  // shared/gguf_linear.metal): the vocabulary projection's staged decode,
  // writing per-(row, GGUF_TILE_COLUMNS) argmax partials instead of logits.
  // One quantized segment, decode rows; a rotated input is rotated into
  // scratch.rotated first. Returns false when the projection cannot run it —
  // a float segment or several segments — and the caller falls back to the
  // logits path.
  [[nodiscard]] bool addGgufHeadArgmax(metal::CommandGraph &graph,
                                       metal::MetalBuffer input, const Projection &projection, uint32_t lanes,
                                       metal::MetalBuffer argmaxValues, metal::MetalBuffer argmaxIndices,
                                       const HeadArgmaxParams &headParams, LinearScratch scratch) const;

private:
  // The device's configuration of the workload; a block plan's tile may follow the formats of the projections it
  // runs.
  [[nodiscard]] LinearConfig baseline(LinearWorkload workload,
                                      std::span<const Projection *const> projections = {}) const;
  // GGUF policy and dispatch (LinearGguf.cpp). Block plans are not tuned.
  [[nodiscard]] LinearConfig ggufBaseline(LinearWorkload workload,
                                          std::span<const Projection *const> projections) const;
  // The scratch of every tile a block decode plan of the workload may take.
  [[nodiscard]] LinearScratchSize ggufDecodeScratchSize(LinearWorkload workload) const;
  void addGguf(metal::CommandGraph &graph, const LinearBuffers &buffers,
               const Projection &projection, const LinearPlan &plan,
               const Projection *gate) const;
  void addGgufStaged(metal::CommandGraph &graph, const LinearBuffers &buffers,
                     const Projection &projection, const LinearPlan &plan,
                     const Projection *gate) const;
  void addGgufPrefill(metal::CommandGraph &graph, const LinearBuffers &buffers,
                      const Projection &projection, const LinearPlan &plan) const;
  void addGgufRegister(metal::CommandGraph &graph, const LinearBuffers &buffers,
                       const Projection &projection, const LinearPlan &plan,
                       const Projection *gate) const;
  void addGgufFloatSegments(metal::CommandGraph &graph, const LinearBuffers &buffers,
                            const Projection &projection, const LinearPlan &plan) const;
  DevicePolicy policy_{};
};

} // namespace richengine::ops
