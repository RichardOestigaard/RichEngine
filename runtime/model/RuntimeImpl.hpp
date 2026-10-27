#pragma once

// Runtime::Impl: the Metal model runtime's private implementation. The
// encode, ANE-predraft and n-gram predraft method bodies live in
// RuntimeEncode.mm, RuntimeAne.mm and RuntimeNgram.mm.

#include "model/Runtime.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/DiffusionSampler.hpp"
#include "model/RuntimeRequest.hpp"
#include "AwakeClock.hpp"
#include "Env.hpp"
#include "Tuning.hpp"
#include "model/AnePredictor.hpp"
#include "model/ModelPolicy.hpp"
#include "model/NullDraft.hpp"
#include "model/DFlashTwo8width.hpp"
#include "model/DFlashTwo16width.hpp"
#include "model/CompositeStateStorage.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/TargetModel.hpp"
#include "model/RuntimeArenas.hpp"
#include "model/VerifyBudget.hpp"

#include "metal/CommandGraph.hpp"
#include "metal/abi/Dfa.h"
#include "metal/abi/StepPack.h"
#include "ops/Embedding.hpp"
#include "ops/KernelNames.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace richengine::model {

using metal::BufferStorage;
using metal::CommandGraph;
using metal::CommandTicket;
using metal::CommandTiming;
using metal::MetalBackend;
using metal::MetalBuffer;
using kv::ChunkedPrefillParams;

namespace detail {

// A batch ticket whose completion runs on wait(): the deferred tickets
// Runtime::Impl returns across Runtime.mm, RuntimeEncode.mm and
// RuntimeNgram.mm.
class DeferredMetalTicket final : public ModelBatchTicket {
public:
  // Runs once the command has completed, and may add work of its own to its
  // timing, as a rerun of the command does.
  using Completion = std::function<std::vector<ModelStepResult>(metal::CommandTiming &)>;

  DeferredMetalTicket(metal::CommandTicket ticket, Completion completion,
                      bool representativePrefillTiming = true)
      : ticket_(std::move(ticket)), completion_(std::move(completion)),
        representativePrefillTiming_(representativePrefillTiming) {}

  bool ready() const noexcept override { return ticket_.ready(); }

  std::vector<ModelStepResult> wait() override {
    if (!completion_) {
      throw std::logic_error("Metal ticket was already consumed");
    }
    metal::CommandTiming timing = ticket_.wait();
    const double commandSeconds = timing.wallSeconds;
    Completion completion = std::move(completion_);
    std::vector<ModelStepResult> results = completion(timing);
    wallMilliseconds_ = timing.wallSeconds * 1000.0;
    // Work the completion added takes wall time, but is no sample of the
    // command's rows.
    if (timing.wallSeconds != commandSeconds)
      representativePrefillTiming_ = false;
    return results;
  }

  double wallMilliseconds() const noexcept override {
    return wallMilliseconds_;
  }
  bool prefillTimingIsRepresentative() const noexcept override {
    return representativePrefillTiming_;
  }

private:
  metal::CommandTicket ticket_;
  Completion completion_;
  double wallMilliseconds_ = 0.0;
  bool representativePrefillTiming_;
};

// A prefill arena's tensors as the target's prefill reads them.
inline TargetModelPrefillBuffers prefillBuffers(const PrefillArena &arena) {
  auto p = [&](PrefillTensor tensor) { return arena.get(tensor); };
  TargetModelPrefillBuffers buffers;
  // Prefill plans read plain bf16 rows; the input and sums slots hold a
  // GGUF chunk's packed plane and exponent bytes (ops::LinearGguf.cpp).
  buffers.linearScratch = {.input = p(PrefillTensor::LinearPacked),
                           .sums = p(PrefillTensor::LinearExponents),
                           .partials = p(PrefillTensor::LinearPartials),
                           .counters = p(PrefillTensor::LinearCounters),
                           .rotated = p(PrefillTensor::LinearRotated),
                           .i8codes = p(PrefillTensor::I8Codes),
                           .i8codesLo = p(PrefillTensor::I8CodesLo),
                           .i8params = p(PrefillTensor::I8Params),
                           .i8paramsLo = p(PrefillTensor::I8ParamsLo)};
  buffers.hidden = {p(PrefillTensor::Hidden0), p(PrefillTensor::Hidden1)};
  buffers.normalized = p(PrefillTensor::Normalized);
  buffers.captured = p(PrefillTensor::Captured);
  buffers.gdnPacked = p(PrefillTensor::GdnPacked);
  buffers.gdnQueries = p(PrefillTensor::GdnQueries);
  buffers.gdnKeys = p(PrefillTensor::GdnKeys);
  buffers.gdnValues = p(PrefillTensor::GdnValues);
  buffers.gdnDecay = p(PrefillTensor::GdnDecay);
  buffers.gdnBeta = p(PrefillTensor::GdnBeta);
  buffers.recurrent = p(PrefillTensor::Recurrent);
  buffers.gdnHidden = p(PrefillTensor::GdnHidden);
  buffers.gdnOutput = p(PrefillTensor::GdnOutput);
  buffers.denseGateScratch = p(PrefillTensor::GateIntermediate);
  buffers.denseIntermediate = p(PrefillTensor::Intermediate);
  buffers.fullPacked = p(PrefillTensor::FullPacked);
  buffers.fullQueries = p(PrefillTensor::FullQueries);
  buffers.fullAttention = p(PrefillTensor::FullAttention);
  buffers.attentionPartials = p(PrefillTensor::AttentionPartials);
  buffers.attentionStatistics = p(PrefillTensor::AttentionStatistics);
  buffers.attentionHidden = p(PrefillTensor::AttentionHidden);
  buffers.attentionOutput = p(PrefillTensor::AttentionOutput);
  buffers.projectionSums = p(PrefillTensor::ProjectionSums);
  buffers.downProjectionSums = p(PrefillTensor::DownProjectionSums);
  buffers.ropeCos = p(PrefillTensor::RopeCos);
  buffers.ropeSin = p(PrefillTensor::RopeSin);
  buffers.ropeCosAlt = p(PrefillTensor::RopeCosAlt);
  buffers.ropeSinAlt = p(PrefillTensor::RopeSinAlt);
  buffers.zeroResidual = p(PrefillTensor::ZeroResidual);
  buffers.chunkKeys = p(PrefillTensor::ChunkKeys);
  buffers.chunkValues = p(PrefillTensor::ChunkValues);
  buffers.gdnChunkScratch = p(PrefillTensor::GdnChunkScratch);
  buffers.moe = arena.moeScratch();
  return buffers;
}

// The draft variant Impl holds.
using DraftModel =
    std::variant<NullDraft, DFlashDraft, DFlashTwo8width, DFlashTwo16width,
                 DFlashV1Draft,
                 DSparkDraft>;

// The DraftModel alternative each draft weights type constructs. The
// mapping is total: a weights type with no specialization fails to compile
// here rather than building the neighboring draft by default.
template <class Weights> struct DraftModelFor;
template <> struct DraftModelFor<NullDraftWeights> { using Type = NullDraft; };
template <> struct DraftModelFor<DFlashDraftWeights> { using Type = DFlashDraft; };
template <> struct DraftModelFor<DFlashV1DraftWeights> {
  using Type = DFlashV1Draft;
};
template <> struct DraftModelFor<DSparkDraftWeights> { using Type = DSparkDraft; };

} // namespace detail


struct Runtime::Impl {  // An image by content: the fields a placement's span identifies it by.
  using ImageKey = richengine::model::RuntimeImageKey;
  using ImageKeyHash = richengine::model::RuntimeImageKeyHash;
  using ImageRows = richengine::model::RuntimeImageRows;
  using ImageState = richengine::model::RuntimeImageState;
  using Request = richengine::model::RuntimeRequest;
  struct DecodeLaneResult final {
    Request *request = nullptr;
    uint32_t retained = 0;
    uint32_t accepted = 0;
    uint32_t currentAnchor = 0;
    uint32_t maximumRetained = 0;
    // The proposals the verify batch was offered this step: the lane's
    // adaptive budget clamped by its retention limit (the bypass's uncapped
    // write keeps the buffer's full width for the acceptance clamp).
    uint32_t proposals = 0;
    // The rows of its block the lane verified (draft_verify_rows).
    uint32_t verifyRows = 0;
    // Whether the lane's proposals came from the draft's own selector this
    // step, so its confidences and accepted count label the verify budget's
    // acceptance model (a predraft, bypass or forced span replaces them).
    bool verifyLabels = false;
    // Why the lane's selection is unusable (invalidSelection), found before
    // any lane commits.
    std::string failure;
  };
  // What a lane's GPU table was last written from. Its entries stay valid
  // while the revision does: KvPool never releases the extent of a page a
  // request holds (PageStorage::releaseExtent).
  struct PageTableBinding final {
    uint64_t requestId = 0;
    uint64_t revision = 0;
  };
  MetalBackend &backend;
  const ModelPackage &package;
  const RuntimeGeometry geometry;
  const ops::ExecutionPlans &operators;
  kv::PageStorage &kvPages;
  CompositeStateStorage &states;
  std::unique_ptr<PrefillArena> prefillArena;
  // Submit-ahead input banks (PrefillArena::get(tensor, bank)): one bit per
  // bank an unconsumed prefill command wrote. At most one submit-ahead
  // command overlaps a running one, so two banks cover the ring.
  uint32_t prefillInputBanksInFlight_ = 0;
  uint32_t claimPrefillInputBank() {
    const uint32_t bank = (prefillInputBanksInFlight_ & 1) ? 1 : 0;
    prefillInputBanksInFlight_ |= 1u << bank;
    return bank;
  }
  void releasePrefillInputBank(uint32_t bank) noexcept {
    prefillInputBanksInFlight_ &= ~(1u << bank);
  }
  // A bank is free while fewer than two prefills are unconsumed.
  bool prefillSubmitAheadAvailable() const noexcept {
    return prefillInputBanksInFlight_ != 3;
  }
  std::unique_ptr<DecodeArena> decodeArena;
  // Every state lane's penalty words, bound whole: a batch lane reads the row
  // of its request's state lane, which need not be its own.
  MetalBuffer penaltyTable;
  // The GDN decode tape (folded commit): per state lane, two parity slots of
  // one step's retained rows per GDN layer — the next step's scan folds them
  // in as its pending prefix. Empty for non-GDN targets.
  MetalBuffer gdnTape_;
  std::array<uint32_t, kLaneCount> gdnPendingRows_{};
  std::array<uint8_t, kLaneCount> gdnTapeParity_{};
  // The bytes of one layer's tape region: the step's rows of conv inputs,
  // their k|v columns, then decay (fp32) and beta (bf16).
  uint64_t gdnTapeLayerBytes() const {
    const uint64_t rows = RICHENGINE_TARGET_VERIFY_ROWS;
    const uint64_t kv =
        (uint64_t{geometry.target.gdnKeyHeads} +
         geometry.target.gdnValueHeads) *
        geometry.target.gdnHeadDimension;
    const uint64_t body =
        rows * (geometry.target.convolutionDimension * 2 + kv * 2 +
                uint64_t{geometry.target.gdnValueHeads} * 4 +
                uint64_t{geometry.target.gdnValueHeads} * 2);
    return (body + 255) / 256 * 256;
  }
  // The process-wide packed token vocabulary (VOCABULARY frame) the GPU-DFA
  // walk kernels index; a request's DFA is dropped while this is absent.
  struct PackedVocab final {
    MetalBuffer bytes;
    MetalBuffer offsets;
    // Bit t names token t special: the marker rule requires its walk to
    // take a marked edge and forbids a regular token from taking one.
    MetalBuffer special;
    uint32_t vocab = 0;
  };
  std::optional<PackedVocab> packedVocab_;
  std::unordered_map<uint64_t, Request> requests;
  // Allocated for images that need an encode, sized for the largest one the
  // start that built it staged, and reclaimable once no image waits for one
  // and no refused start holds it. Injecting already encoded rows needs no
  // vision arena.
  std::shared_ptr<ops::Vision> vision;
  // Every image's rows while anything holds them, so that a placement of
  // the same image anywhere shares them. Entries of rows nothing holds any
  // more go when a lookup or a walk finds them.
  std::unordered_map<ImageKey, std::weak_ptr<ImageRows>, ImageKeyHash> imageRows;
  // Encoded rows kept for reuse once no placement has rows of them left to
  // inject, including prefix hits that land inside an image and still need
  // its remaining rows. Most recently used first, bounded by bytes; the
  // memory reclaimer drops the least recently used entry nothing else holds.
  static constexpr uint64_t kEmbeddingCacheBytes = 512ULL * 1024 * 1024;
  std::list<std::shared_ptr<ImageRows>> embeddingCache;
  uint64_t embeddingCacheBytes = 0;
  // A state in RAM that resumes inside an image, with the rows it needs: its
  // boundary lies less than a page before the image's end, so a restore
  // there injects the image's last rows. The pointer the cache keeps owns
  // both, so the rows go with the state's RAM copy, which the cache drops
  // when it evicts the state or writes it to disk. Held rows and the
  // embedding cache together keep at most kEmbeddingCacheBytes of rows.
  struct HeldState final {
    std::shared_ptr<const CompositeState> state;
    std::shared_ptr<ImageRows> rows;
  };
  std::vector<std::weak_ptr<const HeldState>> stateHolds;
  std::array<PageTableBinding, kLaneCount> pageTableBindings{};
  ModelTelemetry counters;
  ops::Sampling sampling;
  // The acceptance model and cycle costs a MoE target's verify rows follow
  // (ops::DraftSelector::addVerifyRows); everyVerifyRow_ instead verifies
  // every live row, as a dense target's cycles do, so a test's two runs of
  // one schedule emit the same tokens.
  VerifyBudget verifyBudget_;
  bool everyVerifyRow_ = false;
  // DiffusionGemma: its package's target is a DiffusionGemmaWeights, its
  // decodes run the canvas denoising loop (RuntimeDiffusion.mm) instead of
  // draft/verify, single-lane and synchronous.
  const bool diffusion;
  struct DiffusionCanvasArena;
  // The canvas decode's state (RuntimeDiffusion.mm), read only when
  // `diffusion` is set: the step-chunking and exit-policy knobs, captured
  // once from tuning, and the shared canvas arena built lazily on the first
  // canvas decode.
  struct DiffusionState final {
    // Canvas step chunking and exit policy:
    // RICHENGINE_CANVAS_STEPS_PER_CMD (default 2, clamped to 1..4): denoise
    // steps encoded per command buffer; the stats and argmax rings carry one
    // slot per step so a mid-chunk early exit still drains in order — the
    // extra steps past it were dead compute only. 1 restores the old
    // encode/wait-per-step loop.
    const uint32_t canvasStepsPerCmd_ = modelPolicy().canvasStepsPerCmd;
    // RICHENGINE_CANVAS_PREFIX_EXIT (default 1; "0" disables): early exit
    // judges argmax stability over the prefix bounded by the first stop
    // token, so churn past it does not hold the canvas alive.
    const bool canvasPrefixExit_ = modelPolicy().canvasPrefixExit;
    // RICHENGINE_CANVAS_COMMIT_TAIL (default 1; "0" disables): the last
    // scheduled step encodes without canvas_entropy_accept — the commit
    // reads the row-stats argmax directly.
    const bool canvasCommitTail_ = modelPolicy().canvasCommitTail;
    // RICHENGINE_CANVAS_SPECULATIVE_PREFILL (default on; "0" disables): once
    // a drained step leaves the whole-canvas argmax unchanged, the encoder
    // commit prefill is encoded ahead of the formal exit on dedicated
    // buffers and skipped later if the commit mismatches.
    const bool canvasSpecPrefill_ = modelPolicy().canvasSpecPrefill;
    // RICHENGINE_CANVAS_PROFILE ("paper"|"balanced"|"fast"): bundles the
    // speed/quality knobs below for serve's --canvas-profile. Each knob's own
    // RICHENGINE_* variable wins over the profile when set.
    const std::string_view canvasProfile_ = modelPolicy().canvasProfile;
    // RICHENGINE_CANVAS_EXIT_STABLE (default 0.9; "0" disables): the
    // fraction-settled exit — at least this share of argmax-stable rows whose
    // mean entropy is under the confidence threshold commits early. Renoised
    // rows never settle on open-ended prompts, so the paper's strict
    // whole-canvas mean is otherwise unreachable.
    const float canvasExitStable_ = modelPolicy().canvasExitStable;
    // RICHENGINE_CANVAS_EXIT_DRIFT (default 0; "1" enables): fraction-settled
    // exit without the confidence check — commits a mostly-stable canvas's
    // argmax, freezing its churning minority early.
    const bool canvasDriftExit_ = modelPolicy().canvasDriftExit;
    // RICHENGINE_CANVAS_MAX_STEPS (default 0 = the manifest's 48): caps the
    // denoising schedule. Prose canvases never settle inside 48 steps, so
    // the cap is the speed/quality dial — 24 roughly doubles throughput.
    const uint32_t canvasMaxSteps_ = modelPolicy().canvasMaxSteps;
    // RICHENGINE_CANVAS_PROBE ("notrunk"|"notail"|"nosc"): a timing probe
    // that skips the named stage of every canvas step — the trunk pass, the
    // head+row-stats+accept tail, or the self-conditioning chain. Output is
    // garbage; use with CANVAS_TIMING to attribute the per-step cost.
    const std::string_view canvasProbe_ = tuning().canvasProbe;
    // shared_ptr, not unique_ptr: the pointer's deleter binds when the arena
    // is made, so the constructor's unwind path needs no complete type.
    std::shared_ptr<DiffusionCanvasArena> canvasArena_;
  };
  DiffusionState diffusion_;
  [[nodiscard]] DiffusionCanvasArena &canvasArena();
  [[nodiscard]] const DiffusionGemmaWeights &diffusionWeights() const;
  // The encoder pass's per-layer scalars: nonempty only for a diffusion
  // target (encoder_scalars.bin), which its prefills override the packed
  // decoder scalars with.
  [[nodiscard]] std::span<const float> encoderLayerScalars() const;
  std::unique_ptr<ModelBatchTicket>
  decodeDiffusion(const BatchPlan &plan,
                  std::span<const ModelBatchItem> items,
                  std::function<void()> completion);
  // Canvas batching (RuntimeDiffusion.mm): every decode item is a lane
  // with its own tokens/argmax/entropy scratch; each denoising step runs
  // one trunk pass over the concatenated lane rows.
  struct DiffusionLaneArena;
  // One lane of a canvas commit (or speculative commit) prefill.
  struct CommitLane final {
    Request &entry;
    uint64_t position;
    uint32_t rows;
    MetalBuffer pageTable;
    uint32_t pageTableEntries;
    MetalBuffer tokens;
    MetalBuffer ropePositions;
  };
  // A lane's share of one denoising step before the shared trunk: rope
  // positions, the scaled embedding into the lane's hidden slice, the
  // previous step's self-conditioning chain and the rope table slice.
  void encodeCanvasLaneFront(CommandGraph &graph, const Request &entry,
                             DiffusionLaneArena &lane,
                             const DiffusionSchedule &schedule, uint32_t step,
                             uint32_t prefixTokens, uint32_t rows,
                             uint32_t rowOffset,
                             TargetModelPrefillBuffers &buffers);
  // The lane's share after the shared trunk and head: row stats and the
  // entropy-bounded accept on the lane's logits slice and ring buffers.
  // `commitOnly` (the commit-tail flag) ends after the row stats — the
  // accept's outputs have no consumer once the schedule is done.
  void encodeCanvasLaneBack(CommandGraph &graph, DiffusionLaneArena &lane,
                            const DiffusionSchedule &schedule, uint32_t step,
                            uint32_t canvasIndex, uint32_t seed,
                            uint32_t rows, uint32_t rowOffset,
                            MetalBuffer argmaxPrev, MetalBuffer argmaxOut,
                            MetalBuffer statsSlot, MetalBuffer entropySlot,
                            bool commitOnly);
  // The encoder pass over committed canvases: an ordinary causal prefill of
  // every lane's committed tokens at their logical positions, into the
  // requests' real pages, in one trunk pass.
  void encodeCanvasCommitPrefill(CommandGraph &graph,
                                 std::span<const CommitLane> lanes);
  TargetModel targetModel;
  detail::DraftModel draftModel;
  ops::FfnSplit *aneFfn;
  // Tree verify (docs/TREE_VERIFY_DESIGN.md): the selector emits each greedy,
  // unconstrained lane's comb tree when the draft is tree-capable. DFlash2's
  // 87%-acceptance benchmark measured identical tokens at ~2x verify rows,
  // so it stays opt-in; DSpark's low chain acceptance leaves ~19% of its
  // rejections to the comb's sibling leaves, so it defaults on.
  // RICHENGINE_VERIFY_TREE=1 forces on for either kind, =0 forces off.
  const bool verifyTreeEnvSet_ = tuning().verifyTreeSet;
  const bool verifyTreeEnabled = tuning().verifyTree;
  // Adaptive proposal budgets (docs/SPEC_DECODE_BOOST.md L3): a chain lane's
  // accepted-token cap tracks its rolling acceptance EWMA, so a lane that
  // keeps rejecting pays for fewer live verify rows — GDN scan depth and
  // the per-row vocabulary/argmax sweeps — instead of the fixed eight.
  // On by default; RICHENGINE_ADAPTIVE_PROPOSALS=0 disables.
  const bool adaptiveProposals_ =
      modelPolicy().adaptiveProposals && geometry.tuning.adaptiveProposals;
  // Per-step debug gates, read once: getenv scans environ linearly and these
  // ran inside the decode/finalize paths on every command.
  const bool treeDebug_ = tuning().treeDebug;
  const bool draftDebug_ = tuning().draftDebug;
  const bool draftConfDebug_ = tuning().draftConf;
  const bool verifyBudgetDebug_ = tuning().verifyBudgetDebug;
  const std::string treeSkip_ = tuning().treeSkip;
  const bool aneDebug_ = tuning().aneDebug;
  const bool ngramDebug_ = tuning().ngramDebug;
  // Opt-in ANE speculation (docs/ANE_DRAFTING.md): RICHENGINE_ANE_MEDUSA names a
  // CoreML package whose leaf alternates replace a tree batch's sibling
  // leaves; RICHENGINE_ANE_PREDRAFT names one that drafts the next step's
  // proposal chain while the target verifies. Both keep the target's own
  // acceptance authoritative, so a stale or absent result never changes the
  // output — it only wastes ANE time.
  std::unique_ptr<AnePredictor> aneMedusa_;
  std::unique_ptr<AnePredictor> anePredraft_;
  MetalBuffer aneLeafTokens_;  // [kLaneCount][RICHENGINE_DRAFT_PROPOSAL_TOKENS]
  MetalBuffer aneFlag_;        // one u32: the medusa serial last published
  // Three u32s the mask-grid kernel fills under RICHENGINE_GPU_DECODE_GRID:
  // the DFA mask pass's MTLDispatchThreadgroupsIndirectArguments.
  MetalBuffer dfaMaskGrid_;
  // RICHENGINE_GPU_DECODE_GRID (presence): the constrained verify round's
  // GPU-DFA mask dispatch takes its threadgroup count from a GPU-written
  // grid instead of the host's span readback.
  const bool gpuDecodeGridOn_ = tuning().gpuDecodeGrid;
  // RICHENGINE_DFA_MERGE (presence): a Full-mode constrained batch whose
  // lanes all carry a GPU DFA and owe no host-pushed forced span encodes
  // draft, verify and commit as one command — the WaitingMask stage never
  // opens. ConstrainedDecodeTicket::mergedSubmit is the fused encode.
  const bool dfaMergeOn_ = tuning().dfaMerge;
  // RICHENGINE_DFA_SPAN_SKIP_DRAFT (presence): decodeAsync leaves the draft
  // embedding and forward unencoded when every lane of the batch is a
  // greedy DFA lane whose pending forced span covers its proposal budget.
  // Safe because the draft's persistent state — the per-layer KV rings —
  // is written only by the commit stage's context commit, which recomputes
  // K/V from the captured target hidden of retained rows and never reads a
  // draft-decode output.
  const bool dfaSpanSkipDraft_ = tuning().dfaSpanSkipDraft;
  // RICHENGINE_DEVICE_VERIFY_PARAMS (presence): decode_step_pack runs as
  // the last dispatch of every committed decode graph and mirrors each
  // lane's acceptance results into stepPack_, one DecodeStepPackLane per
  // dispatch lane — the device-side copy of what finalizeDecode's
  // readback feeds the next step's host encode. Allocated lazily.
  const bool deviceVerifyParams_ = tuning().deviceVerifyParams;
  // RICHENGINE_SCORE_GATHER=1: an all-score batch's head dispatches only the
  // quant tiles holding the union of its lanes' score columns
  // (TargetModel::addScoreHeadColumns).
  const bool scoreGather_ = tuning().scoreGather;
  // RICHENGINE_SCORE_TENSOR=1 / RICHENGINE_SCORE_FUSED=1: the same
  // restriction as one gather dispatch over the union's columns
  // (TargetModel::addScoreHeadGather), the fused variant folding the final
  // norm into it.
  const bool scoreTensor_ = tuning().scoreTensor;
  const bool scoreFused_ = tuning().scoreFused;
  MetalBuffer stepPack_;
  // The lazily-allocated wanted-column list of a score batch's gather
  // dispatch, sized for the union of a full batch's score options.
  MetalBuffer scoreColumns_;
  [[nodiscard]] const MetalBuffer &scoreColumnsBuffer() {
    if (!scoreColumns_) {
      scoreColumns_ = backend.allocateBuffer(
          sizeof(uint32_t) * kLaneCount * ExecutionLimits::maximumScoreOptions,
          BufferStorage::Shared, "score-columns");
    }
    return scoreColumns_;
  }
  // The lazily-allocated pack block, zeroed so an unwritten lane's zero
  // request id can never match a live request's.
  [[nodiscard]] const MetalBuffer &stepPackBuffer() {
    if (!stepPack_) {
      stepPack_ = backend.allocateBuffer(
          sizeof(DecodeStepPackLane) * kLaneCount, BufferStorage::Shared,
          "decode-step-pack");
      std::memset(stepPack_.contents(), 0, stepPack_.sizeBytes());
    }
    return stepPack_;
  }
  std::atomic<uint32_t> medusaSerial_{0};
  std::atomic<uint32_t> predraftSerial_{0};
  std::atomic<uint32_t> predraftDone_{0};
  std::atomic<uint32_t> predraftLanes_{0};
  std::atomic<bool> predraftValid_{false};
  std::mutex aneMutex_;
  std::condition_variable aneCv_;
  std::array<uint32_t, kLaneCount> aneAnchors_{};
  std::array<uint64_t, kLaneCount> anePositions_{};
  std::array<std::array<uint32_t, RICHENGINE_DRAFT_PROPOSAL_TOKENS>, kLaneCount>
      aneProposals_{};
  const uint32_t aneWaitMs_ = tuning().aneWaitMs;
  // Learning-free predraft (default on; RICHENGINE_NGRAM_PREDRAFT=0
  // disables): an n-gram table over each lane's prompt+generated stream
  // feeds the same ProposedTokens injection the ANE artifact uses — a wrong
  // candidate only wastes verify rows.
  struct NgramPredraft final {
    NgramPredraft()
        : ngramDraftExpect_(modelPolicy().ngramDraftExpect),
          ngramWarmup_(modelPolicy().ngramWarmup),
          ngramTreeMin_(modelPolicy().ngramTreeMin) {}
    const bool ngramPredraft_ = modelPolicy().ngramPredraft;
    // Acceptance gate (SSSD-style): a batch goes n-gram-predrafted when the
    // lanes' expected accepted-token scores total at least what the GPU draft
    // would have delivered. A lane with a weak EWMA or no match drags the
    // total down and can veto; a cold lane re-probes each kNgramProbeTokens.
    // The thresholds take the family's ModelTuning default, an explicitly
    // set RICHENGINE_* variable overriding it.
    const double ngramDraftExpect_;
    const uint32_t ngramWarmup_;
    // The comb tree's admission: a lane's acceptance EWMA must reach this
    // before its table emits sibling leaves — a proposer that rarely lands
    // rescues nothing, and the tree's wider verify would cost for free rows.
    const double ngramTreeMin_;
    static constexpr uint32_t kNgramProbeTokens = 256;
    // Whether the batch left applyNgramPredraft as a comb tree.
    bool predraftedTree_ = false;
    // The batch-maximum live node count of the emitted comb, known at encode
    // time because the host wrote the table — the verify plan and its KV
    // stores size for it instead of the scratch's full capacity.
    uint32_t predraftedTreeNodes_ = RICHENGINE_TREE_VERIFY_NODES;
  };
  NgramPredraft ngram_{};
  // The env switch AND the family's default (ModelDescriptor.tuning).
  [[nodiscard]] bool ngramPredraftEnabled() const noexcept {
    return ngram_.ngramPredraft_ && geometry.tuning.ngramPredraft;
  }
  // Speculative next-step draft on the GPU (RICHENGINE_GPU_PREDRAFT, =0
  // disables): once a committed step's anchors and positions are
  // host-known, the batch's next draft forward is encoded and submitted
  // while the engine consumes this step's results — its GPU work overlaps
  // the host gap instead of sitting inside the next command. The next
  // decode adopts the predrafted proposals when the same lanes continue
  // greedy, unconstrained and unpenalized at the same positions; anything
  // else redrafts. The draft decode is read-only on the rings, so a
  // discarded predraft corrupts nothing.
  struct GpuPredraft final {
    CommandTicket ticket;
    bool submitted = false;
    // Armed inside the step's own command (B-prime): the pack lanes it ran
    // from carry the anchors and positions, so adoption needs no ticket —
    // only the lane bookkeeping finalizeChainedPredraft fills from the
    // readback.
    bool chained = false;
    uint64_t epoch = 0;
    uint32_t lanes = 0;
    std::array<uint64_t, kLaneCount> requestIds{};
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<uint64_t, kLaneCount> positions{};
  };
  GpuPredraft gpuPredraft_;
  const bool gpuPredraftOn_ =
      modelPolicy().gpuPredraft && geometry.tuning.gpuPredraft;
  // RICHENGINE_CHAIN_PREDRAFT=0 keeps the post-finalize second-command
  // predraft: the chained variant needs the step pack's device mirrors,
  // and is off unless the family opts in — it measured slower where the
  // host gap the second command overlaps is wide (Ornith-9B's ANE FFN
  // handoff), while the 35B's tight loop gained it.
  const bool chainPredraft_ =
      modelPolicy().chainPredraft && deviceVerifyParams_;
  // Bumped once per decodeAsync's buffer-writing phase: a predraft's
  // recorded epoch proves no other decode wrote the arena between its
  // submission and the adoption check.
  uint64_t arenaEpoch_ = 0;
  // Waits out a pending speculative draft — the backend admits one
  // in-flight command — and charges its timing into the decode counters.
  // The lane bookkeeping stays for adoptGpuPredraft.
  void settleGpuPredraft() {
    if (!gpuPredraft_.submitted)
      return;
    const CommandTiming timing = gpuPredraft_.ticket.wait();
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    gpuPredraft_.submitted = false;
  }
  // Packs one lane's next-step draft inputs without prepareDecodeLane's
  // readiness asserts or page-table work: the anchor the step just
  // committed plus mask rows, and the draft rotary positions continuing at
  // the lane's new logical position.
  void packDraftPredraftLane(uint32_t lane, uint32_t anchor,
                             uint64_t logicalPosition) {
    const uint32_t queryRows = geometry.draft.queryRows();
    auto *draftInput = contents<uint32_t>(
                           decodeArena->packed(DecodeTensor::DraftInputTokens,
                                               kLaneCount),
                           "draft input tokens") +
                       lane * queryRows;
    draftInput[0] = anchor;
    std::fill(draftInput + 1, draftInput + queryRows,
              geometry.target.maskToken);
    auto *draftPositions =
        contents<uint32_t>(decodeArena->get(lane, DecodeTensor::DraftPositions),
                           "decode draft RoPE positions");
    for (uint32_t row = 0; row < kDecodeRows; ++row)
      draftPositions[row] = static_cast<uint32_t>(logicalPosition + row);
  }
  void launchDraftPredraft(std::span<const DecodeLaneResult> lanes,
                           std::span<const ModelBatchItem> items);
  // Encodes the next step's draft forward into the committed step's own
  // command when every lane can take a chain draft: draft_predraft_pack
  // stages the inputs decode_step_pack just wrote, and the draft graph's
  // attention reads each lane's cache length from the pack. Arms
  // gpuPredraft_'s lane bookkeeping; its anchors/positions are
  // host-known only at finalize, which completeGpuPredraft fills or —
  // on a launch-veto — clears.
  void encodeChainedPredraft(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const ModelBatchItem> items);
  // The finish-callback predraft entry point: a chained arm only needs
  // its lane facts confirmed against the readback (veto drops it);
  // anything else gets the host-launched second command.
  void completeGpuPredraft(std::span<const DecodeLaneResult> lanes,
                           std::span<const ModelBatchItem> items);
  bool adoptGpuPredraft(std::span<Request *const> entries,
                        std::span<const ModelBatchItem> items, uint32_t width,
                        bool allowed);
  // Adaptive draft bypass (default on): when every lane's acceptance EWMA
  // falls below the draft's break-even cost — the draft's per-step weight
  // traffic as a share of the whole step — the batch decodes anchor-only
  // and skips the draft forward, its vocabulary head read and the selector.
  // A lane probes again each kDraftProbeTokens so a recovered EWMA
  // re-enables it. RICHENGINE_DRAFT_BYPASS=0 disables.
  const bool draftBypass_ =
      modelPolicy().draftBypass && geometry.tuning.draftBypass;
  // Break-even measured on the M5 Pro's MiniCPM5-2B-MLX: the draft pass is
  // ~14% of a drafted step, so anchor-only wins once the accepted-token EWMA
  // sits under ~0.16. 0.15 keeps a hair of margin below it.
  const double draftBypassExpect_ = modelPolicy().draftBypassExpect;
  static constexpr uint32_t kDraftProbeTokens = 256;
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        package(value.package),
        geometry(RuntimeGeometry::from(value.package, value.kvPages.layout().format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(value.stateStorage),
        sampling(geometry.target.vocabularySize,
                 geometry.target.paddedVocabularySize),
        // RICHENGINE_DIFFUSION_AR=1 drops a diffusion package onto the
        // ordinary autoregressive decode path: the trunk is the same
        // Gemma4Moe weights, and the model keeps its AR capability. The
        // canvas machinery (arena, warmup, admission) stays parked.
        diffusion(std::holds_alternative<DiffusionGemmaWeights>(
                      value.package.target) &&
                  !tuning().diffusionAr),
        targetModel(std::visit(
                        [&](const TargetModelWeights &weights) {
                          return TargetModel(weights, geometry.target,
                                            value.backend, operators);
                        },
                        value.package.target)),
        draftModel(std::visit(
                       [&](const auto &weights) -> detail::DraftModel {
                         using Draft = typename detail::DraftModelFor<
                             std::decay_t<decltype(weights)>>::Type;
                         // RICHENGINE_NO_DRAFT: force non-speculative
                         // decode — the oracle for verify-stride hunts.
                         if constexpr (std::is_same_v<Draft, NullDraft>)
                           return detail::DraftModel(
                               std::in_place_type<NullDraft>);
                         else {
                           if (envFlag("RICHENGINE_NO_DRAFT"))
                             return detail::DraftModel(
                                 std::in_place_type<NullDraft>);
                           // DFlash2 decoders take a dedicated width
                           // pathway by trained block: block-8 runs the
                           // upstream-native 8-row tile, wider blocks keep
                           // the compiled sixteen.
                           if constexpr (std::is_same_v<Draft, DFlashDraft>)
                             return detail::DraftModel(
                                 geometry.draft.blockSize <=
                                         DFlashTwo8width::kTrainedVerifyWidth
                                     ? detail::DraftModel(
                                           std::in_place_type<
                                               DFlashTwo8width>,
                                           weights, value.backend, operators)
                                     : detail::DraftModel(
                                           std::in_place_type<
                                               DFlashTwo16width>,
                                           weights, value.backend, operators));
                           return detail::DraftModel(std::in_place_type<Draft>,
                                                     weights, value.backend,
                                                     operators);
                         }
                       },
                       value.package.draft)),
        aneFfn(value.aneFfn) {
    if (states.layout() != package.stateLayout() ||
        kvPages.layout() != package.targetKvLayout(kvPages.layout().format)) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded package");
    }
    // A family that fitted its draft's acceptance curve seeds the verify
    // budget's start and shrink target with it (ModelTuning::verifyPrior);
    // unset families keep the upstream models' kPrior.
    if (geometry.tuning.verifyPrior) {
      VerifyBudget::Acceptance seeded{};
      seeded.slope = geometry.tuning.verifyPrior->slope;
      seeded.intercepts = geometry.tuning.verifyPrior->intercepts;
      verifyBudget_.seedAcceptance(seeded);
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
    penaltyTable = decodeArena->packed(DecodeTensor::PenaltyState, kLaneCount);
    // The folded-commit tape is GDN-only: LFM2's conv layers share the
    // state layout but keep the standalone commit.
    if (geometry.target.gdnLayers &&
        !std::getenv("RICHENGINE_NO_GDN_FOLD")) {
      const uint64_t layerBytes = gdnTapeLayerBytes();
      gdnTape_ = backend.allocateBuffer(
          uint64_t{kLaneCount} * 2 * geometry.target.stateLayout.layers *
              layerBytes,
          BufferStorage::Shared, "gdn-decode-tape");
    }
    preparePolicyPipelines();
    if (const char *path = tuning().aneMedusa) {
      std::string error;
      aneMedusa_ = AnePredictor::load(path, error);
      if (!aneMedusa_)
        throw std::invalid_argument("RICHENGINE_ANE_MEDUSA: " + error);
      aneLeafTokens_ = backend.allocateBuffer(
          kLaneCount * RICHENGINE_DRAFT_PROPOSAL_TOKENS * sizeof(uint32_t),
          BufferStorage::Shared, "ane-leaf-tokens");
      aneFlag_ = backend.allocateBuffer(sizeof(uint32_t),
                                        BufferStorage::Shared, "ane-flag");
      *static_cast<uint32_t *>(aneFlag_.contents()) = 0xffffffffu;
    }
    if (const char *path = tuning().anePredraft) {
      std::string error;
      anePredraft_ = AnePredictor::load(path, error);
      if (!anePredraft_)
        throw std::invalid_argument("RICHENGINE_ANE_PREDRAFT: " + error);
    }
  }
  // Warmup selects greedily, so the first sampled, penalized or constrained
  // request would compile the policy's kernels inside its TTFT and stall the
  // engine meanwhile; compile them now. A sampled, penalized and constrained
  // lane and a greedy one reach every kernel the first-token and verify
  // selections dispatch.
  void preparePolicyPipelines() const {
    const ops::SamplingPolicy sampled{.topK = 0,
                                      .temperature = 1.0F,
                                      .topP = 0.95F,
                                      .constrained = true,
                                      .penalties = {1.1F, 0.5F, 0.5F},
                                      .minP = 0.05F};
    const std::array<ops::SamplingPolicy, 2> policies{sampled, {}};
    const std::array<uint32_t, 2> stateLanes{0, 1};
    const ops::PenaltyTable penalties{penaltyTable, stateLanes};
    CommandGraph graph;
    sampling.addInitial(graph, policies, samplingBuffers(2), 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1], penalties);
    sampling.addVerify(graph, policies, samplingBuffers(2),
                       geometry.target.stopTokens[0],
                       geometry.target.stopTokens[1], penalties);
    backend.preparePipelines(graph.dispatches());
  }
  Request &request(uint64_t id) {
    auto found = requests.find(id);
    if (found == requests.end())
      throw std::out_of_range("unknown request");
    return found->second;
  }
  static bool samplingEnabled(const Request &entry) noexcept {
    return entry.sampling.temperature > 0.0F;
  }
  // M-RoPE: text rows advance one counter shared by all three axes;
  // an image's rows spread over (t, h, w) from the counter at the image start
  // and the counter then advances by max(merged height, merged width).
  static std::array<uint32_t, 3> ropePosition(const Request &entry,
                                              uint64_t logical) {
    int64_t delta = 0;
    for (const ImageState &image : entry.images) {
      const ImageSpan &span = image.span;
      if (logical < span.offset)
        break;
      const uint32_t mergedHeight = span.gridHeight / 2;
      const uint32_t mergedWidth = span.gridWidth / 2;
      const uint32_t start =
          static_cast<uint32_t>(static_cast<int64_t>(span.offset) + delta);
      if (logical < span.end()) {
        const uint32_t local = static_cast<uint32_t>(logical - span.offset);
        return {start, start + local / mergedWidth,
                start + local % mergedWidth};
      }
      delta += static_cast<int64_t>(std::max(mergedHeight, mergedWidth)) -
               static_cast<int64_t>(span.tokens);
    }
    const uint32_t position =
        static_cast<uint32_t>(static_cast<int64_t>(logical) + delta);
    return {position, position, position};
  }
  uint64_t embeddingBytes(const ImageSpan &span) const;

  static ImageKey imageKey(const ImageSpan &span) noexcept;

  // The rows of an identical image that something still holds, moved to the
  // front of the embedding cache when it is there; null otherwise.
  std::shared_ptr<ImageRows> findRows(const ImageSpan &span);

  // Keeps encoded rows for reuse as the most recently used, dropping the
  // least recently used while the rows kept for reuse exceed the cache's
  // bytes.
  void retain(const std::shared_ptr<ImageRows> &rows);

  // The bytes of the distinct rows states in RAM hold: all of them, or only
  // those the embedding cache does not hold as well.
  [[nodiscard]] uint64_t heldRowsBytes(bool uncachedOnly) const noexcept;

  // A state in RAM whose boundary lies inside an image, less than a page
  // before its end, holds the image's encoded rows, unless that would take
  // the rows kept for reuse past the cache's bytes: the state returned owns
  // them. Boundaries deeper inside an image keep only the embedding cache.
  std::shared_ptr<const CompositeState>
  holdStraddledRows(const Request &entry, std::shared_ptr<const CompositeState> state);

  // Drops one entry of the embedding cache and returns the bytes it held.
  uint64_t uncache(std::list<std::shared_ptr<ImageRows>>::iterator entry) noexcept;

  // A request lets go of its images; the encoded ones stay in the cache.
  void releaseImages(Request &entry);

  // Frees one cache that can be rebuilt and returns its bytes. The vision
  // arena goes first, when no image waits for its encode and nothing holds
  // it: an image whose rows are encoded never needs it, and the next start
  // that does builds one sized for its own images. Then the least recently
  // used embedding entry nothing else holds, one at a time, since only an
  // encode rebuilds it. An entry something else holds is skipped, since
  // dropping it frees nothing.
  uint64_t releaseOneCache() noexcept;

  // Puts back the encoder a start replaced, or drops the one it built,
  // unless the start completes: an admission granted it, but a later step of
  // the start threw.
  struct VisionRollback final {
    Impl &runtime;
    std::shared_ptr<ops::Vision> previous;
    bool committed = false;
    ~VisionRollback() {
      if (!committed)
        runtime.vision = std::move(previous);
    }
  };
  // What a refused start matched (StateAdmission::held): the rows it would
  // share and, when an image still needs its encode and the live encoder
  // covers it, that encoder.
  struct Matched final {
    std::vector<std::shared_ptr<ImageRows>> rows;
    std::shared_ptr<ops::Vision> encoder;
  };
  // A request's lane with everything else its start allocates, in one
  // admission: the pixel and embedding buffers of the images nothing holds
  // yet and, when an image still needs an encode that the live encoder does
  // not cover, a vision scratch sized for the largest such image. That
  // encoder replaces the live one, which covers fewer patches, so every
  // image waiting on the old one fits the new; a command in flight keeps the
  // old arena until it completes. Rows something holds are shared, encoded
  // or not. Images the restored prefix covers are left out: only their
  // spans are kept. At the budget the engine retries a denied start after
  // each reclaim step, and a denial builds nothing, so no encoder arena,
  // image buffer or lane state is built and dropped every time. The refusal
  // keeps its cause and holds what it matched, so the reclaim before the
  // retry spares it; a grant hands the request's images to `images` and
  // counts the rows it shares as reuses, each once.
  StateAdmission activate(const ModelRequest &request, uint32_t stateLane,
                          std::vector<ImageState> &images);

  // No image waits for its encode, so the vision arena can go.
  [[nodiscard]] bool visionIdle() noexcept;

  // Encodes every image whose rows first appear in this chunk and overwrites
  // the chunk's placeholder embedding rows with the image rows. Text-only
  // requests add no dispatches.
  void addImageRows(CommandGraph &graph, Request &entry,
                    const ModelBatchItem &item, uint32_t rowBegin) {
    const uint64_t chunkBegin = item.logicalPosition;
    const uint64_t chunkEnd = chunkBegin + item.tokenCount;
    for (ImageState &image : entry.images) {
      const uint64_t begin = std::max<uint64_t>(chunkBegin, image.span.offset);
      const uint64_t end = std::min<uint64_t>(chunkEnd, image.span.end());
      if (begin >= end)
        continue;
      if (!image.rows)
        throw std::logic_error("prefill reached an image its activation left out");
      ImageRows &rows = *image.rows;
      if (!rows.encoded && !rows.encoding) {
        if (!vision)
          throw std::logic_error("image request has no vision encoder");
        vision->encode(graph, image.span.grid(), rows.pixels, rows.embeddings);
        rows.encoding = true;
        ++counters.imageEncodes;
      }
      const uint32_t width = package.vision.tensors.layout.outputHiddenSize;
      ops::RowCopy::add(
          graph, rows.embeddings,
          {static_cast<uint32_t>(begin - image.span.offset), width, 0},
          prefillArena->get(PrefillTensor::Hidden0),
          {rowBegin + static_cast<uint32_t>(begin - chunkBegin), width, 0},
          static_cast<uint32_t>(end - begin), width);
    }
  }
  static float nextUniform(Request &entry) noexcept {
    uint64_t value =
        entry.sampling.seed + (++entry.rngCounter) * 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return float(value >> 40) * 0x1p-24F;
  }
  static void stageSamplingCycle(Request &entry) noexcept {
    entry.cycleUniforms.fill(0.0F);
    for (uint32_t index = RICHENGINE_UNIFORM_PROPOSALS;
         index < RICHENGINE_SAMPLING_UNIFORMS; ++index) {
      entry.cycleUniforms[index] = nextUniform(entry);
    }
  }
  [[nodiscard]] MetalBuffer synchronizedPageTable(Request &entry,
                                                  const ModelBatchItem &item);

  // The alt tables, when bound, hold the target's alternate rotary (Gemma's
  // global layers): same positions, altRotaryPairs dims.
  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin, MetalBuffer targetCosAlt = {},
                     MetalBuffer targetSinAlt = {}) const;
  // A constrained lane keeps its final prompt row, which prefill leaves at
  // row 0 of its Hidden0 block, until its first mask arrives.
  void captureFinalHidden(Request &entry, uint32_t lane) const {
    const uint16_t *source =
        contents<uint16_t>(decodeArena->get(lane, DecodeTensor::Hidden0),
                           "target final hidden source");
    entry.finalTargetHidden.assign(source,
                                   source + geometry.target.hiddenSize);
  }
  static DispatchDraftCapturePlan
  activeDraftCaptures(const Request &entry, const ModelBatchItem &item) {
    if (!entry.draftContextPlan) {
      throw std::logic_error("prefill request has no draft context plan");
    }
    const uint64_t next = item.logicalPosition + item.tokenCount;
    return draftCaptureSpansForDispatch(
        *entry.draftContextPlan, static_cast<uint32_t>(item.logicalPosition),
        static_cast<uint32_t>(next));
  }
  static uint32_t captureRows(const DispatchDraftCapturePlan &captures) {
    uint32_t rows = 0;
    for (const auto &capture : captures)
      rows += capture.absoluteEnd - capture.absoluteBegin;
    return rows;
  }
  // The lengths after the draft ring takes rows [begin, end) at target
  // length targetTokens. Unless `reset` starts a new window there, the rows
  // continue the ring, which must hold rows ending at `begin`.
  static LogicalLengths
  advanceDraftContext(const LogicalLengths &previous, uint64_t targetTokens,
                      uint64_t begin, uint64_t end, bool reset,
                      uint32_t draftWindow) {
    if (!reset && (!previous.draftLength || previous.draftEnd() != begin))
      throw std::logic_error("draft capture does not continue the draft ring");
    const uint64_t combined = (reset ? 0 : previous.draftLength) + (end - begin);
    LogicalLengths next = previous;
    next.targetTokens = targetTokens;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, draftWindow));
    next.draftBase = end - next.draftLength;
    return next;
  }
  // Only a sampled lane's draws read its cycle's uniforms.
  void uploadSamplingUniforms(const Request &entry, uint32_t lane) const {
    std::copy(entry.cycleUniforms.begin(), entry.cycleUniforms.end(),
              contents<float>(
                  decodeArena->get(lane, DecodeTensor::SamplingUniforms),
                  "sampling uniforms"));
  }
  // Only a constrained lane's selections read its mask rows.
  std::span<uint32_t> constraintMasks(uint32_t lane) const {
    const MetalBuffer masks =
        decodeArena->get(lane, DecodeTensor::ConstraintMasks);
    return {contents<uint32_t>(masks, "constraint masks"),
            masks.sizeBytes() / sizeof(uint32_t)};
  }
  void uploadConstraintMasks(uint32_t lane,
                             std::span<const uint32_t> masks) const {
    const std::span<uint32_t> rows = constraintMasks(lane);
    if (masks.size() > rows.size())
      throw std::invalid_argument("constraint mask exceeds decode arena");
    std::ranges::copy(masks, rows.begin());
  }
  // A constrained lane whose mask was abandoned admits every token.
  void admitEveryToken(uint32_t lane) const {
    std::ranges::fill(constraintMasks(lane),
                      std::numeric_limits<uint32_t>::max());
  }
  // The whole DfaState table: one u32 automaton state per state lane.
  std::span<uint32_t> dfaStateTable() const {
    const MetalBuffer table =
        decodeArena->packed(DecodeTensor::DfaState, kLaneCount);
    return {contents<uint32_t>(table, "dfa states"), kLaneCount};
  }
  // Uploads a DFA lane's host-mirrored automaton state to its state-lane
  // slot when the mirror is newer (at provideDfa and after a suspension);
  // between those moments decode_dfa_advance keeps the slot authoritative.
  void syncDfaState(Request &entry) const {
    if (!entry.dfaStateDirty)
      return;
    dfaStateTable()[entry.stateLane] = entry.dfaState;
    entry.dfaStateDirty = false;
  }
  // suspend()'s readback: the GPU slot becomes the host mirror's value for
  // whatever lane the resume lands on. A mirror already newer than the slot
  // (the DFA never ran a round on this lane) is kept instead.
  void readbackDfaState(Request &entry) const {
    if (!entry.dfaStateDirty)
      entry.dfaState = dfaStateTable()[entry.stateLane];
    entry.dfaStateDirty = true;
  }
  // Bit i set for each dispatch lane whose request carries a GPU DFA.
  [[nodiscard]] uint32_t
  dfaLaneMask(std::span<Request *const> entries) const {
    uint32_t mask = 0;
    for (uint32_t lane = 0; lane < entries.size(); ++lane)
      if (entries[lane]->dfa)
        mask |= uint32_t{1} << lane;
    return mask;
  }
  // RICHENGINE_DFA_MODE scopes what an installed DFA drives: Full computes
  // the lane's masks and forced spans on the GPU, Forced keeps the host
  // mask round trip and only derives forced spans and the automaton state
  // on-device, Off drops every DFA in provideDfa and stays on the host
  // path entirely.
  enum class DfaMode : uint8_t { Off, Forced, Full };
  static DfaMode dfaModeFromPolicy() {
    // The resolved default: an explicit RICHENGINE_DFA_MODE wins, then the
    // family maker's ModelTuning default, then Off — GPU masks lose to the
    // host round trip inside a 16-row verify loop, so DFA lanes stay on the
    // host path unless asked (a family without a GPU draft never runs that
    // verify loop and may declare "full" itself).
    const std::string_view mode = modelPolicy().dfaMode;
    if (mode == "forced")
      return DfaMode::Forced;
    if (mode == "full")
      return DfaMode::Full;
    return DfaMode::Off;
  }
  const DfaMode dfaMode_ = dfaModeFromPolicy();
  // Whether the entry's masks come from the DFA walk kernels rather than a
  // host mask request: only a Full-mode DFA lane skips the round trip.
  bool gpuDfaMasks(const Request &entry) const noexcept {
    return dfaMode_ == DfaMode::Full && entry.dfa.has_value();
  }
  // gpuDfaMasks over a dispatch: the lanes whose DfaNextStates rows the
  // selection kernels read instead of a constraint bitmask.
  [[nodiscard]] uint32_t
  gpuDfaLaneMask(std::span<Request *const> entries) const {
    return dfaMode_ == DfaMode::Full ? dfaLaneMask(entries) : 0;
  }
  // The greedy lanes of gpuDfaLaneMask, minus any skipped ones: their
  // rounds run the truncation/correction pipeline — an unmasked argmax
  // pass, one truncation row per lane, one masked correction row — while
  // a sampled DFA lane keeps the full next-state rows.
  [[nodiscard]] uint32_t
  greedyDfaLaneMask(std::span<Request *const> entries,
                    uint32_t skip = 0) const {
    const uint32_t mask = gpuDfaLaneMask(entries) & ~skip;
    uint32_t result = 0;
    for (uint32_t lane = 0; lane < entries.size(); ++lane)
      if ((mask & (uint32_t{1} << lane)) &&
          !samplingPolicy(*entries[lane]).samples())
        result |= uint32_t{1} << lane;
    return result;
  }
  // Which per-lane DFA buffer a kernel binds.
  enum class DfaBufferKind : uint8_t { Table, Accept, Marker, Cost };
  // The lane's remaining decode budget for the dfa kernels' parameters:
  // tokens it may still emit, 0xffffffff where the budget rule is off
  // (RICHENGINE_DFA_BUDGET_OFF, its A/B opt-out) and zero past the cap.
  uint32_t dfaTokensLeft(const Request &entry) const noexcept {
    if (tuning().dfaBudgetOff)
      return 0xffffffffu;
    return entry.generatedTokens < entry.maxNewTokens
               ? entry.maxNewTokens - entry.generatedTokens
               : 0;
  }
  // Each dispatch lane's DFA table, accepting bitset or marker mask
  // buffer; lanes without one rebind the first DFA lane's and are masked
  // out. Call only while dfaLaneMask is nonzero.
  [[nodiscard]] std::array<MetalBuffer, kLaneCount>
  dfaBuffers(std::span<Request *const> entries, DfaBufferKind kind);
  // Drops a state lane's pending GPU-forced span (suspend/end/DFA replace):
  // it was derived from the state that lane's request then held.
  void clearDfaForced(uint32_t stateLane) const {
    contents<DfaForcedSpan>(
        decodeArena->packed(DecodeTensor::DfaForced, kLaneCount),
        "dfa forced spans")[stateLane]
        .tokens = 0;
  }
  // A state lane's GPU-forced span for the next round, written by the commit
  // stage's decode_dfa_forced_* kernels and consumed by the next constrained
  // batch's draft stage.
  DfaForcedSpan &dfaForcedSpan(uint32_t stateLane) const {
    return contents<DfaForcedSpan>(
        decodeArena->packed(DecodeTensor::DfaForced, kLaneCount),
        "dfa forced spans")[stateLane];
  }
  // The span lanes whose GPU forced spans the mask-grid kernel consumes
  // on-device under RICHENGINE_GPU_DECODE_GRID, plus each lane's retention
  // limit — the kernel's live-row derivation mirrors the host's.
  struct DfaGridLanes {
    uint32_t gpuSpanMask = 0;
    std::array<uint32_t, kLaneCount> maximumRetained{};
  };
  // The DFA lanes' chain + mask dispatches, encoded after the verify-input
  // tokens pass of the same graph: chain walks the packed InputTokens into
  // DfaRowStates and mask fills the lanes' DfaNextStates rows — each token's
  // post-walk state or -1, one row per live verify row plus the anchor's.
  // A non-null `grid` emits decode_dfa_mask_grid first — it consumes the
  // named GPU forced spans and packs the mask pass's launch on-device —
  // and launches the mask pass through that indirect grid.
  void encodeDfaMasks(CommandGraph &graph, std::span<Request *const> entries,
                      uint32_t lanes, std::span<const uint32_t> liveRows,
                      const DfaGridLanes *grid = nullptr);
  // The greedy DFA lanes' truncation pass (decode_dfa_truncate), encoded
  // after the unmasked selection: names each lane's correction row and
  // patches its GPU forced-span prefix into the selected tokens.
  // proposals caps each lane's compare at the acceptance limit and
  // gpuForced is its GPU-span forced prefix length (0 when the round's
  // proposals did not come from a DfaForcedSpan).
  void encodeDfaTruncation(CommandGraph &graph, uint32_t lanes,
                           uint32_t laneMask,
                           std::span<const uint32_t> proposals,
                           std::span<const uint32_t> gpuForced);
  // The single next-state row each greedy DFA lane's correction needs:
  // decode_dfa_mask in indexed mode writes only mask row trunc+1.
  void encodeDfaCorrectionMasks(CommandGraph &graph,
                                std::span<Request *const> entries,
                                uint32_t lanes, uint32_t laneMask);
  // The masked argmax at each greedy DFA lane's truncation row, the
  // round's correction or bonus token (decode_dfa_argmax_*).
  void encodeDfaCorrections(CommandGraph &graph,
                            std::span<Request *const> entries,
                            uint32_t lanes, uint32_t laneMask,
                            std::span<const uint32_t> liveRows);
  // After acceptance: each DFA lane's persistent automaton state advances
  // by its committed tokens (the anchor plus retained outputs minus the new
  // anchor), ready for the next round's row-0 state.
  void encodeDfaAdvance(CommandGraph &graph,
                        std::span<Request *const> entries, uint32_t lanes);
  // After the state advance, each DFA lane's threadgroup derives the next
  // round's forced-token span on-device into the state lane's DfaForcedSpan
  // (and the dispatch lane's ProposedTokens).
  void encodeDfaForced(CommandGraph &graph,
                       std::span<Request *const> entries, uint32_t lanes);
  // The merged constrained step's on-device span apply
  // (RICHENGINE_DFA_MERGE), dispatched between the draft pass and the
  // verify input of the same command: the in-command twin of the split
  // path's host copy of the pending DfaForcedSpan over ProposedTokens.
  void encodeDfaSpanApply(CommandGraph &graph,
                          std::span<Request *const> entries, uint32_t lanes,
                          uint32_t laneMask,
                          std::span<const uint32_t> maximumRetained);
  // The initial selection's row-0 mask for each DFA lane: the host writes
  // the lane's mirrored state into its DfaRowStates row 0 first, then the
  // mask dispatch produces one next-state row.
  void encodeDfaInitialMasks(CommandGraph &graph,
                             std::span<Request *const> entries,
                             uint32_t lanes);
  static ops::SamplingPenalties samplingPenalties(const Request &entry) noexcept {
    return {entry.sampling.repetitionPenalty, entry.sampling.presencePenalty,
            entry.sampling.frequencyPenalty};
  }
  static ops::SamplingPolicy samplingPolicy(const Request &entry) noexcept {
    return {entry.sampling.topK, entry.sampling.temperature,
            entry.sampling.topP, entry.constraint == ConstraintMode::TokenMask,
            (entry.flags & RequestIgnoreEndOfSequence) != 0,
            samplingPenalties(entry), entry.sampling.minP};
  }
  // Whether the model's draft emits comb-tree tables, per its descriptor's
  // tuning (makeModelDescriptor defaults Null drafts On and the GPU drafts
  // to opt-in; a family maker can override): the env's RICHENGINE_VERIFY_TREE
  // wins whenever it is set.
  bool treeDraftCapable() const {
    // The comb layout halves the node block: the chain's front rows anchor
    // it and each leading position's runner-up becomes a sibling leaf in
    // the back half, so the chain never needs more than half the stride.
    static_assert(RICHENGINE_TREE_VERIFY_NODES >= 2 &&
                      RICHENGINE_TREE_VERIFY_NODES % 2 == 0,
                  "the comb splits the verify nodes into equal halves");
    const auto policy = geometry.tuning.treeVerify;
    if (policy == ModelTuning::TreeVerify::Off)
      return false;
    return verifyTreeEnvSet_ ? verifyTreeEnabled
                             : policy != ModelTuning::TreeVerify::OptIn;
  }
  // Whether a lane's rolling acceptance decides its tree admission: only the
  // maker's Adaptive policy — an explicit RICHENGINE_VERIFY_TREE is a static
  // On/Off that bypasses the band entirely.
  bool treeAdmissionAdaptive() const noexcept {
    return !verifyTreeEnvSet_ &&
           geometry.tuning.treeVerify == ModelTuning::TreeVerify::Adaptive;
  }
  // A batch verifies the selector's comb trees only when every lane can:
  // greedy, unconstrained and unpenalized tree-capable lanes, at most two
  // wide (a tree lane doubles its row block, and four virtual lanes is the
  // row budget), on a target with the tree kernels — GDN, conv and
  // attention mixers, dense FFN, no fp8 KV. Anything else runs the chain
  // verify.
  bool treeVerifyBatch(std::span<Request *const> entries, uint32_t width,
                       bool constrained) const {
    // GPU-drafted sparse MoE keeps the chain verify: the tree's doubled
    // lane block routes its dead slot too, which measured a 2.5x union
    // (31 -> 79 experts/layer) for no retained-token gain on MiniCPM-V-4.7 —
    // 40.9 tok/s against the chain's 59.6. A Null-draft MoE (LFM2.5-24B-A2B)
    // proposes from the n-gram predraft instead: its sparser trees get the
    // gate until a measurement says the union inflates the same way.
    // RICHENGINE_VERIFY_TREE=1 or a family's treeVerify On opts a
    // GPU-drafted sparse-MoE target into trees anyway: the MiniCPM-V
    // measurement keeps the gate by default.
    const bool treeOn = verifyTreeEnvSet_
                            ? verifyTreeEnabled
                            : geometry.tuning.treeVerify ==
                                  ModelTuning::TreeVerify::On;
    if (constrained || !treeDraftCapable() || !width ||
        width > kLaneCount / 2 ||
        (!treeOn && geometry.target.ffnKind == FfnKind::SparseMoe &&
         !std::holds_alternative<NullDraft>(draftModel)) ||
        geometry.target.kvLayout.format == kv::Format::Float8E4M3)
      return false;
    for (uint32_t lane = 0; lane < width; ++lane) {
      const ops::SamplingPolicy policy = samplingPolicy(*entries[lane]);
      if (policy.samples() || policy.constrained ||
          policy.penalties.active())
        return false;
    }
    // Adaptive admission: the batch trees only while every lane's
    // accepted-proposal EWMA sits inside the band — saturated lanes keep
    // the deeper chain (the comb's leaves never win once the top pick
    // lands), starved ones aren't worth the doubled row block.
    if (treeAdmissionAdaptive()) {
      for (uint32_t lane = 0; lane < width; ++lane) {
        const double ewma = entries[lane]->proposalAcceptedAvg;
        if (ewma < modelPolicy().treeAcceptMin ||
            ewma > modelPolicy().treeAcceptMax)
          return false;
      }
    }
    return true;
  }
  ops::SamplingBuffers samplingBuffers(uint32_t lanes) const;

  std::span<uint32_t> penaltyWords(uint32_t stateLane) const;

  // Rebuilds a penalized request's penalty words when it takes a state lane,
  // at activation and at resume, from the history the lane's prefill
  // consumes. No command reads the lane's words yet.
  void bindPenalties(const Request &entry,
                     std::span<const uint32_t> history) const;

  // The one place a token the target selected becomes the pending anchor:
  // tokens are one step's selections in order, the new anchor last. The
  // command that selected them has completed, and the next one that reads
  // the lane's words is encoded after this.
  void commitSelected(Request &entry, std::span<const uint32_t> tokens) {
    if (tokens.empty())
      throw std::logic_error("no selected token to commit");
    if (samplingPenalties(entry).active())
      ops::Sampling::countPenaltyTokens(penaltyWords(entry.stateLane), tokens);
    entry.pendingToken = tokens.back();
    appendNgramTokens(entry, tokens);
  }
  // A selection outside the vocabulary is the sampling kernels' sentinel for a
  // non-finite logit row: a numerical outcome of this request, which it reports
  // as its lane failure (ModelStepResult::failure) so the batch survives.
  [[nodiscard]] std::string invalidSelection(std::span<const uint32_t> tokens) const {
    const auto found = std::find_if(tokens.begin(), tokens.end(), [&](uint32_t token) {
      return token >= geometry.target.vocabularySize;
    });
    if (found == tokens.end())
      return {};
    return "target selected out-of-vocabulary token " + std::to_string(*found) +
           " from a non-finite logit row";
  }
  // A lane of an initial selection, whose final prompt row is at row 0 of
  // its Hidden0 block: one that selects its first token, or a score lane,
  // which needs only the logits.
  struct InitialSelection final {
    Request *entry = nullptr;
    uint32_t lane = 0;
    bool select = false;
  };
  // A sampled lane draws its first token with the first uniform of a fresh
  // cycle.
  void uploadInitialUniform(Request &entry, uint32_t lane) const {
    entry.cycleUniforms.fill(0.0F);
    entry.cycleUniforms[RICHENGINE_UNIFORM_INITIAL] = nextUniform(entry);
    uploadSamplingUniforms(entry, lane);
  }
  // One LM head over the batch's `width` lanes, then one selection of the
  // first token of every selecting lane from its logits row 0, which
  // initialToken() reads. The head computes, and nothing reads, the other
  // rows of each lane and the lanes not listed; a lane that does not select
  // takes the argmax of its row. Sampled lanes' uniforms and constrained
  // lanes' masks are uploaded first.
  void encodeInitialSelections(CommandGraph &graph,
                               std::span<const InitialSelection> lanes,
                               uint32_t width) {
    const uint32_t storage = targetModel.decodeStorageLanes(width);
    auto d = [&](DecodeTensor tensor) {
      return decodeArena->packed(tensor, storage);
    };
    // An all-score batch needs the logits at its lanes' score columns and
    // nowhere else: gather just the quant tiles covering them
    // (TargetModel::addScoreHeadColumns), or, with SCORE_TENSOR /
    // SCORE_FUSED, the wanted columns in one dispatch
    // (TargetModel::addScoreHeadGather). A lane that selects, or a head
    // the restricted dispatch cannot take, keeps the full projection.
    if ((scoreGather_ || scoreTensor_ || scoreFused_) &&
        std::ranges::none_of(lanes, &InitialSelection::select)) {
      std::vector<uint32_t> wanted;
      for (const InitialSelection &selection : lanes)
        for (uint32_t token : selection.entry->scoreTokens)
          wanted.push_back(token);
      std::ranges::sort(wanted);
      const auto [first, last] = std::ranges::unique(wanted);
      wanted.erase(first, last);
      if (scoreTensor_ || scoreFused_) {
        const MetalBuffer &buffer = scoreColumnsBuffer();
        if (wanted.size() <= buffer.sizeBytes() / sizeof(uint32_t)) {
          std::memcpy(buffer.contents(), wanted.data(),
                      wanted.size() * sizeof(uint32_t));
          if (targetModel.addScoreHeadGather(
                  graph, d(DecodeTensor::Hidden0),
                  d(DecodeTensor::FinalHidden), d(DecodeTensor::Logits),
                  width, decodeArena->linearScratch(), buffer,
                  static_cast<uint32_t>(wanted.size()), scoreFused_))
            return;
        }
      }
      if (scoreGather_) {
        std::vector<uint32_t> tiles;
        for (uint32_t token : wanted)
          tiles.push_back(token / QUANT_TILE_ROWS);
        const auto [tfirst, tlast] = std::ranges::unique(tiles);
        tiles.erase(tfirst, tlast);
        if (targetModel.addScoreHeadColumns(
                graph, d(DecodeTensor::Hidden0), d(DecodeTensor::FinalHidden),
                d(DecodeTensor::Logits), width, decodeArena->linearScratch(),
                tiles))
          return;
      }
    }
    targetModel.addHeadBatch(graph, d(DecodeTensor::Hidden0),
                             d(DecodeTensor::FinalHidden),
                             d(DecodeTensor::Logits), width,
                             decodeArena->linearScratch());
    if (std::ranges::none_of(lanes, &InitialSelection::select))
      return;
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    ops::SamplingBuffers buffers = samplingBuffers(width);
    for (const InitialSelection &lane : lanes) {
      if (!lane.select)
        continue;
      policies[lane.lane] = samplingPolicy(*lane.entry);
      stateLanes[lane.lane] = lane.entry->stateLane;
      // A GPU-masked DFA lane's row-0 admission lives in its next-state
      // table; other lanes read a constraint bitmask.
      if (gpuDfaMasks(*lane.entry))
        buffers.dfaLanes |= uint32_t{1} << lane.lane;
    }
    sampling.addInitial(graph, std::span(policies).first(width),
                        buffers, 0,
                        geometry.target.stopTokens[0],
                        geometry.target.stopTokens[1],
                        {penaltyTable, std::span(stateLanes).first(width)});
  }
  // The first token encodeInitialSelections() selected for a batch lane: a
  // selection writes one output token per lane, in lane order.
  uint32_t initialToken(uint32_t lane) const {
    return contents<uint32_t>(
        decodeArena->packed(DecodeTensor::OutputTokens, lane + 1),
        "initial tokens")[lane];
  }
  // Selects the first token of each constrained lane of the plan under the
  // mask it was given, from its final prompt row, in one command. The lanes
  // draft and verify from their next plan on.
  std::unique_ptr<ModelBatchTicket>
  submitInitialSelection(std::span<const ModelBatchItem> items,
                         std::function<void()> completion) {
    const uint32_t width = static_cast<uint32_t>(items.size());
    std::array<InitialSelection, kLaneCount> selections{};
    std::array<Request *, kLaneCount> entries{};
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = request(items[lane].requestId);
      const bool gpuMask = gpuDfaMasks(entry);
      if (entry.decodeStage != DecodeStage::ApplyInitialMask ||
          entry.pendingToken ||
          (gpuMask ? !entry.maskWords.empty()
                   : entry.maskWords.size() != geometry.maskWords()) ||
          entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
        throw std::logic_error("initial selection state is invalid");
      }
      std::ranges::copy(entry.finalTargetHidden,
                        contents<uint16_t>(
                            decodeArena->get(lane, DecodeTensor::Hidden0),
                            "final prompt hidden"));
      if (samplingEnabled(entry))
        uploadInitialUniform(entry, lane);
      if (!gpuMask)
        uploadConstraintMasks(lane, entry.maskWords);
      selections[lane] = {&entry, lane, true};
      entries[lane] = &entry;
    }
    CommandGraph graph;
    // GPU-DFA lanes' row-0 masks come from the walk kernels, concurrent
    // with the head's forward in the same command.
    encodeDfaInitialMasks(graph, {entries.data(), width}, width);
    encodeInitialSelections(graph, std::span(selections).first(width), width);
    CommandTicket command =
        backend.submitCommandAsync(graph.dispatches(), std::move(completion));
    auto finish = [this, selections, width](CommandTiming) {
      std::vector<ModelStepResult> results;
      results.reserve(width);
      for (const InitialSelection &selection :
           std::span(selections).first(width)) {
        Request &entry = *selection.entry;
        ModelStepResult &result = results.emplace_back();
        result.requestId = entry.id;
        const uint32_t token = initialToken(selection.lane);
        // A failed selection leaves no anchor: the engine ends the request
        // before any output, and the other lanes go on.
        result.failure = invalidSelection({&token, 1});
        if (!result.failure.empty())
          continue;
        commitSelected(entry, {&token, 1});
        entry.maskWords.clear();
        entry.finalTargetHidden.clear();
        entry.decodeStage = DecodeStage::Regular;
        emitTerminalAnchor(entry, result);
      }
      return results;
    };
    return std::make_unique<detail::DeferredMetalTicket>(std::move(command),
                                                 std::move(finish));
  }
  ChunkedPrefillParams chunkParams(uint64_t logicalPosition,
                                   uint32_t chunkTokens, uint32_t chunkStride,
                                   std::span<const uint32_t> pages) const {
    return ops::PagedAttention::prefillParams(
        logicalPosition, chunkTokens, chunkStride,
        static_cast<uint32_t>(pages.size()));
  }
  struct PackedPrefillSequence final {
    Request *entry = nullptr;
    const ModelBatchItem *item = nullptr;
    uint32_t lane = 0;
    uint32_t rowBegin = 0;
    uint32_t attentionStride = 0;
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    uint32_t captureBegin = 0;
    ChunkedPrefillParams chunk;
    MetalBuffer pageTable;
    DispatchDraftCapturePlan captures;
    // qwen4exp's host-computed layer inputs: each row's PLE table ids
    // (pleHeads() per row, full-prefix history) and the QSA block records
    // its rows complete. Empty on other targets.
    std::vector<uint32_t> pleRowIds;
    std::vector<BlockIndexerBlock> qsaBlocks;
  };
  struct PackedPrefillBatch final {
    std::vector<PackedPrefillSequence> sequences;
    uint32_t rows = 0;
    uint32_t capturedRows = 0;
  };
  MetalBuffer prefillU16(const MetalBuffer &tensor, uint32_t begin,
                         uint32_t rows, uint32_t width) const {
    return backend.view(tensor, bytesFor<uint16_t>(uint64_t{begin} * width),
                        bytesFor<uint16_t>(uint64_t{rows} * width));
  }
  PackedPrefillBatch
  preparePackedPrefill(std::span<const ModelBatchItem> items,
                       std::array<Request *, kLaneCount> &entries,
                       uint32_t inputBank) {
    PackedPrefillBatch batch;
    batch.sequences.reserve(items.size());
    uint64_t queryOffset = 0;
    uint64_t kvOffset = 0;
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const ModelBatchItem &item = items[lane];
      Request &entry = request(item.requestId);
      if (item.tokenCount > kPrefillRows ||
          item.logicalPosition > entry.promptTokens ||
          item.tokenCount > entry.promptTokens - item.logicalPosition ||
          !entry.resident) {
        throw std::invalid_argument("invalid packed prefill item");
      }
      const LaneMetadata &metadata = states.metadata(entry.stateLane);
      // A chunk submitted ahead of its predecessor's consumption still has
      // the predecessor's rows unapplied: lengths advance only at consume.
      if (metadata.requestId != entry.id ||
          metadata.lengths.targetTokens + entry.prefillUnappliedRows !=
              item.logicalPosition) {
        throw std::logic_error("packed prefill state length is not exact");
      }
      if (item.logicalPosition == 0) {
        states.clearForColdStart(entry.stateLane);
        // Pending tape rows reset with the cold start: a braced block, or
        // the reset rides along every eye-checked non-cold item too.
        gdnPendingRows_[entry.stateLane] = 0;
      }
      if (item.tokenCount > kPrefillRows - batch.rows) {
        throw std::invalid_argument("packed prefill exceeds actual-row budget");
      }
      auto captures = activeDraftCaptures(entry, item);
      const uint32_t capturedRows = captureRows(captures);
      const uint32_t attentionStride =
          ((item.tokenCount + kTileRows - 1) / kTileRows) * kTileRows;
      const ChunkedPrefillParams chunk =
          chunkParams(item.logicalPosition, item.tokenCount, attentionStride,
                      item.pageTable);
      MetalBuffer pageTable = synchronizedPageTable(entry, item);
      PackedPrefillSequence &sequence = batch.sequences.emplace_back(
          PackedPrefillSequence{&entry, &item, lane, batch.rows,
                                attentionStride, queryOffset, kvOffset,
                                batch.capturedRows, chunk, std::move(pageTable),
                                std::move(captures), {}, {}});
      // qwen4exp's n-gram tails and QSA block records span the whole
      // prompt: the chunk's inputTokens alone lacks both.
      if (const auto *qwen4 =
              std::get_if<Qwen4ExpWeights>(&package.target)) {
        sequence.pleRowIds.resize(
            uint64_t{item.tokenCount} * qwen4->layout.pleHeads());
        if (item.requestTokens.size() < item.logicalPosition)
          throw std::invalid_argument(
              "prefill item lacks its request's tokens");
        qwen4ExpPleRows(*qwen4, item.logicalPosition, item.inputTokens,
                        item.requestTokens, sequence.pleRowIds.data());
        for (uint64_t last = item.logicalPosition;
             last < item.logicalPosition + item.tokenCount; ++last) {
          if (last % BLOCK_INDEXER_BLOCK != BLOCK_INDEXER_BLOCK - 1)
            continue;
          const uint64_t first = last - (BLOCK_INDEXER_BLOCK - 1);
          const std::array<uint32_t, 3> rotary =
              ropePosition(entry, first);
          sequence.qsaBlocks.push_back(
              {static_cast<uint32_t>(first),
               {rotary[0], rotary[1], rotary[2]}});
        }
      }
      entries[lane] = &entry;
      batch.rows += item.tokenCount;
      batch.capturedRows += capturedRows;
      queryOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionQueryHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
      kvOffset += bytesFor<uint16_t>(
          uint64_t{geometry.target.attentionKvHeads} * attentionStride *
          geometry.target.attentionHeadDimension);
    }
    if (!batch.rows ||
        queryOffset >
            prefillArena->get(PrefillTensor::FullQueries).sizeBytes() ||
        kvOffset > prefillArena->get(PrefillTensor::ChunkKeys).sizeBytes()) {
      throw std::logic_error("packed prefill scratch geometry overflowed");
    }

    auto *input =
        contents<uint32_t>(prefillArena->get(PrefillTensor::InputTokens,
                                             inputBank),
                           "packed prefill input tokens");
    auto *targetPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::TargetPositions,
                                             inputBank),
                           "target RoPE positions");
    auto *draftPositions =
        contents<uint32_t>(prefillArena->get(PrefillTensor::DraftPositions,
                                             inputBank),
                           "draft RoPE positions");
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      const ModelBatchItem &item = *sequence.item;
      std::copy(item.inputTokens.begin(), item.inputTokens.end(),
                input + sequence.rowBegin);
      for (uint32_t localRow = 0; localRow < item.tokenCount; ++localRow) {
        const uint32_t row = sequence.rowBegin + localRow;
        if (input[row] >= geometry.target.vocabularySize) {
          throw std::invalid_argument("prompt token is out of vocabulary");
        }
        const std::array<uint32_t, 3> rotary =
            ropePosition(*sequence.entry, item.logicalPosition + localRow);
        std::copy(rotary.begin(), rotary.end(), targetPositions + row * 3);
      }
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        for (uint32_t row = capture.absoluteBegin; row < capture.absoluteEnd;
             ++row) {
          const uint32_t compactRow = sequence.captureBegin +
                                      capture.compactDestinationRow + row -
                                      capture.absoluteBegin;
          draftPositions[compactRow] = row;
        }
      }
    }
    return batch;
  }
  void addPackedDraftContext(CommandGraph &graph,
                             const PackedPrefillBatch &batch) {
    if (!batch.capturedRows)
      return;
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    std::array<DFlashPrefillSpan, kLaneCount * 2> spans{};
    uint32_t spanCount = 0;
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      for (const DispatchDraftCaptureSpan &capture : sequence.captures) {
        DFlashPrefillSpan &span = spans.at(spanCount++);
        span.compactRow = sequence.captureBegin + capture.compactDestinationRow;
        span.rows = capture.absoluteEnd - capture.absoluteBegin;
        span.startPosition = capture.absoluteBegin;
        span.ring = states.draft(sequence.entry->stateLane);
      }
    }
    std::visit([&](const auto &model) {
      model.addContextPrefill(
        graph,
        {p(PrefillTensor::Captured), p(PrefillTensor::ProjectionSums),
         p(PrefillTensor::ContextProjected), p(PrefillTensor::ContextHidden),
         p(PrefillTensor::ContextKv), p(PrefillTensor::ContextQkv),
         p(PrefillTensor::DraftRopeCos),
         p(PrefillTensor::DraftRopeSin),
         {.input = p(PrefillTensor::LinearPacked),
          .sums = p(PrefillTensor::LinearExponents),
          .partials = p(PrefillTensor::LinearPartials),
          .counters = p(PrefillTensor::LinearCounters),
          .rotated = p(PrefillTensor::LinearRotated),
          .i8codes = p(PrefillTensor::I8Codes),
          .i8codesLo = p(PrefillTensor::I8CodesLo),
          .i8params = p(PrefillTensor::I8Params),
          .i8paramsLo = p(PrefillTensor::I8ParamsLo)}},
        batch.capturedRows, std::span(spans).first(spanCount));
    }, draftModel);
  }
  // Returns each lane's draft captures, indexed like `entries`.
  std::array<DispatchDraftCapturePlan, kLaneCount>
  encodePackedPrefillGraph(CommandGraph &graph,
                           std::span<const ModelBatchItem> items,
                           std::array<Request *, kLaneCount> &entries,
                           uint32_t inputBank = 0) {
    PackedPrefillBatch batch = preparePackedPrefill(items, entries, inputBank);
    auto p = [&](PrefillTensor tensor) { return prefillArena->get(tensor); };
    auto pi = [&](PrefillTensor tensor) {
      return prefillArena->get(tensor, inputBank);
    };

    addRopeTables(graph, pi(PrefillTensor::TargetPositions), batch.rows,
                  pi(PrefillTensor::DraftPositions), batch.capturedRows,
                  p(PrefillTensor::RopeCos), p(PrefillTensor::RopeSin),
                  p(PrefillTensor::DraftRopeCos),
                  p(PrefillTensor::DraftRopeSin),
                  p(PrefillTensor::RopeCosAlt), p(PrefillTensor::RopeSinAlt));

    targetModel.addEmbedding(graph, pi(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<TargetModelPrefillSequence, kLaneCount> modelSequences{};
    const uint32_t modelSequenceCount =
        static_cast<uint32_t>(batch.sequences.size());
    const uint64_t stateBindingCount = uint64_t{modelSequenceCount} *
                                       geometry.target.stateLayout.layers;
    std::vector<MetalBuffer> convolutionIn(stateBindingCount);
    std::vector<MetalBuffer> convolutionOut(stateBindingCount);
    std::vector<MetalBuffer> recurrentIn(stateBindingCount);
    std::vector<MetalBuffer> recurrentOut(stateBindingCount);
    for (uint32_t lane = 0; lane < batch.sequences.size(); ++lane) {
      const PackedPrefillSequence &sequence = batch.sequences[lane];
      TargetModelPrefillSequence &destination = modelSequences[lane];
      destination.rowBegin = sequence.rowBegin;
      destination.rows = sequence.item->tokenCount;
      destination.attentionStride = sequence.attentionStride;
      destination.queryOffset = sequence.queryOffset;
      destination.kvOffset = sequence.kvOffset;
      destination.chunk = sequence.chunk;
      destination.pageTable = sequence.pageTable;
      const uint32_t gdnLayers = geometry.target.stateLayout.layers;
      const uint64_t stateBegin = uint64_t{lane} * gdnLayers;
      destination.convolutionIn =
          std::span(convolutionIn).subspan(stateBegin, gdnLayers);
      destination.convolutionOut =
          std::span(convolutionOut).subspan(stateBegin, gdnLayers);
      destination.recurrentIn =
          std::span(recurrentIn).subspan(stateBegin, gdnLayers);
      destination.recurrentOut =
          std::span(recurrentOut).subspan(stateBegin, gdnLayers);
      // An odd count of unconsumed chunks means the pending command's output
      // is this chunk's input: bind the parities it will swap between.
      const bool pendingSwap =
          (sequence.entry->prefillUnapplied & 1) != 0;
      const GdnParityBuffers &in = pendingSwap
          ? states.next(sequence.entry->stateLane)
          : states.current(sequence.entry->stateLane);
      const GdnParityBuffers &out = pendingSwap
          ? states.current(sequence.entry->stateLane)
          : states.next(sequence.entry->stateLane);
      for (uint32_t layer = 0; layer < gdnLayers; ++layer) {
        convolutionIn[stateBegin + layer] = in.convolutionLayers[layer];
        convolutionOut[stateBegin + layer] = out.convolutionLayers[layer];
        recurrentIn[stateBegin + layer] = in.recurrentLayers[layer];
        recurrentOut[stateBegin + layer] = out.recurrentLayers[layer];
      }
      // The state cells' auxiliary region (PLE history + QSA pending keys)
      // and the block index plane's entries, for the layouts that carry
      // them — qwen4exp's.
      if (const uint64_t auxiliary =
              geometry.target.stateLayout.auxiliaryBytes) {
        const uint64_t offset =
            geometry.target.stateLayout.convolutionBytes() +
            geometry.target.stateLayout.recurrentBytes();
        destination.auxiliaryIn =
            backend.view(in.stateBase, offset, auxiliary);
        destination.auxiliaryOut =
            backend.view(out.stateBase, offset, auxiliary);
      }
      if (geometry.target.kvLayout.indexDimension)
        destination.indexTable =
            decodeArena->get(sequence.entry->stateLane,
                             DecodeTensor::IndexTable);
      destination.pleRowIds = std::span<const uint32_t>(sequence.pleRowIds);
      destination.qsaBlocks =
          std::span<const BlockIndexerBlock>(sequence.qsaBlocks);
      destination.captureCount = sequence.captures.size();
      for (uint32_t index = 0; index < sequence.captures.size(); ++index) {
        const DispatchDraftCaptureSpan &capture = sequence.captures[index];
        destination.captures[index] = {
            sequence.rowBegin +
                static_cast<uint32_t>(capture.absoluteBegin -
                                      sequence.item->logicalPosition),
            sequence.captureBegin + capture.compactDestinationRow,
            capture.absoluteEnd - capture.absoluteBegin};
      }
    }
    TargetModelPrefillBuffers buffers = detail::prefillBuffers(*prefillArena);
    const MetalBuffer finalHidden = targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvPages.layers(), aneFfn, encoderLayerScalars());
    addPackedDraftContext(graph, batch);

    // A lane that finishes its prompt copies the prompt's last row to row 0
    // of its Hidden0 block. A constrained lane's completion captures that
    // row into finalTargetHidden (captureFinalHidden), which holds it until
    // the first mask. The others share one head: a score lane reads raw
    // logits at the final prompt position, and a policy lane selects its
    // first token.
    std::array<InitialSelection, kLaneCount> selections{};
    uint32_t selectionCount = 0;
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      Request &entry = *sequence.entry;
      const ModelBatchItem &item = *sequence.item;
      if (diffusion || entry.replayingGeneration ||
          item.logicalPosition + item.tokenCount != entry.promptTokens)
        continue;
      const uint32_t hidden = geometry.target.hiddenSize;
      ops::RowCopy::add(
          graph,
          prefillU16(finalHidden, sequence.rowBegin, item.tokenCount, hidden),
          {item.tokenCount - 1, hidden, 0},
          decodeArena->get(sequence.lane, DecodeTensor::Hidden0),
          {0, hidden, 0}, 1, hidden);
      if (entry.constraint != ConstraintMode::None)
        continue;
      const bool scoring = !entry.scoreTokens.empty();
      if (!scoring && samplingEnabled(entry))
        uploadInitialUniform(entry, sequence.lane);
      selections[selectionCount++] = {&entry, sequence.lane, !scoring};
    }
    if (selectionCount) {
      encodeInitialSelections(
          graph, std::span(selections).first(selectionCount),
          static_cast<uint32_t>(batch.sequences.size()));
    }
    std::array<DispatchDraftCapturePlan, kLaneCount> captures{};
    for (const PackedPrefillSequence &sequence : batch.sequences)
      captures[sequence.lane] = sequence.captures;
    return captures;
  }
  void prepareDecodeLane(Request &entry, const ModelBatchItem &item,
                         uint32_t lane);
  // Batch lanes beyond the active width replay the last active request so
  // every padded M32 lane binds valid state.
  static Request &laneEntry(std::span<Request *const> entries, uint32_t lane);

  // Lane bindings for the padded physical width: the real entries in order,
  // then the last lane replayed (laneEntry's rule).
  void bindPageTables(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &pageTables) const;
  // The lanes' block index tables — the same pages' index-plane entries —
  // when the kv layout carries one (qwen4exp's QSA); untouched elsewhere.
  void bindIndexTables(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &indexTables) const;

  void bindGdnStates(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &current,
      std::array<MetalBuffer, kLaneCount> &next) const;

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const;
  // `devicePack` binds the step's DecodeStepPackLane block for the
  // attention splits' cache lengths (the chained predraft); drafts that
  // cannot consume it keep their host-staged lengths.
  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions,
                             const MetalBuffer &devicePack = {});
  // bf16 rows (the arena's hidden storage) to the fp16 the predictor
  // contracts name: a bf16 is the top half of a float's bits.
  static void bf16ToFp16Row(const uint16_t *source, _Float16 *target,
                            uint32_t count);
  // Everything one committed step hands the predictors, copied out of the
  // arena so a queued job never reads buffers the GPU still owns.
  struct AneStepInput final {
    uint32_t lanes = 0;
    std::vector<_Float16> hidden;  // [lane][verify rows][hidden]
    std::array<uint32_t, kLaneCount> anchors{};
    std::array<uint64_t, kLaneCount> positions{};
    std::array<int32_t, kLaneCount> retained{};
  };
  // Feeds the predictors the state this batch just committed: the medusa
  // predictor's leaf alternates apply to the next tree batch, the predraft
  // predictor's proposal chain applies to whatever lane still holds the
  // anchor and position it predicted. Kicking here — before the results
  // reach the engine — gives a job the whole draft+verify of the next step.
  void kickAnePredictors(std::span<const DecodeLaneResult> lanes,
                         std::span<const ModelBatchItem> items);
  // A completed predraft whose assumed anchors and positions match this step
  // replaces the draft forward: its proposals land in the lanes'
  // ProposedTokens and the chain verify consumes them unchanged. Anything
  // else — a running job, a stale or different-laned result, a sampled lane —
  // keeps the GPU draft.
  bool applyAnePredraft(std::span<Request *const> entries,
                        std::span<const ModelBatchItem> items,
                        uint32_t width);
  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected. NgramIndex.hpp owns the
  // table logic.
  void seedNgramHistory(Request &entry, std::span<const uint32_t> prompt);
  void appendNgramTokens(Request &entry, std::span<const uint32_t> tokens);
  // A lane's expected accepted tokens if its n-gram chain is used now:
  // no match contributes nothing; a lane still warming up or due a probe
  // is scored at the draft's expected rate so it can prove itself; after
  // that its observed EWMA speaks.
  double ngramLaneScore(const Request &entry, bool found) const;
  // The learning-free predraft: each lane's chain comes from the followers
  // of its closing 3-gram's last earlier occurrence; a lane without a match
  // repeats its anchor (a wrong guess only wastes its verify rows). The
  // batch goes predrafted when the lanes' expected scores total what the
  // GPU draft would have accepted — mixed-quality lanes no longer veto.
  // Greedy lanes only: injected proposals carry no probabilities. When
  // emitTree is set the lane also writes a comb tree table — chain rows,
  // one sibling leaf per proposal position that has a distinct alternate —
  // and ngram_.predraftedTree_ reports whether the batch left as a tree.
  bool applyNgramPredraft(std::span<Request *const> entries,
                          uint32_t width, bool emitTree);
  void encodeTargetVerifyBatchForward(CommandGraph &graph,
                                      std::span<Request *const> entries,
                                      std::span<const ModelBatchItem> items,
                                      bool tree = false,
                                      std::span<const uint32_t> liveRows = {});
  // The sampling buffers of a tree batch: lane-semantic tensors keep their
  // lanes' strides, while the row-indexed logits and the argmax output take
  // the tree's RICHENGINE_TREE_VERIFY_NODES stride.
  ops::SamplingBuffers samplingTreeBuffers(uint32_t lanes) const;
  void encodeTargetVerifyBatchPolicy(CommandGraph &graph,
                                     std::span<Request *const> entries,
                                     bool tree = false,
                                     std::span<const uint32_t> liveRows = {},
                                     uint32_t dfaLaneSkip = 0);
  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   bool tree = false);
  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained,
                             bool tree = false,
                             std::span<const uint32_t> proposals = {},
                             uint32_t truncLanes = 0);
  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes,
                            uint32_t rowFactor = 1,
                            uint32_t rowStride = kDecodeRows);
  void encodeBatchVerifyInput(CommandGraph &graph,
                              std::span<Request *const> entries,
                              std::span<const ModelBatchItem> items,
                              uint32_t lanes,
                              std::span<const uint32_t> liveRows = {});
  // The committed step's bridge into the next command's encode
  // (RICHENGINE_DEVICE_VERIFY_PARAMS): decode_step_pack reads each lane's
  // RetainedCount/AcceptedCount/OutputTokens and writes the pack lane's
  // next-step anchor, position base, cache length, retention limit and
  // live rows — device-side what the host readback would compute.
  // `budgets` carries the proposal budget the host assigns the lane's
  // next step; `outputStride` is the step's packed OutputTokens lane
  // stride. A no-op when the flag is unset.
  void encodeBatchStepPack(CommandGraph &graph,
                           std::span<Request *const> entries,
                           std::span<const ModelBatchItem> items,
                           std::span<const uint32_t> budgets,
                           uint32_t outputStride);
  // A tree batch's verify inputs: the selector's tables carry every lane's
  // emitted nodes; the pass emits the input tokens, rotary positions and
  // ancestor masks, two decode rows' storage per lane.
  void encodeBatchVerifyTreeInput(CommandGraph &graph,
                                  std::span<Request *const> entries,
                                  std::span<const ModelBatchItem> items,
                                  uint32_t lanes);
  // The tree batch's KV tail: after acceptance, the retained path's slabs
  // move to their committed positions in each attention layer's pages.
  void encodeBatchTreeKvCompact(
      CommandGraph &graph, std::span<Request *const> entries,
      std::span<const ModelBatchItem> items);
  void encodeBatchGdnCommit(CommandGraph &graph,
                            std::span<Request *const> lanes,
                            bool tree = false);
  // A stop token or the last budgeted token needs no target work of its own:
  // the next cycle would only echo it as output. Emitting it as soon as it is
  // selected saves that cycle; the engine is told it has no KV row.
  bool emitTerminalAnchor(Request &entry, ModelStepResult &result) const {
    const bool stop = isStopToken(geometry, *entry.pendingToken);
    if (!stop && entry.maxNewTokens - entry.generatedTokens != 1)
      return false;
    result.outputTokens.push_back(*entry.pendingToken);
    result.outputTokensWithoutKv = 1;
    result.finished = stop;
    ++entry.generatedTokens;
    return true;
  }
  std::vector<ModelStepResult> finalizeDecode(
      std::span<DecodeLaneResult> lanes, std::span<const ModelBatchItem> items,
      CommandTiming timing, bool tree = false) {
    // RICHENGINE_MOE_STATS: the step's MoE layers left a record each in the
    // shared tile-count scratch; this readback is why the flag is debug-only.
    ops::MoE::logStats(decodeArena->moeScratch(1).tileCount);
    static const bool finTiming = envFlag("RICHENGINE_DECODE_TIMING");
    const auto f0 = std::chrono::steady_clock::now();
    // The acceptance kernels pack a lane's emitted rows at the chain's
    // verify stride (the tree's at its node stride), not the lane's
    // capacity slot — reads follow the packed stride, never get(lane).
    const uint32_t rowStride =
        tree ? uint32_t{ExecutionLimits::targetVerifyRows}
             : geometry.draft.verifyRows();
    const uint32_t *packedOutputTokens = contents<uint32_t>(
        decodeArena->packed(DecodeTensor::OutputTokens,
                            static_cast<uint32_t>(items.size())),
        "target output tokens");
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
      // A chain batch's verify rows (draft_verify_rows) bound its retention;
      // a tree batch's acceptance caps at its node count alone.
      if (!tree) {
        laneResult.verifyRows = *contents<uint32_t>(
            d(DecodeTensor::VerifyRows), "GPU verify rows");
        if (!laneResult.verifyRows ||
            laneResult.verifyRows > laneResult.maximumRetained)
          throw std::runtime_error("draft selector produced invalid verify rows");
        if (laneResult.retained > laneResult.verifyRows)
          throw std::runtime_error("target policy retained an unverified row");
      }
      const uint32_t *outputTokens = packedOutputTokens + lane * rowStride;
      if (!laneResult.retained || laneResult.retained > kDecodeRows) {
        if (treeDebug_) {
          const uint32_t *out = contents<uint32_t>(d(DecodeTensor::OutputTokens), "out");
          const uint32_t *counts = contents<uint32_t>(d(DecodeTensor::TreeCounts), "counts");
          fprintf(stderr,
                  "tree-debug lane=%u retained=%u accepted=%u tree_count=%u "
                  "out=[%u %u %u %u %u %u %u %u]\n",
                  lane, laneResult.retained, laneResult.accepted, counts[0],
                  out[0], out[1], out[2], out[3], out[4], out[5], out[6],
                  out[7]);
        }
        throw std::runtime_error("target policy produced invalid retention");
      }
      if (laneResult.accepted > kDraftProposalTokens)
        throw std::runtime_error(
            "target accepted more than the draft proposed");
      // The retained target tokens end with the next anchor. A non-finite
      // target row can also accept a sentinel draft proposal as an interior
      // token, so every retained token is checked.
      const uint32_t *targetTokens = outputTokens;
      if (draftDebug_ && !tree) {
        const uint32_t *props = contents<uint32_t>(
            d(DecodeTensor::ProposedTokens), "proposed tokens");
        const uint16_t *hidden = contents<uint16_t>(
            decodeArena->packed(DecodeTensor::FinalHidden,
                                uint32_t(lanes.size())),
            "final hidden") + lane * rowStride * geometry.target.hiddenSize;
        uint64_t h0 = 1469598103934665603ULL;
        for (uint32_t e = 0; e < geometry.target.hiddenSize; ++e) {
          h0 = (h0 ^ hidden[e]) * 1099511628211ULL;
        }
        const float *logits = contents<float>(
            decodeArena->packed(DecodeTensor::Logits,
                                uint32_t(lanes.size())),
            "logits") + lane * rowStride * geometry.target.paddedVocabularySize;
        float lmax = -INFINITY; uint32_t lmaxi = 0; uint32_t lnans = 0;
        for (uint32_t e = 0; e < geometry.target.vocabularySize; ++e) {
          const float v = logits[e];
          if (!std::isfinite(v)) { ++lnans; continue; }
          if (v > lmax) { lmax = v; lmaxi = e; }
        }
        const uint32_t *flatIn = contents<uint32_t>(
            decodeArena->packed(DecodeTensor::InputTokens,
                                uint32_t(lanes.size())),
            "flat in");
        fprintf(stderr, "  lane=%u lmax=%f@%u nans=%u in0=%u flat=[%u %u %u %u]\n",
                lane, lmax, lmaxi, lnans,
                flatIn[lane * rowStride],
                outputTokens[0], outputTokens[1],
                outputTokens[2], outputTokens[rowStride - 1]);
        // Top-2 margin on the correction row (the logits the retained
        // tail is sampled from): a near-tie flips on tiling noise; a wide
        // gap means a genuinely different row.
        {
          const float *corr =
              logits + laneResult.accepted * geometry.target.paddedVocabularySize;
          float m1 = -INFINITY, m2 = -INFINITY;
          uint32_t i1 = 0, i2 = 0;
          for (uint32_t e = 0; e < geometry.target.vocabularySize; ++e) {
            const float v = corr[e];
            if (v > m1) { m2 = m1; i2 = i1; m1 = v; i1 = e; }
            else if (v > m2) { m2 = v; i2 = e; }
          }
          fprintf(stderr, "  corr row=%u m1=%.4f@%u m2=%.4f@%u margin=%.4f\n",
                  laneResult.accepted, m1, i1, m2, i2, m1 - m2);
        }
        std::string p, t;
        for (uint32_t i = 0; i < kDraftProposalTokens; ++i)
          p += (i ? " " : "") + std::to_string(props[i]);
        for (uint32_t i = 0; i < laneResult.retained && i < 9; ++i)
          t += (i ? " " : "") + std::to_string(targetTokens[i]);
        fprintf(stderr, "draft lane=%u pos=%llu props=[%s] retained=%u accepted=%u out=[%s] h0=%016llx\n",
                lane, static_cast<unsigned long long>(items[lane].logicalPosition), p.c_str(),
                laneResult.retained, laneResult.accepted, t.c_str(),
                static_cast<unsigned long long>(h0));
        // RICHENGINE_VERIFY_DUMP: per-tensor FNV hashes for stride-mismatch
        // hunts — lane N's hash should equal lane 0's at width 1 when the
        // same prompt is replayed (B1 vs B2 lane 0 is the oracle).
        if (envFlag("RICHENGINE_VERIFY_DUMP")) {
          const auto fnv = [](const void *ptr, size_t bytes) {
            uint64_t h = 1469598103934665603ULL;
            const auto *b = static_cast<const uint8_t *>(ptr);
            for (size_t i = 0; i < bytes; ++i)
              h = (h ^ b[i]) * 1099511628211ULL;
            return h;
          };
          const uint32_t vs = rowStride, qr = geometry.draft.queryRows();
          const uint32_t ret = laneResult.retained;
          const uint16_t *capAll = contents<uint16_t>(
              decodeArena->packed(DecodeTensor::CapturedTargetHidden,
                                  uint32_t(lanes.size())),
              "captured hidden all");
          const uint32_t cw = geometry.target.capturedHiddenSize();
          const uint16_t *ckv = contents<uint16_t>(
              decodeArena->packed(DecodeTensor::ContextKv,
                                  uint32_t(lanes.size())),
              "context kv");
          const uint32_t *din = contents<uint32_t>(
              decodeArena->packed(DecodeTensor::DraftInputTokens,
                                  uint32_t(lanes.size())),
              "draft input");
          const uint16_t *dfhAll = contents<uint16_t>(
              decodeArena->packed(DecodeTensor::DraftFinalHidden,
                                  uint32_t(lanes.size())),
              "draft final hidden all");
          // Fused-head argmax partials for the correction row: the top-2
          // tile maxima bound the margin that flipped the emitted token.
          const uint32_t headTiles =
              geometry.target.paddedVocabularySize / 64;
          const float *hv = contents<float>(
              decodeArena->packed(DecodeTensor::HeadArgmaxValues,
                                  uint32_t(lanes.size())),
              "head argmax values");
          const uint32_t *hi = contents<uint32_t>(
              decodeArena->packed(DecodeTensor::HeadArgmaxIndices,
                                  uint32_t(lanes.size())),
              "head argmax indices");
          const uint32_t crow = laneResult.accepted;
          const auto bf2s = [](uint16_t b) {
            uint32_t u = uint32_t(b) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
          };
          const uint16_t *fh =
              hidden + size_t(crow) * geometry.target.hiddenSize;
          fprintf(stderr,
                  "  fh lane=%u crow=%u [%.4f %.4f %.4f %.4f %.4f %.4f]\n",
                  lane, crow, bf2s(fh[0]), bf2s(fh[1]), bf2s(fh[2]),
                  bf2s(fh[3]), bf2s(fh[4]), bf2s(fh[5]));
          float am1 = -INFINITY, am2 = -INFINITY;
          uint32_t ai1 = 0, ai2 = 0;
          for (uint32_t t = 0; t < headTiles; ++t) {
            const float v = hv[size_t(lane) * vs * headTiles + crow * headTiles + t];
            const uint32_t i = hi[size_t(lane) * vs * headTiles + crow * headTiles + t];
            if (v > am1) { am2 = am1; ai2 = ai1; am1 = v; ai1 = i; }
            else if (v > am2) { am2 = v; ai2 = i; }
          }
          fprintf(stderr,
                  "  hax lane=%u crow=%u m1=%.4f@%u m2=%.4f@%u margin=%.4f "
                  "ot=[%u %u %u %u %u %u %u %u]\n",
                  lane, crow, am1, ai1, am2, ai2, am1 - am2,
                  outputTokens[0], outputTokens[1], outputTokens[2],
                  outputTokens[3], outputTokens[4], outputTokens[5],
                  outputTokens[6], outputTokens[7]);
          fprintf(stderr,
                  "vdump lane=%u pos=%llu ret=%u acc=%u vin=%016llx log=%016llx "
                  "cap=%016llx ctxkv=%016llx din=%016llx dfh=%016llx\n",
                  lane,
                  static_cast<unsigned long long>(items[lane].logicalPosition),
                  ret, laneResult.accepted,
                  static_cast<unsigned long long>(
                      fnv(flatIn + lane * vs, vs * 4)),
                  static_cast<unsigned long long>(
                      fnv(logits, size_t(ret) *
                                    geometry.target.paddedVocabularySize * 4)),
                  static_cast<unsigned long long>(
                      fnv(capAll + size_t(lane) * vs * cw,
                          size_t(ret) * cw * 2)),
                  static_cast<unsigned long long>(
                      fnv(ckv + size_t(lane) * vs *
                              geometry.draft.contextKvSize(),
                          size_t(ret) * geometry.draft.contextKvSize() * 2)),
                  static_cast<unsigned long long>(
                      fnv(din + size_t(lane) * qr, qr * 4)),
                  static_cast<unsigned long long>(
                      fnv(dfhAll + size_t(lane) * qr * geometry.draft.hiddenSize,
                          size_t(qr) * geometry.draft.hiddenSize * 2)));
          std::string lr;
          for (uint32_t r = 0; r < vs && r < 8; ++r) {
            char tmp[24];
            std::snprintf(tmp, sizeof(tmp), "%016llx ",
                          static_cast<unsigned long long>(fnv(
                              logits + size_t(r) *
                                           geometry.target.paddedVocabularySize,
                              geometry.target.paddedVocabularySize * 4)));
            lr += tmp;
          }
          fprintf(stderr, "  lrow lane=%u %s\n", lane, lr.c_str());
        }
        if (lane == 0) {
          const uint16_t *cap = contents<uint16_t>(
              d(DecodeTensor::CapturedTargetHidden), "captured hidden");
          const auto bf2f = [](uint16_t b) {
            uint32_t u = uint32_t(b) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
          };
          const uint16_t *dfh = contents<uint16_t>(
              d(DecodeTensor::DraftFinalHidden), "draft final hidden");
          uint32_t nans = 0;
          for (uint32_t e = 0; e < 2048; ++e)
            nans += !std::isfinite(bf2f(dfh[e]));
          fprintf(stderr,
                  "  cap0=[%.3f %.3f %.3f %.3f] dfh=[%.4f %.4f %.4f %.4f] nans=%u\n",
                  bf2f(cap[0]), bf2f(cap[1]), bf2f(cap[2]), bf2f(cap[3]),
                  bf2f(dfh[0]), bf2f(dfh[1]), bf2f(dfh[2]), bf2f(dfh[3]),
                  nans);
          Request &rq = request(items[lane].requestId);
          const auto &ring = states.draft(rq.stateLane)[0];
          const uint16_t *rk = contents<uint16_t>(ring.keys, "draft ring keys");
          const uint64_t slot =
              items[lane].logicalPosition > 0
                  ? items[lane].logicalPosition - 1
                  : 0;
          const uint16_t *kv = rk + slot * 64;
          fprintf(stderr, "  ring@%llu=[%.3f %.3f %.3f %.3f %.3f %.3f]\n",
                  static_cast<unsigned long long>(slot), bf2f(kv[0]),
                  bf2f(kv[1]), bf2f(kv[2]), bf2f(kv[3]), bf2f(kv[4]),
                  bf2f(kv[5]));
          const uint16_t *att = contents<uint16_t>(
              d(DecodeTensor::DraftAttention), "draft attention out");
          fprintf(stderr, "  att0=[%.3f %.3f %.3f %.3f]\n",
                  bf2f(att[0]), bf2f(att[1]), bf2f(att[2]), bf2f(att[3]));
          const uint32_t *tops = contents<uint32_t>(
              d(DecodeTensor::TopPartialIds), "top partial ids");
          const float *tps = contents<float>(
              d(DecodeTensor::ProposalProbs), "proposal probs");
          fprintf(stderr, "  tops=[%u %u %u %u %u %u] probs=[%.3f %.3f %.3f %.3f]\n",
                  tops[0], tops[1], tops[2], tops[3], tops[4], tops[5],
                  tps[0], tps[1], tps[2], tps[3]);
          const uint16_t *pcap = contents<uint16_t>(
              prefillArena->get(PrefillTensor::Captured), "prefill captured");
          const uint32_t cw = geometry.target.capturedHiddenSize();
          const uint16_t *r16 = pcap + size_t(slot) * cw;
          fprintf(stderr, "  pcap@%llu=[%.4f %.4f %.4f %.4f] pcap@0=[%.4f %.4f %.4f %.4f]\n",
                  static_cast<unsigned long long>(slot), bf2f(r16[0]),
                  bf2f(r16[1]), bf2f(r16[2]), bf2f(r16[3]), bf2f(pcap[0]),
                  bf2f(pcap[1]), bf2f(pcap[2]), bf2f(pcap[3]));
          const uint16_t *pproj = contents<uint16_t>(
              prefillArena->get(PrefillTensor::ContextProjected),
              "prefill projected");
          const uint16_t *pj = pproj + size_t(slot) * 2048;
          fprintf(stderr, "  proj@%llu=[%.3f %.3f %.3f %.3f] proj@0=[%.3f %.3f %.3f %.3f]\n",
                  static_cast<unsigned long long>(slot), bf2f(pj[0]),
                  bf2f(pj[1]), bf2f(pj[2]), bf2f(pj[3]), bf2f(pproj[0]),
                  bf2f(pproj[1]), bf2f(pproj[2]), bf2f(pproj[3]));
          const uint16_t *pkv = contents<uint16_t>(
              prefillArena->get(PrefillTensor::ContextKv), "prefill ctxkv");
          const uint16_t *phid = contents<uint16_t>(
              prefillArena->get(PrefillTensor::ContextHidden),
              "prefill ctx hidden");
          const uint16_t *kv16 = pkv + size_t(slot) * 1024;
          const uint16_t *hd16 = phid + size_t(slot) * 2048;
          fprintf(stderr, "  ctxhid@%llu=[%.3f %.3f %.3f %.3f] ctxkv@%llu=[%.3f %.3f %.3f %.3f | %.3f %.3f %.3f %.3f]\n",
                  static_cast<unsigned long long>(slot), bf2f(hd16[0]),
                  bf2f(hd16[1]), bf2f(hd16[2]), bf2f(hd16[3]),
                  static_cast<unsigned long long>(slot), bf2f(kv16[0]),
                  bf2f(kv16[1]), bf2f(kv16[2]), bf2f(kv16[3]),
                  bf2f(kv16[512]), bf2f(kv16[513]), bf2f(kv16[514]),
                  bf2f(kv16[515]));
          if (slot == 16) {
            for (int what = 0; what < 3; ++what) {
              const uint16_t *src = what == 2 ? r16 : (what ? hd16 : pj);
              const uint32_t width = what == 2 ? cw : 2048;
              char path[128];
              snprintf(path, sizeof(path), "/tmp/engine_%s_%llu.bin",
                       what == 2 ? "pcap" : (what ? "ctxhid" : "proj"),
                       static_cast<unsigned long long>(slot));
              FILE *f = fopen(path, "wb");
              if (f) {
                for (uint32_t i = 0; i < width; ++i) {
                  float v = bf2f(src[i]);
                  fwrite(&v, 4, 1, f);
                }
                fclose(f);
              }
            }
            const float *rcos = contents<float>(
                prefillArena->get(PrefillTensor::DraftRopeCos),
                "draft rope cos");
            const float *rsin = contents<float>(
                prefillArena->get(PrefillTensor::DraftRopeSin),
                "draft rope sin");
            FILE *fr = fopen("/tmp/engine_rope_16.bin", "wb");
            if (fr) {
              fwrite(rcos + slot * 32, 4, 32, fr);
              fwrite(rsin + slot * 32, 4, 32, fr);
              fclose(fr);
            }
          }
        }
      }
      // RICHENGINE_DRAFT_CONF: the confidence head ships loaded but unscored.
      // Score each emitted proposal three ways — its own markov_w1 and w2
      // rows, and its predecessor's w1 (the anchor's at position 0) — against
      // the position's final hidden, to learn which tail makes the head
      // predict the chain's accept prefix (accepted is that prefix's length).
      if (draftConfDebug_ && !tree) {
        if (const auto *dspark = std::get_if<DSparkDraft>(&draftModel)) {
          const DSparkDraftWeights &w = dspark->weights();
          if (w.confidenceWeight.storage() != BufferStorage::Shared ||
              w.confidenceBias.storage() != BufferStorage::Shared ||
              w.markovEmbedding.storage() != BufferStorage::Shared ||
              w.markovProjection.storage() != BufferStorage::Shared) {
            fprintf(stderr, "conf lane=%u confidence weights not CPU-visible\n",
                    lane);
          } else {
          const uint32_t hiddenSize = w.layout.hiddenSize;
          const uint32_t rank = w.layout.markovRank;
          const uint32_t vocab =
              uint32_t(w.markovEmbedding.sizeBytes() / (2 * size_t{rank}));
          const auto bf = [](uint16_t bits) {
            return std::bit_cast<float>(uint32_t{bits} << 16);
          };
          const uint16_t *hidden = contents<uint16_t>(
              decodeArena->packed(DecodeTensor::DraftFinalHidden,
                                  uint32_t(lanes.size())),
              "draft final hidden") +
              lane * geometry.draft.queryRows() * hiddenSize;
          const uint32_t *props = contents<uint32_t>(
              d(DecodeTensor::ProposedTokens), "proposed tokens");
          const uint32_t *anchor = contents<uint32_t>(
              decodeArena->packed(DecodeTensor::InputTokens,
                                  uint32_t(lanes.size())),
              "decode anchor") + lane * rowStride;
          const uint16_t *cw = contents<uint16_t>(w.confidenceWeight,
                                                  "confidence weight");
          const uint16_t *w1 = contents<uint16_t>(w.markovEmbedding,
                                                  "markov w1");
          const uint16_t *w2 = contents<uint16_t>(w.markovProjection,
                                                  "markov w2");
          const float bias = bf(*contents<uint16_t>(w.confidenceBias,
                                                    "confidence bias"));
          std::string c1, c2, c3;
          for (uint32_t p = 0; p < kDraftProposalTokens; ++p) {
            const uint16_t *h = hidden + p * hiddenSize;
            const uint32_t token = props[p];
            const uint32_t prev = p ? props[p - 1] : anchor[0];
            if (token >= vocab || prev >= vocab)
              break;
            for (uint32_t table = 0; table < 3; ++table) {
              const uint16_t *m =
                  (table == 0 ? w1 + size_t{token} * rank
                              : table == 1 ? w2 + size_t{token} * rank
                                           : w1 + size_t{prev} * rank);
              float logit = bias;
              for (uint32_t i = 0; i < hiddenSize; ++i)
                logit += bf(cw[i]) * bf(h[i]);
              for (uint32_t i = 0; i < rank; ++i)
                logit += bf(cw[hiddenSize + i]) * bf(m[i]);
              char number[16];
              std::snprintf(number, sizeof(number), "%s%.3f", p ? " " : "",
                            1.0F / (1.0F + std::exp(-logit)));
              (table == 0 ? c1 : table == 1 ? c2 : c3) += number;
            }
          }
          fprintf(stderr,
                  "conf lane=%u retained=%u accepted=%u w1=[%s] w2=[%s] prev=[%s]\n",
                  lane, laneResult.retained, laneResult.accepted, c1.c_str(),
                  c2.c_str(), c3.c_str());
          // Rerank probe: score every pool slot with the same head. At the
          // first rejected position the target's pick is known — if it sat
          // in the pool and out-scored the selected candidate on confidence,
          // a conf-reranked chain would have extended the accept.
          const uint32_t *cands = contents<uint32_t>(
              d(DecodeTensor::TopPartialIds), "draft pool candidates");
          const float *poolUnary = contents<float>(
              d(DecodeTensor::TopPartialValues), "draft pool scores");
          constexpr uint32_t kPool = RICHENGINE_DSPARK_POOL;
          std::string line;
          char cell[128];
          for (uint32_t p = 0; p < kDraftProposalTokens; ++p) {
            const uint16_t *h = hidden + p * hiddenSize;
            auto confOf = [&](uint32_t token, const uint16_t *table) {
              const uint16_t *m = table + size_t{token} * rank;
              float logit = bias;
              for (uint32_t i = 0; i < hiddenSize; ++i)
                logit += bf(cw[i]) * bf(h[i]);
              for (uint32_t i = 0; i < rank; ++i)
                logit += bf(cw[hiddenSize + i]) * bf(m[i]);
              return 1.0F / (1.0F + std::exp(-logit));
            };
            const uint32_t *row = cands + p * kPool;
            const uint32_t sel = props[p];
            int selIdx = -1;
            float bestConf = -1.0F;
            uint32_t bestIdx = 0;
            for (uint32_t c = 0; c < kPool; ++c) {
              if (row[c] >= vocab)
                continue;
              if (row[c] == sel)
                selIdx = int(c);
              const float s = confOf(row[c], w2);
              if (s > bestConf) {
                bestConf = s;
                bestIdx = c;
              }
            }
            std::snprintf(cell, sizeof(cell), " p%u:sel@%d best@%u/%.3f", p,
                          selIdx, bestIdx, bestConf);
            line += cell;
            // The target's pick exists only at the first rejected position.
            if (p == laneResult.accepted && p < kDraftProposalTokens) {
              const uint32_t tgt = targetTokens[p];
              // Exact margin when the pick is in-pool: its biased score
              // minus the selected's. Out of pool only the bias advantage
              // over the headroom to the weakest pool score is knowable.
              float margin = -999.0F;
              const uint32_t prevTok = p ? props[p - 1] : anchor[0];
              int tgtIdx = -1;
              const float *urow = poolUnary + p * kPool;
              for (uint32_t c = 0; c < kPool; ++c)
                if (row[c] == tgt)
                  tgtIdx = int(c);
              int biasedRank = -1;
              if (tgt < vocab && prevTok < vocab && selIdx >= 0 &&
                  tgtIdx >= 0) {
                const uint16_t *fp = w1 + size_t{prevTok} * rank;
                auto biasOf = [&](uint32_t tok) {
                  const uint16_t *m = w2 + size_t{tok} * rank;
                  float b = 0.0F;
                  for (uint32_t i = 0; i < rank; ++i)
                    b += bf(fp[i]) * bf(m[i]);
                  return b;
                };
                const float tgtBiased = urow[tgtIdx] + biasOf(tgt);
                margin = tgtBiased - (urow[selIdx] + biasOf(sel));
                // The pick's rank among the pool's biased scores is what a
                // sibling-leaf verify could reach.
                biasedRank = 1;
                for (uint32_t c = 0; c < kPool; ++c) {
                  if (row[c] >= vocab || int(c) == tgtIdx)
                    continue;
                  if (urow[c] + biasOf(row[c]) > tgtBiased)
                    ++biasedRank;
                }
              }
              int tgtRank = -1;
              float tgtConf = 0.0F;
              std::vector<std::pair<float, uint32_t>> scored;
              scored.reserve(kPool);
              for (uint32_t c = 0; c < kPool; ++c) {
                if (row[c] >= vocab)
                  continue;
                const float s = confOf(row[c], w2);
                scored.emplace_back(s, c);
                if (row[c] == tgt) {
                  tgtIdx = int(c);
                  tgtConf = s;
                }
              }
              if (tgtIdx >= 0) {
                std::sort(scored.begin(), scored.end(),
                          [](auto a, auto b) { return a.first > b.first; });
                for (uint32_t c = 0; c < scored.size(); ++c)
                  if (scored[c].second == uint32_t(tgtIdx))
                    tgtRank = int(c) + 1;
              }
              std::snprintf(cell, sizeof(cell),
                            " REJ tgt=%u idx=%d margin=%.2f brank=%d conf=%.3f rank=%d selconf=%.3f",
                            tgt, tgtIdx, margin, biasedRank, tgtConf, tgtRank,
                            selIdx >= 0 ? confOf(row[selIdx], w2) : -1.0F);
              line += cell;
            }
          }
          fprintf(stderr, "cands lane=%u acc=%u%s\n", lane,
                  laneResult.accepted, line.c_str());
          }
        }
      }
      laneResult.failure = invalidSelection({targetTokens, laneResult.retained});
      if (tree && treeDebug_) {
        const uint32_t *treeTok = contents<uint32_t>(
            d(DecodeTensor::TreeTokens), "tree tokens");
        const uint32_t *sel = contents<uint32_t>(
            d(DecodeTensor::TreeSelected), "tree selected");
        const uint32_t dbgWidth = static_cast<uint32_t>(items.size());
        const uint32_t *pos = contents<uint32_t>(
            decodeArena->packed(DecodeTensor::Positions,
                                dbgWidth <= 2 ? 2 * dbgWidth : dbgWidth),
            "positions") + lane * 48;
        const uint32_t *inp = contents<uint32_t>(
            decodeArena->packed(DecodeTensor::InputTokens,
                                dbgWidth <= 2 ? 2 * dbgWidth : dbgWidth),
            "inputs") + lane * 16;
        const uint32_t *msk = contents<uint32_t>(
            d(DecodeTensor::TreeMasks), "masks");
        const uint32_t *rpath = contents<uint32_t>(
            d(DecodeTensor::RetainedPath), "retained path");
        const uint32_t medusaFlag = aneFlag_
            ? *static_cast<uint32_t *>(aneFlag_.contents())
            : 0u;
        if (lane == 0) {
          const uint32_t *nd = contents<uint32_t>(
              d(DecodeTensor::TreeNodes), "tree nodes");
          std::string dump;
          for (uint32_t r = 0; r < 16; ++r) {
            char c[64];
            std::snprintf(c, sizeof(c), " %u:t%u/p%ud%u/x%x", r, treeTok[r],
                          RICHENGINE_TREE_NODE_PARENT(nd[r]),
                          RICHENGINE_TREE_NODE_DEPTH(nd[r]), msk[r]);
            dump += c;
          }
          fprintf(stderr, "tree-nodes lane=0%s\n", dump.c_str());
        }
        fprintf(stderr,
                "tree-step lane=%u retained=%u accepted=%u out=[%u %u %u %u %u %u %u %u] "
                "tree=[%u %u %u %u %u] sel=[%u %u %u %u %u] "
                "in=[%u %u %u] pos=[%u %u %u] mask=[%x %x %x] "
                "path=[%u %u %u %u] mflag=%u fail=%s\n",
                lane, laneResult.retained, laneResult.accepted,
                targetTokens[0], targetTokens[1], targetTokens[2],
                targetTokens[3], targetTokens[4], targetTokens[5],
                targetTokens[6], targetTokens[7],
                treeTok[0], treeTok[1], treeTok[2],
                treeTok[8], treeTok[9], sel[0], sel[1], sel[2], sel[8], sel[9],
                inp[0], inp[1], inp[9], pos[0], pos[3], pos[27],
                msk[1], msk[9], msk[15], rpath[0], rpath[1], rpath[2], rpath[3],
                medusaFlag, laneResult.failure.c_str());
      }
    }

    // The chain cycle teaches the verify budget its lanes' labels and its
    // cost: proposals verified up to the first rejection, and the wall time
    // at the rows the cycle verified in all.
    if (!tree &&
        geometry.target.ffnKind == FfnKind::SparseMoe &&
        std::all_of(lanes.begin(),
                    lanes.begin() + static_cast<std::ptrdiff_t>(items.size()),
                    [](const DecodeLaneResult &laneResult) {
                      return laneResult.failure.empty();
                    })) {
      const bool constrained = std::any_of(
          lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(items.size()),
          [](const DecodeLaneResult &laneResult) {
            return laneResult.request->constraint == ConstraintMode::TokenMask;
          });
      std::array<VerifyBudget::Lane, kLaneCount> observed{};
      for (uint32_t lane = 0; lane < items.size(); ++lane) {
        const float *confidence = contents<float>(
            decodeArena->get(lane, DecodeTensor::VerifyConfidence),
            "draft proposal confidence");
        std::copy_n(confidence, kDraftProposalTokens,
                    observed[lane].confidence.begin());
        observed[lane].verifyRows = lanes[lane].verifyRows;
        observed[lane].accepted = lanes[lane].accepted;
        observed[lane].labels = lanes[lane].verifyLabels;
      }
      verifyBudget_.observe(std::span(observed).first(items.size()),
                            constrained, timing.wallSeconds * 1000.0);
      if (verifyBudgetDebug_) {
        const VerifyBudget::Acceptance &a = verifyBudget_.acceptance();
        fprintf(stderr, "verify-budget slope=%.3f intercepts=[", a.slope);
        for (uint32_t p = 0; p < VerifyBudget::kPositions; ++p)
          fprintf(stderr, "%s%.2f", p ? " " : "", a.intercepts[p]);
        fprintf(stderr, "]");
        for (uint32_t lane = 0; lane < items.size(); ++lane)
          fprintf(stderr, " l%u:rows=%u,acc=%u", lane,
                  observed[lane].verifyRows, observed[lane].accepted);
        const auto costs = verifyBudget_.costs(
            static_cast<uint32_t>(items.size()), constrained);
        fprintf(stderr, " cost=[%.2f %.2f %.2f]",
                costs[items.size()], costs[items.size() * 8],
                costs[items.size() * VerifyBudget::kBlockRows]);
        fprintf(stderr, "\n");
      }
    }

    const auto f1 = std::chrono::steady_clock::now();
    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      Request &entry = *laneResult.request;
      if (!laneResult.failure.empty()) {
        // The cycle's state and tokens are not committed; the engine ends
        // the request.
        entry.maskWords.clear();
        entry.ngramInFlight = false;
        entry.verifyMaskInFlight = false;
        entry.verifyMaskRows = 0;
        entry.gpuMaskInFlight = false;
        results.push_back({.requestId = entry.id,
                           .failure = std::move(laneResult.failure)});
        continue;
      }
      const uint32_t *targetTokens =
          packedOutputTokens + lane * rowStride;
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));

      states.swapParity(entry.stateLane);
      if (gdnTape_) {
        gdnPendingRows_[entry.stateLane] = laneResult.retained;
        gdnTapeParity_[entry.stateLane] ^= 1;
      }
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      states.updateLengths(
          entry.stateLane,
          advanceDraftContext(states.metadata(entry.stateLane).lengths,
                              nextLength, items[lane].logicalPosition,
                              nextLength, false, geometry.draft.draftWindow()));
      entry.generatedTokens += laneResult.retained;
      if (entry.draftBypassed) {
        // A bypassed step offered no draft proposals: a fake zero would
        // poison the EWMA, so only the next probe is scheduled.
        entry.draftBypassed = false;
        entry.draftProbeAt = entry.generatedTokens + kDraftProbeTokens;
      } else {
        // The draft's acceptance EWMA is the bypass and n-gram predraft
        // gates' measure of what the GPU draft would have scored, so it
        // trains whether or not the budget adapts — and never on a step a
        // different proposer (n-gram, ANE) supplied.
        if (!entry.ngramInFlight)
          entry.proposalAcceptedAvg =
              0.75 * entry.proposalAcceptedAvg + 0.25 * laneResult.accepted;
        if (adaptiveProposals_ && !tree) {
          if (laneResult.accepted >= entry.proposalBudget) {
            // A capped count can't show how much further the chain would
            // have run; widen until the trained block's cap stops binding.
            entry.proposalBudget =
                std::min<uint32_t>(geometry.draft.proposalLimit(),
                                   entry.proposalBudget + 1);
          } else {
            // Shrink toward the EWMA target one row per step: a
            // single-visit clamp to avg+2 oscillates the live verify width
            // whenever the rolling rate hovers across an integer boundary.
            const uint32_t target = std::clamp<uint32_t>(
                static_cast<uint32_t>(entry.proposalAcceptedAvg) + 2, 1,
                kDraftProposalTokens);
            if (target < entry.proposalBudget) --entry.proposalBudget;
          }
        }
      }
      if (entry.ngramInFlight) {
        entry.ngramInFlight = false;
        const double observed =
            std::min(laneResult.accepted, laneResult.retained - 1);
        entry.ngramAcceptedAvg = entry.ngramRounds
                                     ? 0.75 * entry.ngramAcceptedAvg +
                                           0.25 * observed
                                     : observed;
        ++entry.ngramRounds;
        entry.ngramProbeAt =
            entry.generatedTokens + NgramPredraft::kNgramProbeTokens;
      }
      commitSelected(entry, {targetTokens, laneResult.retained});
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      entry.verifyMaskRows = 0;
      entry.gpuMaskInFlight = false;
      results.push_back(
          {entry.id, 0, std::move(output), false, DecodeStage::Regular,
           // A chain lane's drafted count is the proposals it verified: the
           // offered ones clamped to its verify rows, which the budget may
           // have shortened below its trained row stride. A tree's is the
           // full comb width, as before.
           tree ? kDraftProposalTokens
                : std::min(laneResult.proposals,
                           laneResult.verifyRows - 1),
           std::min(laneResult.accepted, laneResult.retained - 1)});
      ModelStepResult &result = results.back();
      if (entry.generatedTokens < entry.maxNewTokens)
        emitTerminalAnchor(entry, result);
    }

    counters.lastDecodeWidth = static_cast<uint32_t>(items.size());
    counters.lastDecodeGpuSeconds = timing.gpuSeconds;
    counters.totalDecodeGpuSeconds += timing.gpuSeconds;
    counters.lastDecodeWallSeconds = timing.wallSeconds;
    counters.totalDecodeWallSeconds += timing.wallSeconds;
    kickAnePredictors(lanes, items);
    if (finTiming) {
      const auto f2 = std::chrono::steady_clock::now();
      fprintf(stderr, "finalize-detail loop=%.3fms tail=%.3fms\n",
              std::chrono::duration<double, std::milli>(f1 - f0).count(),
              std::chrono::duration<double, std::milli>(f2 - f1).count());
    }
    return results;
  }
  // A constrained DFlash cycle has one host dependency between three Metal
  // commands: draft proposals define the grammar simulation, while the target
  // forward is independent of the resulting mask.  This ticket keeps the
  // scheduler batch (and therefore its DecodeArena lanes) owned across that
  // dependency.  All state transitions run on the engine thread; completion
  // handlers only wake it, so they capture the wake hook and never the ticket.
  class ConstrainedDecodeTicket final : public ModelBatchTicket {
  public:
    ConstrainedDecodeTicket(Impl &impl, std::vector<DecodeLaneResult> lanes,
                            std::span<const ModelBatchItem> items,
                            CommandGraph draft,
                            std::function<void()> completion, bool merged)
        : impl_(impl), lanes_(std::move(lanes)),
          items_(items.begin(), items.end()),
          wake_(std::move(completion)) {
      if (merged) {
        mergedSubmit(draft);
        return;
      }
      submit(draft);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        std::array<Request *, kLaneCount> entries{};
        // RICHENGINE_GPU_DECODE_GRID records the lanes whose GPU forced
        // spans the target command's grid kernel consumes on-device; the
        // host keeps its own clear for the direct mask dispatch.
        DfaGridLanes dfaGrid;
        // Constrained lanes skip the adaptive budget, so each lane's live
        // verify rows are the proposal cap clipped by its retention limit —
        // the same arithmetic the unconstrained path computes up front.
        const uint32_t cap =
            modelPolicy().proposalCap.value_or(
                impl_.geometry.draft.proposalLimit());
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          DecodeLaneResult &laneResult = lanes_[lane];
          Request &entry = *laneResult.request;
          entries[lane] = &entry;
          // A free lane asks for no mask and none of its selections read
          // one: it keeps the live rows decodeAsync computed and clears
          // every mask-inflight flag the suspension checks read.
          if (!constrained(lane)) {
            gpuMask_[lane] = false;
            forced_[lane] = 0;
            gpuForced_[lane] = 0;
            entry.gpuMaskInFlight = false;
            entry.verifyMaskInFlight = false;
            liveRows_[lane] = laneResult.proposals + 1;
            continue;
          }
          uint32_t *proposed = contents<uint32_t>(
              impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
              "constrained draft proposals");
          entry.maskWords.clear();
          // A Full-mode DFA lane's masks come from the walk kernels: no
          // mask request goes out and verifyMaskInFlight stays clear, so
          // stale host responses cannot corrupt it. A Forced-mode lane
          // still takes the host round trip — verifyMaskInFlight — while
          // gpuMaskInFlight guards suspension against the commit stage's
          // automaton-state and forced-span writes either mode encodes.
          gpuMask_[lane] = impl_.gpuDfaMasks(entry);
          entry.gpuMaskInFlight = entry.dfa.has_value();
          entry.verifyMaskInFlight = !gpuMask_[lane];

          // A DFA lane's forced span comes from the commit stage's
          // on-device derivation (Full and Forced modes alike): the state
          // lane's DfaForcedSpan, written from the committed state itself —
          // no anchor check applies — and consumed once. Host-pushed spans
          // serve host-constrained lanes: its first token anchors it at the
          // pending token, so a span whose anchor no longer matches is
          // dropped unused.
          forced_[lane] = 0;
          gpuForced_[lane] = 0;
          bool gpuSpan = false;
          if (entry.dfa) {
            DfaForcedSpan &span = impl_.dfaForcedSpan(entry.stateLane);
            if (!abandoned_[lane] && entry.pendingToken && span.tokens) {
              gpuSpan = true;
              forced_[lane] = std::min<uint32_t>(
                  span.tokens,
                  std::min<uint32_t>(kDraftProposalTokens,
                                     laneResult.maximumRetained - 1));
              std::copy_n(span.token_ids, forced_[lane], proposed);
              // Under the GPU-grid gate the mask pass's grid kernel
              // consumes the span on-device and sizes the mask launch
              // from it; the host keeps the clear for the direct path.
              if (impl_.gpuDecodeGridOn_) {
                dfaGrid.gpuSpanMask |= uint32_t{1} << lane;
                dfaGrid.maximumRetained[lane] = laneResult.maximumRetained;
              } else {
                span.tokens = 0;
              }
              ++impl_.counters.gpuForcedSpansApplied;
              impl_.counters.forcedTokensEmitted += forced_[lane];
            }
          }
          // A span from the on-device derivation is byte-forced: the
          // truncation pass auto-accepts its rows. A host-pushed span
          // keeps the argmax check even on a DFA lane.
          if (gpuSpan)
            gpuForced_[lane] = forced_[lane];
          if (!entry.forcedTokens.empty()) {
            if (gpuSpan) {
              // A host span racing the on-device derivation loses to it.
              entry.forcedTokens.clear();
              ++impl_.counters.forcedSpansDroppedStale;
            } else {
              // A DFA lane whose round produced no span still takes a host
              // push; host-constrained lanes are served as before.
              std::vector<uint32_t> span = std::move(entry.forcedTokens);
              entry.forcedTokens.clear();
              if (!abandoned_[lane] && entry.pendingToken && span.size() > 1 &&
                  span.front() == *entry.pendingToken) {
                forced_[lane] = std::min<uint32_t>(
                    span.size() - 1,
                    std::min<uint32_t>(kDraftProposalTokens,
                                       laneResult.maximumRetained - 1));
                std::copy_n(span.begin() + 1, forced_[lane], proposed);
                ++impl_.counters.forcedSpansApplied;
                impl_.counters.forcedTokensEmitted += forced_[lane];
              } else {
                ++impl_.counters.forcedSpansDroppedStale;
              }
            }
          }

          // The lane's live verify rows: its forced prefix, or the
          // proposal cap clipped by its retention limit, plus the anchor.
          liveRows_[lane] =
              forced_[lane]
                  ? forced_[lane] + 1
                  : std::min(cap, laneResult.maximumRetained - 1) + 1;
          // A forced span replaced the lane's proposals: its confidences do
          // not describe them, so it labels no acceptance for the budget.
          if (forced_[lane])
            laneResult.verifyLabels = false;
          if (!abandoned_[lane] && !gpuMask_[lane]) {
            // The mask response carries one row per simulated token plus
            // the pre-anchor row, so the simulation stops at the lane's
            // live verify rows — the selection never reads the mask rows
            // past them. The row count goes on the entry for provideMask's
            // validation. Slots a short-block draft never emitted are
            // stale tokens the grammar's out-of-range truncation skips.
            ModelMaskRequest request;
            request.requestId = entry.id;
            request.simulationTokens.reserve(liveRows_[lane]);
            request.simulationTokens.push_back(*entry.pendingToken);
            request.simulationTokens.insert(request.simulationTokens.end(),
                                            proposed,
                                            proposed + liveRows_[lane] - 1);
            entry.verifyMaskRows = liveRows_[lane] + 1;
            requests.push_back(std::move(request));
          }
        }

        CommandGraph target;
        const uint32_t width = static_cast<uint32_t>(lanes_.size());
        try {
          impl_.encodeBatchVerifyInput(target, {entries.data(), width},
                                       items_, width,
                                       {liveRows_.data(), width});
          impl_.encodeDfaMasks(target, {entries.data(), width}, width,
                               {liveRows_.data(), width},
                               dfaGrid.gpuSpanMask ? &dfaGrid : nullptr);
          impl_.encodeBatchEmbedding(target, DecodeTensor::InputTokens,
                                     DecodeTensor::Hidden0, width, 1,
                                     impl_.geometry.draft.verifyRows());
          impl_.encodeTargetVerifyBatchForward(
              target, {entries.data(), lanes_.size()}, items_);
          submit(target);
        } catch (...) {
          // A span the grid kernel would have consumed stays pending when
          // the command never reaches the GPU; drop it like the host
          // clear would have.
          for (uint32_t lane = 0; lane < width; ++lane)
            if (dfaGrid.gpuSpanMask & (uint32_t{1} << lane))
              impl_.clearDfaForced(entries[lane]->stateLane);
          throw;
        }
        stage_ = Stage::TargetForward;
      }

      if (stage_ == Stage::TargetForward && command_.ready()) {
        const CommandTiming forward = command_.wait();
        addTiming(forward);
        targetForwardGpuSeconds_ += forward.gpuSeconds;
        maskWaitStarted_ = AwakeClock::now();
        stage_ = Stage::WaitingMask;
      }

      if (stage_ == Stage::WaitingMask) {
        bool masksReady = true;
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          masksReady = masksReady &&
                       (!constrained(lane) || abandoned_[lane] ||
                        gpuMask_[lane] ||
                        !lanes_[lane].request->maskWords.empty());
        }
        if (masksReady) {
          maskWaitSeconds_ +=
              std::chrono::duration<double>(AwakeClock::now() -
                                            *maskWaitStarted_)
                  .count();
          maskWaitStarted_.reset();
          std::array<Request *, kLaneCount> entries{};
          std::array<uint32_t, kLaneCount> maximumRetained{};
          std::array<uint32_t, kLaneCount> proposals{};
          uint32_t dfaSkip = 0;
          const uint32_t cap =
              modelPolicy().proposalCap.value_or(
                  impl_.geometry.draft.proposalLimit());
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            DecodeLaneResult &laneResult = lanes_[lane];
            Request &entry = *laneResult.request;
            entries[lane] = &entry;
            maximumRetained[lane] = laneResult.maximumRetained;
            // Rows past a lane's live budget carry anchor duplicates; they
            // must never be retained, or the draft context commits a row
            // whose token its hidden state never saw.
            proposals[lane] =
                constrained(lane)
                    ? (forced_[lane] ? forced_[lane]
                                     : std::min(
                                           cap, laneResult.maximumRetained - 1))
                    : laneResult.proposals;
            laneResult.proposals = proposals[lane];
            if (!constrained(lane))
              continue;
            if (abandoned_[lane])
              impl_.admitEveryToken(lane);
            else if (!gpuMask_[lane])
              impl_.uploadConstraintMasks(lane, entry.maskWords);
            // An abandoned DFA lane falls back to the filled bitmask: the
            // selection reads its next-state rows only while dfaLanes holds.
            if (abandoned_[lane] && gpuMask_[lane])
              dfaSkip |= uint32_t{1} << lane;
          }

          CommandGraph commit;
          impl_.encodeTargetVerifyBatchPolicy(
              commit, {entries.data(), lanes_.size()}, false,
              {liveRows_.data(), lanes_.size()}, dfaSkip);
          // Greedy Full-mode DFA lanes ran the unmasked pass: name each
          // lane's correction row, write its single next-state row and
          // take the masked argmax there; their accepted prefixes cap at
          // the truncation rows in acceptance.
          const uint32_t truncMask = impl_.greedyDfaLaneMask(
              {entries.data(), lanes_.size()}, dfaSkip);
          const uint32_t width =
              static_cast<uint32_t>(lanes_.size());
          impl_.encodeDfaTruncation(commit, width, truncMask,
                                    {proposals.data(), lanes_.size()},
                                    {gpuForced_.data(), lanes_.size()});
          impl_.encodeDfaCorrectionMasks(
              commit, {entries.data(), lanes_.size()}, width, truncMask);
          impl_.encodeDfaCorrections(
              commit, {entries.data(), lanes_.size()}, width, truncMask,
              {liveRows_.data(), lanes_.size()});
          impl_.encodeBatchAcceptance(commit, {entries.data(), lanes_.size()},
                                      {maximumRetained.data(), lanes_.size()},
                                      false,
                                      {proposals.data(), lanes_.size()},
                                      truncMask);
          impl_.encodeDfaAdvance(commit, {entries.data(), lanes_.size()},
                                 static_cast<uint32_t>(lanes_.size()));
          impl_.encodeDfaForced(commit, {entries.data(), lanes_.size()},
                                static_cast<uint32_t>(lanes_.size()));
          impl_.encodeBatchGdnCommit(commit, {entries.data(), lanes_.size()});
          impl_.encodeDraftStateCommitBatch(
              commit, {entries.data(), lanes_.size()}, items_);
          // Constrained lanes skip the adaptive budget; free lanes keep
          // the budget decodeAsync clamped for their step.
          std::array<uint32_t, kLaneCount> budgets{};
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
            budgets[lane] =
                constrained(lane) || !impl_.adaptiveProposals_
                    ? cap
                    : std::clamp(lanes_[lane].request->proposalBudget,
                                 std::min<uint32_t>(1, cap), cap);
          impl_.encodeBatchStepPack(
              commit, {entries.data(), lanes_.size()}, items_,
              {budgets.data(), lanes_.size()},
              impl_.geometry.draft.verifyRows());
          submit(commit);
          stage_ = Stage::Commit;
        }
      }
      return requests;
    }

    // Forced tokens fed as proposals this cycle, per lane; zero means the
    // lane runs the draft's own proposals.
    std::array<uint32_t, kLaneCount> forced_{};
    // Each lane's GPU-DfaForcedSpan prefix length — the rows the
    // truncation pass auto-accepts — or zero when the lane's proposals
    // did not come from the on-device derivation.
    std::array<uint32_t, kLaneCount> gpuForced_{};
    // Lanes whose masks the GPU-DFA kernels compute (Full mode): they
    // never emit a ModelMaskRequest, and WaitingMask treats them as
    // satisfied. A Forced-mode DFA lane keeps this clear: its masks still
    // come from the host.
    std::array<bool, kLaneCount> gpuMask_{};
    // Each lane's live verify rows, computed in the Draft stage from its
    // forced span or proposal budget and reused by the policy commit.
    std::array<uint32_t, kLaneCount> liveRows_{};

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::Done)
        return false;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        if (lanes_[lane].request->id == requestId)
          return constrained(lane);
      return false;
    }

    void abandonMask(uint64_t requestId) noexcept override {
      if (stage_ == Stage::Done)
        return;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
        if (lanes_[lane].request->id == requestId) {
          abandoned_[lane] = true;
          lanes_[lane].request->maskWords.clear();
          return;
        }
      }
    }

    bool ready() const noexcept override {
      return stage_ == Stage::Commit && command_.ready();
    }

    std::vector<ModelStepResult> wait() override {
      if (!ready())
        throw std::logic_error("constrained decode ticket is not complete");
      addTiming(command_.wait());
      stage_ = Stage::Done;
      ModelTelemetry &counters = impl_.counters;
      ++counters.constrainedMaskOverlapBatches;
      for (uint32_t lane = 0; lane < lanes_.size(); ++lane)
        counters.constrainedMaskOverlapRequests += constrained(lane);
      counters.lastConstrainedTargetForwardGpuSeconds =
          targetForwardGpuSeconds_;
      counters.totalConstrainedTargetForwardGpuSeconds +=
          targetForwardGpuSeconds_;
      counters.lastConstrainedMaskWaitSeconds = maskWaitSeconds_;
      counters.totalConstrainedMaskWaitSeconds += maskWaitSeconds_;
      return impl_.finalizeDecode(lanes_, items_, timing_);
    }

    double wallMilliseconds() const noexcept override {
      return timing_.wallSeconds * 1000.0;
    }

  private:
    enum class Stage : uint8_t {
      Draft,
      TargetForward,
      WaitingMask,
      Commit,
      Done
    };

    void submit(const CommandGraph &graph) {
      command_ = impl_.backend.submitCommandAsync(graph.dispatches(), wake_);
    }

    // Whether the lane's selections read a mask from the host or the GPU
    // DFA walk.
    bool constrained(uint32_t lane) const noexcept {
      return lanes_[lane].request->constraint == ConstraintMode::TokenMask;
    }

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
    }

    // RICHENGINE_DFA_MERGE: the whole step — draft, verify input, DFA
    // masks, target forward, selection and commits — appends onto the
    // draft graph, one command in place of the split path's three. The
    // gate admits only Full-mode lanes that all carry a GPU DFA and owe
    // no host-pushed forced span, so no host mask request exists to
    // answer: the pending GPU forced spans apply on-device
    // (decode_dfa_span_apply) between the draft pass and the verify
    // input. The span itself is host-readable at encode time — the
    // commit that wrote it retired before this step was planned — so
    // the forced and live constants stay host-derived exactly as the
    // split path's, and the mask pass keeps the direct launch its
    // host-computed rows already give it. A settle racing the single
    // command changes nothing the GPU computes: the abandoned lane's
    // admit-everything fallback exists only for a mask wait this path
    // never opens, and a dead lane's committed tokens never publish.
    void mergedSubmit(CommandGraph &graph) {
      const uint32_t width = static_cast<uint32_t>(lanes_.size());
      const uint32_t cap =
          modelPolicy().proposalCap.value_or(
              impl_.geometry.draft.proposalLimit());
      std::array<Request *, kLaneCount> entries{};
      std::array<uint32_t, kLaneCount> maximumRetained{};
      std::array<uint32_t, kLaneCount> proposals{};
      uint32_t spanMask = 0;
      for (uint32_t lane = 0; lane < width; ++lane) {
        DecodeLaneResult &laneResult = lanes_[lane];
        Request &entry = *laneResult.request;
        entries[lane] = &entry;
        entry.maskWords.clear();
        // Every lane took the gate's GPU-mask branch, so this is the
        // bookkeeping the split path's Draft stage does for one.
        gpuMask_[lane] = true;
        entry.gpuMaskInFlight = true;
        entry.verifyMaskInFlight = false;
        // The span bookkeeping the split path does host-side: the forced
        // prefix length, its auto-accept record and the lane's live
        // verify rows, plus the counters the apply reports.
        forced_[lane] = 0;
        gpuForced_[lane] = 0;
        DfaForcedSpan &span = impl_.dfaForcedSpan(entry.stateLane);
        if (entry.pendingToken && span.tokens) {
          forced_[lane] = std::min<uint32_t>(
              span.tokens,
              std::min<uint32_t>(kDraftProposalTokens,
                                 laneResult.maximumRetained - 1));
          gpuForced_[lane] = forced_[lane];
          spanMask |= uint32_t{1} << lane;
          ++impl_.counters.gpuForcedSpansApplied;
          impl_.counters.forcedTokensEmitted += forced_[lane];
        }
        liveRows_[lane] =
            forced_[lane]
                ? forced_[lane] + 1
                : std::min(cap, laneResult.maximumRetained - 1) + 1;
        maximumRetained[lane] = laneResult.maximumRetained;
        proposals[lane] = liveRows_[lane] - 1;
        laneResult.proposals = proposals[lane];
      }
      try {
        impl_.encodeDfaSpanApply(graph, {entries.data(), width}, width,
                               spanMask, {maximumRetained.data(), width});
        impl_.encodeBatchVerifyInput(graph, {entries.data(), width}, items_,
                                     width, {liveRows_.data(), width});
        impl_.encodeDfaMasks(graph, {entries.data(), width}, width,
                             {liveRows_.data(), width});
        impl_.encodeBatchEmbedding(graph, DecodeTensor::InputTokens,
                                   DecodeTensor::Hidden0, width, 1,
                                   impl_.geometry.draft.verifyRows());
        impl_.encodeTargetVerifyBatchForward(graph, {entries.data(), width},
                                             items_);
        impl_.encodeTargetVerifyBatchPolicy(graph, {entries.data(), width},
                                            false,
                                            {liveRows_.data(), width});
        const uint32_t truncMask =
            impl_.greedyDfaLaneMask({entries.data(), width});
        impl_.encodeDfaTruncation(graph, width, truncMask,
                                {proposals.data(), width},
                                {gpuForced_.data(), width});
        impl_.encodeDfaCorrectionMasks(graph, {entries.data(), width},
                                       width, truncMask);
        impl_.encodeDfaCorrections(graph, {entries.data(), width}, width,
                                   truncMask, {liveRows_.data(), width});
        impl_.encodeBatchAcceptance(graph, {entries.data(), width},
                                    {maximumRetained.data(), width}, false,
                                    {proposals.data(), width}, truncMask);
        impl_.encodeDfaAdvance(graph, {entries.data(), width}, width);
        impl_.encodeDfaForced(graph, {entries.data(), width}, width);
        impl_.encodeBatchGdnCommit(graph, {entries.data(), width});
        impl_.encodeDraftStateCommitBatch(graph, {entries.data(), width},
                                          items_);
        // Constrained lanes skip the adaptive budget, so the step pack
        // gets the flat proposal cap as each lane's next-step budget.
        std::array<uint32_t, kLaneCount> budgets{};
        budgets.fill(cap);
        impl_.encodeBatchStepPack(graph, {entries.data(), width}, items_,
                                  {budgets.data(), width},
                                  impl_.geometry.draft.verifyRows());
        submit(graph);
      } catch (...) {
        // A span the apply kernel would have consumed stays pending when
        // the command never reaches the GPU; drop it like the split
        // path's catch does for the grid kernel's spans.
        for (uint32_t lane = 0; lane < width; ++lane)
          if (spanMask & (uint32_t{1} << lane))
            impl_.clearDfaForced(entries[lane]->stateLane);
        throw;
      }
      stage_ = Stage::Commit;
    }

    Impl &impl_;
    std::vector<DecodeLaneResult> lanes_;
    std::vector<ModelBatchItem> items_;
    Stage stage_ = Stage::Draft;
    CommandTicket command_;
    CommandTiming timing_;
    std::array<bool, kLaneCount> abandoned_{};
    double targetForwardGpuSeconds_ = 0.0;
    double maskWaitSeconds_ = 0.0;
    std::optional<AwakeClock::time_point> maskWaitStarted_;
    std::function<void()> wake_;
  };
};

#include "model/RuntimeBindings.hpp"

} // namespace richengine::model
