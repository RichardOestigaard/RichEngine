#pragma once

// Runtime::Impl: the Metal model runtime's private implementation. The
// encode, ANE-predraft and n-gram predraft method bodies live in
// RuntimeEncode.mm, RuntimeAne.mm and RuntimeNgram.mm.

#include "model/Runtime.hpp"
#include "model/RuntimeRequest.hpp"
#include "AwakeClock.hpp"
#include "Env.hpp"
#include "model/AnePredictor.hpp"
#include "model/NullDraft.hpp"
#include "model/QwenState.hpp"
#include "model/QwenTarget.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/CommandGraph.hpp"
#include "ops/Embedding.hpp"
#include "ops/Linear.hpp"
#include "ops/PagedAttention.hpp"
#include "ops/PagedKv.hpp"
#include "ops/RoPE.hpp"
#include "ops/RowCopy.hpp"
#include "ops/Sampling.hpp"
#include "ops/Vision.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
  using Completion = std::function<std::vector<ModelStepResult>(metal::CommandTiming)>;

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
    wallMilliseconds_ = timing.wallSeconds * 1000.0;
    Completion completion = std::move(completion_);
    return completion(timing);
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
  QwenStateStorage &states;
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
  QwenTarget targetModel;
  std::variant<NullDraft, DFlashDraft, PlainDraft, DSparkDraft> draftModel;
  // Tree verify (docs/TREE_VERIFY_DESIGN.md): the selector emits each greedy,
  // unconstrained lane's comb tree when the draft is a DFlash2. Measured a
  // net loss on the decode benchmark (identical accepted tokens at ~2x
  // verify rows), so it is opt-in: RICHENGINE_VERIFY_TREE=1 enables.
  const bool verifyTreeEnabled = envFlagOn("RICHENGINE_VERIFY_TREE");
  // Adaptive proposal budgets (docs/SPEC_DECODE_BOOST.md L3): a chain lane's
  // accepted-token cap tracks its rolling acceptance EWMA, so a lane that
  // keeps rejecting pays for fewer live verify rows — GDN scan depth and
  // the per-row vocabulary/argmax sweeps — instead of the fixed eight.
  // On by default; RICHENGINE_ADAPTIVE_PROPOSALS=0 disables.
  const bool adaptiveProposals_ =
      ![] {
        const char *value = std::getenv("RICHENGINE_ADAPTIVE_PROPOSALS");
        return value && std::string_view(value) == "0";
      }();
  // Per-step debug gates, read once: getenv scans environ linearly and these
  // ran inside the decode/finalize paths on every command.
  const bool treeDebug_ = envFlag("RICHENGINE_TREE_DEBUG");
  const bool draftDebug_ = envFlag("RICHENGINE_DRAFT_DEBUG");
  const std::string treeSkip_ = [] {
    const char *value = std::getenv("RICHENGINE_TREE_SKIP");
    return value ? std::string(value) : std::string();
  }();
  const bool aneDebug_ = envFlag("RICHENGINE_ANE_DEBUG");
  const bool ngramDebug_ = envFlag("RICHENGINE_NGRAM_DEBUG");
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
  const uint32_t aneWaitMs_ = envUint("RICHENGINE_ANE_WAIT_MS", 3);
  // Learning-free predraft (opt-in, RICHENGINE_NGRAM_PREDRAFT=1): an n-gram
  // table over each lane's prompt+generated stream feeds the same
  // ProposedTokens injection the ANE artifact uses — a wrong candidate
  // only wastes verify rows.
  const bool ngramPredraft_ = envFlagOn("RICHENGINE_NGRAM_PREDRAFT");
  // Acceptance gate (SSSD-style): a batch goes n-gram-predrafted when the
  // lanes' expected accepted-token scores total at least what the GPU draft
  // would have delivered. A lane with a weak EWMA or no match drags the
  // total down and can veto; a cold lane re-probes each kNgramProbeTokens.
  const double ngramDraftExpect_ = envDouble("RICHENGINE_NGRAM_DRAFT_EXPECT", 4.0);
  const uint32_t ngramWarmup_ = envUint("RICHENGINE_NGRAM_WARMUP", 8);
  static constexpr uint32_t kNgramProbeTokens = 256;
  explicit Impl(RuntimeContext value)
      : backend(value.backend),
        package(value.package),
        geometry(RuntimeGeometry::from(value.package, value.kvPages.layout().format)),
        operators(value.operators),
        kvPages(value.kvPages),
        states(value.stateStorage),
        sampling(geometry.target.vocabularySize),
        targetModel(std::visit(
                        [&](const auto &weights) {
                          return QwenTarget(weights, geometry.target,
                                            value.backend, operators);
                        },
                        value.package.target)),
        draftModel(std::visit(
                       [&](const auto &weights)
                           -> std::variant<NullDraft, DFlashDraft, PlainDraft,
                                           DSparkDraft> {
                         using W = std::decay_t<decltype(weights)>;
                         using Draft = std::conditional_t<
                             std::is_same_v<W, NullDraftWeights>, NullDraft,
                             std::conditional_t<
                                 std::is_same_v<W, DFlashDraftWeights>,
                                 DFlashDraft,
                                 std::conditional_t<
                                     std::is_same_v<W, PlainDraftWeights>,
                                     PlainDraft, DSparkDraft>>>;
                         if constexpr (std::is_same_v<Draft, NullDraft>)
                           return std::variant<NullDraft, DFlashDraft,
                                               PlainDraft, DSparkDraft>(
                               std::in_place_type<Draft>);
                         else
                           return std::variant<NullDraft, DFlashDraft,
                                               PlainDraft, DSparkDraft>(
                               std::in_place_type<Draft>, weights,
                               value.backend, operators);
                       },
                       value.package.draft)) {
    if (states.layout() != package.stateLayout() ||
        kvPages.layout() != package.targetKvLayout(kvPages.layout().format)) {
      throw std::invalid_argument(
          "model runtime resources do not match the loaded package");
    }
    prefillArena = std::make_unique<PrefillArena>(backend, geometry, operators);
    decodeArena = std::make_unique<DecodeArena>(backend, geometry, operators);
    penaltyTable = decodeArena->packed(DecodeTensor::PenaltyState, kLaneCount);
    preparePolicyPipelines();
    if (const char *path = std::getenv("RICHENGINE_ANE_MEDUSA")) {
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
    if (const char *path = std::getenv("RICHENGINE_ANE_PREDRAFT")) {
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
  // Qwen3.5 M-RoPE: text rows advance one counter shared by all three axes;
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

  void addRopeTables(CommandGraph &graph, MetalBuffer targetPositions,
                     uint32_t targetRows, MetalBuffer draftPositions,
                     uint32_t draftRows, MetalBuffer targetCos,
                     MetalBuffer targetSin, MetalBuffer draftCos,
                     MetalBuffer draftSin) const;
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
  static QwenLogicalLengths
  advanceDraftContext(const QwenLogicalLengths &previous, uint64_t targetTokens,
                      uint64_t begin, uint64_t end, bool reset) {
    if (!reset && (!previous.draftLength || previous.draftEnd() != begin))
      throw std::logic_error("draft capture does not continue the draft ring");
    const uint64_t combined = (reset ? 0 : previous.draftLength) + (end - begin);
    QwenLogicalLengths next = previous;
    next.targetTokens = targetTokens;
    next.draftLength =
        static_cast<uint32_t>(std::min<uint64_t>(combined, kDraftCacheStride));
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
  // A batch verifies the selector's comb trees only when every lane can:
  // greedy, unconstrained and unpenalized DFlash lanes, at most two wide
  // (a tree lane doubles its row block, and four virtual lanes is the row
  // budget), on a target with the tree kernels — GDN and attention mixers,
  // dense FFN, no fp8 KV. Anything else runs the chain verify.
  bool treeVerifyBatch(std::span<Request *const> entries, uint32_t width,
                       bool constrained) const {
    if (constrained || !verifyTreeEnabled || !width ||
        width > kLaneCount / 2 ||
        !std::holds_alternative<DFlashDraft>(draftModel) ||
        geometry.target.convLayers ||
        geometry.target.ffnKind == QwenFfnKind::SparseMoe ||
        geometry.target.kvLayout.format == kv::Format::Float8E4M3)
      return false;
    for (uint32_t lane = 0; lane < width; ++lane) {
      const ops::SamplingPolicy policy = samplingPolicy(*entries[lane]);
      if (policy.samples() || policy.constrained ||
          policy.penalties.active())
        return false;
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
    targetModel.addHeadBatch(graph, d(DecodeTensor::Hidden0),
                             d(DecodeTensor::FinalHidden),
                             d(DecodeTensor::Logits), width,
                             decodeArena->linearScratch());
    if (std::ranges::none_of(lanes, &InitialSelection::select))
      return;
    std::array<ops::SamplingPolicy, kLaneCount> policies{};
    std::array<uint32_t, kLaneCount> stateLanes{};
    for (const InitialSelection &lane : lanes) {
      if (!lane.select)
        continue;
      policies[lane.lane] = samplingPolicy(*lane.entry);
      stateLanes[lane.lane] = lane.entry->stateLane;
    }
    sampling.addInitial(graph, std::span(policies).first(width),
                        samplingBuffers(width), 0,
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
    for (uint32_t lane = 0; lane < width; ++lane) {
      Request &entry = request(items[lane].requestId);
      if (entry.decodeStage != DecodeStage::ApplyInitialMask ||
          entry.pendingToken ||
          entry.maskWords.size() != geometry.maskWords() ||
          entry.finalTargetHidden.size() != geometry.target.hiddenSize) {
        throw std::logic_error("initial selection state is invalid");
      }
      std::ranges::copy(entry.finalTargetHidden,
                        contents<uint16_t>(
                            decodeArena->get(lane, DecodeTensor::Hidden0),
                            "final prompt hidden"));
      if (samplingEnabled(entry))
        uploadInitialUniform(entry, lane);
      uploadConstraintMasks(lane, entry.maskWords);
      selections[lane] = {&entry, lane, true};
    }
    CommandGraph graph;
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
        throw std::invalid_argument("invalid packed Qwen prefill item");
      }
      const QwenLaneMetadata &metadata = states.metadata(entry.stateLane);
      // A chunk submitted ahead of its predecessor's consumption still has
      // the predecessor's rows unapplied: lengths advance only at consume.
      if (metadata.requestId != entry.id ||
          metadata.lengths.targetTokens + entry.prefillUnappliedRows !=
              item.logicalPosition) {
        throw std::logic_error("packed prefill state length is not exact");
      }
      if (item.logicalPosition == 0)
        states.clearForColdStart(entry.stateLane);
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
      batch.sequences.push_back({&entry, &item, lane, batch.rows,
                                 attentionStride, queryOffset, kvOffset,
                                 batch.capturedRows, chunk, std::move(pageTable),
                                 std::move(captures)});
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
         p(PrefillTensor::ContextKv), p(PrefillTensor::DraftRopeCos),
         p(PrefillTensor::DraftRopeSin),
         {.i8codes = p(PrefillTensor::I8Codes),
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
                  p(PrefillTensor::DraftRopeSin));

    targetModel.addEmbedding(graph, pi(PrefillTensor::InputTokens),
                             p(PrefillTensor::Hidden0), batch.rows);
    for (const PackedPrefillSequence &sequence : batch.sequences) {
      addImageRows(graph, *sequence.entry, *sequence.item, sequence.rowBegin);
    }

    std::array<QwenTargetPrefillSequence, kLaneCount> modelSequences{};
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
      QwenTargetPrefillSequence &destination = modelSequences[lane];
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
    QwenTargetPrefillBuffers buffers;
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
    buffers.chunkKeys = p(PrefillTensor::ChunkKeys);
    buffers.chunkValues = p(PrefillTensor::ChunkValues);
    buffers.gdnChunkScratch = p(PrefillTensor::GdnChunkScratch);
    buffers.moe = prefillArena->moeScratch();
    const MetalBuffer finalHidden = targetModel.addPrefill(
        graph, std::move(buffers),
        std::span(modelSequences).first(batch.sequences.size()), batch.rows,
        kvPages.layers());
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
      if (entry.replayingGeneration ||
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

  void bindGdnStates(
      std::span<Request *const> entries,
      std::array<MetalBuffer, kLaneCount> &current,
      std::array<MetalBuffer, kLaneCount> &next) const;

  void bindDraftRings(
      std::span<Request *const> entries,
      std::vector<std::array<MetalBuffer, kLaneCount>> &keys,
      std::vector<std::array<MetalBuffer, kLaneCount>> &values) const;
  void encodeDraftBatchGraph(CommandGraph &graph,
                             std::span<Request *const> entries,
                             std::span<const uint64_t> logicalPositions);
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
  // The vocabulary is under 2^18, so three tokens pack into one key.
  static uint64_t ngramKey(const uint32_t *tokens) {
    return uint64_t{tokens[0]} | (uint64_t{tokens[1]} << 18) |
           (uint64_t{tokens[2]} << 36);
  }
  static constexpr uint32_t kNgramNone = ~0u;
  void noteNgramAt(Request &entry, uint32_t start);
  // (Re)seeds a lane's n-gram state from its prompt at admission; emitted
  // tokens then append through commitSelected.
  void seedNgramHistory(Request &entry, std::span<const uint32_t> prompt);
  void appendNgramTokens(Request &entry, std::span<const uint32_t> tokens);
  // Follows the most recent earlier occurrences of the stream's closing
  // 3-gram — the stream's own tail is a key's newest start, so each key
  // keeps two. The candidate with the longest backward extension wins.
  uint32_t ngramLookup(const Request &entry, uint32_t *out) const;
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
  // Greedy lanes only: injected proposals carry no probabilities.
  bool applyNgramPredraft(std::span<Request *const> entries,
                          uint32_t width);
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
                                     std::span<const uint32_t> liveRows = {});
  void encodeDraftStateCommitBatch(CommandGraph &graph,
                                   std::span<Request *const> entries,
                                   std::span<const ModelBatchItem> items,
                                   bool tree = false);
  void encodeBatchAcceptance(CommandGraph &graph,
                             std::span<Request *const> lanes,
                             std::span<const uint32_t> maximumRetained,
                             bool tree = false,
                             std::span<const uint32_t> proposals = {});
  void encodeBatchEmbedding(CommandGraph &graph, DecodeTensor tokens,
                            DecodeTensor output, uint32_t lanes,
                            uint32_t rowFactor = 1);
  void encodeBatchVerifyInput(CommandGraph &graph, uint32_t lanes);
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
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      DecodeLaneResult &laneResult = lanes[lane];
      auto d = [&](DecodeTensor tensor) {
        return decodeArena->get(lane, tensor);
      };
      laneResult.retained = *contents<uint32_t>(d(DecodeTensor::RetainedCount),
                                                "GPU retained token count");
      laneResult.accepted = *contents<uint32_t>(d(DecodeTensor::AcceptedCount),
                                                "GPU accepted draft count");
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
      const uint32_t *targetTokens = contents<uint32_t>(
          d(DecodeTensor::OutputTokens), "target output tokens");
      if (draftDebug_ && !tree) {
        const uint32_t *props = contents<uint32_t>(
            d(DecodeTensor::ProposedTokens), "proposed tokens");
        std::string p, t;
        for (uint32_t i = 0; i < kDraftProposalTokens; ++i)
          p += (i ? " " : "") + std::to_string(props[i]);
        for (uint32_t i = 0; i < laneResult.retained && i < 9; ++i)
          t += (i ? " " : "") + std::to_string(targetTokens[i]);
        fprintf(stderr, "draft lane=%u pos=%llu props=[%s] retained=%u accepted=%u out=[%s]\n",
                lane, static_cast<unsigned long long>(items[lane].logicalPosition), p.c_str(),
                laneResult.retained, laneResult.accepted, t.c_str());
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
        fprintf(stderr,
                "tree-step lane=%u retained=%u accepted=%u out=[%u %u %u %u] "
                "tree=[%u %u %u %u %u] sel=[%u %u %u %u %u] "
                "in=[%u %u %u] pos=[%u %u %u] mask=[%x %x %x] "
                "path=[%u %u %u %u] mflag=%u fail=%s\n",
                lane, laneResult.retained, laneResult.accepted,
                targetTokens[0], targetTokens[1], targetTokens[2],
                targetTokens[3], treeTok[0], treeTok[1], treeTok[2],
                treeTok[8], treeTok[9], sel[0], sel[1], sel[2], sel[8], sel[9],
                inp[0], inp[1], inp[9], pos[0], pos[3], pos[27],
                msk[1], msk[9], msk[15], rpath[0], rpath[1], rpath[2], rpath[3],
                medusaFlag, laneResult.failure.c_str());
      }
    }

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
        results.push_back({.requestId = entry.id,
                           .failure = std::move(laneResult.failure)});
        continue;
      }
      const uint32_t *targetTokens =
          contents<uint32_t>(decodeArena->get(lane, DecodeTensor::OutputTokens),
                             "target output tokens");
      std::vector<uint32_t> output;
      output.reserve(laneResult.retained);
      output.push_back(laneResult.currentAnchor);
      output.insert(output.end(), targetTokens,
                    targetTokens + (laneResult.retained - 1));

      states.swapParity(entry.stateLane);
      const uint64_t nextLength =
          items[lane].logicalPosition + laneResult.retained;
      states.updateLengths(
          entry.stateLane,
          advanceDraftContext(states.metadata(entry.stateLane).lengths,
                              nextLength, items[lane].logicalPosition,
                              nextLength, false));
      entry.generatedTokens += laneResult.retained;
      if (adaptiveProposals_ && !tree) {
        entry.proposalAcceptedAvg =
            0.75 * entry.proposalAcceptedAvg + 0.25 * laneResult.accepted;
        if (laneResult.accepted >= entry.proposalBudget) {
          // A capped count can't show how much further the chain would have
          // run; widen until the cap stops binding.
          entry.proposalBudget =
              std::min<uint32_t>(kDraftProposalTokens,
                                 entry.proposalBudget + 1);
        } else {
          entry.proposalBudget = std::clamp<uint32_t>(
              static_cast<uint32_t>(entry.proposalAcceptedAvg) + 2, 1,
              kDraftProposalTokens);
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
            entry.generatedTokens + Impl::kNgramProbeTokens;
      }
      commitSelected(entry, {targetTokens, laneResult.retained});
      entry.maskWords.clear();
      entry.verifyMaskInFlight = false;
      results.push_back({entry.id,
                         0,
                         std::move(output),
                         false,
                         DecodeStage::Regular,
                         kDraftProposalTokens,
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
                            const CommandGraph &draft,
                            std::function<void()> completion)
        : impl_(impl), lanes_(std::move(lanes)),
          items_(items.begin(), items.end()),
          wake_(std::move(completion)) {
      submit(draft);
    }

    std::vector<ModelMaskRequest> takeMaskRequests() override {
      std::vector<ModelMaskRequest> requests;
      if (stage_ == Stage::Draft && command_.ready()) {
        addTiming(command_.wait());
        std::array<Request *, kLaneCount> entries{};
        for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
          DecodeLaneResult &laneResult = lanes_[lane];
          Request &entry = *laneResult.request;
          const uint32_t *proposed = contents<uint32_t>(
              impl_.decodeArena->get(lane, DecodeTensor::ProposedTokens),
              "constrained draft proposals");
          entry.maskWords.clear();
          entry.verifyMaskInFlight = true;
          entries[lane] = &entry;

          if (!abandoned_[lane]) {
            ModelMaskRequest request;
            request.requestId = entry.id;
            request.simulationTokens.reserve(kDecodeRows);
            request.simulationTokens.push_back(*entry.pendingToken);
            request.simulationTokens.insert(request.simulationTokens.end(),
                                            proposed,
                                            proposed + kDraftProposalTokens);
            requests.push_back(std::move(request));
          }
        }

        CommandGraph target;
        const uint32_t width = static_cast<uint32_t>(lanes_.size());
        impl_.encodeBatchVerifyInput(target, width);
        impl_.encodeBatchEmbedding(target, DecodeTensor::InputTokens,
                                   DecodeTensor::Hidden0, width);
        impl_.encodeTargetVerifyBatchForward(
            target, {entries.data(), lanes_.size()}, items_);
        submit(target);
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
          masksReady = masksReady && (abandoned_[lane] ||
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
          for (uint32_t lane = 0; lane < lanes_.size(); ++lane) {
            DecodeLaneResult &laneResult = lanes_[lane];
            Request &entry = *laneResult.request;
            entries[lane] = &entry;
            maximumRetained[lane] = laneResult.maximumRetained;
            if (abandoned_[lane])
              impl_.admitEveryToken(lane);
            else
              impl_.uploadConstraintMasks(lane, entry.maskWords);
          }

          CommandGraph commit;
          impl_.encodeTargetVerifyBatchPolicy(commit,
                                              {entries.data(), lanes_.size()});
          impl_.encodeBatchAcceptance(commit, {entries.data(), lanes_.size()},
                                      {maximumRetained.data(), lanes_.size()});
          impl_.encodeBatchGdnCommit(commit, {entries.data(), lanes_.size()});
          impl_.encodeDraftStateCommitBatch(
              commit, {entries.data(), lanes_.size()}, items_);
          submit(commit);
          stage_ = Stage::Commit;
        }
      }
      return requests;
    }

    bool ownsMaskWait(uint64_t requestId) const noexcept override {
      if (stage_ == Stage::Draft || stage_ == Stage::Done)
        return false;
      return std::any_of(lanes_.begin(), lanes_.end(),
                         [requestId](const DecodeLaneResult &lane) {
                           return lane.request->id == requestId;
                         });
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
      counters.constrainedMaskOverlapRequests += lanes_.size();
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

    void addTiming(CommandTiming value) noexcept {
      timing_.gpuSeconds += value.gpuSeconds;
      timing_.wallSeconds += value.wallSeconds;
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
