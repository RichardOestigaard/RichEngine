// DiffusionGemma denoising decode: the per-request canvas loop. One decode
// step is one canvas per request — up to maxDenoisingSteps denoising passes
// over the canvas positions, chunked into shared synchronous commands —
// then the commit, which argmaxes the last logits, and the encoder pass
// that writes the committed canvas's KV into the request's pages ahead of
// the next canvas. The canvas KV lands in the request's own pages at and
// past the committed length: those slots are uncommitted (the engine
// admits the canvas ahead of the step) and the commit pass rewrites them
// with the committed tokens' keys and values, so no scratch extent is
// needed.
//
// Batched lanes: when the scheduler hands over several diffusion requests,
// their canvases denoise in the same step — each lane keeps its own
// tokens/argmax/entropy buffers, but the trunk pass runs once over the
// concatenated rows (one addPrefill with a sequence per lane), so the
// weight traffic of a step is shared across lanes. Rows of a lane that
// exited earlier stay inside the row span as dead compute — the weight
// reads, which dominate the step, are unaffected.

#include "model/RuntimeImpl.hpp"
#include "Tuning.hpp"

#include "ops/Canvas.hpp"
#include "ops/Embedding.hpp"
#include "ops/KernelNames.hpp"
#include "ops/Normalization.hpp"

#include <cstring>
#include <vector>

namespace richengine::model {
namespace {

// The tokens that end a committed canvas; the trailing positions take the
// pad id.
constexpr std::array<uint32_t, 3> kCanvasStopTokens = {1, 106, 50};

} // namespace

// One lane's canvas scratch: the tokens ping-pong, the per-step argmax and
// entropy rings a chunked command drains through, the commit/spec inputs
// (kept apart from `positions` so an in-flight commit is never rewritten),
// and the self-conditioning chain's intermediates.
struct Runtime::Impl::DiffusionLaneArena final {
  MetalBuffer tokens[2];
  MetalBuffer argmaxRing[CanvasStepDriver::kMaxStepsPerCommand];
  MetalBuffer argmaxCur;
  MetalBuffer sampled;
  // [kMaxStepsPerCommand][rows] fp32 — slot s at float offset s*rows.
  MetalBuffer entropy;
  // {mean entropy, argmax-stable flag} per step of the chunk: slot i sits
  // at float offset 2i and is bound as a view.
  MetalBuffer stats;
  MetalBuffer positions;
  MetalBuffer commitPositions;
  MetalBuffer specPositions;
  MetalBuffer commitTokens;
  MetalBuffer specTokens;
  // The self-conditioning block: the previous step's soft embeddings, its
  // normed gate/up inputs and the down projection's output.
  MetalBuffer softEmbeds;
  MetalBuffer scNormed;
  MetalBuffer scGate;
  MetalBuffer scUp;
  MetalBuffer scDown;
};

// The diffusion decode's tensors, allocated once per runtime and shared by
// every request's canvas (decodes never overlap). Per-lane scratch lives in
// `lanes`; `headHidden`/`logits` are the concatenated row space the batched
// trunk and LM head write — one buffer, lane l's slice at row offset l's
// cumulative rows. Everything else rides the ordinary prefill arena.
struct Runtime::Impl::DiffusionCanvasArena final {
  DiffusionCanvasArena(MetalBackend &backend,
                       const RuntimeGeometry &geometry, uint32_t rows)
      : backend_(backend), geometry_(geometry), rows_(rows) {
    auto shared = [&](uint64_t bytes, const char *name) {
      return backend.allocateBuffer(bytes, BufferStorage::Shared, name);
    };
    tokensZero = shared(bytesFor<uint32_t>(rows),
                        "diffusion-canvas-tokens-zero");
    std::memset(tokensZero.contents(), 0, bytesFor<uint32_t>(rows));
    const uint64_t hiddenBytes =
        bytesFor<uint16_t>(uint64_t{rows} * geometry.target.hiddenSize);
    scZero = shared(hiddenBytes, "diffusion-sc-zero");
    std::memset(scZero.contents(), 0, hiddenBytes);
  }

  // Grow the lane scratch and the concat tails so `count` lanes can run one
  // batched canvas pass. Buffers sized for `rows` canvas rows per lane
  // (the schedule's full canvas length; lanes dispatch fewer rows when the
  // budget shrinks the canvas).
  void ensureLanes(uint32_t count) {
    const uint32_t rows = rows_;
    const uint32_t hidden = geometry_.target.hiddenSize;
    const uint32_t width = geometry_.target.denseIntermediateSize;
    const uint64_t u32Row = bytesFor<uint32_t>(rows);
    auto shared = [&](uint64_t bytes, const char *name) {
      return backend_.allocateBuffer(bytes, BufferStorage::Shared, name);
    };
    auto device = [&](uint64_t bytes, const char *name) {
      return backend_.allocateBuffer(bytes, BufferStorage::Private, name);
    };
    while (lanes.size() < count) {
      DiffusionLaneArena lane;
      for (MetalBuffer &buffer : lane.tokens)
        buffer = shared(u32Row, "diffusion-canvas-tokens");
      for (MetalBuffer &buffer : lane.argmaxRing)
        buffer = shared(u32Row, "diffusion-canvas-argmax");
      lane.argmaxCur = shared(u32Row, "diffusion-canvas-argmax-cur");
      lane.sampled = shared(u32Row, "diffusion-canvas-sampled");
      lane.entropy = shared(
          bytesFor<float>(rows * CanvasStepDriver::kMaxStepsPerCommand),
          "diffusion-canvas-entropy");
      lane.stats = shared(
          bytesFor<float>(2 * CanvasStepDriver::kMaxStepsPerCommand),
          "diffusion-canvas-stats");
      lane.positions = shared(bytesFor<uint32_t>(rows * 3),
                              "diffusion-canvas-positions");
      lane.commitPositions = shared(bytesFor<uint32_t>(rows * 3),
                                    "diffusion-commit-positions");
      lane.specPositions = shared(bytesFor<uint32_t>(rows * 3),
                                  "diffusion-spec-commit-positions");
      lane.commitTokens = shared(u32Row, "diffusion-commit-tokens");
      lane.specTokens = shared(u32Row, "diffusion-spec-commit-tokens");
      const uint64_t hiddenBytes =
          bytesFor<uint16_t>(uint64_t{rows} * hidden);
      const uint64_t wideBytes =
          bytesFor<uint16_t>(uint64_t{rows} * width);
      lane.softEmbeds = device(hiddenBytes, "diffusion-soft-embeds");
      lane.scNormed = device(hiddenBytes, "diffusion-sc-normed");
      lane.scGate = device(wideBytes, "diffusion-sc-gate");
      lane.scUp = device(wideBytes, "diffusion-sc-up");
      lane.scDown = device(hiddenBytes, "diffusion-sc-down");
      lanes.push_back(std::move(lane));
    }
    if (count > capacity_) {
      capacity_ = count;
      headHidden = shared(
          bytesFor<uint16_t>(uint64_t{capacity_} * rows * hidden),
          "diffusion-head-normed");
      logits = shared(bytesFor<uint16_t>(uint64_t{capacity_} * rows *
                                         geometry_.target.vocabularySize),
                      "diffusion-canvas-logits");
    }
  }

  MetalBackend &backend_;
  const RuntimeGeometry &geometry_;
  const uint32_t rows_; // per-lane buffer rows (the schedule's canvas length)
  uint32_t capacity_ = 0;
  std::vector<DiffusionLaneArena> lanes;
  MetalBuffer tokensZero; // the first step's argmax_prev: an empty canvas
  MetalBuffer scZero;     // step 0's absent soft embeddings
  // bf16 [capacity_*rows][hidden] — the final-norm output the batched LM
  // head reads; the head writes into `logits`.
  MetalBuffer headHidden;
  // bf16 [capacity_*rows][vocabulary] — also the next step's soft-embedding
  // input; a lane's slice starts at its cumulative row offset.
  MetalBuffer logits;
};

Runtime::Impl::DiffusionCanvasArena &Runtime::Impl::canvasArena() {
  if (!diffusion_.canvasArena_) {
    const DiffusionSchedule &schedule =
        std::get<DiffusionGemmaLayout>(package.descriptor.target).diffusion;
    diffusion_.canvasArena_ = std::make_shared<DiffusionCanvasArena>(
        backend, geometry, schedule.canvasLength);
  }
  return *diffusion_.canvasArena_;
}

const DiffusionGemmaWeights &Runtime::Impl::diffusionWeights() const {
  return std::get<DiffusionGemmaWeights>(package.target);
}

std::span<const float> Runtime::Impl::encoderLayerScalars() const {
  if (!diffusion)
    return {};
  return diffusionWeights().encoderLayerScalars;
}

// One lane's share of a denoising step, up to the shared trunk: the rope
// positions for its canvas, the scaled embedding of its current canvas
// written into the lane's slice of hidden[0], the previous step's
// self-conditioning injection on the lane's own buffers, and the rope
// tables written at the lane's row offset.
void Runtime::Impl::encodeCanvasLaneFront(
    CommandGraph &graph, const Request &entry, DiffusionLaneArena &lane,
    const DiffusionSchedule &schedule, uint32_t step,
    uint32_t prefixTokens, uint32_t rows, uint32_t rowOffset,
    TargetModelPrefillBuffers &buffers) {
  const DiffusionGemmaWeights &weights = diffusionWeights();
  DiffusionCanvasArena &a = canvasArena();
  const ops::Linear &linear = operators.linear();
  const uint32_t hidden = geometry.target.hiddenSize;
  const uint32_t width = geometry.target.denseIntermediateSize;
  const uint32_t phase = step & 1;
  const MetalBuffer tokensIn = lane.tokens[phase];

  // Canvas positions continue after the encoder prefix.
  auto *positions =
      contents<uint32_t>(lane.positions, "canvas rope positions");
  for (uint32_t row = 0; row < rows; ++row) {
    const std::array<uint32_t, 3> rotary =
        ropePosition(entry, uint64_t{prefixTokens} + row);
    std::copy(rotary.begin(), rotary.end(), positions + row * 3);
  }
  const uint64_t hiddenRow = uint64_t{hidden} * sizeof(uint16_t);
  const MetalBuffer hiddenSlice = backend.view(
      buffers.hidden[0], rowOffset * hiddenRow, uint64_t{rows} * hiddenRow);
  targetModel.addEmbedding(graph, tokensIn, hiddenSlice, rows);
  // Self-conditioning: soft_emb = softmax(prev logits) @ embedding *
  // sqrt(hidden) — canvas_soft_embed_topk — through the GeGLU block, fused
  // into the embeddings by canvas_self_condition's residual+scaleless-norm.
  // The norm is applied on every step, including the first (no embeddings):
  // it erases the embedding scale, matching the reference.
  const bool fusedTail = ops::Canvas::fusedLogitTail();
  const float softcap = geometry.target.logitSoftcap;
  const MetalBuffer sc =
      step == schedule.maxDenoisingSteps ? a.scZero : lane.scDown;
  if (step != schedule.maxDenoisingSteps && diffusion_.canvasProbe_ != "nosc") {
    const float embedScale = geometry.target.embeddingScale
                                 ? geometry.target.embeddingScale
                                 : std::sqrt(static_cast<float>(hidden));
    // The previous step's temperature applies to its logits: the fused
    // kernel transforms on load with prevInverseTemperature.
    const float prevInverseTemperature =
        1.0F / schedule.temperature(step + 1);
    const uint64_t vocabRow =
        uint64_t{geometry.target.vocabularySize} * sizeof(uint16_t);
    const MetalBuffer laneLogits = backend.view(
        a.logits, rowOffset * vocabRow, uint64_t{rows} * vocabRow);
    ops::Canvas::addSoftEmbed(graph, laneLogits, weights.tokenEmbedding,
                              lane.softEmbeds, geometry.target.vocabularySize,
                              rows, /*topK=*/64, embedScale,
                              /*exact=*/false,
                              fusedTail ? softcap : 0.0F,
                              fusedTail ? prevInverseTemperature : 0.0F);
    ops::Normalization::addRmsWithQ4Sums(
        graph, lane.softEmbeds, weights.selfConditioning.preNorm,
        lane.scNormed, buffers.projectionSums, hidden, rows);
    linear.addPrefill(graph, lane.scNormed, weights.selfConditioning.gate,
                      lane.scGate, buffers.projectionSums, rows,
                      buffers.linearScratch, {}, true);
    linear.addPrefill(graph, lane.scNormed, weights.selfConditioning.up,
                      lane.scUp, buffers.projectionSums, rows,
                      buffers.linearScratch, {}, true);
    const uint32_t intermediate = rows * width;
    graph.add(std::string(ops::kGegluMultiply), {lane.scGate, lane.scUp, lane.scGate},
              intermediate, {(intermediate + 255) / 256, 1, 1}, {256, 1, 1});
    linear.addPrefillSums(graph, lane.scGate, buffers.projectionSums,
                          weights.selfConditioning.down, rows);
    linear.addPrefill(graph, lane.scGate, weights.selfConditioning.down,
                      lane.scDown, buffers.projectionSums, rows,
                      buffers.linearScratch, {}, true);
  }
  if (diffusion_.canvasProbe_ != "nosc")
    ops::Canvas::addSelfCondition(graph, hiddenSlice, sc, hiddenSlice, hidden,
                                  rows);

  // No draft side: the positions/table slots borrow the target's own
  // buffers (the kernel writes zero draft rows). The rope tables land at
  // the lane's row offset inside the shared row space.
  const uint64_t ropeRow =
      uint64_t{geometry.target.rotaryPairs} * sizeof(float);
  const MetalBuffer ropeCos = backend.view(
      buffers.ropeCos, rowOffset * ropeRow, uint64_t{rows} * ropeRow);
  const MetalBuffer ropeSin = backend.view(
      buffers.ropeSin, rowOffset * ropeRow, uint64_t{rows} * ropeRow);
  MetalBuffer ropeCosAlt, ropeSinAlt;
  if (buffers.ropeCosAlt) {
    const uint64_t altRow =
        uint64_t{geometry.target.altRotaryPairs} * sizeof(float);
    ropeCosAlt = backend.view(buffers.ropeCosAlt, rowOffset * altRow,
                              uint64_t{rows} * altRow);
    ropeSinAlt = backend.view(buffers.ropeSinAlt, rowOffset * altRow,
                              uint64_t{rows} * altRow);
  }
  addRopeTables(graph, lane.positions, rows, lane.positions, 0, ropeCos,
                ropeSin, ropeCos, ropeSin, ropeCosAlt, ropeSinAlt);
}

// The lane's share after the batched trunk: the head already wrote the
// lane's logits slice; the row stats and the entropy-bounded accept run on
// the lane's own ring buffers and write the lane's next canvas.
void Runtime::Impl::encodeCanvasLaneBack(
    CommandGraph &graph, DiffusionLaneArena &lane,
    const DiffusionSchedule &schedule, uint32_t step, uint32_t canvasIndex,
    uint32_t seed, uint32_t rows, uint32_t rowOffset,
    MetalBuffer argmaxPrev, MetalBuffer argmaxOut, MetalBuffer statsSlot,
    MetalBuffer entropySlot, bool commitOnly) {
  DiffusionCanvasArena &a = canvasArena();
  const uint64_t vocabRow =
      uint64_t{geometry.target.vocabularySize} * sizeof(uint16_t);
  const MetalBuffer laneLogits = backend.view(
      a.logits, rowOffset * vocabRow, uint64_t{rows} * vocabRow);
  const bool fusedTail = ops::Canvas::fusedLogitTail();
  const float softcap = geometry.target.logitSoftcap;
  const float inverseTemperature = 1.0F / schedule.temperature(step);
  const uint64_t logitCount = uint64_t{rows} * geometry.target.vocabularySize;
  // The fused kernels transform on load; the standalone passes are only
  // needed for the unfused fallbacks.
  if (!fusedTail) {
    if (softcap > 0.0F)
      ops::Canvas::addLogitSoftcap(graph, laneLogits, softcap, logitCount);
    ops::Canvas::addLogitsScale(graph, laneLogits, inverseTemperature,
                                logitCount);
  }
  const uint32_t stepSeed =
      seed ^ (step * 0x9E3779B9u) ^ canvasIndex * 0x85EBCA6Bu;
  ops::Canvas::addRowStats(graph, laneLogits, lane.sampled, lane.argmaxCur,
                           entropySlot, geometry.target.vocabularySize,
                           stepSeed, rows,
                           fusedTail ? softcap : 0.0F,
                           fusedTail ? inverseTemperature : 0.0F);
  if (commitOnly)
    return;
  const uint32_t phase = step & 1;
  ops::Canvas::addEntropyAccept(
      graph, std::move(entropySlot), lane.sampled, std::move(argmaxPrev),
      lane.tokens[phase ^ 1],
      std::move(argmaxOut), std::move(statsSlot), lane.argmaxCur,
      schedule.entropyBound, geometry.target.vocabularySize, stepSeed,
      rows);
}

// The encoder pass over the committed canvas: an ordinary causal prefill of
// the committed tokens at their logical positions, into the request's real
// pages, with the encoder's layer scalars. Each argument list entry is one
// lane; the trunk runs once over the concatenated rows.
void Runtime::Impl::encodeCanvasCommitPrefill(
    CommandGraph &graph, std::span<const CommitLane> lanes) {
  if (lanes.empty())
    return;
  TargetModelPrefillBuffers buffers = detail::prefillBuffers(*prefillArena);
  const uint64_t hiddenRow =
      uint64_t{geometry.target.hiddenSize} * sizeof(uint16_t);
  const uint64_t ropeRow =
      uint64_t{geometry.target.rotaryPairs} * sizeof(float);
  std::vector<TargetModelPrefillSequence> sequences;
  sequences.reserve(lanes.size());
  uint32_t rowOffset = 0;
  for (const CommitLane &lane : lanes) {
    auto *positions =
        contents<uint32_t>(lane.ropePositions, "commit rope positions");
    for (uint32_t row = 0; row < lane.rows; ++row) {
      const std::array<uint32_t, 3> rotary =
          ropePosition(lane.entry, lane.position + row);
      std::copy(rotary.begin(), rotary.end(), positions + row * 3);
    }
    targetModel.addEmbedding(
        graph, lane.tokens,
        backend.view(buffers.hidden[0], rowOffset * hiddenRow,
                     uint64_t{lane.rows} * hiddenRow),
        lane.rows);
    addRopeTables(
        graph, lane.ropePositions, lane.rows, lane.ropePositions, 0,
        backend.view(buffers.ropeCos, rowOffset * ropeRow,
                     uint64_t{lane.rows} * ropeRow),
        backend.view(buffers.ropeSin, rowOffset * ropeRow,
                     uint64_t{lane.rows} * ropeRow),
        buffers.ropeCos, buffers.ropeSin,
        buffers.ropeCosAlt, buffers.ropeSinAlt);
    TargetModelPrefillSequence sequence;
    sequence.rowBegin = rowOffset;
    sequence.rows = lane.rows;
    sequence.attentionStride = lane.rows;
    sequence.chunk = ops::PagedAttention::prefillParams(
        lane.position, lane.rows, lane.rows, lane.pageTableEntries);
    sequence.pageTable = lane.pageTable;
    sequences.push_back(std::move(sequence));
    rowOffset += lane.rows;
  }
  static_cast<void>(
      targetModel.addPrefill(graph, std::move(buffers), sequences, rowOffset,
                             kvPages.layers(), nullptr,
                             encoderLayerScalars()));
}

std::unique_ptr<ModelBatchTicket>
Runtime::Impl::decodeDiffusion(const BatchPlan &plan,
                               std::span<const ModelBatchItem> items,
                               std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const DiffusionGemmaWeights &weights = diffusionWeights();
  DiffusionSchedule schedule = weights.layout.diffusion;
  if (diffusion_.canvasMaxSteps_)
    schedule.maxDenoisingSteps =
        std::min(schedule.maxDenoisingSteps, diffusion_.canvasMaxSteps_);
  DiffusionCanvasArena &a = canvasArena();
  a.ensureLanes(static_cast<uint32_t>(items.size()));

  // The per-request results computed by the steps that already ran, applied
  // when the trailing commit command completes.
  struct LaneResult final {
    uint64_t requestId = 0;
    std::vector<uint32_t> tokens;
    uint32_t stopIndex = 0;
    bool finished = false;
    std::string failure;
  };

  // One batched canvas lane: the request's sampler, step driver and ring
  // state for this canvas, plus the lane's row offset inside the shared
  // hidden/logits space the trunk pass writes.
  struct CanvasLane final {
    CanvasLane(Request &requestEntry, const ModelBatchItem &batchItem,
               const DiffusionSchedule &base, uint32_t laneRows,
               const CanvasStepDriver::Config &config,
               std::span<const uint32_t> stops, uint32_t offset)
        : entry(&requestEntry),
          sampler(adaptiveSchedule(base, requestEntry, laneRows)),
          driver(sampler.schedule(), config, stops),
          position(batchItem.logicalPosition),
          rowOffset(offset) {}

    static DiffusionSchedule adaptiveSchedule(const DiffusionSchedule &base,
                                              const Request &entry,
                                              uint32_t laneRows) {
      DiffusionSchedule steps = base;
      if (entry.warmup) {
        // Warmup canvases run a two-step schedule: they exist to compile
        // every canvas kernel once, not to denoise — the bootstrap ladder
        // would otherwise pay 11 full canvases before the server is ready.
        steps.maxDenoisingSteps = 2;
      } else {
        // A real canvas shrinks to the request's remaining token budget
        // (rounded up to a KV page, floored at two pages, capped by the
        // lane buffers): the tail rows are dead denoising work. This is
        // canvas-only sizing — nothing outside the canvas sees it.
        const uint64_t remaining =
            entry.maxNewTokens > entry.generatedTokens
                ? uint64_t{entry.maxNewTokens} - entry.generatedTokens
                : 0;
        if (remaining && remaining < steps.canvasLength) {
          const uint32_t pages = static_cast<uint32_t>(
              (remaining + kv::kPageTokens - 1) / kv::kPageTokens);
          steps.canvasLength = std::min(
              laneRows,
              std::max(2 * kv::kPageTokens, pages * kv::kPageTokens));
        }
      }
      return steps;
    }

    Request *entry;
    DiffusionSampler sampler;
    CanvasStepDriver driver;
    MetalBuffer pageTable;
    uint32_t entries = 0;
    uint32_t canvasSeed = 0;
    uint64_t position = 0;
    uint32_t rowOffset = 0;
    // Ring slot holding the previous step's argmax; empty means the first
    // step (argmax_prev = tokensZero). Steps descending alternate the
    // canvas tokens ping-pong by parity, so consecutive steps inside one
    // command chain cleanly.
    MetalBuffer argmaxPrev;
    std::vector<uint32_t> committed;
    CanvasStepDriver::Chunk chunk{};
    bool canvasDone = false;
    bool committedOut = false;
    // The speculative commit prefill (RICHENGINE_CANVAS_SPECULATIVE_
    // PREFILL): armed when a drained step's whole-canvas argmax was
    // unchanged, encoded on dedicated tokens/positions buffers, discarded
    // when the formal commit's tokens differ.
    std::optional<CommandTicket> specPrefill;
    std::vector<uint32_t> specSnapshot;
    bool stateApplied = false;
    LaneResult result;
  };

  std::vector<CanvasLane> lanes;
  lanes.reserve(items.size());
  double gpuSeconds = 0.0;
  double wallSeconds = 0.0;
  const CanvasStepDriver::Config driverConfig{
      diffusion_.canvasStepsPerCmd_, diffusion_.canvasPrefixExit_, diffusion_.canvasCommitTail_,
      diffusion_.canvasExitStable_, diffusion_.canvasDriftExit_};
  uint32_t rowOffset = 0;
  for (const ModelBatchItem &item : items) {
    Request &entry = request(item.requestId);
    if (!entry.resident || !entry.promptComplete ||
        entry.decodeStage != DecodeStage::Regular)
      throw std::logic_error("diffusion decode request is not decode-ready");
    if (entry.constraint != ConstraintMode::None || !entry.scoreTokens.empty())
      throw std::invalid_argument(
          "diffusion decode does not take constrained or score lanes");
    CanvasLane &lane =
        lanes.emplace_back(entry, item, schedule, a.rows_, driverConfig,
                           kCanvasStopTokens, rowOffset);
    lane.pageTable = synchronizedPageTable(entry, item);
    lane.entries = static_cast<uint32_t>(item.pageTable.size());
    lane.canvasSeed = static_cast<uint32_t>(
        entry.sampling.seed ^ (uint64_t{entry.canvasIndex} * 2654435761u));
    lane.committed.assign(lane.sampler.schedule().canvasLength, 0);
    lane.result.requestId = entry.id;
    rowOffset += lane.sampler.schedule().canvasLength;
  }

  // Canvas init: uniform noise per lane, one command; the first step's
  // argmax_prev is zeros.
  {
    CommandGraph graph;
    for (size_t i = 0; i < lanes.size(); ++i) {
      ops::Canvas::addUniformNoise(
          graph, a.lanes[i].tokens[0], geometry.target.vocabularySize,
          lanes[i].canvasSeed, lanes[i].sampler.schedule().canvasLength);
    }
    const CommandTiming timing =
        backend.submitCommandAsync(graph.command()).wait();
    gpuSeconds += timing.gpuSeconds;
    wallSeconds += timing.wallSeconds;
  }

  const ops::Linear &linear = operators.linear();
  const uint32_t hidden = geometry.target.hiddenSize;
  size_t pending = lanes.size();
  while (pending) {
    // Arm the next chunk for every lane still denoising.
    uint32_t slots = 0;
    for (CanvasLane &lane : lanes) {
      if (lane.canvasDone)
        continue;
      lane.chunk = lane.driver.nextChunk();
      if (lane.chunk.count)
        slots = std::max(slots, lane.chunk.count);
    }
    if (slots) {
      CommandGraph graph;
      for (uint32_t slot = 0; slot < slots; ++slot) {
        TargetModelPrefillBuffers buffers =
            detail::prefillBuffers(*prefillArena);
        std::vector<TargetModelPrefillSequence> sequences;
        sequences.reserve(lanes.size());
        uint32_t spanRows = 0;
        // Front halves: each lane writes its hidden slice, self-conditioning
        // chain and rope tables, then contributes one canvas sequence.
        for (size_t i = 0; i < lanes.size(); ++i) {
          CanvasLane &lane = lanes[i];
          if (lane.canvasDone || slot >= lane.chunk.count)
            continue;
          const uint32_t step = lane.chunk.firstStep - slot;
          const uint32_t laneRows = lane.sampler.schedule().canvasLength;
          encodeCanvasLaneFront(graph, *lane.entry, a.lanes[i],
                                lane.sampler.schedule(), step,
                                static_cast<uint32_t>(lane.position),
                                laneRows, lane.rowOffset, buffers);
          TargetModelPrefillSequence sequence;
          sequence.rowBegin = lane.rowOffset;
          sequence.rows = laneRows;
          sequence.attentionStride = laneRows;
          sequence.chunk = ops::PagedAttention::prefillParams(
              lane.position, laneRows, laneRows, lane.entries);
          sequence.pageTable = lane.pageTable;
          sequence.canvas = true;
          sequences.push_back(std::move(sequence));
          spanRows = std::max(spanRows, lane.rowOffset + laneRows);
        }
        // The shared pass: one trunk sweep over the concatenated rows and
        // one LM-head GEMM — the step's weight traffic is read once for
        // every lane.
        const MetalBuffer headSums = buffers.projectionSums;
        MetalBuffer finalHidden = buffers.hidden[0];
        if (diffusion_.canvasProbe_ != "notrunk")
          finalHidden =
              targetModel.addPrefill(graph, std::move(buffers), sequences,
                                     spanRows, kvPages.layers());
        ops::Normalization::addRmsWithQ4Sums(
            graph, finalHidden, weights.finalNorm, a.headHidden, headSums,
            hidden, spanRows);
        if (diffusion_.canvasProbe_ != "notail")
        {
          TargetModelPrefillBuffers head =
              detail::prefillBuffers(*prefillArena);
          linear.addPrefill(graph, a.headHidden, weights.logitsProjection,
                            a.logits, head.projectionSums, spanRows,
                            head.linearScratch, {}, true);
        }
        // Per-lane tails: row stats and the entropy accept on the lane's
        // own logits slice and ring buffers.
        for (size_t i = 0; i < lanes.size() && diffusion_.canvasProbe_ != "notail";
             ++i) {
          CanvasLane &lane = lanes[i];
          if (lane.canvasDone || slot >= lane.chunk.count)
            continue;
          const uint32_t step = lane.chunk.firstStep - slot;
          const uint32_t laneRows = lane.sampler.schedule().canvasLength;
          const bool commitOnly =
              lane.chunk.commitTail && slot + 1 == lane.chunk.count;
          DiffusionLaneArena &la = a.lanes[i];
          const MetalBuffer prev =
              slot ? la.argmaxRing[slot - 1]
                   : (lane.argmaxPrev ? lane.argmaxPrev : a.tokensZero);
          const MetalBuffer statsSlot =
              backend.view(la.stats, slot * 2 * sizeof(float),
                           2 * sizeof(float));
          const MetalBuffer entropySlot =
              backend.view(la.entropy, slot * laneRows * sizeof(float),
                           laneRows * sizeof(float));
          encodeCanvasLaneBack(graph, la, lane.sampler.schedule(), step,
                               lane.entry->canvasIndex, lane.canvasSeed,
                               laneRows, lane.rowOffset, prev,
                               la.argmaxRing[slot], statsSlot, entropySlot,
                               commitOnly);
        }
      }
      const CommandTiming timing =
          backend.submitCommandAsync(graph.command()).wait();
      gpuSeconds += timing.gpuSeconds;
      wallSeconds += timing.wallSeconds;
    }

    // Drain the ring slots in encode order per lane; the first exit ends
    // the lane's canvas — steps encoded past it were dead compute.
    for (size_t i = 0; i < lanes.size(); ++i) {
      CanvasLane &lane = lanes[i];
      if (lane.canvasDone)
        continue;
      DiffusionLaneArena &la = a.lanes[i];
      const uint32_t laneRows = lane.sampler.schedule().canvasLength;
      const float *ring = contents<float>(la.stats, "canvas stats ring");
      const float *entropies =
          contents<float>(la.entropy, "canvas entropy ring");
      for (uint32_t slot = 0; slot < lane.chunk.count; ++slot) {
        const bool commitOnly =
            lane.chunk.commitTail && slot + 1 == lane.chunk.count;
        const uint32_t *argmax = contents<uint32_t>(
            commitOnly ? la.argmaxCur : la.argmaxRing[slot],
            "canvas argmax");
        lane.committed.assign(argmax, argmax + laneRows);
        const CanvasStepDriver::Outcome outcome = lane.driver.drain(
            slot, {ring[2 * slot], ring[2 * slot + 1] != 0.0F},
            lane.committed,
            {entropies + slot * laneRows, static_cast<size_t>(laneRows)});
        if (tuning().canvasTiming)
          fprintf(stderr,
                  "  lane %zu step %u: entropy=%.4f stable=%u "
                  "settled=%.2f/%.4f\n",
                  i, lane.chunk.firstStep - slot, ring[2 * slot],
                  unsigned(ring[2 * slot + 1] != 0.0F),
                  outcome.stableFraction, outcome.stableMeanEntropy);
        if (!commitOnly)
          lane.argmaxPrev = la.argmaxRing[slot];
        if (outcome.exited || outcome.scheduleEnd) {
          lane.canvasDone = true;
          break;
        }
      }
      if (diffusion_.canvasSpecPrefill_ && !lane.specPrefill && !lane.canvasDone &&
          lane.driver.speculationArmed()) {
        lane.specSnapshot =
            lane.sampler.commit(lane.committed, kCanvasStopTokens).tokens;
        std::memcpy(la.specTokens.contents(), lane.specSnapshot.data(),
                    laneRows * sizeof(uint32_t));
        CommandGraph specGraph;
        const CommitLane spec{*lane.entry, lane.position, laneRows,
                              lane.pageTable, lane.entries, la.specTokens,
                              la.specPositions};
        encodeCanvasCommitPrefill(specGraph, {&spec, 1});
        lane.specPrefill = backend.submitCommandAsync(specGraph.command());
      }
    }

    // Commit every lane whose canvas finished this round in one encoder
    // pass: the committed canvases' KV joins each request's prefix.
    std::vector<CommitLane> commits;
    for (size_t i = 0; i < lanes.size(); ++i) {
      CanvasLane &lane = lanes[i];
      if (!lane.canvasDone || lane.committedOut)
        continue;
      DiffusionLaneArena &la = a.lanes[i];
      const uint32_t laneRows = lane.sampler.schedule().canvasLength;
      DiffusionSampler::Commit commit =
          lane.sampler.commit(lane.committed, kCanvasStopTokens);
      if (tuning().canvasTiming)
        fprintf(stderr, "canvas %u lane %zu: wall=%.0fms gpu=%.0fms stop=%u\n",
                lane.entry->canvasIndex, i, wallSeconds * 1000.0,
                gpuSeconds * 1000.0, commit.stopIndex);
      if (tuning().opTimings) metal::dumpOpTimings();
      bool skipPrefill = false;
      if (lane.specPrefill) {
        const CommandTiming timing = lane.specPrefill->wait();
        gpuSeconds += timing.gpuSeconds;
        wallSeconds += timing.wallSeconds;
        // Same queue, in order: a stale speculation wrote KV the real
        // prefill simply overwrites, so only the token match decides.
        skipPrefill = lane.specSnapshot == commit.tokens;
      }
      if (!skipPrefill) {
        std::memcpy(la.commitTokens.contents(), commit.tokens.data(),
                    laneRows * sizeof(uint32_t));
        commits.push_back({*lane.entry, lane.position, laneRows,
                           lane.pageTable, lane.entries, la.commitTokens,
                           la.commitPositions});
      }
      lane.committedOut = true;
      --pending;
      lane.result.stopIndex = commit.stopIndex;
      // Emit through the first stop token inclusive; the padded rows past
      // it are KV-only. The output budget may cap earlier.
      const uint32_t emitted = std::min<uint32_t>(
          lane.sampler.emitBudget(lane.entry->generatedTokens,
                                  lane.entry->maxNewTokens),
          commit.stop ? commit.stopIndex + 1 : laneRows);
      lane.result.tokens.assign(commit.tokens.begin(),
                                commit.tokens.begin() + emitted);
      lane.result.finished =
          lane.sampler.finished(commit, emitted) &&
          !(lane.entry->flags & RequestIgnoreEndOfSequence);
      lane.entry->generatedTokens += emitted;
      lane.entry->canvasIndex += 1;
    }
    if (!commits.empty()) {
      CommandGraph graph;
      encodeCanvasCommitPrefill(graph, commits);
      const CommandTiming timing =
          backend.submitCommandAsync(graph.command()).wait();
      gpuSeconds += timing.gpuSeconds;
      wallSeconds += timing.wallSeconds;
    }
    // The committed canvases' KV is part of each request's prefix now.
    for (size_t i = 0; i < lanes.size(); ++i) {
      CanvasLane &lane = lanes[i];
      if (!lane.committedOut || lane.stateApplied)
        continue;
      lane.stateApplied = true;
      states.swapParity(lane.entry->stateLane);
      LogicalLengths lengths =
          states.metadata(lane.entry->stateLane).lengths;
      lengths.targetTokens = static_cast<uint32_t>(lane.position) +
                             lane.sampler.schedule().canvasLength;
      states.updateLengths(lane.entry->stateLane, lengths);
    }
  }

  std::vector<LaneResult> laneResults;
  laneResults.reserve(lanes.size());
  for (CanvasLane &lane : lanes)
    laneResults.push_back(std::move(lane.result));

  auto finish = [laneResults = std::move(laneResults),
                 gpuSeconds, wallSeconds](CommandTiming &timing) mutable {
    timing.gpuSeconds += gpuSeconds;
    timing.wallSeconds += wallSeconds;
    std::vector<ModelStepResult> results;
    for (size_t index = 0; index < laneResults.size(); ++index) {
      LaneResult &lane = laneResults[index];
      ModelStepResult result;
      result.requestId = lane.requestId;
      result.outputTokens = std::move(lane.tokens);
      result.finished = lane.finished;
      result.failure = std::move(lane.failure);
      results.push_back(std::move(result));
    }
    return results;
  };
  // Every canvas step already ran to completion; the ticket reports the
  // batch synchronously through a command holding one harmless dispatch
  // (the backend rejects command-less submissions). The write lands in a
  // specTokens buffer, which is always refilled before use.
  CommandGraph empty;
  ops::Canvas::addUniformNoise(empty, a.lanes[0].specTokens,
                               geometry.target.vocabularySize, 0,
                               schedule.canvasLength);
  CommandTicket command =
      backend.submitCommandAsync(empty.command(), std::move(completion));
  return std::make_unique<detail::DeferredMetalTicket>(std::move(command),
                                                     std::move(finish));
}

} // namespace richengine::model
