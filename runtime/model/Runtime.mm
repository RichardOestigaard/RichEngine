#include "model/Runtime.hpp"
#include "AwakeClock.hpp"
#include "Env.hpp"
#include "model/AnePredictor.hpp"
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

#include "model/RuntimeImpl.hpp"

namespace richengine::model {

Runtime::Runtime(RuntimeContext context)
    : impl_(std::make_unique<Impl>(context)) {}

Runtime::~Runtime() = default;

void Runtime::checkHealth() { impl_->backend.checkHealth(); }

void Runtime::beginColdRequest(const ModelRequest &request,
                               uint32_t stateLane) {
  if (const StateAdmission admission = beginAt(request, stateLane); !admission.granted()) {
    throw metal::MetalAllocationError(
        std::string("unable to allocate a lane's state: ") +
            metal::allocationFailureName(admission.allocationFailure),
        admission.allocationFailure);
  }
  try {
    setDraftContextPlan(
        request.id,
        planDraftContext(0, static_cast<uint32_t>(request.prompt.size()), {}));
  } catch (...) {
    end(request.id);
    throw;
  }
}

StateAdmission Runtime::begin(const ModelRequest &request) {
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  StateAdmission admission = admitIdleLane(
      impl_->states, [&](uint32_t lane) { return beginAt(request, lane); });
  rollback.committed = admission.granted();
  return admission;
}

void Runtime::suspend(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || entry.verifyMaskInFlight) {
    throw std::logic_error("Qwen request cannot be suspended");
  }
  impl_->states.releaseLane(entry.stateLane, requestId);
  impl_->pageTableBindings[entry.stateLane] = {};
  impl_->releaseImages(entry);
  entry.draftContextPlan.reset();
  entry.replayingGeneration |= entry.promptComplete;
  entry.promptComplete = false;
  entry.resident = false;
}

StateAdmission Runtime::resume(const ModelRequest &request) {
  Impl::Request &entry = impl_->request(request.id);
  if (entry.resident) {
    throw std::logic_error("Qwen request is not suspended");
  }
  if (request.prompt.size() < entry.promptTokens) {
    throw std::invalid_argument("recomputed history cannot shorten the prompt");
  }
  Impl::VisionRollback rollback{*impl_, impl_->vision};
  std::vector<Impl::ImageState> images;
  StateAdmission admission = admitIdleLane(impl_->states, [&](uint32_t lane) {
    return impl_->activate(request, lane, images);
  });
  if (admission.granted()) {
    entry.stateLane = *admission.lane;
    entry.resident = true;
    entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
    entry.images = std::move(images);
    entry.restoredTokens = request.restoredTokens;
    impl_->bindPenalties(entry, request.prompt);
    impl_->seedNgramHistory(entry, request.prompt);
  }
  rollback.committed = admission.granted();
  return admission;
}

StateAdmission Runtime::beginAt(const ModelRequest &request, uint32_t stateLane) {
  if (!request.id || stateLane >= kLaneCount || request.prompt.empty()) {
    throw std::invalid_argument("invalid executor request activation");
  }
  if (impl_->requests.contains(request.id)) {
    throw std::logic_error("request is already active");
  }
  Impl::Request entry;
  entry.id = request.id;
  entry.promptTokens = static_cast<uint32_t>(request.prompt.size());
  entry.maxNewTokens = request.maxNewTokens;
  entry.sampling = request.sampling;
  entry.constraint = request.constraint;
  entry.flags = request.flags;
  entry.scoreTokens.assign(request.scoreTokens.begin(),
                           request.scoreTokens.end());
  std::vector<Impl::ImageState> images;
  const StateAdmission admission = impl_->activate(request, stateLane, images);
  if (!admission.granted())
    return admission;
  entry.stateLane = stateLane;
  entry.resident = true;
  entry.images = std::move(images);
  entry.restoredTokens = request.restoredTokens;
  impl_->bindPenalties(entry, request.prompt);
  impl_->seedNgramHistory(entry, request.prompt);
  auto [_, inserted] = impl_->requests.emplace(request.id, std::move(entry));
  if (!inserted) {
    throw std::logic_error("request insertion lost uniqueness");
  }
  return admission;
}

std::unique_ptr<StateRestore> Runtime::beginRestore(
    uint64_t requestId, uint32_t boundary,
    std::shared_ptr<const CompositeState> state, bool restoreDraft,
    std::function<void()> completion) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || !state || boundary >= entry.promptTokens)
    throw std::invalid_argument("invalid state restore");
  return impl_->states.beginRestore(entry.stateLane, *state, restoreDraft,
      std::move(completion), [this, requestId, boundary, restoreDraft] {
        finishRestore(requestId, boundary, restoreDraft);
      });
}

void Runtime::finishRestore(uint64_t requestId, uint32_t restoredPrefixLength,
                            bool restoreDraftState) {
  Impl::Request &entry = impl_->request(requestId);
  // A shorter restore would replay rows of images that were never staged.
  if (restoredPrefixLength < entry.restoredTokens)
    throw std::invalid_argument("restore stops before images its activation left out");
  if (!restoreDraftState)
    ++impl_->counters.draftStateRestoreSkipped;
  const QwenLogicalLengths &lengths =
      impl_->states.metadata(entry.stateLane).lengths;
  if (lengths.targetTokens != restoredPrefixLength ||
      (restoreDraftState &&
       !lengths.hasCompleteDraftWindow(kDraftCacheStride)) ||
      (!restoreDraftState && lengths.draftLength != 0)) {
    throw std::invalid_argument("prefix logical length does not match state");
  }
  // Activation left out the images ModelRequest::restoredTokens covers; a
  // restore further in releases the rest here: warmup and direct callers
  // activate with 0, as does an engine start that let its cache lease go and
  // found a state when it looked up again. Their spans stay because rotary
  // positions after them depend on their grids.
  for (Impl::ImageState &image : entry.images) {
    if (image.span.end() <= restoredPrefixLength) {
      image.rows.reset();
    }
  }
  entry.promptComplete = false;
  if (!entry.replayingGeneration) {
    entry.finalTargetHidden.clear();
    entry.pendingToken.reset();
  }
  entry.draftContextPlan.reset();
}

void Runtime::setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident || plan.replayEnd != entry.promptTokens) {
    throw std::invalid_argument("draft context plan does not match request");
  }
  const uint64_t current =
      impl_->states.metadata(entry.stateLane).lengths.targetTokens;
  if (plan.replayBegin != current)
    throw std::invalid_argument("draft context plan restore boundary is stale");
  entry.draftContextPlan = std::move(plan);
}

std::vector<ModelStepResult>
Runtime::prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return prefillAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                std::function<void()> completion) {
  switch (plan.kind) {
  case WorkKind::Prefill:
    return prefillAsync(plan, items, std::move(completion));
  case WorkKind::Decode:
    return decodeAsync(plan, items, std::move(completion));
  }
  throw std::logic_error("unknown model work kind");
}

bool Runtime::prefillSubmitAheadAvailable() const noexcept {
  return impl_->prefillSubmitAheadAvailable();
}

std::unique_ptr<ModelBatchTicket>
Runtime::prefillAsync(const BatchPlan &plan,
                      std::span<const ModelBatchItem> items,
                      std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Prefill);
  if (plan.decodeStage != DecodeStage::Regular) {
    throw std::invalid_argument("Qwen prefill cannot resume a mask plan");
  }

  std::array<Impl::Request *, kLaneCount> entries{};
  CommandGraph graph;
  // The submit-ahead ring admits a second in-flight prefill; its host-written
  // input tensors come from the other bank so the running chunk's inputs are
  // untouched. Bank 0 serves whenever nothing is in flight.
  const uint32_t inputBank = impl_->claimPrefillInputBank();
  std::array<DispatchDraftCapturePlan, kLaneCount> captures;
  try {
    captures = impl_->encodePackedPrefillGraph(graph, items, entries,
                                               inputBank);
  } catch (...) {
    impl_->releasePrefillInputBank(inputBank);
    throw;
  }
  const bool encodesImages = std::any_of(
      entries.begin(), entries.begin() + items.size(), [](const auto *entry) {
        return std::any_of(entry->images.begin(), entry->images.end(),
                           [](const auto &image) {
                             return image.rows && image.rows->encoding;
                           });
      });
  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  CommandTicket command;
  try {
    command = impl_->backend.submitCommandAsync(graph.dispatches(),
                                              std::move(completion));
  } catch (...) {
    impl_->releasePrefillInputBank(inputBank);
    throw;
  }
  for (uint32_t lane = 0; lane < items.size(); ++lane) {
    Impl::Request &entry = *entries[lane];
    ++entry.prefillUnapplied;
    entry.prefillUnappliedRows += items[lane].tokenCount;
  }
  Impl *impl = impl_.get();
  auto finish = [impl, entries, captures, inputBank,
                 items = std::move(copiedItems)](CommandTiming timing) mutable {
    impl->releasePrefillInputBank(inputBank);
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      const uint64_t chunkEnd = items[lane].logicalPosition + items[lane].tokenCount;
      for (Impl::ImageState &image : entries[lane]->images) {
        if (!image.rows)
          continue;
        Impl::ImageRows &rows = *image.rows;
        if (rows.encoding) {
          rows.encoding = false;
          rows.encoded = true;
          rows.pixels = MetalBuffer{};
        }
        // Its last row is injected: the cache owns the rows from now on, so
        // reclaim can free them while the request decodes.
        if (image.span.end() <= chunkEnd) {
          impl->retain(image.rows);
          image.rows.reset();
        }
      }
    }

    std::vector<ModelStepResult> results;
    results.reserve(items.size());
    for (uint32_t lane = 0; lane < items.size(); ++lane) {
      Impl::Request &entry = *entries[lane];
      const ModelBatchItem &item = items[lane];
      const uint64_t nextLength = item.logicalPosition + item.tokenCount;
      --entry.prefillUnapplied;
      entry.prefillUnappliedRows -= item.tokenCount;
      // The anchor this chunk selects when it completes a generation prompt.
      std::optional<uint32_t> selected;
      if (nextLength == entry.promptTokens && !entry.replayingGeneration &&
          entry.scoreTokens.empty() && entry.constraint == ConstraintMode::None) {
        selected = impl->initialToken(lane);
        if (std::string failure = impl->invalidSelection({&*selected, 1});
            !failure.empty()) {
          // The chunk's state is not committed; the engine ends the
          // request.
          results.push_back({.requestId = entry.id,
                             .consumedPromptTokens = item.tokenCount,
                             .failure = std::move(failure)});
          continue;
        }
      }
      impl->states.swapParity(entry.stateLane);
      QwenLogicalLengths lengths = impl->states.metadata(entry.stateLane).lengths;
      lengths.targetTokens = nextLength;
      for (const DispatchDraftCaptureSpan &capture : captures[lane]) {
        lengths = Impl::advanceDraftContext(lengths, nextLength,
                                            capture.absoluteBegin,
                                            capture.absoluteEnd,
                                            capture.resetDraftState);
        impl->counters.draftContextRowsActive += capture.activeRows;
        impl->counters.draftContextRowsMaterialization +=
            capture.materializationRows;
        if (capture.resetDraftState)
          ++impl->counters.draftStateResets;
      }
      impl->counters.targetPrefillRows += item.tokenCount;
      impl->counters.draftContextRowsAvoided +=
          item.tokenCount - Impl::captureRows(captures[lane]);
      impl->states.updateLengths(entry.stateLane, lengths);
      entry.promptComplete = nextLength == entry.promptTokens;
      ModelStepResult result{entry.id, item.tokenCount, {}, false,
                             DecodeStage::Regular, 0, 0};
      if (entry.promptComplete && !entry.replayingGeneration) {
        entry.pendingToken.reset();
        if (!entry.scoreTokens.empty()) {
          // Score-only: read the raw fp32 logits at the final prompt position
          // (the lane's logits row 0) in requested order.
          const float *row = contents<float>(
              impl->decodeArena->get(lane, DecodeTensor::Logits),
              "score logits");
          result.scoreLogits.reserve(entry.scoreTokens.size());
          for (uint32_t token : entry.scoreTokens) {
            const float logit = row[token];
            if (!std::isfinite(logit)) {
              // A numerical outcome for this request, not a broken invariant:
              // report it as a lane failure so the engine drops this request
              // before cache publication or output and the batch survives.
              result.scoreLogits.clear();
              result.failure = "score logit is not finite";
              break;
            }
            result.scoreLogits.push_back(logit);
          }
          result.finished = true;
        } else if (selected) {
          impl->commitSelected(entry, {&*selected, 1});
          impl->emitTerminalAnchor(entry, result);
        } else {
          // The first token waits for the request's first mask. A replay
          // never gets here: it keeps its stage, and a request that holds
          // its mask asks for none.
          impl->captureFinalHidden(entry, lane);
          entry.decodeStage = DecodeStage::ApplyInitialMask;
          result.nextDecodeStage = DecodeStage::ApplyInitialMask;
        }
      }
      if (entry.promptComplete)
        entry.replayingGeneration = false;
      results.push_back(std::move(result));
    }
    impl->counters.lastPrefillWallSeconds = timing.wallSeconds;
    impl->counters.totalPrefillWallSeconds += timing.wallSeconds;
    impl->counters.lastPrefillGpuSeconds = timing.gpuSeconds;
    impl->counters.totalPrefillGpuSeconds += timing.gpuSeconds;
    return results;
  };
  return std::make_unique<detail::DeferredMetalTicket>(
      std::move(command), std::move(finish), !encodesImages);
}

std::vector<ModelStepResult>
Runtime::decode(const BatchPlan &plan, std::span<const ModelBatchItem> items) {
  return decodeAsync(plan, items, {})->wait();
}

std::unique_ptr<ModelBatchTicket>
Runtime::decodeAsync(const BatchPlan &plan,
                     std::span<const ModelBatchItem> items,
                     std::function<void()> completion) {
  validatePlan(plan, items, WorkKind::Decode);
  const auto encodeStart = std::chrono::steady_clock::now();
  const bool constrained = plan.constrained;
  if (plan.decodeStage != DecodeStage::Regular) {
    if (!constrained) {
      throw std::invalid_argument(
          "only constrained decode uses a specialized decode stage");
    }
    return impl_->submitInitialSelection(items, std::move(completion));
  }

  const uint32_t width = static_cast<uint32_t>(items.size());
  std::vector<Impl::DecodeLaneResult> lanes(width);
  std::array<Impl::Request *, kLaneCount> requests{};
  std::array<uint64_t, kLaneCount> logicalPositions{};
  std::array<uint32_t, kLaneCount> maximumRetained{};
  std::array<uint32_t, kLaneCount> proposals{};
  std::array<uint32_t, kLaneCount> liveRows{};
  for (uint32_t lane = 0; lane < width; ++lane) {
    const ModelBatchItem &item = items[lane];
    Impl::Request &entry = impl_->request(item.requestId);
    if ((entry.constraint == ConstraintMode::TokenMask) != constrained) {
      throw std::invalid_argument(
          "request does not belong to the batch's constraint mode");
    }
    if (entry.decodeStage != DecodeStage::Regular) {
      throw std::logic_error("request decode stage does not match decode plan");
    }
    if (!entry.pendingToken)
      throw std::logic_error("decode request has no current anchor");
    const uint32_t remaining = entry.maxNewTokens - entry.generatedTokens;
    if (!remaining)
      throw std::logic_error("completed request was decoded");
    if (isStopToken(impl_->geometry, *entry.pendingToken) || remaining == 1) {
      throw std::logic_error("terminal anchor was not emitted on selection");
    }

    if (constrained && !entry.maskWords.empty())
      throw std::logic_error("constrained request has stale mask state");
    if (Impl::samplingEnabled(entry)) {
      Impl::stageSamplingCycle(entry);
      impl_->uploadSamplingUniforms(entry, lane);
    }

    // DFlash has one physical graph: anchor + seven proposal rows. A shorter
    // output budget only lowers the token-exact commit count; it never
    // changes the Metal graph shape.
    Impl::DecodeLaneResult &laneResult = lanes[lane];
    laneResult.request = &entry;
    laneResult.currentAnchor = *entry.pendingToken;
    laneResult.maximumRetained = std::min(remaining, kDecodeRows);

    // The lane's proposal budget caps acceptance at one row below its
    // retention limit anyway; a lane near its output cap verifies only the
    // rows it could still keep.
    const uint32_t budget =
        impl_->adaptiveProposals_ && !constrained
            ? std::clamp(entry.proposalBudget, uint32_t{1},
                         uint32_t{kDraftProposalTokens})
            : uint32_t{kDraftProposalTokens};
    proposals[lane] =
        std::min(budget, laneResult.maximumRetained - 1);
    liveRows[lane] = proposals[lane] + 1;
    // A Null draft's n-gram proposals carry no draft probabilities, so a
    // sampled lane cannot verify them: it decodes its anchor row alone.
    if (std::holds_alternative<NullDraft>(impl_->draftModel) &&
        Impl::samplingEnabled(entry)) {
      proposals[lane] = 0;
      liveRows[lane] = 1;
    }

    impl_->prepareDecodeLane(entry, item, lane);
    requests[lane] = &entry;
    logicalPositions[lane] = item.logicalPosition;
    maximumRetained[lane] = laneResult.maximumRetained;
  }

  const std::span<Impl::Request *const> entries(requests.data(), width);
  // A completed ANE predraft stands in for the draft forward when its assumed
  // anchors and positions match every lane — the chain path consumes its
  // proposals; there is no tree table, so a predrafted batch stays a chain.
  // A Null draft (Granite) has nothing to encode even for a constrained
  // batch: the n-gram predraft writes its ProposedTokens in its place.
  const bool nullDraft =
      std::holds_alternative<NullDraft>(impl_->draftModel);
  const bool predrafted =
      (!constrained || nullDraft) &&
      (impl_->applyAnePredraft(entries, items, width) ||
       impl_->applyNgramPredraft(entries, width));
  // The tree decision needs every lane's policy: a tree batch is all greedy,
  // unconstrained, unpenalized DFlash lanes, at most two wide.
  const bool tree =
      !predrafted && !constrained &&
      impl_->treeVerifyBatch(entries, width, constrained);
  const uint32_t ropeRows = width * kDecodeRows;
  CommandGraph commandGraph;
  if (tree) {
    // The draft's tables come from its host-written positions before the
    // draft graph; the target's positions are the tree input pass's output,
    // so its tables are built after that pass runs.
    impl_->addRopeTables(
        commandGraph,
        impl_->decodeArena->packed(DecodeTensor::Positions, 2 * width), 0,
        impl_->decodeArena->packed(DecodeTensor::DraftPositions, width),
        ropeRows,
        impl_->decodeArena->packed(DecodeTensor::RopeCos, 2 * width),
        impl_->decodeArena->packed(DecodeTensor::RopeSin, 2 * width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeSin, width));
  } else {
    impl_->addRopeTables(
        commandGraph,
        impl_->decodeArena->packed(DecodeTensor::Positions, width), ropeRows,
        impl_->decodeArena->packed(DecodeTensor::DraftPositions, width),
        ropeRows, impl_->decodeArena->packed(DecodeTensor::RopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::RopeSin, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeSin, width));
  }
  if (!predrafted) {
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::DraftInputTokens,
                                DecodeTensor::DraftHidden0, width);
    impl_->encodeDraftBatchGraph(commandGraph, entries,
                                 {logicalPositions.data(), width});
  }
  if (constrained) {
    return std::make_unique<Impl::ConstrainedDecodeTicket>(
        *impl_, std::move(lanes), items, commandGraph, std::move(completion));
  }
  if (tree) {
    const bool skipForward = impl_->treeSkip_.find('f') != std::string::npos;
    const bool skipPolicy = impl_->treeSkip_.find('p') != std::string::npos;
    const bool skipCommits = impl_->treeSkip_.find('c') != std::string::npos;
    const bool skipKv = impl_->treeSkip_.find('k') != std::string::npos;
    const bool skipGdn = impl_->treeSkip_.find('g') != std::string::npos;
    const bool skipDraft = impl_->treeSkip_.find('d') != std::string::npos;
    // An ANE medusa result published before this dispatch executes replaces
    // the draft's sibling leaves; a job still running keeps them.
    if (impl_->aneMedusa_ && impl_->medusaSerial_)
      impl_->sampling.addTreeLeafPatch(
          commandGraph,
          impl_->decodeArena->packed(DecodeTensor::TreeTokens, width),
          impl_->aneLeafTokens_, impl_->aneFlag_,
          impl_->decodeArena->packed(DecodeTensor::TreeCounts, width),
          impl_->medusaSerial_.load(), width);
    impl_->encodeBatchVerifyTreeInput(commandGraph, entries, items, width);
    impl_->addRopeTables(
        commandGraph,
        impl_->decodeArena->packed(DecodeTensor::Positions, 2 * width),
        width * RICHENGINE_TREE_VERIFY_NODES,
        impl_->decodeArena->packed(DecodeTensor::DraftPositions, width), 0,
        impl_->decodeArena->packed(DecodeTensor::RopeCos, 2 * width),
        impl_->decodeArena->packed(DecodeTensor::RopeSin, 2 * width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeCos, width),
        impl_->decodeArena->packed(DecodeTensor::DraftRopeSin, width));
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                                DecodeTensor::Hidden0, width, 2);
    if (!skipForward)
      impl_->encodeTargetVerifyBatchForward(commandGraph, entries, items,
                                            true);
    if (!skipPolicy)
      impl_->encodeTargetVerifyBatchPolicy(commandGraph, entries, true);
    impl_->encodeBatchAcceptance(commandGraph, entries,
                                 {maximumRetained.data(), width}, true);
    if (!skipCommits && !skipKv)
      impl_->encodeBatchTreeKvCompact(commandGraph, entries, items);
    if (!skipCommits && !skipGdn)
      impl_->encodeBatchGdnCommit(commandGraph, entries, true);
    if (!skipCommits && !skipDraft)
      impl_->encodeDraftStateCommitBatch(commandGraph, entries, items, true);
  } else {
    impl_->encodeBatchVerifyInput(commandGraph, width);
    impl_->encodeBatchEmbedding(commandGraph, DecodeTensor::InputTokens,
                                DecodeTensor::Hidden0, width);
    impl_->encodeTargetVerifyBatchForward(commandGraph, entries, items, false,
                                          {liveRows.data(), width});
    impl_->encodeTargetVerifyBatchPolicy(commandGraph, entries, false,
                                         {liveRows.data(), width});
    impl_->encodeBatchAcceptance(commandGraph, entries,
                                 {maximumRetained.data(), width}, false,
                                 {proposals.data(), width});
    impl_->encodeBatchGdnCommit(commandGraph, entries);
    impl_->encodeDraftStateCommitBatch(commandGraph, entries, items);
  }

  std::vector<ModelBatchItem> copiedItems(items.begin(), items.end());
  Impl *impl = impl_.get();
  static const bool decodeTiming = envFlag("RICHENGINE_DECODE_TIMING");
  const auto encodeEnd = std::chrono::steady_clock::now();
  auto finish = [impl, lanes = std::move(lanes),
                 items = std::move(copiedItems), tree,
                 encodeStart, encodeEnd](CommandTiming timing) mutable {
    const auto t0 = std::chrono::steady_clock::now();
    auto result = impl->finalizeDecode(lanes, items, timing, tree);
    if (decodeTiming) {
      const auto t1 = std::chrono::steady_clock::now();
      fprintf(stderr,
              "decode-timing encode=%.3fms finalize=%.3fms gpu=%.3fms "
              "wall=%.3fms\n",
              std::chrono::duration<double, std::milli>(encodeEnd - encodeStart).count(),
              std::chrono::duration<double, std::milli>(t1 - t0).count(),
              timing.gpuSeconds * 1e3, timing.wallSeconds * 1e3);
    }
    return result;
  };
  CommandTicket command = impl_->backend.submitCommandAsync(
      commandGraph.dispatches(), std::move(completion));
  return std::make_unique<detail::DeferredMetalTicket>(std::move(command),
                                               std::move(finish));
}

uint32_t Runtime::residentLane(uint64_t requestId) {
  Impl::Request &entry = impl_->request(requestId);
  if (!entry.resident)
    throw std::logic_error("request is not resident");
  return entry.stateLane;
}

std::shared_ptr<const CompositeState> Runtime::snapshot(uint64_t requestId) {
  std::shared_ptr<const CompositeState> state =
      impl_->states.snapshot(residentLane(requestId));
  if (!state)
    return state;
  return impl_->holdStraddledRows(impl_->request(requestId), std::move(state));
}

uint64_t Runtime::snapshotBytes() const noexcept {
  return impl_->states.layout().cachedBytes();
}

bool Runtime::canSnapshotToDisk() const noexcept {
  return impl_->states.canSnapshotToDisk();
}

std::unique_ptr<StateOffload>
Runtime::snapshotToDisk(uint64_t requestId, std::function<void()> completion) {
  return impl_->states.snapshotToDisk(residentLane(requestId), std::move(completion));
}

uint32_t Runtime::statesToActivate() const noexcept {
  return impl_->states.statesToActivate();
}

uint64_t Runtime::reclaimIdleState(bool keepLane, IdleMemory scope) noexcept {
  // One unit per call, so a denied allocation frees only what it needs;
  // rebuildable caches go once the pool has nothing more to give.
  if (const uint64_t buffer = impl_->states.releaseOneIdle(keepLane))
    return buffer;
  return scope == IdleMemory::BuffersThenCaches ? impl_->releaseOneCache() : 0;
}

std::optional<std::string>
Runtime::provideMask(uint64_t requestId, std::span<const uint32_t> words) {
  Impl::Request &entry = impl_->request(requestId);
  const bool acceptsMask =
      waitsForMask(entry.decodeStage) || entry.verifyMaskInFlight;
  // Initial-mask replies can race resource preemption. They belong to the
  // host continuation, not the released device state.
  if (entry.constraint != ConstraintMode::TokenMask || !acceptsMask ||
      !entry.maskWords.empty()) {
    throw std::logic_error("request is not waiting for a token mask");
  }
  const uint32_t maskWords = impl_->geometry.maskWords();
  uint64_t expected = entry.verifyMaskInFlight
                          ? uint64_t{kDecodeRows + 1} * maskWords
                          : maskWords;
  // The native loop matches each response's word count to its request.
  if (words.size() != expected) {
    throw std::logic_error("token mask has the wrong word count");
  }
  const uint32_t rows = static_cast<uint32_t>(words.size() / maskWords);
  for (uint32_t row = 0; row < rows; ++row) {
    auto begin = words.begin() + uint64_t{row} * maskWords;
    if (std::none_of(begin, begin + maskWords,
                     [](uint32_t word) { return word != 0; })) {
      return "token mask row permits no vocabulary token";
    }
  }
  if (entry.verifyMaskInFlight) {
    if (!entry.pendingToken || (words[*entry.pendingToken / 32] &
                                (1U << (*entry.pendingToken % 32))) == 0) {
      return "verify mask is not synchronized to the pending anchor";
    }
  }
  entry.maskWords.assign(words.begin(), words.end());
  return std::nullopt;
}

void Runtime::end(uint64_t requestId) {
  auto found = impl_->requests.find(requestId);
  if (found == impl_->requests.end())
    return;
  impl_->releaseImages(found->second);
  if (found->second.resident) {
    impl_->states.releaseLane(found->second.stateLane, requestId);
    impl_->pageTableBindings[found->second.stateLane] = {};
  }
  impl_->requests.erase(found);
}

namespace {

// A warmup step whose lane failed (a non-finite logit row) fails the warmup
// there, with the lane's reason.
void requireLanesSucceeded(std::span<const ModelStepResult> results) {
  for (const ModelStepResult &result : results)
    if (!result.failure.empty())
      throw std::runtime_error(result.failure);
}

// Warmup runs on the startup runway the engine's KV pool allocated
// (ExecutionLimits::warmupKvPages); it never allocates KV.
void requireRunwayPages(const kv::PageStorage &storage,
                        std::span<const uint32_t> pages) {
  for (uint32_t page : pages) {
    if (page >= ExecutionLimits::warmupKvPages || !storage.isAllocated(page)) {
      throw std::logic_error("warmup KV page " + std::to_string(page) +
                             " is outside the startup runway");
    }
  }
}

// A warmup request's batch item. Each warmup residency keeps one page list,
// so its revision stays 1.
ModelBatchItem warmupItem(uint64_t id, uint64_t position, uint32_t tokens,
                          std::span<const uint32_t> pages) {
  return {.requestId = id,
          .logicalPosition = position,
          .tokenCount = tokens,
          .pageTable = pages,
          .pageTableRevision = 1};
}

} // namespace

void Runtime::prepareWarmupDecode(uint64_t requestId, uint32_t anchor) {
  // Teacher-force a valid input so EOS selected by synthetic prefill cannot
  // prevent the warmup from exercising the real draft/verify/commit graph.
  while (anchor < impl_->geometry.target.vocabularySize &&
         isStopToken(impl_->geometry, anchor))
    ++anchor;
  if (anchor >= impl_->geometry.target.vocabularySize)
    throw std::logic_error("decode warmup has no non-terminal input token");
  auto &entry = impl_->request(requestId);
  entry.pendingToken = anchor;
  entry.generatedTokens = 0;
}

WarmupStepResult Runtime::warmupPrefill(uint32_t rows) {
  if (!rows || rows > kPrefillRows)
    throw std::invalid_argument("invalid prefill warmup row count");
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 100;
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::vector<uint32_t> warmupPrompt(rows, 0);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 16;
  beginColdRequest(request, 0);
  try {
    std::vector<uint32_t> pages((rows + kv::kPageTokens - 1) / kv::kPageTokens);
    std::iota(pages.begin(), pages.end(), 0u);
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, rows}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, rows, pages);
    item.inputTokens = request.prompt;
    const auto phaseStart = AwakeClock::now();
    auto result = prefill(plan, std::span<const ModelBatchItem>(&item, 1));
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    requireLanesSucceeded(result);
    if (result.size() != 1 || result[0].consumedPromptTokens != rows) {
      throw std::runtime_error("prefill warmup result mismatch");
    }
    lanes.push_back({std::move(result[0]), impl_->request(id).pendingToken,
                     impl_->states.metadata(0).lengths.targetTokens});
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real " + std::to_string(rows) +
              "-row packed KV target+draft prefill [M32]",
          wallSeconds, std::move(lanes)};
}

WarmupStepResult Runtime::warmupDecodeBatch(uint32_t width) {
  if (!width || width > kLaneCount) {
    throw std::invalid_argument("invalid decode warmup width");
  }
  constexpr uint64_t firstId = std::numeric_limits<uint64_t>::max() - 110;
  // Plan order is deliberately unrelated to state-lane order. DecodeArena
  // lanes follow the explicit BatchPlan, while recurrent and KV state stay
  // addressed by each request's state lane; batching must never assume lanes
  // 0..3.
  constexpr std::array<uint32_t, kLaneCount> stateLaneOrder{2, 0, 3, 1};
  double wallSeconds = 0.0;
  std::vector<WarmupLaneResult> lanes;
  std::array<std::vector<uint32_t>, kLaneCount> pages;
  try {
    for (uint32_t lane = 0; lane < width; ++lane) {
      std::vector<uint32_t> warmupPrompt{lane};
      ModelRequest request;
      request.id = firstId + lane;
      request.prompt = warmupPrompt;
      request.maxNewTokens = 16;
      beginColdRequest(request, stateLaneOrder[lane]);
      pages[lane] = {5 + lane};
      requireRunwayPages(impl_->kvPages, pages[lane]);
      BatchPlan prefillPlan{.kind = WorkKind::Prefill,
                            .items = {{request.id, 1}},
                            .decodeStage = DecodeStage::Regular};
      ModelBatchItem item = warmupItem(request.id, 0, 1, pages[lane]);
      item.inputTokens = request.prompt;
      requireLanesSucceeded(
          prefill(prefillPlan, std::span<const ModelBatchItem>(&item, 1)));
      prepareWarmupDecode(request.id, warmupPrompt.back());
    }
    BatchPlan plan;
    plan.kind = WorkKind::Decode;
    std::vector<ModelBatchItem> items;
    for (uint32_t lane = 0; lane < width; ++lane) {
      plan.items.push_back({firstId + lane, 0});
      items.push_back(warmupItem(firstId + lane, 1, 0, pages[lane]));
    }
    const auto phaseStart = AwakeClock::now();
    auto decoded = decode(plan, items);
    wallSeconds = std::chrono::duration<double>(AwakeClock::now() - phaseStart).count();
    requireLanesSucceeded(decoded);
    bool committedEveryLane = decoded.size() == width;
    for (uint32_t lane = 0; committedEveryLane && lane < width; ++lane) {
      const auto &lengths = impl_->states.metadata(stateLaneOrder[lane]).lengths;
      committedEveryLane = !decoded[lane].outputTokens.empty() &&
                           lengths.targetTokens > 1 &&
                           lengths.targetTokens ==
                               1 + decoded[lane].outputTokens.size() -
                                   decoded[lane].outputTokensWithoutKv &&
                           lengths.hasCompleteDraftWindow(kDraftCacheStride);
    }
    if (!committedEveryLane || impl_->counters.lastDecodeWidth != width) {
      throw std::runtime_error(
          "decode warmup B" + std::to_string(width) +
          " mismatch [committed=" + std::to_string(committedEveryLane) +
          ",width=" + std::to_string(impl_->counters.lastDecodeWidth) + "]");
    }
    for (uint32_t lane = 0; lane < width; ++lane) {
      lanes.push_back({std::move(decoded[lane]),
                       impl_->request(firstId + lane).pendingToken,
                       impl_->states.metadata(stateLaneOrder[lane]).lengths.targetTokens});
      end(firstId + lane);
    }
  } catch (...) {
    for (uint32_t lane = 0; lane < width; ++lane)
      end(firstId + lane);
    throw;
  }
  return {"real B" + std::to_string(width) + " draft/verify/commit decode",
          wallSeconds, std::move(lanes)};
}

WarmupStepResult Runtime::warmupCompositeStateRestore() {
  constexpr uint64_t id = std::numeric_limits<uint64_t>::max() - 121;
  constexpr uint32_t prefixTokens = 2 * kv::kPageTokens;
  constexpr uint32_t suffixTokens = kDecodeRows;
  constexpr uint32_t promptTokens = prefixTokens + suffixTokens;
  std::vector<uint32_t> warmupPrompt(promptTokens, 2);
  ModelRequest request;
  request.id = id;
  request.prompt = warmupPrompt;
  request.maxNewTokens = 8;
  std::shared_ptr<const CompositeState> cachedState;
  double wallSeconds = 0.0;
  beginColdRequest(request, 0);
  try {
    // Deliberately non-contiguous physical ids exercise page-table lookup.
    const std::vector<uint32_t> pages{12, 10, 11};
    requireRunwayPages(impl_->kvPages, pages);
    BatchPlan plan{.kind = WorkKind::Prefill,
                   .items = {{id, prefixTokens}},
                   .decodeStage = DecodeStage::Regular};
    ModelBatchItem item = warmupItem(id, 0, prefixTokens, pages);
    item.inputTokens =
        std::span<const uint32_t>(request.prompt).first(prefixTokens);
    static_cast<void>(prefill(plan, std::span<const ModelBatchItem>(&item, 1)));
    wallSeconds = impl_->counters.lastPrefillWallSeconds;
    cachedState = snapshot(id);
    if (!cachedState)
      throw metal::MetalAllocationError("prefix warmup state allocation failed");
    end(id);
    beginColdRequest(request, 1);
    if (beginRestore(id, prefixTokens, cachedState, true, {}))
      throw std::logic_error("a resident state restore returned a read");
    setDraftContextPlan(id, planDraftContext(prefixTokens, promptTokens, {}));
    const auto &restored = impl_->states.metadata(1).lengths;
    if (restored.targetTokens != prefixTokens ||
        !restored.hasCompleteDraftWindow(kDraftCacheStride)) {
      throw std::runtime_error("prefix restore length mismatch");
    }

    // Continue from committed KV history. This M8 command teacher-forces a
    // new chunk, then the real speculative cycle overwrites its speculative
    // page suffix and advances only the accepted commit length.
    BatchPlan suffixPlan{.kind = WorkKind::Prefill,
                         .items = {{id, suffixTokens}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem suffix = warmupItem(id, prefixTokens, suffixTokens, pages);
    suffix.inputTokens = std::span<const uint32_t>(request.prompt)
                             .subspan(prefixTokens, suffixTokens);
    requireLanesSucceeded(
        prefill(suffixPlan, std::span<const ModelBatchItem>(&suffix, 1)));
    prepareWarmupDecode(id, warmupPrompt.back());
    const double continuationWallSeconds =
        impl_->counters.lastPrefillWallSeconds;
    wallSeconds += continuationWallSeconds;
    BatchPlan decodePlan{.kind = WorkKind::Decode,
                         .items = {{id, 0}},
                         .decodeStage = DecodeStage::Regular};
    ModelBatchItem decodeItem = warmupItem(id, promptTokens, 0, pages);
    auto decoded =
        decode(decodePlan, std::span<const ModelBatchItem>(&decodeItem, 1));
    requireLanesSucceeded(decoded);
    const double historicalDecodeWallSeconds =
        impl_->counters.lastDecodeWallSeconds;
    wallSeconds += historicalDecodeWallSeconds;
    const auto &continued = impl_->states.metadata(1).lengths;
    if (decoded.size() != 1 || decoded[0].outputTokens.empty() ||
        !continued.hasCompleteDraftWindow(kDraftCacheStride) ||
        continued.targetTokens <= promptTokens ||
        continued.targetTokens !=
            promptTokens + decoded[0].outputTokens.size() -
                decoded[0].outputTokensWithoutKv) {
      throw std::runtime_error(
          "restored historical prefix did not continue exactly");
    }
    end(id);
  } catch (...) {
    end(id);
    throw;
  }
  return {"real paged-KV state restore, arbitrary page table, lane move, "
          "bounded restore continuation, and decode",
          wallSeconds, {}};
}

ModelMemoryActual Runtime::actualRuntimeMemory() const {
  return {impl_->states.actualAllocatedBytes(), impl_->prefillArena->bytes(),
          impl_->decodeArena->bytes(), impl_->states.stagingBytes()};
}

ModelTelemetry Runtime::telemetry() const noexcept {
  ModelTelemetry result = impl_->counters;
  result.stateAllocatedBytes = impl_->states.actualAllocatedBytes();
  result.idleGdnCells = impl_->states.idleCells();
  result.idleDraftRings = impl_->states.idleRings();
  result.visionArenaBytes = impl_->vision ? impl_->vision->arenaBytes() : 0;
  result.embeddingCacheBytes = impl_->embeddingCacheBytes;
  result.stateHeldImageBytes = impl_->heldRowsBytes(false);
  for (const auto &[_, held] : impl_->imageRows) {
    if (const std::shared_ptr<const Impl::ImageRows> rows = held.lock())
      result.imageRowsBytes += rows->pixels.sizeBytes() + rows->embeddings.sizeBytes();
  }
  return result;
}

ModelMemoryPlan plannedRuntimeMemory(const DeviceCapabilities &device,
                                     const ModelPackage &package,
                                     const ops::ExecutionPlans &operators,
                                     kv::Format format) {
  requireCompatibleModelPackage(package);
  if (device.appleGpuFamily < DeviceCapabilities::kMinimumAppleGpuFamily) {
    throw std::invalid_argument("model runtime requires Apple tensor BF16");
  }
  const RuntimeGeometry geometry = RuntimeGeometry::from(package, format);
  return {package.stateLayout().laneBytes(),
          plannedPrefillBytes(geometry, operators),
          plannedDecodeBytes(geometry, operators)};
}

std::unique_ptr<RuntimeModel> createRuntime(RuntimeContext context) {
  return std::make_unique<Runtime>(std::move(context));
}

} // namespace richengine::model
