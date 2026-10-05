#pragma once

// The runtime's per-request record and the admission helpers that act on
// it: the lane a start is given (laneAdmission, admitIdleLane) and the
// batch-plan and shared-buffer checks admission and encode share. Used by
// Runtime::Impl (RuntimeImpl.hpp), which aliases the types under their
// nested names.

#include "model/Runtime.hpp"
#include "model/QwenState.hpp"
#include "model/RuntimeArenas.hpp"

#include "metal/MetalBackend.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace richengine::model {

using metal::BufferStorage;
using metal::MetalBuffer;

struct RuntimeImageKey final {
  uint64_t digestLo = 0;
  uint64_t digestHi = 0;
  uint32_t gridHeight = 0;
  uint32_t gridWidth = 0;

  bool operator==(const RuntimeImageKey &) const = default;
};
struct RuntimeImageKeyHash final {
  // The digest is already a content hash.
  size_t operator()(const RuntimeImageKey &key) const noexcept {
    return static_cast<size_t>(key.digestLo ^ key.digestHi);
  }
};
// One image's encoded rows, shared by every placement that still has rows
// to inject (repeated placements and concurrent requests alike) and by
// the embedding cache. Whichever placement's chunk reaches the image
// first encodes it and the others inject after it; the pixels go once
// the encode has completed.
struct RuntimeImageRows final {
  RuntimeImageKey key;
  MetalBuffer pixels;
  MetalBuffer embeddings;
  bool encoding = false;
  bool encoded = false;
  // Its entry in the embedding cache while the cache holds it.
  std::optional<std::list<std::shared_ptr<RuntimeImageRows>>::iterator>
      cached;
};
// A placement keeps its rows until its last row is injected; its span
// stays, because rotary positions after it depend on its grid.
struct RuntimeImageState final {
  ImageSpan span;
  std::shared_ptr<RuntimeImageRows> rows;
};

struct RuntimeRequest final {
  uint64_t id = 0;
  uint32_t stateLane = 0;
  bool resident = false;
  bool promptComplete = false;
  // Rebuild state from already-emitted tokens without sampling an initial
  // anchor, consuming RNG, or replaying output to the caller.
  bool replayingGeneration = false;
  uint32_t promptTokens = 0;
  uint32_t maxNewTokens = 0;
  uint32_t generatedTokens = 0;
  SamplingParameters sampling;
  ConstraintMode constraint = ConstraintMode::None;
  // RequestFlag bits.
  uint32_t flags = 0;
  std::optional<uint32_t> pendingToken;
  // A constrained request's final prompt row, held from its prompt's end
  // until its first token is selected under its first mask, suspensions
  // included. Composite cache state never stores it; every cache hit
  // replays one input token and regenerates this value.
  std::vector<uint16_t> finalTargetHidden;
  std::array<float, kSamplingUniformCount> cycleUniforms{};
  // Nonempty selects score-only mode: the final prefill chunk computes raw
  // logits at these token ids instead of selecting an anchor.
  std::vector<uint32_t> scoreTokens;
  std::vector<uint32_t> maskWords;
  // Set only while the current scheduler-owned ticket overlaps grammar-mask
  // computation with target verification. This is model runtime state, not a
  // scheduler decode stage.
  bool verifyMaskInFlight = false;
  uint64_t rngCounter = 0;
  DecodeStage decodeStage = DecodeStage::Regular;
  std::optional<DraftContextPlan> draftContextPlan;
  std::vector<RuntimeImageState> images;
  // What its activation took from a cached state: the images that end
  // there were left out (ModelRequest::restoredTokens).
  uint32_t restoredTokens = 0;
  // Learning-free predraft state (RICHENGINE_NGRAM_PREDRAFT): the lane's
  // token stream and each 3-gram's last two starts.
  std::vector<uint32_t> ngramHistory;
  // Each 3-gram key keeps its last four starts; the lookup walks them all.
  std::unordered_map<uint64_t, std::array<uint32_t, 4>> ngramIndex;
  // Acceptance gating: EWMA of accepted tokens per n-gram round, set in
  // finalizeDecode for steps where this lane's proposals were used.
  uint32_t ngramRounds = 0;
  double ngramAcceptedAvg = 0;
  // Next generatedTokens count at which a gated-off lane may probe again.
  uint32_t ngramProbeAt = 0;
  bool ngramInFlight = false;
  // Adaptive proposal length (opt-in, RICHENGINE_ADAPTIVE_PROPOSALS): EWMA of
  // accepted draft tokens per chain verify step and the lane's current
  // budget — the acceptance cap and live verify rows that step.
  double proposalAcceptedAvg = kDraftProposalTokens;
  uint32_t proposalBudget = kDraftProposalTokens;
  // Prefill chunks submitted but not yet consumed (submit-ahead ring): the
  // count flips the parity binding an encode selects, and the rows relax the
  // packed-prefill length check by the unapplied advance.
  uint32_t prefillUnapplied = 0;
  uint32_t prefillUnappliedRows = 0;
};

namespace {

inline bool isStopToken(const RuntimeGeometry &geometry, uint32_t token) noexcept {
  return token == geometry.target.stopTokens[0] ||
         token == geometry.target.stopTokens[1];
}

inline void requireShared(const MetalBuffer &buffer, std::string_view label) {
  if (!buffer || buffer.storage() != BufferStorage::Shared ||
      !buffer.contents()) {
    throw std::logic_error(std::string(label) + " is not CPU-visible");
  }
}

template <class T>
inline T *contents(const MetalBuffer &buffer, std::string_view label) {
  requireShared(buffer, label);
  return static_cast<T *>(buffer.contents());
}

[[maybe_unused]] inline void validatePlan(const BatchPlan &plan, std::span<const ModelBatchItem> items,
                  WorkKind expected) {
  if (plan.kind != expected || plan.empty() || plan.width() > kLaneCount ||
      items.size() != plan.items.size()) {
    throw std::invalid_argument("model runtime received an invalid batch plan");
  }
  for (size_t index = 0; index < items.size(); ++index) {
    if (items[index].requestId != plan.items[index].requestId ||
        (expected == WorkKind::Prefill &&
         (!plan.items[index].tokenCount ||
          plan.items[index].tokenCount != items[index].tokenCount ||
          items[index].inputTokens.size() != items[index].tokenCount)) ||
        (expected == WorkKind::Decode &&
         (plan.items[index].tokenCount || items[index].tokenCount ||
          !items[index].inputTokens.empty()))) {
      throw std::invalid_argument("batch items do not match explicit plan");
    }
  }
}

// The lane a start's admission gave it, or the cause of its refusal.
inline StateAdmission laneAdmission(uint32_t lane, const metal::AllocationResult &result) {
  if (result)
    return {lane, StateFailure::None};
  return {{}, StateFailure::MemoryPressure, result.failure};
}

// Any unassigned lane works: its buffers come from the storage's pool, and
// the governor is asked only for what the pool lacks.
template <class Activate>
inline StateAdmission admitIdleLane(const QwenStateStorage &states,
                             Activate activate) {
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    if (!states.metadata(lane).assigned())
      return activate(lane);
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

} // namespace

} // namespace richengine::model
