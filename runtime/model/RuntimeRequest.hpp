#pragma once

// The runtime's per-request record and the admission helpers that act on
// it: the lane a start is given (laneAdmission, admitIdleLane) and the
// batch-plan and shared-buffer checks admission and encode share. Used by
// Runtime::Impl (RuntimeImpl.hpp), which aliases the types under their
// nested names.

#include "model/NgramIndex.hpp"
#include "model/Runtime.hpp"
#include "model/CompositeStateStorage.hpp"
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

// One audio clip's mel and its encoded rows, shared by every placement of
// the same content that still has rows to inject. `mel` holds the float32
// [128 x melFrames] payload; `embeddings` one bf16 outputHiddenSize row
// per placeholder token. No embedding cache: identical clips across
// requests re-encode once their last placement releases the rows.
struct RuntimeAudioRows final {
  RuntimeImageKey key;
  MetalBuffer mel;
  MetalBuffer embeddings;
  bool encoding = false;
  bool encoded = false;
};
// A placement keeps its rows until its last row is injected; audio rows
// take ordinary sequential rotary positions, so nothing else depends on
// the span once its tokens are prefilled.
struct RuntimeAudioState final {
  AudioSpan span;
  std::shared_ptr<RuntimeAudioRows> rows;
};

// A request's compiled byte-DFA constraint (CONSTRAINT_DFA frame): the
// row-major states x 256 i16 transition table (-1 = dead) and its accepting
// bitset, kept host-side for resend after a weight release and uploaded to
// request-owned Metal buffers for the dfa walk kernels.
struct RequestDfa final {
  std::vector<int16_t> table;
  // The EOS-admitting bitset (the grammar's accepting ∪ free-text states)
  // and the marker mask — 32 bytes per state, bit b marking edge (s, b)
  // special-only — from the CONSTRAINT_DFA frame.
  std::vector<uint8_t> accepting;
  std::vector<uint8_t> marker;
  MetalBuffer tableBuffer;
  MetalBuffer acceptBuffer;
  MetalBuffer markerBuffer;
  // Per state, the minimum bytes of an acceptance-requiring unmarked-edge
  // walk (32767: none a regular token can take) — the token-cost bound the
  // decode budget rule gates a mask row with. A frame without the cost
  // section keeps an all-zero buffer, whose zeros read as "terminates
  // here", which gates nothing.
  std::vector<int16_t> costs;
  MetalBuffer costBuffer;
  // The grammar's EOS ids (the contract's, not the engine's stop tokens);
  // at an EOS-admitting state a member ends the walk at the stop state.
  std::array<uint32_t, 8> eosIds{};
  uint32_t eosCount = 0;
  uint32_t states = 0;
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
  // DiffusionGemma: how many canvases the request has committed; each
  // canvas's RNG seed is the request seed mixed by this counter.
  uint32_t canvasIndex = 0;
  std::vector<uint32_t> maskWords;
  // Set only while the current scheduler-owned ticket overlaps grammar-mask
  // computation with target verification. This is model runtime state, not a
  // scheduler decode stage.
  bool verifyMaskInFlight = false;
  // The mask rows the in-flight verify request asked for — the simulated
  // tokens plus the pre-anchor row, the lane's live verify rows + 1 — so
  // provideMask validates the response against the count actually sent.
  // Zero means no row count was recorded (a wait opened before the record
  // or an abandoned one), which keeps the legacy full-table expectation.
  uint32_t verifyMaskRows = 0;
  // The request's GPU-DFA constraint: while set, decode computes the lane's
  // masks with the dfa kernels and no host mask request goes out.
  std::optional<RequestDfa> dfa;
  // Host mirror of the lane's persistent automaton state — the state the
  // pending anchor walks from. Authoritative while dfaStateDirty (set at
  // provideDfa and across a suspension); otherwise the state lane's
  // DecodeTensor::DfaState slot is authoritative and the mirror is stale.
  uint32_t dfaState = 0;
  bool dfaStateDirty = true;
  // The DFA lane's verifyMaskInFlight: set while a constrained batch's GPU
  // work may still write its DfaState slot, so suspend() waits it out.
  bool gpuMaskInFlight = false;
  // The host-pushed grammar-forced span: its first token anchors it at
  // pendingToken, the rest replace draft proposals while they match.
  std::vector<uint32_t> forcedTokens;
  uint64_t rngCounter = 0;
  // A bootstrap warmup request: the diffusion decode truncates its canvas
  // schedule — warmup needs one compiled pass over every kernel, not 48.
  bool warmup = false;
  DecodeStage decodeStage = DecodeStage::Regular;
  std::optional<DraftContextPlan> draftContextPlan;
  std::vector<RuntimeImageState> images;
  std::vector<RuntimeAudioState> audios;
  // What its activation took from a cached state: the images that end
  // there were left out (ModelRequest::restoredTokens).
  uint32_t restoredTokens = 0;
  // Learning-free predraft state (RICHENGINE_NGRAM_PREDRAFT): the lane's
  // token stream and each 3-gram key's last four starts, which the lookup
  // walks. NgramIndex.hpp owns the logic.
  ngram::Table ngram;
  // Acceptance gating: EWMA of accepted tokens per n-gram round, set in
  // finalizeDecode for steps where this lane's proposals were used.
  uint32_t ngramRounds = 0;
  double ngramAcceptedAvg = 0;
  // Next generatedTokens count at which a gated-off lane may probe again.
  uint32_t ngramProbeAt = 0;
  bool ngramInFlight = false;
  // Cross-request corpus eligibility (RICHENGINE_NGRAM_CORPUS): armed at
  // admission for every lane in mode 1, by the hint flag or a tail-probe
  // hit in mode 2, and later by the starvation probe when the lane's own
  // 3-gram finds nothing. corpusProbeAt paces those starvation probes.
  bool corpusArmed = false;
  uint32_t corpusProbeAt = 0;
  // Weak session key (opening-prompt hash) and, on corpus docs this lane
  // committed, self-exclusion: origin skips the lane's own pushed stream,
  // session prefers same-conversation docs on ties.
  uint64_t corpusSession = 0;
  // Adaptive proposal length (opt-in, RICHENGINE_ADAPTIVE_PROPOSALS): EWMA of
  // accepted draft tokens per chain verify step and the lane's current
  // budget — the acceptance cap and live verify rows that step.
  double proposalAcceptedAvg = kDraftProposalTokens;
  uint32_t proposalBudget = kDraftProposalTokens;
  // Adaptive draft bypass (RICHENGINE_DRAFT_BYPASS): while the acceptance
  // EWMA sits below the draft's break-even cost the lane skips the draft
  // pass entirely; this step's flag skips the EWMA update (a fake zero
  // would poison it), and the probe counter schedules a drafted step every
  // kDraftProbeTokens so a recovered lane re-enables.
  uint32_t draftProbeAt = 0;
  bool draftBypassed = false;
  // Prefill chunks submitted but not yet consumed (submit-ahead ring): the
  // count flips the parity binding an encode selects, and the rows relax the
  // packed-prefill length check by the unapplied advance.
  uint32_t prefillUnapplied = 0;
  uint32_t prefillUnappliedRows = 0;
  // LFM3-Omni speech output (docs in dev/omni-reference/NOTES.md section 6).
  // Every emitted audio position is one LM slot whose input embedding is
  // the last frame's feedback row instead of a token embedding: while the
  // pending anchor is the audio placeholder (125032), the step's Hidden0
  // anchor row is patched with `feedback` (bf16 row, staged per lane).
  // `mode`: 0 text, 1 audio, 2 interleaved. Audio slots run a single
  // host-controlled step — the lane's proposals are anchor-only, its lm_head
  // output is discarded, the sampled codec frame drives the emission.
  struct SpeechState final {
    uint32_t mode = 0;           // 0 TEXT, 1 AUDIO, 2 INTERLEAVED
    uint32_t slotsLeft = 0;      // interleaved: slots until the block flips
    bool speaking = false;       // interleaved: current block is audio
    bool textDone = false;       // interleaved: text half ended (text_end)
    // bf16 feedback embedding of the last sampled frame: the next audio or
    // post-audio text slot's anchor input while pendingToken == 125032.
    std::vector<uint16_t> feedback;
    MetalBuffer feedbackBuffer;  // its Metal staging slot for the row patch
    uint32_t frames = 0;         // frames emitted this request (PCM offset)
  };
  SpeechState speech;
  // The request's speech decoder (depthformer + codec + ISTFT state), lazily
  // bound while a speech-capable model serves the turn. Type-erased: the
  // ops type is complete only where the runtime compiles it.
  std::shared_ptr<void> speechGen;
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
inline StateAdmission admitIdleLane(const CompositeStateStorage &states,
                             Activate activate) {
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    if (!states.metadata(lane).assigned())
      return activate(lane);
  }
  return {{}, StateFailure::ConcurrencyLimit};
}

} // namespace

} // namespace richengine::model
