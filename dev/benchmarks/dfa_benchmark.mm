// Constrained-decode (DFA) benchmark: drives real constrained generation
// through engine::Engine the way server/backend.py does — TOKEN_MASK
// requests, MASK_REQUEST round trips answered by an in-process byte-DFA
// mirror of server/constraints.py's DfaConstraint, optional host-pushed
// FORCED_TOKENS spans, and CONSTRAINT_DFA + VOCABULARY installs for the
// GPU-DFA paths RICHENGINE_DFA_MODE=forced|full runs.
//
// Scenarios (--scenario, comma-separated; default all):
//   unconstrained  baseline greedy decode, no constraint
//   mask           TokenMask + host DFA masks (production's host path)
//   mask-forced    TokenMask + host masks + host-pushed forced-token spans
//   dfa            TokenMask + CONSTRAINT_DFA installed; what the DFA then
//                  drives depends on RICHENGINE_DFA_MODE at process start:
//                  off = host path, forced = host masks + on-device spans,
//                  full = GPU masks (RICHENGINE_DFA_MERGE and
//                  RICHENGINE_DFA_SPAN_SKIP_DRAFT apply)
//
// The pack (--pack) comes from dev/benchmarks/dfa_pack.py: the real packed
// vocabulary, the production-compiled DFA for a tool-call-style JSON
// schema, and a tokenized prompt.

#include "engine/Engine.hpp"
#include "engine/Bootstrap.hpp"
#include "engine/wire/Json.hpp"
#include "engine/RuntimeResources.hpp"
#include "model/Runtime.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef RICHENGINE_BUILD_ID
#error "dfa_benchmark requires the production RICHENGINE_BUILD_ID"
#endif

using namespace richengine;
using namespace richengine::engine;

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point value) {
  return std::chrono::duration<double, std::milli>(value.time_since_epoch())
      .count();
}

// ---------------------------------------------------------------- pack --

struct Pack final {
  uint32_t vocabulary = 0;
  std::vector<uint8_t> tokenBytes;
  std::vector<int32_t> tokenOffsets;  // vocabulary + 1 entries
  std::vector<uint8_t> specialBits;   // (vocabulary + 7) / 8 bytes
  uint32_t dfaStates = 0;
  std::vector<int16_t> dfaTable;      // states * 256, row-major, -1 dead
  std::vector<uint8_t> accepting;     // EOS-admitting bitset (accepting|eos_ok)
  std::vector<uint8_t> marker;        // states * 32 bytes
  std::vector<uint32_t> eosIds;
  std::vector<uint32_t> prompt;
  // The budget rule's per-state termination bounds (n_states i16 entries),
  // the frame's optional cost section.
  std::vector<int16_t> costs;
};

class PackReader final {
public:
  explicit PackReader(std::vector<uint8_t> data) : data_(std::move(data)) {}

  template <typename T>
  void read(T &value, const char *field) {
    if (cursor_ + sizeof(T) > data_.size())
      fail(field);
    std::memcpy(&value, data_.data() + cursor_, sizeof(T));
    cursor_ += sizeof(T);
  }

  template <typename T>
  void readArray(std::vector<T> &value, size_t count, const char *field) {
    if (count > (data_.size() - cursor_) / sizeof(T))
      fail(field);
    value.resize(count);
    std::memcpy(value.data(), data_.data() + cursor_, count * sizeof(T));
    cursor_ += count * sizeof(T);
  }

  void finish() const {
    if (cursor_ != data_.size())
      throw std::runtime_error("dfa pack has trailing bytes");
  }

private:
  [[noreturn]] void fail(const char *field) const {
    throw std::runtime_error(std::string("dfa pack truncated at ") + field);
  }

  std::vector<uint8_t> data_;
  size_t cursor_ = 0;
};

Pack loadPack(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open dfa pack " + path.string());
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
  PackReader reader(std::move(data));
  std::array<char, 8> magic{};
  for (char &byte : magic)
    reader.read(byte, "magic");
  if (std::string_view(magic.data(), magic.size()) != "DFAPACK\x02")
    throw std::runtime_error("dfa pack magic mismatch");
  Pack pack;
  uint32_t tokenByteCount = 0;
  uint32_t acceptingBytes = 0;
  uint32_t markerBytes = 0;
  uint32_t eosCount = 0;
  uint32_t specialBytes = 0;
  uint32_t promptTokens = 0;
  reader.read(pack.vocabulary, "vocabulary");
  reader.read(tokenByteCount, "token byte count");
  reader.read(pack.dfaStates, "dfa states");
  reader.read(acceptingBytes, "accepting bytes");
  reader.read(markerBytes, "marker bytes");
  reader.read(eosCount, "eos count");
  reader.read(specialBytes, "special bytes");
  reader.read(promptTokens, "prompt tokens");
  if (!pack.vocabulary || !pack.dfaStates || eosCount > 8)
    throw std::runtime_error("dfa pack header is invalid");
  if (acceptingBytes != (pack.dfaStates + 7) / 8 ||
      markerBytes != pack.dfaStates * 32 ||
      specialBytes != (pack.vocabulary + 7) / 8)
    throw std::runtime_error("dfa pack section sizes do not match the header");
  reader.readArray(pack.tokenBytes, tokenByteCount, "token bytes");
  reader.readArray(pack.tokenOffsets, pack.vocabulary + 1, "token offsets");
  reader.readArray(pack.specialBits, specialBytes, "special bits");
  reader.readArray(pack.dfaTable, size_t{pack.dfaStates} * 256, "dfa table");
  reader.readArray(pack.accepting, acceptingBytes, "accepting");
  reader.readArray(pack.marker, markerBytes, "marker");
  reader.readArray(pack.eosIds, eosCount, "eos ids");
  reader.readArray(pack.prompt, promptTokens, "prompt");
  // The budget rule's per-state termination bounds — n_states i16 entries,
  // exactly the frame's optional cost section.
  reader.readArray(pack.costs, size_t{pack.dfaStates}, "cost table");
  reader.finish();
  if (pack.tokenOffsets.front() != 0 ||
      uint64_t(pack.tokenOffsets.back()) != pack.tokenBytes.size() ||
      !std::is_sorted(pack.tokenOffsets.begin(), pack.tokenOffsets.end()))
    throw std::runtime_error("dfa pack token offsets are not monotone");
  for (uint32_t token : pack.prompt)
    if (token >= pack.vocabulary)
      throw std::runtime_error("dfa pack prompt has out-of-vocabulary ids");
  return pack;
}

// ------------------------------------------------- host-side DFA mirror --

// server/constraints.py DfaConstraint: the consuming state after the
// committed tokens, Dfa.advance_token/_walk semantics for mask rows, and
// Dfa.forced_tokens_after's forced-byte chain + canonical cover for the
// host-pushed span (greedy longest-prefix like decode_dfa_forced_tokens).
class DfaMirror final {
public:
  explicit DfaMirror(const Pack &pack)
      : pack_(pack),
        maskWords_(model::maskWordsPerToken(pack.vocabulary)) {
    for (uint32_t id : pack.eosIds)
      eos_.insert(id);
  }

  [[nodiscard]] uint32_t maskWords() const noexcept { return maskWords_; }

  // The state a token's byte walk leaves from `state`, or -1: eos ids are
  // admitted only at EOS-admitting states and land on the stop state; a
  // regular token dies on a marked edge, a special token dies taking none.
  [[nodiscard]] int64_t advance(int64_t state, uint32_t token) const {
    const bool live = state >= 0 && uint64_t(state) < pack_.dfaStates;
    if (eos_.count(token))
      return live && eosAdmits(uint64_t(state)) ? int64_t(pack_.dfaStates)
                                              : int64_t(-1);
    if (!live || token >= pack_.vocabulary ||
        pack_.tokenOffsets[token] == pack_.tokenOffsets[token + 1])
      return -1;
    return walk(uint64_t(state), token);
  }

  void commit(std::span<const uint32_t> tokens) {
    for (uint32_t token : tokens) {
      if (stopped_)
        continue;
      int64_t next = advance(state_, token);
      if (next < 0) {
        // A special token the DFA cannot consume at an EOS-admitting state
        // maps to the stop state, as DfaConstraint._foreign_stop.
        if (special(token) && state_ < pack_.dfaStates &&
            eosAdmits(state_)) {
          stopped_ = true;
          state_ = pack_.dfaStates;
          continue;
        }
        ++invalidCommits_;
        stopped_ = true;
        state_ = pack_.dfaStates;
        continue;
      }
      state_ = uint64_t(next);
      if (next == int64_t(pack_.dfaStates))
        stopped_ = true;
    }
  }

  // The token bitmask one state admits: eos ids when EOS-admitting, every
  // non-eos token whose walk stays live. Rows are memoized per state.
  [[nodiscard]] const std::vector<uint32_t> &maskRow(uint64_t state) {
    auto found = rowCache_.find(state);
    if (found != rowCache_.end())
      return found->second;
    std::vector<uint32_t> words(maskWords_, 0);
    if (state < pack_.dfaStates) {
      if (eosAdmits(state))
        for (uint32_t id : pack_.eosIds)
          setBit(words, id);
      for (uint32_t token = 0; token < pack_.vocabulary; ++token) {
        if (eos_.count(token))
          continue;
        if (walk(state, token) >= 0)
          setBit(words, token);
      }
    }
    return rowCache_.emplace(state, std::move(words)).first->second;
  }

  // The mask rows a MASK_REQUEST answers: row 0 at the consuming state,
  // one row after each simulated token; a token the walk rejects stops the
  // simulation and its row repeats, as DfaConstraint.masks.
  [[nodiscard]] std::vector<uint32_t>
  masks(std::span<const uint32_t> simulationTokens, uint64_t &degenerateRows) {
    const uint32_t rows = static_cast<uint32_t>(simulationTokens.size()) + 1;
    std::vector<int64_t> states{int64_t(state_)};
    int64_t cursor = int64_t(state_);
    for (uint32_t token : simulationTokens) {
      if (token >= pack_.vocabulary)
        break;
      cursor = advance(cursor, token);
      if (cursor < 0)
        break;
      states.push_back(cursor);
    }
    std::vector<uint32_t> words(size_t{rows} * maskWords_, 0);
    for (size_t row = 0; row < states.size(); ++row) {
      if (uint64_t(states[row]) >= pack_.dfaStates) {
        // The stop state admits nothing; emit the eos set so the row is
        // nonzero like the degenerate rows below.
        ++degenerateRows;
        for (uint32_t id : pack_.eosIds)
          setBit(words, row * maskWords_, id);
        continue;
      }
      const std::vector<uint32_t> &mask = maskRow(uint64_t(states[row]));
      std::memcpy(words.data() + row * maskWords_, mask.data(),
                  maskWords_ * sizeof(uint32_t));
    }
    for (size_t row = states.size(); row < rows; ++row) {
      ++degenerateRows;
      std::memcpy(words.data() + row * maskWords_,
                  words.data() + (states.size() - 1) * maskWords_,
                  maskWords_ * sizeof(uint32_t));
    }
    return words;
  }

  // The forced span at the consuming state, anchored like the wire's
  // FORCED_TOKENS: tokens[0] fills the pending slot, so the byte chain
  // starts before the pending anchor and its first token is the anchor.
  [[nodiscard]] std::vector<uint32_t> forcedSpan() const {
    if (stopped_ || state_ >= pack_.dfaStates)
      return {};
    // The single-live-byte chain behind the consuming state, capped like
    // RICHENGINE_DFA_FORCED_BYTES.
    std::vector<uint8_t> chain;
    int64_t cursor = int64_t(state_);
    while (chain.size() < 64 && cursor >= 0 &&
           uint64_t(cursor) < pack_.dfaStates) {
      int32_t next = -1;
      uint8_t byte = 0;
      const int16_t *row = pack_.dfaTable.data() + uint64_t(cursor) * 256;
      for (uint32_t b = 0; b < 256; ++b) {
        if (row[b] >= 0) {
          if (next >= 0) {
            next = -2;
            break;
          }
          next = row[b];
          byte = static_cast<uint8_t>(b);
        }
      }
      if (next < 0)
        break;
      chain.push_back(byte);
      cursor = next;
    }
    if (chain.empty())
      return {};
    // Greedy longest-prefix cover of the chain, lowest id on ties, each
    // emitted token checked live under the marker rule — the on-device
    // decode_dfa_forced_tokens derivation.
    std::vector<uint32_t> tokens;
    int64_t q = int64_t(state_);
    size_t offset = 0;
    while (offset < chain.size() &&
           tokens.size() < model::ExecutionLimits::draftProposalTokens &&
           q >= 0 && uint64_t(q) < pack_.dfaStates) {
      const size_t remaining = chain.size() - offset;
      uint32_t best = pack_.vocabulary;
      uint32_t bestLength = 0;
      for (uint32_t token = 0; token < pack_.vocabulary; ++token) {
        const int32_t begin = pack_.tokenOffsets[token];
        const uint32_t length =
            uint32_t(pack_.tokenOffsets[token + 1] - begin);
        if (!length || length > remaining || length < bestLength)
          continue;
        if (std::memcmp(chain.data() + offset, pack_.tokenBytes.data() + begin,
                        length) == 0 &&
            (length > bestLength || token < best)) {
          best = token;
          bestLength = length;
        }
      }
      if (!bestLength)
        break;
      const int64_t next = walk(uint64_t(q), best);
      if (next < 0)
        break;
      tokens.push_back(best);
      offset += bestLength;
      q = next;
    }
    // Like forced_tokens_after: an anchor plus at least one more token.
    if (tokens.size() < 2)
      return {};
    return tokens;
  }

  [[nodiscard]] uint64_t invalidCommits() const noexcept {
    return invalidCommits_;
  }
  [[nodiscard]] uint64_t cachedRows() const noexcept { return rowCache_.size(); }

private:
  [[nodiscard]] bool eosAdmits(uint64_t state) const {
    return (pack_.accepting[state >> 3] >> (state & 7)) & 1;
  }
  [[nodiscard]] bool special(uint32_t token) const {
    return (pack_.specialBits[token >> 3] >> (token & 7)) & 1;
  }
  [[nodiscard]] bool marked(uint64_t state, uint8_t byte) const {
    return (pack_.marker[state * 32 + (byte >> 3)] >> (byte & 7)) & 1;
  }
  static void setBit(std::vector<uint32_t> &words, uint32_t token) {
    words[token / 32] |= 1U << (token % 32);
  }
  void setBit(std::vector<uint32_t> &words, size_t offset,
              uint32_t token) const {
    words[offset + token / 32] |= 1U << (token % 32);
  }
  [[nodiscard]] int64_t walk(uint64_t state, uint32_t token) const {
    const int32_t end = pack_.tokenOffsets[token + 1];
    bool tookMarked = false;
    for (int32_t index = pack_.tokenOffsets[token]; index < end; ++index) {
      const uint8_t byte = pack_.tokenBytes[index];
      const int16_t next = pack_.dfaTable[state * 256 + byte];
      if (next < 0)
        return -1;
      if (marked(state, byte)) {
        if (!special(token))
          return -1;
        tookMarked = true;
      }
      state = uint64_t(next);
    }
    if (special(token) && !tookMarked)
      return -1;
    return int64_t(state);
  }

  const Pack &pack_;
  const uint32_t maskWords_;
  std::unordered_set<uint32_t> eos_;
  uint64_t state_ = 0;
  bool stopped_ = false;
  uint64_t invalidCommits_ = 0;
  std::unordered_map<uint64_t, std::vector<uint32_t>> rowCache_;
};

// ------------------------------------------------------------- events ---

struct Observation final {
  std::vector<uint32_t> outputTokens;
  bool completed = false;
  std::string failure;
};

struct MaskAsk final {
  uint64_t requestId = 0;
  std::vector<uint32_t> simulationTokens;
};

struct BatchObservation final {
  WorkKind kind = WorkKind::Decode;
  uint32_t width = 0;
  uint32_t outputTokens = 0;
  double wallMilliseconds = 0.0;
};

class Events final : public EngineEventSink {
public:
  std::function<void(uint64_t, std::span<const uint32_t>)> onTokens;

  void batchCompleted(WorkKind kind, uint32_t width, uint32_t,
                      uint32_t outputTokens, uint32_t, uint32_t,
                      double wallMilliseconds, double) override {
    ++batchSequence_;
    lastBatch_ = {kind, width, outputTokens, wallMilliseconds};
  }

  void started(uint64_t, uint32_t, uint32_t) override {}

  void tokens(uint64_t requestId, std::span<const uint32_t> values) override {
    Observation &value = observations_[requestId];
    value.outputTokens.insert(value.outputTokens.end(), values.begin(),
                              values.end());
    if (onTokens)
      onTokens(requestId, values);
  }

  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulationTokens) override {
    maskAsks_.push_back(
        {requestId,
         std::vector<uint32_t>(simulationTokens.begin(),
                               simulationTokens.end())});
    ++maskRequests_;
    maskRows_ += simulationTokens.size() + 1;
  }

  void completed(uint64_t requestId, EngineFinishReason, uint32_t,
                 uint32_t, uint32_t, uint32_t,
                 std::span<const float>) override {
    observations_[requestId].completed = true;
  }

  void failed(uint64_t requestId, LaneOutcome outcome,
              std::string message) override {
    observations_[requestId].failure =
        std::string(laneOutcomeWire(outcome).code) + ":" + message;
  }

  [[nodiscard]] const Observation &get(uint64_t requestId) const {
    auto found = observations_.find(requestId);
    if (found == observations_.end())
      throw std::logic_error("request produced no events");
    return found->second;
  }

  [[nodiscard]] bool maskAsksEmpty() const noexcept {
    return maskAsks_.empty();
  }

  [[nodiscard]] MaskAsk takeMaskAsk() {
    MaskAsk ask = std::move(maskAsks_.front());
    maskAsks_.erase(maskAsks_.begin());
    return ask;
  }

  [[nodiscard]] uint64_t batchSequence() const noexcept {
    return batchSequence_;
  }
  [[nodiscard]] const BatchObservation &lastBatch() const noexcept {
    return lastBatch_;
  }
  [[nodiscard]] uint64_t maskRequests() const noexcept { return maskRequests_; }
  [[nodiscard]] uint64_t maskRows() const noexcept { return maskRows_; }

private:
  std::unordered_map<uint64_t, Observation> observations_;
  std::vector<MaskAsk> maskAsks_;
  BatchObservation lastBatch_;
  uint64_t batchSequence_ = 0;
  uint64_t maskRequests_ = 0;
  uint64_t maskRows_ = 0;
};

class Driver final {
public:
  Driver(engine::Engine &engine, Events &events)
      : engine_(engine), events_(events) {
    engine_.setCompletionNotifier([wake = wake_] {
      {
        std::lock_guard lock(wake->mutex);
        wake->notified = true;
      }
      wake->condition.notify_one();
    });
  }

  // maskProvider answers each queued MASK_REQUEST with its rows and returns
  // how many it served; it runs between ticks like the server's mask worker.
  void untilIdle(
      const std::function<void(const BatchObservation &)> &observer,
      const std::function<uint32_t()> &maskProvider) {
    const auto deadline = Clock::now() + std::chrono::hours(2);
    uint64_t observedBatchSequence = events_.batchSequence();
    for (;;) {
      const bool progressed = engine_.tick(milliseconds(Clock::now()));
      const uint64_t sequence = events_.batchSequence();
      if (sequence != observedBatchSequence) {
        if (sequence != observedBatchSequence + 1)
          throw std::logic_error("benchmark skipped a completed batch");
        observedBatchSequence = sequence;
        if (observer)
          observer(events_.lastBatch());
      }
      const uint32_t answered = maskProvider ? maskProvider() : 0;
      if (!events_.maskAsksEmpty())
        throw std::logic_error("mask provider left a request unanswered");
      if (drained())
        return;
      // An answered mask is progress the next tick must consume; only sleep
      // when neither the engine nor the provider moved this round.
      if (progressed || answered)
        continue;
      if (Clock::now() >= deadline)
        throw std::runtime_error("dfa benchmark timed out");
      auto wakeup = deadline;
      if (auto next = engine_.nextWakeupMilliseconds()) {
        wakeup = std::min(wakeup, Clock::time_point{
            std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double, std::milli>(*next))});
      }
      std::unique_lock lock(wake_->mutex);
      wake_->condition.wait_until(lock, wakeup, [this] {
        if (!wake_->notified)
          return false;
        wake_->notified = false;
        return true;
      });
    }
  }

private:
  struct WakeState final {
    std::mutex mutex;
    std::condition_variable condition;
    bool notified = false;
  };

  // Every submitted request has ended, no command is in flight and no mask
  // request waits unanswered — a lane parked in WaitingMask with no ticket
  // outstanding leaves commandInFlight() false but must not drain the loop.
  [[nodiscard]] bool drained() const {
    const engine::EngineSnapshot counts = engine_.snapshot();
    return counts.submitted ==
               counts.completed + counts.cancelled + counts.failed &&
           !engine_.commandInFlight() && events_.maskAsksEmpty();
  }

  engine::Engine &engine_;
  Events &events_;
  std::shared_ptr<WakeState> wake_ = std::make_shared<WakeState>();
};

// ---------------------------------------------------------- scenarios ----

enum class Scenario : uint8_t { Unconstrained, Mask, MaskForced, Dfa, Mixed };

struct ScenarioResult final {
  std::string scenario;
  uint32_t sample = 0;
  uint32_t width = 0;
  uint32_t steps = 0;
  uint32_t outputTokens = 0;
  uint64_t decodeBatches = 0;
  uint64_t decodeOutputTokens = 0;
  uint64_t maskRequests = 0;
  uint64_t maskRows = 0;
  uint64_t degenerateRows = 0;
  uint64_t pushedSpans = 0;
  uint64_t invalidCommits = 0;
  double requestWallMilliseconds = 0.0;
  double prefillWallMilliseconds = 0.0;
  double prefillGpuMilliseconds = 0.0;
  double decodeWallMilliseconds = 0.0;
  double decodeGpuMilliseconds = 0.0;
  double maskWaitSeconds = 0.0;
  double maskAnswerSeconds = 0.0;
  double constrainedTargetGpuSeconds = 0.0;
  uint64_t forcedSpansApplied = 0;
  uint64_t forcedSpansDroppedStale = 0;
  uint64_t forcedTokensEmitted = 0;
  uint64_t gpuForcedSpansApplied = 0;
  uint64_t maskOverlapBatches = 0;
  uint64_t maskOverlapRequests = 0;
  bool dfaInstalled = false;
  bool gpuMasked = false;
  std::vector<uint32_t> firstLaneTokens;
};

struct TelemetryDelta final {
  double prefillWallSeconds = 0.0;
  double prefillGpuSeconds = 0.0;
  double decodeWallSeconds = 0.0;
  double decodeGpuSeconds = 0.0;
  double maskWaitSeconds = 0.0;
  double constrainedTargetGpuSeconds = 0.0;
  uint64_t forcedSpansApplied = 0;
  uint64_t forcedSpansDroppedStale = 0;
  uint64_t forcedTokensEmitted = 0;
  uint64_t gpuForcedSpansApplied = 0;
  uint64_t maskOverlapBatches = 0;
  uint64_t maskOverlapRequests = 0;
};

TelemetryDelta delta(const model::ModelTelemetry &before,
                     const model::ModelTelemetry &after) {
  TelemetryDelta result;
  result.prefillWallSeconds =
      after.totalPrefillWallSeconds - before.totalPrefillWallSeconds;
  result.prefillGpuSeconds =
      after.totalPrefillGpuSeconds - before.totalPrefillGpuSeconds;
  result.decodeWallSeconds =
      after.totalDecodeWallSeconds - before.totalDecodeWallSeconds;
  result.decodeGpuSeconds =
      after.totalDecodeGpuSeconds - before.totalDecodeGpuSeconds;
  result.maskWaitSeconds =
      after.totalConstrainedMaskWaitSeconds -
      before.totalConstrainedMaskWaitSeconds;
  result.constrainedTargetGpuSeconds =
      after.totalConstrainedTargetForwardGpuSeconds -
      before.totalConstrainedTargetForwardGpuSeconds;
  result.forcedSpansApplied =
      after.forcedSpansApplied - before.forcedSpansApplied;
  result.forcedSpansDroppedStale =
      after.forcedSpansDroppedStale - before.forcedSpansDroppedStale;
  result.forcedTokensEmitted =
      after.forcedTokensEmitted - before.forcedTokensEmitted;
  result.gpuForcedSpansApplied =
      after.gpuForcedSpansApplied - before.gpuForcedSpansApplied;
  result.maskOverlapBatches =
      after.constrainedMaskOverlapBatches -
      before.constrainedMaskOverlapBatches;
  result.maskOverlapRequests =
      after.constrainedMaskOverlapRequests -
      before.constrainedMaskOverlapRequests;
  return result;
}

void submitRequest(engine::Engine &engine, uint64_t id,
                   const std::vector<uint32_t> &prompt, uint32_t maxNewTokens,
                   bool constrained, double deadlineMilliseconds) {
  EngineRequest request;
  request.id = id;
  request.prompt = prompt;
  request.maxNewTokens = maxNewTokens;
  request.sampling.temperature = 0.0F;
  request.sampling.topP = 1.0F;
  request.constraint =
      constrained ? ConstraintMode::TokenMask : ConstraintMode::None;
  request.deadlineMilliseconds = deadlineMilliseconds;
  engine.submit(std::move(request));
}

ScenarioResult runScenario(engine::Engine &engine, Driver &driver,
                           model::RuntimeModel &executor, Events &events,
                           engine::Cache &cache, const Pack &pack,
                           Scenario scenario, uint32_t width, uint32_t steps,
                           uint32_t sample, uint64_t &requestId) {
  static_cast<void>(cache.evictAll());
  const bool constrained = scenario != Scenario::Unconstrained;
  // The mixed burst alternates DFA-installed constrained lanes with free
  // ones — what RICHENGINE_MIXED_DECODE_PACK packs into one command under
  // RICHENGINE_DFA_MODE=full.
  const bool mixed = scenario == Scenario::Mixed;
  const bool installDfa = scenario == Scenario::Dfa || mixed;
  const bool pushForced = scenario == Scenario::MaskForced;
  const auto laneConstrained = [&](uint32_t lane) {
    return constrained && (!mixed || lane % 2 == 0);
  };
  // The lanes share one prompt; warm it once so every lane resumes from the
  // published prefix in a packed prefill and decodes in lockstep, as the
  // decode-throughput scenario of backend_benchmark does.
  submitRequest(engine, requestId++, pack.prompt, 1, false,
                milliseconds(Clock::now() + std::chrono::hours(2)));
  driver.untilIdle({}, {});
  if (!events.get(requestId - 1).completed)
    throw std::runtime_error("dfa benchmark warm request failed: " +
                             events.get(requestId - 1).failure);

  std::vector<DfaMirror> mirrors(width, DfaMirror(pack));
  std::unordered_map<uint64_t, uint32_t> laneOf;
  const uint64_t asksBefore = events.maskRequests();
  const uint64_t rowsBefore = events.maskRows();
  uint64_t degenerateRows = 0;
  uint64_t pushedSpans = 0;
  double maskAnswerSeconds = 0.0;

  if (constrained)
    events.onTokens = [&](uint64_t id, std::span<const uint32_t> tokens) {
      auto lane = laneOf.find(id);
      if (lane == laneOf.end())
        return;
      if (!laneConstrained(lane->second))
        return;
      mirrors[lane->second].commit(tokens);
      if (pushForced) {
        if (std::vector<uint32_t> span = mirrors[lane->second].forcedSpan();
            !span.empty()) {
          engine.provideForcedTokens(id, span);
          ++pushedSpans;
        }
      }
    };
  const auto maskProvider = [&]() -> uint32_t {
    uint32_t answered = 0;
    while (!events.maskAsksEmpty()) {
      const MaskAsk ask = events.takeMaskAsk();
      auto lane = laneOf.find(ask.requestId);
      if (lane == laneOf.end())
        throw std::logic_error("mask request for an unknown request");
      const auto maskBegan = Clock::now();
      std::vector<uint32_t> words =
          mirrors[lane->second].masks(ask.simulationTokens, degenerateRows);
      maskAnswerSeconds +=
          std::chrono::duration<double>(Clock::now() - maskBegan).count();
      engine.provideMask(ask.requestId, words);
      ++answered;
    }
    return answered;
  };

  const auto began = Clock::now();
  const model::ModelTelemetry before = executor.telemetry();
  std::vector<uint64_t> requestIds;
  requestIds.reserve(width);
  for (uint32_t lane = 0; lane < width; ++lane) {
    const uint64_t id = requestId++;
    laneOf.emplace(id, lane);
    requestIds.push_back(id);
    submitRequest(engine, id, pack.prompt, steps, laneConstrained(lane),
                  milliseconds(began + std::chrono::hours(2)));
    if (installDfa && laneConstrained(lane)) {
      // Queued requests record the frame for admission to deliver, exactly
      // like a CONSTRAINT_DFA frame outrunning its lane.
      static_cast<void>(
          engine.provideDfa(id, pack.dfaTable, pack.accepting, pack.marker,
                            pack.eosIds, pack.costs));
    }
  }

  uint64_t decodeBatches = 0;
  uint64_t decodeOutputTokens = 0;
  driver.untilIdle(
      [&](const BatchObservation &batch) {
        if (batch.kind != WorkKind::Decode)
          return;
        ++decodeBatches;
        decodeOutputTokens += batch.outputTokens;
      },
      maskProvider);
  const auto ended = Clock::now();
  events.onTokens = {};
  const TelemetryDelta model = delta(before, executor.telemetry());

  ScenarioResult result;
  result.sample = sample;
  result.width = width;
  result.steps = steps;
  result.requestWallMilliseconds =
      milliseconds(ended) - milliseconds(began);
  result.prefillWallMilliseconds = model.prefillWallSeconds * 1000.0;
  result.prefillGpuMilliseconds = model.prefillGpuSeconds * 1000.0;
  result.decodeWallMilliseconds = model.decodeWallSeconds * 1000.0;
  result.decodeGpuMilliseconds = model.decodeGpuSeconds * 1000.0;
  result.maskWaitSeconds = model.maskWaitSeconds;
  result.maskAnswerSeconds = maskAnswerSeconds;
  result.constrainedTargetGpuSeconds = model.constrainedTargetGpuSeconds;
  result.forcedSpansApplied = model.forcedSpansApplied;
  result.forcedSpansDroppedStale = model.forcedSpansDroppedStale;
  result.forcedTokensEmitted = model.forcedTokensEmitted;
  result.gpuForcedSpansApplied = model.gpuForcedSpansApplied;
  result.maskOverlapBatches = model.maskOverlapBatches;
  result.maskOverlapRequests = model.maskOverlapRequests;
  result.decodeBatches = decodeBatches;
  result.decodeOutputTokens = decodeOutputTokens;
  result.maskRequests = events.maskRequests() - asksBefore;
  result.maskRows = events.maskRows() - rowsBefore;
  result.degenerateRows = degenerateRows;
  result.pushedSpans = pushedSpans;
  result.dfaInstalled = installDfa;
  // A Full-mode DFA lane never asks the host: no new mask requests means the
  // install became GPU-masked.
  result.gpuMasked = installDfa && !result.maskRequests;
  for (uint64_t id : requestIds) {
    const Observation &observed = events.get(id);
    if (!observed.completed || !observed.failure.empty())
      throw std::runtime_error("constrained request failed: " +
                               observed.failure);
    result.outputTokens +=
        static_cast<uint32_t>(observed.outputTokens.size());
    result.invalidCommits += mirrors[laneOf.at(id)].invalidCommits();
  }
  result.firstLaneTokens = events.get(requestIds.front()).outputTokens;
  switch (scenario) {
  case Scenario::Unconstrained:
    result.scenario = "unconstrained";
    break;
  case Scenario::Mask:
    result.scenario = "mask";
    break;
  case Scenario::MaskForced:
    result.scenario = "mask-forced";
    break;
  case Scenario::Dfa:
    result.scenario = "dfa";
    break;
  case Scenario::Mixed:
    result.scenario = "mixed";
    break;
  }
  return result;
}

// ------------------------------------------------------------ output -----

const char *envValue(const char *name) {
  const char *value = std::getenv(name);
  return value ? value : "";
}

void emitResult(const ScenarioResult &result) {
  const double tokensPerSecond =
      result.requestWallMilliseconds > 0.0
          ? double(result.outputTokens) * 1000.0 /
                result.requestWallMilliseconds
          : 0.0;
  const double stepMilliseconds =
      result.decodeBatches
          ? result.decodeWallMilliseconds / double(result.decodeBatches)
          : 0.0;
  std::cout << std::setprecision(10)
            << "{\"scenario\":" << json::quote(result.scenario)
            << ",\"sample\":" << result.sample
            << ",\"width\":" << result.width
            << ",\"steps_requested\":" << result.steps
            << ",\"output_tokens\":" << result.outputTokens
            << ",\"decode_batches\":" << result.decodeBatches
            << ",\"decode_output_tokens\":" << result.decodeOutputTokens
            << ",\"wall_ms\":" << result.requestWallMilliseconds
            << ",\"prefill_wall_ms\":" << result.prefillWallMilliseconds
            << ",\"prefill_gpu_ms\":" << result.prefillGpuMilliseconds
            << ",\"decode_wall_ms\":" << result.decodeWallMilliseconds
            << ",\"decode_gpu_ms\":" << result.decodeGpuMilliseconds
            << ",\"ms_per_decode_batch\":" << stepMilliseconds
            << ",\"request_wall_tokens_per_s\":" << tokensPerSecond
            << ",\"mask_requests\":" << result.maskRequests
            << ",\"mask_rows\":" << result.maskRows
            << ",\"degenerate_rows\":" << result.degenerateRows
            << ",\"mask_wait_s\":" << result.maskWaitSeconds
            << ",\"mask_answer_s\":" << result.maskAnswerSeconds
            << ",\"constrained_target_gpu_s\":"
            << result.constrainedTargetGpuSeconds
            << ",\"forced_spans_applied\":" << result.forcedSpansApplied
            << ",\"forced_spans_dropped_stale\":"
            << result.forcedSpansDroppedStale
            << ",\"forced_tokens_emitted\":" << result.forcedTokensEmitted
            << ",\"gpu_forced_spans_applied\":"
            << result.gpuForcedSpansApplied
            << ",\"mask_overlap_batches\":" << result.maskOverlapBatches
            << ",\"mask_overlap_requests\":" << result.maskOverlapRequests
            << ",\"host_spans_pushed\":" << result.pushedSpans
            << ",\"invalid_commits\":" << result.invalidCommits
            << ",\"dfa_installed\":" << (result.dfaInstalled ? "true" : "false")
            << ",\"gpu_masked\":" << (result.gpuMasked ? "true" : "false")
            << ",\"env\":{\"RICHENGINE_DFA_MODE\":"
            << json::quote(envValue("RICHENGINE_DFA_MODE"))
            << ",\"RICHENGINE_DFA_MERGE\":"
            << json::quote(envValue("RICHENGINE_DFA_MERGE"))
            << ",\"RICHENGINE_DFA_SPAN_SKIP_DRAFT\":"
            << json::quote(envValue("RICHENGINE_DFA_SPAN_SKIP_DRAFT"))
            << ",\"RICHENGINE_GPU_DECODE_GRID\":"
            << json::quote(envValue("RICHENGINE_GPU_DECODE_GRID"))
            << "},\"first_lane_tokens\":[";
  for (size_t index = 0; index < result.firstLaneTokens.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << result.firstLaneTokens[index];
  }
  std::cout << "]}\n";
  std::cout.flush();
}

uint32_t parseUint(std::string_view value, const char *option,
                   uint32_t maximum) {
  uint32_t parsed = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (error != std::errc{} || end != value.data() + value.size() || !parsed ||
      parsed > maximum)
    throw std::invalid_argument(std::string(option) +
                                " takes an integer in [1, " +
                                std::to_string(maximum) + "]");
  return parsed;
}

std::vector<Scenario> parseScenarios(std::string_view value) {
  std::vector<Scenario> selected;
  for (;;) {
    const size_t comma = value.find(',');
    const std::string_view name = value.substr(0, comma);
    const Scenario scenario =
        name == "unconstrained" ? Scenario::Unconstrained
        : name == "mask"        ? Scenario::Mask
        : name == "mask-forced" ? Scenario::MaskForced
        : name == "dfa"         ? Scenario::Dfa
        : name == "mixed"       ? Scenario::Mixed
                                : throw std::invalid_argument(
                                      "unknown benchmark scenario");
    selected.push_back(scenario);
    if (comma == std::string_view::npos)
      break;
    value.remove_prefix(comma + 1);
  }
  return selected;
}

kv::Format parseKvFormat(std::string_view value) {
  for (const kv::Format format :
       {kv::Format::Int4, kv::Format::Int8, kv::Format::BFloat16,
        kv::Format::Float8E4M3})
    if (kv::formatName(format) == value)
      return format;
  throw std::invalid_argument("--kv-format takes int4, int8, bf16 or fp8e4m3");
}

double parseAneFfnShare(std::string_view value) {
  double share = 0.0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), share);
  if (error != std::errc{} || end != value.data() + value.size() ||
      !(share >= 0.0 && share < 1.0))
    throw std::invalid_argument("--ane-ffn-share takes a share in [0, 1)");
  return share;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      std::cerr << "usage: dfa-benchmark METALLIB MODEL_ROOT --pack PACK "
                   "[--scenario LIST] [--width W] [--steps N] [--samples S] "
                   "[--ane-ffn-share SHARE] [--kv-format FORMAT]\n"
                   "  LIST: unconstrained,mask,mask-forced,dfa "
                   "(default: all)\n"
                   "  PACK: dev/benchmarks/dfa_pack.py output\n";
      return 2;
    }
    std::filesystem::path packPath;
    std::vector<Scenario> scenarios{Scenario::Unconstrained, Scenario::Mask,
                                    Scenario::MaskForced, Scenario::Dfa};
    uint32_t width = 1;
    uint32_t steps = 128;
    uint32_t samples = 1;
    std::optional<double> aneFfnShare;
    kv::Format kvFormat = kv::Format::Int8;
    for (int index = 3; index < argc; index += 2) {
      if (index + 1 >= argc)
        throw std::invalid_argument("benchmark option requires a value");
      const std::string_view option(argv[index]);
      const std::string_view value(argv[index + 1]);
      if (option == "--pack") {
        packPath = value;
      } else if (option == "--scenario") {
        scenarios = parseScenarios(value);
      } else if (option == "--width") {
        width = parseUint(value, "--width",
                         model::ExecutionLimits::maximumBatchWidth);
      } else if (option == "--steps") {
        steps = parseUint(value, "--steps", 4096);
      } else if (option == "--samples") {
        samples = parseUint(value, "--samples", 100);
      } else if (option == "--ane-ffn-share") {
        aneFfnShare = parseAneFfnShare(value);
      } else if (option == "--kv-format") {
        kvFormat = parseKvFormat(value);
      } else {
        throw std::invalid_argument("unknown benchmark option");
      }
    }
    if (packPath.empty())
      throw std::invalid_argument("dfa-benchmark requires --pack");
    const Pack pack = loadPack(packPath);

    engine::RuntimeBootstrapConfig bootstrapConfig;
    auto &config = bootstrapConfig.resources;
    config.metallibPath = std::filesystem::path(argv[1]);
    config.modelRoot = std::filesystem::path(argv[2]);
    config.model = model::inspectModelPackage(config.modelRoot);
    config.buildId = RICHENGINE_BUILD_ID;
    config.aneFfn = engine::AneFfnSetting::fromGiven(aneFfnShare, std::nullopt);
    config.kvFormat = kvFormat;
    if (config.model.capabilities.vocabularySize != pack.vocabulary)
      throw std::runtime_error(
          "dfa pack vocabulary does not match the model");
    std::unique_ptr<engine::RuntimeBootstrap> bootstrap;
    const auto memoryDeadline =
        Clock::now() + std::chrono::seconds(180);
    bool waitingAnnounced = false;
    while (!bootstrap) {
      try {
        bootstrap = engine::RuntimeBootstrap::start(
            bootstrapConfig, [](std::span<const uint8_t>) {},
            []() -> std::string {
              throw std::logic_error(
                  "benchmark does not accept native status requests");
            });
      } catch (const engine::RuntimeBootstrapError &error) {
        if (error.report().resourceFailure !=
                engine::RuntimeResourceFailure::HostCapacity ||
            Clock::now() >= memoryDeadline)
          throw;
        if (!waitingAnnounced) {
          std::cerr << "dfa-benchmark: waiting for host memory to settle "
                       "before loading the model\n";
          waitingAnnounced = true;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
      }
    }
    if (!bootstrap->nativeLoop().ready() ||
        !bootstrap->nativeLoop().engineHealthy() ||
        bootstrap->nativeLoop().commandInFlight())
      throw std::runtime_error(
          "dfa benchmark bootstrap did not finish idle and ready");
    auto *resources = &bootstrap->resources();
    auto *executor = &bootstrap->modelRuntime();

    Events events;
    engine::EngineConfig engineConfig;
    engineConfig.maxContext =
        bootstrap->nativeLoop().snapshot().maximumContextTokens;
    engineConfig.vocabularySize = pack.vocabulary;
    engine::connectToGovernor(engineConfig, resources->memoryGovernor());
    engine::Engine engine(engineConfig, resources->cache(), *executor,
                          events);
    Driver driver(engine, events);
    // The process-wide packed vocabulary the GPU-DFA walk kernels index,
    // like the wire's VOCABULARY frame.
    engine.setVocabulary(pack.tokenBytes, pack.tokenOffsets,
                         pack.specialBits);

    std::cout << "{\"event\":\"started\",\"build_id\":"
              << json::quote(RICHENGINE_BUILD_ID)
              << ",\"vocabulary\":" << pack.vocabulary
              << ",\"dfa_states\":" << pack.dfaStates
              << ",\"prompt_tokens\":" << pack.prompt.size() << "}\n";
    std::cout.flush();

    uint64_t requestId = 1;
    for (uint32_t sample = 0; sample < samples; ++sample) {
      for (Scenario scenario : scenarios) {
        std::cerr << "dfa-benchmark: scenario="
                  << (scenario == Scenario::Unconstrained ? "unconstrained"
                      : scenario == Scenario::Mask         ? "mask"
                      : scenario == Scenario::MaskForced   ? "mask-forced"
                      : scenario == Scenario::Dfa          ? "dfa"
                                                           : "mixed")
                  << " sample=" << sample << " width=" << width
                  << " steps=" << steps << '\n';
        ScenarioResult result =
            runScenario(engine, driver, *executor, events, resources->cache(),
                        pack, scenario, width, steps, sample, requestId);
        emitResult(result);
      }
    }
    std::cout << "{\"event\":\"finished\"}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "dfa-benchmark: " << error.what() << '\n';
    return 1;
  }
}
