#include "model/DiffusionSampler.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace richengine::model {

DiffusionSampler::DiffusionSampler(DiffusionSchedule schedule)
    : schedule_(schedule) {
  if (!schedule_.valid()) {
    throw std::invalid_argument("invalid diffusion schedule");
  }
}

std::vector<bool>
DiffusionSampler::acceptMask(std::span<const float> entropies) const {
  if (entropies.size() != schedule_.canvasLength) {
    throw std::invalid_argument("entropy rows do not match the canvas");
  }
  std::vector<uint32_t> order(entropies.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
    return entropies[a] < entropies[b];
  });
  std::vector<bool> accepted(entropies.size(), false);
  float prefix = 0.0F;
  for (uint32_t rank = 0; rank < order.size(); ++rank) {
    // Exclusive prefix: the position is the cheapest-ranked survivor while
    // the budget still covers everything accepted before it.
    if (!(prefix <= schedule_.entropyBound))
      break;
    accepted[order[rank]] = true;
    prefix += entropies[order[rank]];
  }
  return accepted;
}

bool DiffusionSampler::earlyExit(const StepStats &stats) noexcept {
  return earlyExit(stats, stats.argmaxStable);
}

bool DiffusionSampler::earlyExit(const StepStats &stats,
                                 bool stable) noexcept {
  if (firstStep_) {
    firstStep_ = false;
    stableSteps_ = 0;
    return false;
  }
  stableSteps_ = stable ? stableSteps_ + 1 : 0;
  return stableSteps_ >= schedule_.stabilityThreshold &&
         stats.meanEntropy < schedule_.confidenceThreshold;
}

uint32_t DiffusionSampler::firstStop(
    std::span<const uint32_t> tokens,
    std::span<const uint32_t> stopTokens) noexcept {
  for (uint32_t index = 0; index < tokens.size(); ++index) {
    if (std::find(stopTokens.begin(), stopTokens.end(), tokens[index]) !=
        stopTokens.end())
      return index;
  }
  return static_cast<uint32_t>(tokens.size());
}

DiffusionSampler::Commit
DiffusionSampler::commit(std::span<const uint32_t> argmax,
                         std::span<const uint32_t> stopTokens) const {
  if (argmax.size() != schedule_.canvasLength) {
    throw std::invalid_argument("argmax rows do not match the canvas");
  }
  Commit result;
  result.tokens.assign(argmax.begin(), argmax.end());
  result.stopIndex = firstStop(result.tokens, stopTokens);
  result.stop = result.stopIndex < result.tokens.size();
  if (result.stop) {
    std::fill(result.tokens.begin() + result.stopIndex + 1,
              result.tokens.end(), schedule_.padToken);
  }
  return result;
}

uint32_t DiffusionSampler::emitBudget(uint64_t emittedTokens,
                                      uint32_t maxNewTokens) const noexcept {
  const uint64_t remaining =
      maxNewTokens > emittedTokens ? maxNewTokens - emittedTokens : 0;
  return static_cast<uint32_t>(
      std::min<uint64_t>(remaining, schedule_.canvasLength));
}

bool DiffusionSampler::finished(const Commit &commit,
                                uint32_t emitted) const noexcept {
  return commit.stop && commit.stopIndex < emitted;
}

uint32_t DiffusionSampler::canvasPages() const noexcept {
  return schedule_.canvasLength / kv::kPageTokens;
}

std::vector<RichKvPage>
DiffusionSampler::canvasPageTable(std::span<const RichKvPage> prefixEntries,
                                  uint64_t scratchExtentBase) const {
  if (scratchExtentBase & RICHENGINE_KV_PAGE_INDEX_MASK) {
    throw std::invalid_argument(
        "canvas scratch extent is not page-table aligned");
  }
  std::vector<RichKvPage> table(prefixEntries.begin(), prefixEntries.end());
  const uint32_t pages = canvasPages();
  for (uint32_t page = 0; page < pages; ++page) {
    table.push_back(scratchExtentBase + page);
  }
  return table;
}

CanvasStepDriver::CanvasStepDriver(const DiffusionSchedule &schedule,
                                   Config config,
                                   std::span<const uint32_t> stopTokens)
    : schedule_(schedule),
      config_(config),
      stopTokens_(stopTokens.begin(), stopTokens.end()),
      sampler_(schedule),
      nextStep_(schedule.maxDenoisingSteps) {
  if (!schedule_.valid())
    throw std::invalid_argument("invalid diffusion schedule");
  if (!config_.stepsPerCommand)
    config_.stepsPerCommand = 1;
  config_.stepsPerCommand =
      std::min(config_.stepsPerCommand, kMaxStepsPerCommand);
  prevArgmax_.resize(schedule_.canvasLength);
}

CanvasStepDriver::Chunk CanvasStepDriver::nextChunk() {
  if (finished_)
    return {};
  if (current_.count && drained_ != current_.count)
    throw std::logic_error("canvas chunk left the stats ring undrained");
  const uint32_t count =
      std::min(config_.stepsPerCommand, nextStep_);
  current_.firstStep = nextStep_;
  current_.count = count;
  // The last encoded step of the schedule may drop its accept tail: its
  // tokensOut, stats and renoise have no consumer either way.
  current_.commitTail = config_.commitTail && nextStep_ == count;
  nextStep_ -= count;
  drained_ = 0;
  return current_;
}

CanvasStepDriver::Outcome
CanvasStepDriver::drain(uint32_t slot,
                        const DiffusionSampler::StepStats &stats,
                        std::span<const uint32_t> argmax,
                        std::span<const float> entropies) {
  if (finished_)
    throw std::logic_error("canvas step driver drained past the exit");
  if (slot != drained_)
    throw std::logic_error("canvas stats ring drained out of order");
  ++drained_;
  if (!argmax.empty() && argmax.size() != schedule_.canvasLength)
    throw std::invalid_argument("canvas argmax rows do not match");
  Outcome outcome;
  outcome.stopIndex = argmax.empty()
                          ? schedule_.canvasLength
                          : DiffusionSampler::firstStop(argmax, stopTokens_);
  const bool tailSlot =
      current_.commitTail && slot + 1 == current_.count;
  if (tailSlot) {
    // The commit-only encode wrote no stats; the argmax is the commit.
    outcome.scheduleEnd = true;
    finished_ = true;
    return outcome;
  }
  // Whole-canvas argmax-stable stats, whenever a previous argmax exists:
  // the fraction-settled gate reads them, and the prefix exit uses the
  // share as a maturity guard — a fresh stop on a still-churning canvas is
  // premature, the denoise has barely started.
  uint32_t stableCount = 0;
  float stableSum = 0.0F;
  const bool haveRows = prevArgmaxValid_ && !argmax.empty();
  const bool haveEntropy = haveRows && entropies.size() >= argmax.size();
  if (haveRows) {
    for (size_t i = 0; i < argmax.size(); ++i) {
      if (argmax[i] == prevArgmax_[i]) {
        ++stableCount;
        if (haveEntropy)
          stableSum += entropies[i];
      }
    }
    outcome.stableFraction =
        static_cast<float>(stableCount) / argmax.size();
    outcome.stableMeanEntropy =
        haveEntropy && stableCount ? stableSum / stableCount : 0.0F;
  }
  const bool mature = config_.stableExitFraction <= 0.0F ||
                      (haveRows && outcome.stableFraction >=
                                       config_.stableExitFraction);
  bool stable;
  float liveFraction = outcome.stableFraction;
  bool prefixScope =
      config_.prefixExit && !argmax.empty() &&
      outcome.stopIndex < schedule_.canvasLength;
  if (prefixScope) {
    // Stable over [0, firstStop): the rows past a stop token pad out on
    // commit, so their churn must not hold the canvas alive — but only a
    // mostly-settled canvas may commit an early stop. Drift mode judges
    // the live prefix alone: dead rows past the stop churn forever, so a
    // whole-canvas fraction would never settle.
    if (config_.driftExit) {
      uint32_t prefixStable = 0;
      if (prevArgmaxValid_)
        for (uint32_t i = 0; i < outcome.stopIndex; ++i)
          prefixStable += argmax[i] == prevArgmax_[i];
      liveFraction = outcome.stopIndex
          ? static_cast<float>(prefixStable) / outcome.stopIndex
          : 1.0F;
      stable = prevArgmaxValid_ &&
               liveFraction >= config_.stableExitFraction;
    } else {
      stable = mature && prevArgmaxValid_ &&
               std::equal(prevArgmax_.begin(),
                          prevArgmax_.begin() + outcome.stopIndex,
                          argmax.begin());
    }
  } else if (config_.prefixExit && !argmax.empty()) {
    // No stop token yet: the whole canvas is the prefix.
    stable = prevArgmaxValid_ &&
             std::equal(prevArgmax_.begin(), prevArgmax_.end(),
                        argmax.begin());
  } else {
    stable = stats.argmaxStable;
  }
  outcome.argmaxStable = stable;
  DiffusionSampler::StepStats gate = stats;
  if (prefixScope && entropies.size() >= outcome.stopIndex &&
      outcome.stopIndex > 0) {
    // The mean entropy over [0, firstStop) — the kernel's stats entry is
    // whole-canvas, and the tail's noise would never let it settle.
    float sum = 0.0F;
    for (uint32_t i = 0; i < outcome.stopIndex; ++i)
      sum += entropies[i];
    gate.meanEntropy = sum / static_cast<float>(outcome.stopIndex);
  } else if (prefixScope && outcome.stopIndex == 0) {
    gate.meanEntropy = 0.0F;
  }
  // Fraction-settled exit (no stop yet): when the argmax-stable share of
  // the canvas clears the configured fraction, the churning remainder
  // commits at argmax — the same tokens an exhausted schedule would emit.
  // The confident-rows check applies unless driftExit waives it.
  if (!stable && !prefixScope && config_.stableExitFraction > 0.0F &&
      stableCount && outcome.stableFraction >= config_.stableExitFraction &&
      (config_.driftExit ||
       (haveEntropy &&
        outcome.stableMeanEntropy <= schedule_.confidenceThreshold))) {
    stable = true;
    // The sampler's own confidence clause stays satisfied on a drift exit.
    gate.meanEntropy = config_.driftExit ? 0.0F : outcome.stableMeanEntropy;
  }
  // Drift mode: a stable canvas commits its argmax regardless of the rows'
  // confidence — the same tokens an exhausted schedule would emit. The
  // live fraction is prefix-local when a stop bounds the canvas.
  if (config_.driftExit && stable && stableCount &&
      liveFraction >= config_.stableExitFraction)
    gate.meanEntropy = 0.0F;
  outcome.exited = sampler_.earlyExit(gate, stable);
  if (!argmax.empty()) {
    std::copy(argmax.begin(), argmax.end(), prevArgmax_.begin());
    prevArgmaxValid_ = true;
  }
  // The speculation gate tracks the kernel's whole-canvas flag, not the
  // prefix signal — see speculationArmed().
  lastStable_ = stats.argmaxStable;
  if (outcome.exited)
    finished_ = true;
  else if (drained_ == current_.count && !nextStep_)
    finished_ = true;
  return outcome;
}

} // namespace richengine::model
