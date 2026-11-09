#include "engine/memory/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <algorithm>
#include <optional>
#include <sstream>

namespace richengine::engine {

std::string MemoryStatusReporter::update(const ResourceWaitSnapshot &wait,
                                         bool hostGrowthAllowed) {
  const unsigned state = (!hostGrowthAllowed ? 1u : 0u) |
                         (wait.memory ? 2u : 0u) |
                         (wait.suspended ? 4u : 0u) |
                         (wait.draining ? 8u : 0u) |
                         (wait.heldBehindRefusal ? 16u : 0u);
  if (state == state_)
    return {};
  state_ = state;
  if (!state)
    return "Memory: growth available; resource wait cleared";
  std::ostringstream out;
  out << "Memory: growth " << (hostGrowthAllowed ? "available" : "paused")
      << "; waiting=" << wait.memory << "; suspended=" << wait.suspended;
  if (wait.heldBehindRefusal)
    out << "; held=" << wait.heldBehindRefusal;
  if (wait.draining)
    out << "; waiting for resident requests to finish";
  return out.str();
}

bool MemoryControl::run(MemoryPressure pressure) {
  loop_.releaseIdleWeights();
  governor_.setPressure(pressure);
  const double now = loop_.monotonicMilliseconds();
  static_cast<void>(backend_.refreshMemoryStats());
  const MemoryGovernorSnapshot memory = governor_.snapshot();
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    writeStderrLine(diagnostic);
  // Requests held back by a refusal wait for memory too, the refused one
  // included while a pass defers it.
  const bool requestWaiting =
      wait.memory || wait.suspended || wait.heldBehindRefusal;
  const std::optional<MemoryReclaimDirective> directive =
      policy_.update(memory, now, requestWaiting);
  noteElasticShrinks(now);
  if (!directive) {
    growElastic(memory, requestWaiting, now);
    return false;
  }
  MemoryReclaimResult reclaim = loop_.reclaimMemory(*directive);
  // The elastic memory's chunks above its floor go once the engine's own
  // reclaim has nothing left to give, and only to a real shortage: a
  // critical directive takes them all, and a directive a request waits on
  // takes them until its target is met. A pass that defers to transfers in
  // flight does not touch them: what the transfer holds back ranks below
  // the store, and the pass that continues once it lands takes it first.
  if (elastic_ &&
      (directive->critical ||
       (requestWaiting && reclaim.outcome == ReclaimOutcome::Exhausted))) {
    const uint64_t shortfall =
        directive->targetBytes -
        std::min(directive->targetBytes, reclaim.releasedBytes);
    reclaim.releasedBytes +=
        releaseElastic(directive->critical, shortfall);
    // A chunk the store gave counts toward the target like the engine's own.
    if (!directive->critical && directive->targetBytes &&
        reclaim.releasedBytes >= directive->targetBytes)
      reclaim.outcome = ReclaimOutcome::Met;
  }
  noteElasticShrinks(now);
  policy_.reclaimed(*directive, reclaim);
  governor_.reclaimed(reclaim.outcome);
  static_cast<void>(backend_.refreshMemoryStats());
  // What transfers held back continues at the next command-free point.
  return reclaim.outcome == ReclaimOutcome::Pending;
}

void MemoryControl::noteElasticShrinks(double nowMilliseconds) {
  if (!elastic_)
    return;
  const uint64_t shrinks = elastic_->shrinkCount();
  if (shrinks == elasticShrinks_)
    return;
  elasticShrinks_ = shrinks;
  elasticShrinkMilliseconds_ = nowMilliseconds;
}

uint64_t MemoryControl::releaseElastic(bool toFloor, uint64_t shortfallBytes) {
  if (toFloor)
    return elastic_->releaseToFloor();
  uint64_t released = 0;
  while (released < shortfallBytes) {
    const uint64_t chunk = elastic_->releaseAboveFloor();
    if (!chunk)
      break;
    released += chunk;
  }
  return released;
}

void MemoryControl::growElastic(const MemoryGovernorSnapshot &memory,
                                bool requestWaiting, double now) {
  if (!elastic_ || requestWaiting || !memory.hostGrowthAllowed ||
      (elasticShrinkMilliseconds_ &&
       now - *elasticShrinkMilliseconds_ < kElasticGrowthCooldownMilliseconds))
    return;
  const uint64_t bytes = elastic_->growthBytes();
  if (!bytes)
    return;
  // Free room only: the host's headroom beyond the recovery margin, so the
  // chunk does not bring it back to the margin where reclaim would take the
  // chunk again, and the engine's headroom under its limit. Upstream lets an
  // idle engine exchange cache for what the chunk lacks of either; this
  // seam cannot reach it (MemoryControl's class comment), so the chunk
  // waits for room rather than make it.
  const uint64_t spare =
      memory.hostHeadroomBytes -
      std::min(memory.hostHeadroomBytes, kHostRecoveryMarginBytes);
  if (std::min(spare, memory.headroomBytes) < bytes)
    return;
  if (elastic_->growTowardCap(governor_.allocationAdmission()))
    static_cast<void>(backend_.refreshMemoryStats());
}

} // namespace richengine::engine
