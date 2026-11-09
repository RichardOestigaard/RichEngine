#pragma once

#include "engine/memory/ElasticMemory.hpp"
#include "engine/memory/MemoryGovernor.hpp"
#include "engine/wire/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace richengine::engine {

// How long after reclaim last gave back a chunk of the elastic memory the
// control pass waits before it takes one back: a host short once is likely
// short again soon, and each chunk that goes moves and evicts experts, and
// each that comes back is refilled from the SSD.
inline constexpr double kElasticGrowthCooldownMilliseconds = 30000.0;

// Emits only transitions; retry counts and queue depth do not produce logs.
class MemoryStatusReporter final {
public:
  [[nodiscard]] std::string update(const ResourceWaitSnapshot &wait,
                                    bool hostGrowthAllowed);
private:
  unsigned state_ = 0;
};

// The engine's memory control between commands: it releases the weights of
// an engine left without a request (NativeRuntime::releaseIdleWeights),
// records the system's pressure, logs a change in what requests wait for,
// and runs the reclaim pass the pressure policy asks for, telling the
// governor what it found.
//
// A registered elastic memory (setElasticMemory: the expert store's chunks
// above its floor) joins the pass: a critical directive takes it straight to
// its floor, and a directive the engine's own reclaim cannot meet takes its
// chunks one at a time while a real shortage presses (the directive is
// critical, or a request waits for memory — a speculative warning pass with
// nothing waiting never shrinks it). Upstream's order puts the store after
// the cache nothing in use holds and before what is in use, inside
// Engine::reclaimMemory; this tree's seam sits a level up, so the store
// gives only once the engine's reclaim reports Exhausted, which keeps it
// resident longer but never blocks a directive the engine meets alone.
//
// A pass the policy asks nothing of takes back one chunk while memory
// allows it: no request waits for memory, the host lets growth go on
// (MemoryGovernorSnapshot::hostGrowthAllowed), and
// kElasticGrowthCooldownMilliseconds have passed since the pass first found
// a chunk given back. The chunk takes from the host only what its headroom
// holds beyond the recovery margin and from the engine only the room its
// limit has free — upstream also lets an idle engine trade ordinary cache
// for what the chunk lacks (NativeRuntime::reclaimForExperts), an exchange
// the engine owns and this seam cannot reach, so here a chunk waits for
// free room rather than make it.
class MemoryControl final {
public:
  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop) noexcept
      : governor_(governor), backend_(backend), loop_(loop) {}

  // The elastic memory the reclaim passes shrink and regrow, registered
  // once the model's adapter exists; it stays the model's to free. Null
  // runs the passes without one, as a model without streamed experts does.
  void setElasticMemory(ElasticMemory *elastic) noexcept { elastic_ = elastic; }

  // One memory-control pass at a command-free point. True while transfers
  // in flight hold back part of the target: the transport runs it again at
  // the next command-free point.
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  // Records when a pass first finds a chunk of the elastic memory that
  // reclaim gave back (its own releases and the rest alike).
  void noteElasticShrinks(double nowMilliseconds);
  // The store's chunks above the floor toward `shortfallBytes`:
  // releaseToFloor() on a critical directive, else releaseAboveFloor() until
  // the shortfall is covered or the floor is reached. Returns the bytes
  // given back.
  uint64_t releaseElastic(bool toFloor, uint64_t shortfallBytes);
  void growElastic(const MemoryGovernorSnapshot &memory, bool requestWaiting,
                   double nowMilliseconds);

  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  MemoryStatusReporter reporter_;
  MemoryPressurePolicy policy_;
  ElasticMemory *elastic_ = nullptr;
  uint64_t elasticShrinks_ = 0;
  std::optional<double> elasticShrinkMilliseconds_;
};

} // namespace richengine::engine
