#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>

namespace richengine::engine {

// Memory that gives back while the engine is short and takes it again while
// it is ample, between commands: upstream's elastic expert store ported to
// the qwen4 tree (qwen4/model/ExpertStore.hpp's releaseAboveFloor(),
// releaseToFloor() and growTowardCap()), which the store's owner satisfies
// through an adapter so that the memory control reaches it without a
// reverse dependency on the store's type or namespace. Calls run on the
// engine thread at command-free points only, as the store's own contract
// requires (its guards return no memory while a command is in flight, so an
// adapter need not check one itself).
class ElasticMemory {
public:
  virtual ~ElasticMemory() = default;

  // The bytes the next growth would take, the next released chunk's: zero
  // while the memory is at its cap or cannot grow (while released, a
  // command in flight, or a failed load), which a pass then skips.
  [[nodiscard]] virtual uint64_t growthBytes() const noexcept = 0;
  // While memory is short: gives back the last live chunk above the floor,
  // the experts it holds moved to free slots below it and the coldest
  // evicted to make room. Returns the chunk's bytes; zero at the floor,
  // while released, or while a command is in flight. A reclaim that needs
  // more calls it again.
  virtual uint64_t releaseAboveFloor() = 0;
  // While memory is critically short: gives back every chunk above the
  // floor in one pass, as releaseAboveFloor() would one at a time. Returns
  // their bytes.
  virtual uint64_t releaseToFloor() = 0;
  // While memory is ample: takes back the next released chunk under
  // `admit`, its slots free. The refusal leaves the chunk released and is
  // returned; the memory stays as it is at the cap, while released, while a
  // command is in flight and after a failed load.
  virtual metal::AllocationResult
  growTowardCap(const metal::AllocationAdmission &admit) = 0;
  // How many chunks have been given back so far: it moves whenever memory
  // the control pass does not itself release shrinks (an idle release
  // included), so the growth cooldown restarts from the last change the
  // pass saw rather than only from its own releases.
  [[nodiscard]] virtual uint64_t shrinkCount() const noexcept = 0;
};

} // namespace richengine::engine
