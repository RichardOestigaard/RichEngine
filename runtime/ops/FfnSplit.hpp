#pragma once

#include "metal/CommandGraph.hpp"
#include "ops/AneFfnCalibration.hpp"
#include "ops/Linear.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace richengine::ops {

// The prefill FFN's dense-channel split with the Neural Engine: the GPU runs
// the leading intermediate channels of each layer and the agent runs the
// rest, with the GPU's staging of the agent's weights a layer ahead and its
// join of the agent's partial output, all inside the one prefill command.
// ops::AneFfn implements it over the private MIL/surfaces path;
// ops::CoreAiFfn over the public CoreAI framework's encode().
class FfnSplit {
public:
  // A chunk runs from kMinimumRows to the prefill token budget.
  static constexpr uint32_t kMinimumRows = 512;
  static constexpr uint32_t kMaximumRows = RICHENGINE_PREFILL_TOKEN_BUDGET;

  virtual ~FfnSplit() = default;

  // The fraction of intermediate channels the agent takes.
  [[nodiscard]] virtual double share() const = 0;
  // What Metal allocated for the split.
  [[nodiscard]] virtual uint64_t allocatedBytes() const = 0;
  // The least rows of a chunk the split takes.
  [[nodiscard]] virtual uint32_t minimumRows() const = 0;
  virtual void setMinimumRows(uint32_t rows) = 0;
  // Stops the split once it loses to the GPU alone: `breaker` judges each
  // command finish() finds split, and `lost` runs once when it trips.
  virtual void setBreaker(ane_ffn::Breaker breaker, std::function<void()> lost) = 0;

  // The commands finish() found split since the split was built, their
  // evaluations, and the agent's milliseconds over them.
  struct Served final {
    uint64_t commands = 0, evaluations = 0;
    double milliseconds = 0.0;
  };
  [[nodiscard]] virtual const Served &served() const = 0;

  // Checks the split against the GPU alone on `layers`; see the
  // implementations. Throws on failure, which stops the split.
  [[nodiscard]] virtual double verify(std::span<const SwiGluProjections> layers, const PrefillFfnBuffers &ffn,
                                      const std::array<metal::MetalBuffer, 2> &hidden) = 0;

  // Whether the split takes a chunk of `rows` rows.
  [[nodiscard]] virtual bool splits(uint32_t rows) const = 0;
  // Whether the split stopped, and why.
  [[nodiscard]] virtual bool retired() const = 0;
  [[nodiscard]] virtual std::string reason() const = 0;

  // Starts encoding a command; false once the split stopped.
  virtual bool begin() = 0;
  // Layer `layer`'s FFN of a chunk of `rows` rows, encoded in layer order.
  virtual void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                   metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) = 0;
  // Starts the agent's evaluations of the command encoded since begin(),
  // then submits it. A submission that throws stops the split first.
  [[nodiscard]] virtual metal::CommandTicket commit(const metal::CommandGraph &graph,
                                                    metal::CommandCompletion completion) = 0;
  // After the command committed last has completed: whether its outputs hold
  // the split's FFN. False when its agent work failed, which stops the
  // split; the caller runs the chunk again on the GPU alone.
  [[nodiscard]] virtual bool finish() = 0;

  // The engine's idle release and the load that undoes it
  // (engine::ReleasableMemory). See the implementations.
  [[nodiscard]] virtual bool release() = 0;
  virtual void restore() noexcept = 0;
};

} // namespace richengine::ops
