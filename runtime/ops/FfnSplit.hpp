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
// ops::AneFfn implements it over the private MIL/surfaces path, a command of
// enough rows as two pipelined steps of half the rows each;
// ops::CoreAiFfn over the public CoreAI framework's encode(), one step only.
class FfnSplit {
public:
  // A command's chunk, or each step of a pipelined one, runs from
  // kMinimumRows to the prefill token budget.
  static constexpr uint32_t kMinimumRows = 512;
  static constexpr uint32_t kMaximumRows = RICHENGINE_PREFILL_TOKEN_BUDGET;

  // The kinds of command: one step, or two pipelined steps.
  enum class Steps : uint8_t { One, Two };
  // The agent's units of the units of intermediate channels in a command of
  // each kind; none where the split takes no command of that kind.
  struct Shares final {
    uint32_t oneStep = 0, twoSteps = 0;
    [[nodiscard]] uint32_t most() const noexcept { return oneStep > twoSteps ? oneStep : twoSteps; }
  };
  // The least rows of a command the split takes in one step, and of one it
  // takes in two, from which every larger command runs in two; none of
  // either kind it takes no command of.
  struct Rows final {
    uint32_t oneStep = 0, twoSteps = 0;
  };

  virtual ~FfnSplit() = default;

  // The fraction of intermediate channels the agent takes.
  [[nodiscard]] virtual double share() const = 0;
  // What Metal allocated for the split.
  [[nodiscard]] virtual uint64_t allocatedBytes() const = 0;
  // The least rows of a command the split takes, of either kind.
  [[nodiscard]] virtual uint32_t minimumRows() const = 0;
  virtual void setMinimumRows(uint32_t rows) = 0;
  // The least rows of a command of each kind the split takes; a one-kind
  // split takes `oneStep` alone (ops::AneFfn implements both).
  virtual void setRows(Rows rows) { setMinimumRows(rows.oneStep); }
  // Pauses the split for `pause` once it loses to the GPU alone, and stops
  // it once a probe loses too (ane_ffn::Pause): `breaker` judges each command
  // finish() finds split. Until this is called, none does.
  virtual void setBreaker(ane_ffn::Breaker breaker,
                        AwakeClock::duration pause = ane_ffn::Pause::kLength) = 0;

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

  // Whether the split takes a command of `rows` rows.
  [[nodiscard]] virtual bool splits(uint32_t rows) const = 0;
  // The kind of command a chunk of `rows` rows runs.
  [[nodiscard]] virtual Steps steps(uint32_t) const noexcept { return Steps::One; }
  // The rows a command's first step holds (rows for one-step implementations).
  [[nodiscard]] virtual uint32_t firstStepRows(uint32_t rows) const noexcept { return rows; }
  // Whether the split stopped, and why.
  [[nodiscard]] virtual bool retired() const = 0;
  [[nodiscard]] virtual std::string reason() const = 0;

  // Starts encoding a one-step command; false once the split stopped.
  bool begin() { return begin(Steps::One); }
  // Starts encoding a command of `steps` steps; false once the split stopped.
  virtual bool begin(Steps steps) = 0;
  // A step's FFN of `rows` rows, encoded in layer order; each step adds its
  // layers in order with that step's own rows, and join() comes between
  // steps' adds.
  virtual void add(metal::CommandGraph &graph, uint32_t layer, const PrefillFfnBuffers &ffn,
                   metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t rows) = 0;
  // Encodes the join of the step added last, if not joined. No-op default.
  virtual void join(metal::CommandGraph &) {}
  // Starts the agent's evaluations of the command encoded since begin(),
  // then submits it. A submission that throws stops the split first.
  [[nodiscard]] virtual metal::CommandTicket commit(const metal::CommandGraph &graph,
                                                    metal::CommandCompletion completion) = 0;
  // After the command committed last has completed, in `timing`: whether its
  // outputs hold the split's FFN. False when its agent work failed, which
  // stops the split; the caller runs the chunk again on the GPU alone.
  [[nodiscard]] virtual bool finish(const metal::CommandTiming &timing) = 0;

  // The engine's idle release and the load that undoes it
  // (engine::ReleasableMemory). See the implementations.
  [[nodiscard]] virtual bool release() = 0;
  virtual void restore() noexcept = 0;
};

} // namespace richengine::ops
