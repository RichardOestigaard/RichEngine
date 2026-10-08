#pragma once

#include "model/DiffusionGemma.hpp"
#include "ops/PagedKv.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace richengine::model {

// Host-side decisions of the DiffusionGemma denoising loop, mirrored against
// the canvas kernels' device work. The kernels produce per-step entropy,
// samples and argmax rows on the GPU; this code owns the host policy: the
// accept mask, the early-exit counters, the commit truncation and how many
// canvases the request's remaining budget admits. It is CPU-only so the
// engine tests can drive it without a device.
class DiffusionSampler final {
public:
  explicit DiffusionSampler(DiffusionSchedule schedule);

  // The canvas kernel's decision, on the host: position i is accepted iff
  // the exclusive prefix sum of the entropies sorted ascending, evaluated at
  // i's rank, stays within the bound (the cheapest positions first).
  [[nodiscard]] std::vector<bool>
  acceptMask(std::span<const float> entropies) const;

  [[nodiscard]] float temperature(uint32_t step) const noexcept {
    return schedule_.temperature(step);
  }

  // Per-step record the device's stats pair feeds: {mean entropy over the
  // canvas, every row's argmax equal to the previous step's}.
  struct StepStats final {
    float meanEntropy = 0.0F;
    bool argmaxStable = false;
  };
  // One step's outcome: early exit when the argmax canvas has stayed
  // identical for stabilityThreshold consecutive steps while the mean
  // entropy is below confidenceThreshold. The first step never exits.
  [[nodiscard]] bool earlyExit(const StepStats &stats) noexcept;
  // Same, with the stability signal supplied by the caller: the kernel's
  // whole-canvas flag, or the step driver's first-stop-bounded prefix
  // compare (RICHENGINE_CANVAS_PREFIX_EXIT).
  [[nodiscard]] bool earlyExit(const StepStats &stats, bool stable) noexcept;

  // Index of the first stop token in `tokens`, or tokens.size() when none.
  // commit() shares it; the step driver scans each drained argmax with it.
  [[nodiscard]] static uint32_t
  firstStop(std::span<const uint32_t> tokens,
            std::span<const uint32_t> stopTokens) noexcept;

  // The committed canvas: argmax tokens with everything past the first stop
  // token padded. `emitted` counts how many tokens the stream takes (all of
  // them, so the emitted stream and the KV span stay aligned); `stop` marks
  // a stop token among them.
  struct Commit final {
    std::vector<uint32_t> tokens;
    bool stop = false;
    // Index of the first stop token in `tokens`, canvasLength when none.
    uint32_t stopIndex = 0;
  };
  [[nodiscard]] Commit
  commit(std::span<const uint32_t> argmax,
         std::span<const uint32_t> stopTokens) const;

  // Whole-request state: the running canvas count, tokens already emitted
  // and the request's remaining budget; the previous step's argmax rows feed
  // the accept kernel's argmax_prev.
  struct Session final {
    uint32_t canvases = 0;
    uint64_t emittedTokens = 0;
    bool prevArgmaxValid = false;
  };

  // How many tokens this canvas may emit (the whole canvas while the budget
  // allows). Zero ends the request by length.
  [[nodiscard]] uint32_t emitBudget(uint64_t emittedTokens,
                                    uint32_t maxNewTokens) const noexcept;
  // True when the committed canvas ends the request: a stop token within the
  // emitted prefix.
  [[nodiscard]] bool finished(const Commit &commit,
                              uint32_t emitted) const noexcept;

  // Canvas scratch pages: one page per kPageTokens canvas positions, appended
  // after the prefix's pages in the step's page table.
  [[nodiscard]] uint32_t canvasPages() const noexcept;
  // The canvas step's page table: the request's `prefixPages` entries, then
  // the scratch extent's pages (its entries are extentBase + index).
  [[nodiscard]] std::vector<RichKvPage>
  canvasPageTable(std::span<const RichKvPage> prefixEntries,
                  uint64_t scratchExtentBase) const;

  [[nodiscard]] const DiffusionSchedule &schedule() const noexcept {
    return schedule_;
  }

private:
  DiffusionSchedule schedule_;
  uint32_t stableSteps_ = 0;
  bool firstStep_ = true;
};

// The host side of one canvas's chunked denoise (RuntimeDiffusion.mm).
// Steps descend from maxDenoisingSteps to 1; up to stepsPerCommand
// consecutive steps encode into one command buffer, each writing a
// per-slot stats ring entry and a per-slot argmax ring buffer, so the
// host drains every slot after a single wait. The driver owns the encode
// plan, the exit decision and the commit-tail marker — CPU-only, so the
// engine tests drive it without a device.
class CanvasStepDriver final {
public:
  static constexpr uint32_t kMaxStepsPerCommand = 4;

  struct Config final {
    // RICHENGINE_CANVAS_STEPS_PER_CMD: clamped to 1..kMaxStepsPerCommand.
    uint32_t stepsPerCommand = 1;
    // RICHENGINE_CANVAS_PREFIX_EXIT: stability judged on the argmax prefix
    // bounded by this step's first stop token, not the whole canvas.
    bool prefixExit = false;
    // RICHENGINE_CANVAS_COMMIT_TAIL: the schedule's last step (1) encodes
    // without the accept dispatch — its canvas, stats and renoise are dead;
    // the row-stats argmax is the commit.
    bool commitTail = false;
    // RICHENGINE_CANVAS_EXIT_STABLE: fraction-settled exit. Renoised rows
    // churn forever on open-ended prompts, so the paper's whole-canvas
    // mean-entropy gate is unreachable there — when at least this fraction
    // of the canvas is argmax-stable and the stable rows' mean entropy sits
    // under the confidence threshold, the churning tail commits at argmax,
    // identical to exhausting the schedule. 0 disables.
    float stableExitFraction = 0.0F;
    // RICHENGINE_CANVAS_EXIT_DRIFT (default 0; "1" enables): drops the
    // confidence requirement from the fraction exit — the commit reads
    // argmax either way, so a fraction-settled canvas emits the same tokens
    // it would at the cap; only the still-churning minority is frozen at
    // its current argmax. A speed/quality trade the confidence gate would
    // otherwise never allow.
    bool driftExit = false;
  };

  // One command's encode plan: steps firstStep, firstStep-1, ... for
  // `count` ring slots. commitTail marks the last slot as step 1 encoded
  // commit-only (no canvas_entropy_accept, no stats slot write).
  struct Chunk final {
    uint32_t firstStep = 0;
    uint32_t count = 0;
    bool commitTail = false;
  };

  CanvasStepDriver(const DiffusionSchedule &schedule, Config config,
                   std::span<const uint32_t> stopTokens);

  // The next chunk to encode; consumes it from the schedule. Empty
  // (count == 0) once finished(). Throws logic_error when the previous
  // chunk was never fully drained — the ring slots would alias.
  [[nodiscard]] Chunk nextChunk();

  struct Outcome final {
    // Early exit fired at this slot: drain stops, this slot's argmax is
    // the commit. Slots encoded past it in the same chunk were dead
    // compute.
    bool exited = false;
    // The stability signal used for the exit decision (whole-canvas or
    // first-stop prefix depending on the config).
    bool argmaxStable = false;
    // This slot was the commit-only encode; the schedule is done.
    bool scheduleEnd = false;
    // First stop token index in this slot's argmax, canvasLength if none.
    uint32_t stopIndex = 0;
    // Diagnostics for the fraction-settled gate: the share of rows whose
    // argmax held from the previous step and their mean entropy (0 when no
    // previous argmax exists).
    float stableFraction = 0.0F;
    float stableMeanEntropy = 0.0F;
  };
  // Feed one drained slot, in encode order. `stats` is the slot's ring
  // entry (ignored for a commit-tail slot, which wrote none); `argmax` is
  // the slot's argmax canvas readback — the ring slot, or the row-stats
  // argmax output for the tail slot. `entropies`, when it covers the
  // canvas, lets the prefix-exit gate use the mean entropy of the prefix
  // up to the first stop token instead of the whole-canvas mean — the
  // tail rows keep churning noise that is dead output. With
  // stableExitFraction set, the same spans drive the fraction-settled
  // exit: stable rows' share of the argmax plus their mean entropy.
  Outcome drain(uint32_t slot, const DiffusionSampler::StepStats &stats,
                std::span<const uint32_t> argmax,
                std::span<const float> entropies = {});

  [[nodiscard]] bool finished() const noexcept { return finished_; }
  // The last drained step's whole-canvas argmax was unchanged while the
  // canvas continues: a speculative commit prefill may arm on this slot's
  // padded snapshot (RICHENGINE_CANVAS_SPECULATIVE_PREFILL). Uses the
  // kernel's whole-canvas flag even under prefixExit — a merely
  // prefix-stable canvas can still change past the stop.
  [[nodiscard]] bool speculationArmed() const noexcept {
    return lastStable_ && !finished_;
  }

private:
  DiffusionSchedule schedule_;
  Config config_;
  std::vector<uint32_t> stopTokens_;
  DiffusionSampler sampler_; // the exit counters and their first-step rule
  std::vector<uint32_t> prevArgmax_;
  bool prevArgmaxValid_ = false;
  uint32_t nextStep_ = 0;
  Chunk current_;
  uint32_t drained_ = 0;
  bool finished_ = false;
  bool lastStable_ = false;
};

} // namespace richengine::model
