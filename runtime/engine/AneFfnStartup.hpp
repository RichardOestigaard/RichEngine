#pragma once

#include "engine/memory/MemoryPlan.hpp"
#include "ops/AneFfn.hpp"
#include "ops/AneFfnCalibration.hpp"
#include "ops/FfnSplit.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace richengine::engine {

// Whether the prefill FFN's Neural Engine split may run (richengine serve
// --disable-ane turns it off), and the split a dev tool runs instead of
// calibrating one.
struct AneFfnSetting final {
  // The ANE takes the whole channel units nearest `share` of them
  // (ops::ane_ffn::nearestUnits), over chunks of `minimumRows` rows or more.
  struct Given final {
    double share = 0.0;
    uint32_t minimumRows = ops::FfnSplit::kMinimumRows;
  };
  bool enabled = true;
  std::optional<Given> given;
  // A d1 decision model's split as given (RICHENGINE_ANE_D1): takes the
  // share whatever a calibration would say — measured up to 2x slower than
  // the GPU (docs/ANE_EXPERIMENTS.md), kept for the GPU work it frees.
  std::optional<Given> d1;
  // A CoreAI-backed split as given (RICHENGINE_COREAI_FFN): same contract,
  // the public framework's encode() in place of the private program —
  // measured ~30% faster per evaluation (docs/ANE_EXPERIMENTS.md). Takes
  // the share whatever a calibration would say, like `given`.
  std::optional<Given> coreai;
  // Calibrate this Mac's again: none of what it remembered of the model
  // serves (--recalibrate-ane, after an OS update or for a calibration whose
  // GPU alone this Mac no longer takes).
  bool recalibrate = false;

  // The setting of a dev tool's --ane-ffn-share and --ane-ffn-minimum-rows:
  // none calibrates the split, and a share of 0 runs the GPU alone. Throws
  // for least rows without a share.
  [[nodiscard]] static AneFfnSetting fromGiven(std::optional<double> share, std::optional<uint32_t> minimumRows);
};

// The split of a model, prepared: the split, taking every chunk its
// functions hold; its chunks as timed, if they were; and what
// AneFfn::verify returned.
struct AneFfnPrepared final {
  std::unique_ptr<ops::FfnSplit> split;
  ops::ane_ffn::ChunkTimings chunks;
  double error = 0.0;
};

// The model's side of the split, as functions, so that a test can stand in
// for it (engine::aneFfnModel() in RuntimeResources.hpp).
struct AneFfnModel final {
  // Whether the target has dense FFN layers, which the split could take; a
  // MoE target has none.
  bool dense = false;
  // Whether the target is a d1 decision model (one forward per judgment),
  // which a d1 given applies to.
  bool decisionModel = false;
  // Why the split does not take them; empty when it does.
  std::string unsupported;
  // Why this Mac cannot run the split (ane::unavailable), asked only of a
  // split that may run; none when it can.
  std::function<std::optional<std::string>()> unavailable;
  // The channel units of the split (ops::AneFfn::units), and the model's
  // layers for its two-step fit (ops::AneFfn::fits).
  uint32_t units = 0, layers = 0;
  // The split's Metal memory with the ANE taking `aneUnits` of them.
  std::function<uint64_t(uint32_t aneUnits)> plannedBytes;
  // The same for the CoreAI split; null falls back to plannedBytes.
  std::function<uint64_t(uint32_t aneUnits)> plannedBytesCoreAi;
  // Calibration's timings (ops::ane_ffn::Measurement::time), all but the
  // mixer, and the same of the split with the ANE taking `aneUnits`, on the
  // layers they timed, each beside the GPU alone read right before it.
  std::function<ops::ane_ffn::Timings()> time;
  std::function<ops::ane_ffn::Timing(uint32_t aneUnits)> timeAt;
  // The split that serves at `shares` in commands of `rows`, verified, with
  // its chunks timed (ops::ane_ffn::Measurement::chunks) when `timeChunks`. A
  // model a test plays prepares no split.
  std::function<AneFfnPrepared(ops::AneFfn::Shares shares,
                               ops::AneFfn::Rows rows, bool timeChunks)> prepare;
  // The same over the CoreAI framework (AneFfnSetting::coreai); null where
  // the machine lacks the framework or the model's shape no CoreAI asset
  // covers, which a coreai given reports as unavailable.
  std::function<AneFfnPrepared(uint32_t aneUnits)> prepareCoreAi;
  // This Mac's calibration of the split within `maxAneUnits` units, if one
  // is remembered; remembering one, true once kept.
  std::function<std::optional<ops::ane_ffn::Calibration>(uint32_t maxAneUnits)> recall;
  std::function<bool(uint32_t maxAneUnits, const ops::ane_ffn::Calibration &calibration)> remember;
  // Whether the Metal backend still serves.
  std::function<bool()> healthy;
};

// How a start's split came out: off as given, not taken by the model or this
// Mac, refused by the memory plan, gaining too little, failed, running, or
// set up once the engine's runtime has prefilled a full chunk on the GPU
// alone (finishAneFfn), and why or how. Startup logs it (startAneFfn,
// finishAneFfn).
struct AneFfnOutcome final {
  enum class Kind : uint8_t { Off, Unsupported, Refused, NoGain, Unavailable, Split, Pending };
  Kind kind = Kind::Off;
  std::string reason;
  // The context the engine serves without --max-context, and the one the
  // memory plan holds without the split. While the split may run, the first
  // is what the plan holds with the split this Mac's calibration of the model
  // takes, none if it found no gain, whatever the start makes of it, so that
  // every start of the model on this Mac serves the same; with no calibration
  // made, with the largest split the model could take. Otherwise both are the
  // same. For a split refused a --max-context, the first is the most context
  // a split leaves instead.
  uint32_t context = 0, contextWithout = 0;
};
// The kind's name in lowercase, as "no_gain".
[[nodiscard]] std::string_view aneFfnOutcomeName(AneFfnOutcome::Kind kind) noexcept;

// The split a start runs and the memory plan that sets its memory aside,
// both none without one, and the outcome. A start whose split is Pending
// holds the largest split's memory and sets the split up after its
// runtime's prefill of a full chunk (finishAneFfn): within `maxAneUnits`,
// the units the plan holds, from this Mac's `calibration` of the model as
// recalled, if one.
struct AneFfnStart final {
  std::unique_ptr<ops::FfnSplit> split;
  std::optional<EngineMemoryPlan> plan;
  AneFfnOutcome outcome;
  uint32_t maxAneUnits = 0;
  std::optional<ops::ane_ffn::Calibration> calibration;
  // The context the start was asked to leave, which the log line weighs.
  uint32_t requested = 0;
};

// The prefill FFN's Neural Engine split of `model` as `setting` allows,
// within the memory plan `planMemory` makes with the split's bytes set aside:
// one whose plan still holds `requestedContextTokens`, or with zero the
// automatic context (AneFfnOutcome::context). Every step is arithmetic but a
// given split's, which is set up here and takes the given least chunk, no
// breaker, and nothing remembered. Where the Neural Engine refused this
// Mac's split of the model, within the units the plan holds, twice, the GPU
// runs alone; otherwise the split is Pending: the plan holds the most units,
// and finishAneFfn sets it up once the engine's runtime has prefilled a full
// chunk on the GPU alone. Logs the outcome. A failure leaves the GPU alone,
// unless `cancelled` returns true, the wait for the ANE was interrupted or
// the backend no longer serves, which the failure's exception reports.
// Throws std::invalid_argument for a given split of no units or of least
// rows no function holds.
[[nodiscard]] AneFfnStart
startAneFfn(const AneFfnModel &model, const AneFfnSetting &setting, uint32_t requestedContextTokens,
            const std::function<EngineMemoryPlanResult(uint64_t aneFfnBytes)> &planMemory,
            const std::function<bool()> &cancelled);

// The split a start left Pending (startAneFfn), once the engine's runtime has
// prefilled a full chunk on the GPU alone in `prefillMilliseconds` of GPU
// time, and how it came out, which it logs; the context and plan stay the
// start's. It takes this Mac's calibration of the model, as recalled, unless
// the calibration's mixer reads more than a tenth longer than the one this
// start's prefill gives: one taken on a busier Mac. Else it is calibrated
// (AneFfnStartup.cpp), the mixer the rest of that prefill's layer, and a
// split it finds remembered, no split not. The split judges itself against
// the GPU alone as calibrated (ops::ane_ffn::Breaker), and a split that stops
// for losing keeps the calibration; the Neural Engine's refusals count
// (ops::ane_ffn::Calibration). Fails as startAneFfn does.
struct AneFfnFinished final {
  std::unique_ptr<ops::FfnSplit> split;
  AneFfnOutcome outcome;
};
[[nodiscard]] AneFfnFinished finishAneFfn(const AneFfnModel &model, const AneFfnStart &start,
                                        double prefillMilliseconds, const std::function<bool()> &cancelled);

} // namespace richengine::engine
