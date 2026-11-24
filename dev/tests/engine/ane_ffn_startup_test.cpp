// What a start and the set-up it leaves pending make of the prefill FFN's Neural Engine split
// (engine/AneFfnStartup.cpp) on a model the test plays: each outcome, the memory plan it adopts, the automatic
// context, the calibration it remembers or takes again and the line it logs.

#include "TestChecks.hpp"
#include "TestModel.hpp"
#include "TestStderr.hpp"
#include "ane/Program.hpp"
#include "engine/AneFfnStartup.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace richengine;
using namespace richengine::engine;
using Kind = AneFfnOutcome::Kind;
using richengine::ops::ane_ffn::Calibration;
using richengine::ops::ane_ffn::ChunkTimings;
using richengine::ops::ane_ffn::Timing;
using richengine::ops::ane_ffn::Timings;
using richengine::test::rejects;
using richengine::test::require;

constexpr std::string_view kSettingUp =
    "Setting up the Neural Engine FFN split (richengine serve --disable-ane keeps the FFN on the GPU).";

DeviceCapabilities device() {
  DeviceCapabilities result;
  result.deviceName = "test";
  result.appleGpuFamily = 9;
  result.macosMajor = 27;
  result.macosMinor = 4;
  result.physicalMemoryBytes = 16 * kGiB;
  result.recommendedMaxWorkingSetBytes = 12 * kGiB;
  result.maxBufferLengthBytes = 8 * kGiB;
  result.maxThreadgroupMemoryBytes = 32 * 1024;
  result.maxThreadgroupWidth = 1024;
  result.hasUnifiedMemory = true;
  return result;
}

// The memory plan of a model of 4 GiB of weights with `bytes` set aside for the split.
EngineMemoryPlanResult planMemory(uint64_t bytes) {
  ModelMemoryProfile profile = test::modelMemoryProfile(2 * kGiB, kGiB, kGiB);
  profile.footprint.aneFfnBytes = bytes;
  return evaluateEngineMemoryPlan(device(), profile, 0);
}
uint32_t contextWith(uint64_t bytes) {
  const EngineMemoryPlanResult result = planMemory(bytes);
  return result.plan ? result.plan->maximumContextTokens() : 0;
}

// `tokens` with its thousands grouped, as the log prints a context.
std::string grouped(uint32_t tokens) {
  std::string digits = std::to_string(tokens);
  for (size_t at = digits.size(); at > 3; at -= 3) digits.insert(at - 3, ",");
  return digits;
}

// An M5 Max's timings of Qwen3.8-27B's 34 units, which choose 8 of them in one step (ane_ffn_calibration_test's
// kMeasured): no mid share, so the fit's two-share case. Its half step at the high share (two steps' H) is set at
// 70 ms, three times the M5 Max's 22.3, so that two steps gain only beside a mixer of more than about 2 ms: none
// beside the 2 ms a Fake's prefill gives by default. The mixer is the set-up's to read, not the timing's.
Timings measured() { return {{14.0 / 34, 11.699, 23.484}, {27.0 / 34, 5.033, 44.683}, 18.990, 70.0}; }

// The chunks of a split timed over its functions of 2048 down to 512 rows: the GPU alone 5 ms at 512 rows and 20 at
// 2048, each function's split layer 0.8 of the GPU's at its rows but function 2048's at `most` ms and function 640's
// at 5.5, which splits its chunks from 553 rows (ane_ffn_calibration_test).
ChunkTimings timedChunks(double most) {
  const auto gpu = [](uint32_t rows) { return 5.0 + (rows - 512) * 15.0 / 1536; };
  ChunkTimings chunks;
  chunks.gpu = {{{512, gpu(512)}, {2048, gpu(2048)}}};
  for (uint32_t rows = 2048; rows >= 512; rows -= 128)
    chunks.functions.push_back({rows == 512 ? 512 : rows - 127, rows,
                                rows == 2048 ? most : rows == 640 ? 5.5 : 0.8 * gpu(rows)});
  return chunks;
}

// A model the test plays: a dense one of 34 units and five layers, each `unitBytes` of the split's memory, what its
// timings and its prepared split return, and what the start and the set-up asked of it.
struct Fake {
  bool dense = true;
  std::string unsupported;
  std::optional<std::string> unavailable;
  uint64_t unitBytes = 256 * kMiB;
  uint32_t layers = 64;
  // The rest of a full chunk's layer of the model's prefill on the GPU
  // alone, beside its FFN.
  double mixer = 2.0;
  std::function<Timings()> time = measured;
  // The timing at the units the set-up first chooses: a GPU alone of none
  // leaves its refitted model no choice, so the first share serves.
  std::function<Timing(uint32_t)> timeAt = [](uint32_t aneUnits) {
    return Timing{static_cast<double>(aneUnits) / 34, 0.0, 20.0, 0.0};
  };
  std::function<AneFfnPrepared(uint32_t, bool)> prepare = [](uint32_t, bool timed) {
    AneFfnPrepared prepared;
    if (timed) prepared.chunks = timedChunks(15.5);
    prepared.error = 0.025;
    return prepared;
  };
  // This Mac's calibration as the model keeps it, whatever the most units; the most units the start recalled one
  // for, and the calibration it remembered and under which most units.
  std::optional<Calibration> stored, remembered;
  std::optional<uint32_t> recalled, rememberedMost;
  bool healthy = true;
  // The calls the start and the set-up made: of unavailable() and time(), and of prepare() with its arguments.
  uint32_t asked = 0, timings = 0;
  std::optional<std::pair<uint32_t, bool>> prepared;

  [[nodiscard]] uint64_t bytes(uint32_t aneUnits) const { return aneUnits * unitBytes; }
  // What the plan holds with the largest split the model could take, the most units whose plan holds any: the
  // context a start that leaves its split pending assumes, whatever the set-up makes of it.
  [[nodiscard]] uint32_t largestSplitContext() const {
    for (uint32_t aneUnits = 33; aneUnits > 0; --aneUnits)
      if (const uint32_t context = contextWith(bytes(aneUnits))) return context;
    return 0;
  }

  AneFfnModel model() {
    AneFfnModel result;
    result.dense = dense;
    result.unsupported = unsupported;
    result.unavailable = [this] {
      ++asked;
      return unavailable;
    };
    result.units = 34;
    result.layers = layers;
    result.plannedBytes = [this](uint32_t aneUnits) { return bytes(aneUnits); };
    result.time = [this] {
      ++timings;
      return time();
    };
    result.timeAt = [this](uint32_t aneUnits) { return timeAt(aneUnits); };
    result.prepare = [this](ops::AneFfn::Shares shares, ops::AneFfn::Rows, bool timeChunks) {
      prepared = {shares.oneStep, timeChunks};
      return prepare(shares.oneStep, timeChunks);
    };
    result.recall = [this](uint32_t maxAneUnits) {
      recalled = maxAneUnits;
      return stored;
    };
    result.remember = [this](uint32_t maxAneUnits, const Calibration &calibration) {
      rememberedMost = maxAneUnits;
      stored = remembered = calibration;
      return true;
    };
    result.healthy = [this] { return healthy; };
    return result;
  }
};

struct Started {
  AneFfnStart start;
  std::string log;
};
Started start(Fake &fake, AneFfnSetting setting = {}, uint32_t context = 0, std::function<bool()> cancelled = {}) {
  Started result;
  result.log = test::capturedStderr(
      [&] { result.start = startAneFfn(fake.model(), setting, context, planMemory, cancelled); });
  return result;
}

struct Finished {
  AneFfnFinished finished;
  std::string log;
};
Finished finish(Fake &fake, const AneFfnStart &start, double prefillMilliseconds,
                std::function<bool()> cancelled = {}) {
  Finished result;
  result.log = test::capturedStderr(
      [&] { result.finished = finishAneFfn(fake.model(), start, prefillMilliseconds, cancelled); });
  return result;
}

// The engine's prefill of a full chunk on the GPU alone that the set-up waited for, in milliseconds: each of the
// model's layers its FFN and the model's mixer beside it. The FFN is the GPU alone's full chunk as the calibration
// the start recalled timed it, so that the prefill gives the model's mixer beside it (a recalled calibration whose
// mixer reads more than a tenth longer calibrates again); else as time() times it, none where time() fails, as the
// finish's own timing then does.
double prefillMilliseconds(Fake &fake, const AneFfnStart &start) {
  if (start.calibration && start.calibration->aneUnits)
    return (start.calibration->gpu[1].milliseconds + fake.mixer) * fake.layers;
  try {
    return (fake.time().gpuAlone + fake.mixer) * fake.layers;
  } catch (const std::exception &) {
    return 0.0;
  }
}

// The start and the set-up it left pending, the set-up's prefill of a full
// chunk taking `prefill` of GPU time (prefillMilliseconds where zero): `finished`
// is empty where the start settled the split itself.
struct Served {
  Started started;
  Finished finished;
  // The outcome the whole start came to.
  [[nodiscard]] const AneFfnOutcome &outcome() const {
    return finished.finished.split || finished.log.size() ? finished.finished.outcome
                                                        : started.start.outcome;
  }
};
Served served(Fake &fake, AneFfnSetting setting = {}, uint32_t context = 0, double prefill = 0,
              std::function<bool()> cancelled = {}) {
  Served result;
  result.started = start(fake, setting, context, cancelled);
  if (result.started.start.outcome.kind == Kind::Pending)
    result.finished =
        finish(fake, result.started.start,
                   prefill ? prefill : prefillMilliseconds(fake, result.started.start), cancelled);
  return result;
}

bool has(const std::string &log, std::string_view text) { return log.find(text) != std::string::npos; }
// Whether `log` is one line that holds `text`.
bool line(const std::string &log, std::string_view text) {
  return has(log, text) && log.find('\n') == log.size() - 1;
}
bool nothingStarted(const Fake &fake, const Started &started) {
  return !fake.timings && !fake.prepared && !fake.recalled && !fake.remembered && !started.start.split &&
         !started.start.plan &&
         !has(started.log, kSettingUp);
}

// --disable-ane leaves the GPU alone, at the context the plan holds without the split, without asking whether this Mac
// could run it, and says so where the split could run.
void testOff() {
  Fake fake;
  Started started = start(fake, {.enabled = false});
  require(started.start.outcome.kind == Kind::Off && nothingStarted(fake, started) && !fake.asked &&
              started.start.outcome.context == contextWith(0) &&
              started.start.outcome.contextWithout == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone, as given."),
          "--disable-ane was not taken as given");
  fake.unsupported = "ANE FFN split needs affine Q4 projections or unrotated quantized GGUF tensors";
  started = start(fake, {.enabled = false});
  require(started.start.outcome.kind == Kind::Off && started.log.empty(),
          "--disable-ane was logged for a model the split does not take");
}

// A model of no dense FFN layers logs nothing; one whose layers the split does not take, or on a Mac without the
// private Neural Engine interface, says why. Each serves the context the plan holds without the split.
void testUnsupported() {
  Fake fake;
  fake.dense = false;
  Started started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && started.log.empty() &&
              started.start.outcome.context == contextWith(0),
          "a MoE model was logged");
  fake = {};
  fake.unsupported = "ANE FFN split needs affine Q4 projections or unrotated quantized GGUF tensors";
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && !fake.asked &&
              started.start.outcome.context == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone: " + fake.unsupported + "."),
          "a dense model the split does not take did not say why");
  fake = {};
  fake.unavailable = "AppleNeuralEngine lacks _ANEClient";
  started = start(fake);
  require(started.start.outcome.kind == Kind::Unsupported && nothingStarted(fake, started) && fake.asked == 1 &&
              started.start.outcome.context == contextWith(0) &&
              line(started.log, "The GPU runs the prefill FFN alone: AppleNeuralEngine lacks _ANEClient."),
          "a Mac without the Neural Engine's interface did not say so: " + started.log);
}

// A --max-context no split leaves: the GPU runs alone, nothing is timed or prepared, and a warning names the most
// any split leaves and the way out. One the GPU alone does not hold either fails the start on its own.
void testRefusedContext() {
  Fake fake;
  const uint32_t without = contextWith(0), most = contextWith(fake.bytes(1));
  require(most < without, "the split's memory does not come out of the KV cache");
  Started started = start(fake, {}, most + 1);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == most &&
              started.start.outcome.contextWithout == without && nothingStarted(fake, started),
          "a context no split leaves did not refuse the split");
  require(line(started.log, "Warning · Neural Engine FFN split off: --max-context " + std::to_string(most + 1) +
                                " cannot be held with it, and any split leaves at most ") &&
              has(started.log, "; pass --max-context " + std::to_string(most) +
                                   " or less to run it, or --disable-ane, which also silences this line."),
          "the refusal did not warn with the way out: " + started.log);
  started = start(fake, {}, without + 1);
  require(started.start.outcome.kind == Kind::Refused && nothingStarted(fake, started) && started.log.empty(),
          "a context the GPU alone does not hold was warned of");
  // A given share is refused on its own units.
  const uint32_t given = contextWith(fake.bytes(7));
  started = start(fake, {.given = AneFfnSetting::Given{0.2}}, given + 1);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == given &&
              has(started.log, ", and the given share leaves at most "),
          "a given share was not refused on its own units: " + started.log);
}

// Without --max-context a split whose fewest units leave no context is refused, with a warning, before anything is
// timed, and the engine serves the context the plan holds without it.
void testRefusedAutomatic() {
  Fake fake;
  fake.unitBytes = 16 * kGiB;
  const Started started = start(fake);
  require(started.start.outcome.kind == Kind::Refused && started.start.outcome.context == contextWith(0) &&
              nothingStarted(fake, started) &&
              line(started.log, "Warning · Neural Engine FFN split off: it leaves no memory for context; "
                                "--disable-ane silences this line."),
          "a split that leaves no context was not refused: " + started.log);
}

// Without --max-context a split the model can run waits for the runtime's prefill of a full chunk on the GPU alone:
// the start leaves it Pending, holds the plan of the most units the plan holds any context with, times nothing and
// says nothing, and recalls this Mac's calibration for the set-up to weigh.
void testPending() {
  Fake fake;
  const Started started = start(fake);
  const AneFfnStart &pending = started.start;
  require(pending.outcome.kind == Kind::Pending && !pending.split && pending.plan && pending.maxAneUnits &&
              !pending.calibration && !fake.timings && !fake.prepared && fake.recalled == pending.maxAneUnits &&
              pending.plan->breakdown().aneFfnBytes == fake.bytes(pending.maxAneUnits) &&
              pending.outcome.context == contextWith(fake.bytes(pending.maxAneUnits)) &&
              pending.outcome.context == fake.largestSplitContext() &&
              pending.outcome.contextWithout == contextWith(0) && started.log.empty(),
          "the split was not left to set up over the largest split's plan: " + started.log);
  // Only a start left Pending is set up.
  const Started off = start(fake, {.enabled = false});
  rejects([&] { static_cast<void>(finish(fake, off.start, 100.0)); }, "was not left to set up",
          "a start that left nothing pending was set up");
}

// The set-up the start left: the prefill's layer less its FFN's is the mixer two steps run beside, which the record
// keeps beside its own GPU alone's full chunk. Without --max-context the split takes its memory from the KV cache:
// the start's plan stands, the set-up remembers its calibration, and the engine serves the automatic context, which
// the log names beside the GPU's alone.
void testAutomaticContext() {
  Fake fake;
  const Started started = start(fake);
  require(started.start.outcome.kind == Kind::Pending, "the split was not left pending: " + started.log);
  const Finished finished = finish(fake, started.start, prefillMilliseconds(fake, started.start));
  const AneFfnOutcome &outcome = finished.finished.outcome;
  require(outcome.kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, true) &&
              fake.remembered && fake.remembered->aneUnits == 8 && fake.remembered->minimumRows == 553 &&
              fake.remembered->gpu[1].milliseconds == 20.0 &&
              std::abs(fake.remembered->mixer - 0.99) < 1e-9 &&
              fake.rememberedMost == fake.recalled,
          "the split of 8 units was not prepared and remembered with its mixer");
  require(started.start.plan && outcome.context == contextWith(fake.bytes(started.start.maxAneUnits)) &&
              outcome.context < contextWith(0) && outcome.contextWithout == contextWith(0) &&
              started.start.plan->maximumContextTokens() == outcome.context,
          "the plan with the split was not adopted at the automatic context");
  const size_t split = finished.log.find("Neural Engine FFN split at share 0.24 for chunks of 553 rows or more: 15.5 ms "
                                         "per 2048-row FFN layer against 20.0 on the GPU alone, 2.5% from the GPU alone "
                                         "on the Neural Engine's part (calibrated in ");
  require(finished.log.find(kSettingUp) < split && split != std::string::npos &&
              has(finished.log, "; context " + grouped(outcome.context) + " tokens (" +
                                    grouped(contextWith(0)) + " with --disable-ane)."),
          "the split was not set up and logged with both contexts: " + finished.log);
}

// Every start of the model on a Mac serves the automatic context of the plan the pending start holds, whatever the
// set-up makes of the split: the most units the model could take, and the memory a smaller split leaves stays in the
// cache. A start refused its split twice leaves the GPU alone, at the GPU alone's context.
void testDeterministicContext() {
  const auto noGain = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.0;
    return timings;
  };
  const auto slowChunks = [](uint32_t, bool) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  const auto fails = [](uint32_t, bool) -> AneFfnPrepared { throw std::runtime_error("an evaluation failed"); };
  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  const Calibration nine{9, 700, gpu};
  const uint32_t most = Fake{}.largestSplitContext();
  require(most && most < contextWith(Fake{}.bytes(1)), "the largest split holds no less context");
  struct Case {
    Fake fake;
    AneFfnSetting setting;
    Kind kind;
    uint32_t context;
  };
  std::vector<Case> cases(9);
  cases[1].fake.stored = nine;
  cases[2].fake.time = noGain;
  cases[3].fake.prepare = slowChunks;
  cases[4].fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  cases[5].fake.prepare = fails;
  cases[6].setting = {.given = AneFfnSetting::Given{0.2}};
  cases[7].fake.stored = nine;
  cases[7].fake.prepare = fails;
  cases[8].fake.stored = Calibration{};
  const auto split = [](uint32_t aneUnits) { return contextWith(Fake{}.bytes(aneUnits)); };
  const std::array<std::pair<Kind, uint32_t>, 9> expected{{{Kind::Split, most},
                                                           {Kind::Split, most},
                                                           {Kind::NoGain, most},
                                                           {Kind::NoGain, most},
                                                           {Kind::Unavailable, most},
                                                           {Kind::Unavailable, most},
                                                           {Kind::Split, split(7)},
                                                           {Kind::Unavailable, most},
                                                           {Kind::Unavailable, contextWith(0)}}};
  for (size_t index = 0; index < cases.size(); ++index) {
    Case &test = cases[index];
    const auto &[kind, context] = expected[index];
    const Served ran = served(test.fake, test.setting);
    const AneFfnOutcome &outcome = ran.finished.finished.split || ran.finished.finished.outcome.kind != Kind::Off
                                     ? ran.finished.finished.outcome
                                     : ran.started.start.outcome;
    const std::string log = ran.started.log + ran.finished.log;
    const bool named = has(log, "; context " + grouped(context) + " tokens (" + grouped(contextWith(0)) +
                                    " with --disable-ane)");
    require(outcome.kind == kind && outcome.context == context && outcome.contextWithout == contextWith(0) &&
                ran.started.start.plan.has_value() == (kind != Kind::Unavailable || context == most) &&
                named == (context != contextWith(0)),
            "start " + std::to_string(index) + " did not serve the pending plan's automatic context: " + log);
  }
  // A start that takes the calibration a set-up remembered serves the context that start served.
  Fake fake;
  const uint32_t first = served(fake).outcome().context;
  const Served again = served(fake);
  require(fake.prepared == std::pair(8u, false) && again.outcome().context == first,
          "a start that took the remembered calibration served another context");
  // With --max-context the start holds the context asked for, which the log does not name.
  fake = {};
  const Served started = served(fake, {}, 4096);
  require(started.outcome().kind == Kind::Split && started.started.start.plan &&
              started.started.start.plan->maximumContextTokens() >= 4096 && !has(started.finished.log, "with --disable-ane"),
          "a start given --max-context named the automatic context");
}

// With --max-context the split takes at most the units whose plan still holds it, and the set-up calibrates within
// them.
void testExplicitContext() {
  Fake fake;
  const uint32_t context = contextWith(fake.bytes(5));
  const Served ran = served(fake, {}, context);
  require(ran.started.start.outcome.kind == Kind::Pending && ran.started.start.maxAneUnits == 5 &&
              ran.started.start.plan && ran.started.start.plan->maximumContextTokens() >= context &&
              ran.outcome().kind == Kind::Split && fake.prepared == std::pair(5u, true) &&
              fake.remembered && fake.remembered->aneUnits == 5 && fake.recalled == 5u &&
              fake.rememberedMost == 5u && !has(ran.finished.log, "with --disable-ane"),
          "the split took more units than the plan holds with the context asked for");
}

// A failure of the timings or of the prepared split leaves the GPU alone, the pending plan as it was and nothing
// remembered, and warns; cancellation, an interrupted wait and a backend that stopped serving end the set-up.
void testFailures() {
  Fake fake;
  fake.time = []() -> Timings { throw std::runtime_error("calibration timings are not usable"); };
  Served ran = served(fake);
  require(ran.started.start.outcome.kind == Kind::Pending && ran.outcome().kind == Kind::Unavailable &&
              ran.started.start.plan && !ran.finished.finished.split && !fake.prepared && !fake.remembered,
          "a timing that failed did not leave the GPU alone");
  require(has(ran.finished.log, "Warning · Neural Engine FFN split unavailable (calibration timings are not usable); the "
                                "GPU runs the prefill FFN alone; context ") &&
              has(ran.finished.log, " with --disable-ane). --disable-ane silences this line."),
          "a timing that failed did not warn: " + ran.finished.log);
  fake = {};
  fake.prepare = [](uint32_t, bool) -> AneFfnPrepared {
    throw std::runtime_error("ANE FFN split of 640 rows failed (the Neural Engine's output or its scales were not "
                             "finite)");
  };
  ran = served(fake);
  require(ran.outcome().kind == Kind::Unavailable && !fake.remembered &&
              has(ran.finished.log, "unavailable (ANE FFN split of 640 rows failed"),
          "a split that failed verify() was not unavailable");
  // Chunks that were not timed are not read.
  fake.prepare = [](uint32_t, bool) { return AneFfnPrepared{}; };
  ran = served(fake);
  require(ran.outcome().kind == Kind::Unavailable && !fake.remembered &&
              has(ran.finished.log, "unavailable (the split's chunks were not timed)"),
          "a split of no chunks timed was not unavailable: " + ran.finished.log);

  bool cancelled = false;
  fake = {};
  fake.prepare = [&](uint32_t, bool) -> AneFfnPrepared {
    cancelled = true;
    throw std::runtime_error("the Metal command was not submitted");
  };
  rejects(
      [&] {
        const Started started = start(fake);
        static_cast<void>(finish(fake, started.start, 100.0, [&] { return cancelled; }));
      },
      "not submitted", "a cancelled set-up went on");
  fake = {};
  fake.prepare = [](uint32_t, bool) -> AneFfnPrepared { throw ane::Interrupted("the wait was interrupted"); };
  rejects(
      [&] {
        const Started started = start(fake);
        static_cast<void>(finish(fake, started.start, 100.0));
      },
      "interrupted", "an interrupted wait went on");
  fake = {};
  fake.prepare = [&](uint32_t, bool) -> AneFfnPrepared {
    fake.healthy = false;
    throw std::runtime_error("the backend stopped serving");
  };
  rejects(
      [&] {
        const Started started = start(fake);
        static_cast<void>(finish(fake, started.start, 100.0));
      },
      "stopped serving", "a set-up went on on a backend that stopped");
}

// No share predicted to gain enough, or no chunk timed to, leaves the GPU alone — and keeps nothing of it, so the
// next start calibrates again.
void testNoGain() {
  Fake fake;
  fake.time = [] {
    Timings timings = measured();
    timings.gpuAlone = 14.5;
    return timings;
  };
  Served ran = served(fake);
  require(ran.outcome().kind == Kind::NoGain && !fake.prepared && !fake.remembered &&
              has(ran.finished.log, "The GPU runs the prefill FFN alone: no Neural Engine split beats its 14.5 ms per "
                                    "2048-row FFN layer by enough (calibrated in ") &&
              has(ran.finished.log, "the next start calibrates again"),
          "a share predicted to gain too little split: " + ran.finished.log);
  fake = {};
  fake.prepare = [](uint32_t, bool) {
    AneFfnPrepared prepared;
    prepared.chunks = timedChunks(21.0);
    return prepared;
  };
  ran = served(fake);
  require(ran.outcome().kind == Kind::NoGain && !fake.remembered &&
              has(ran.finished.log, "no chunk of the Neural Engine split beats it by enough, 21.0 ms per 2048-row FFN "
                                    "layer against 20.0 on the GPU alone (calibrated in "),
          "a split of no chunk that gains split: " + ran.finished.log);
}

// A set-up takes this Mac's calibration once remembered: it neither times nor remembers, prepares the remembered
// share untimed with the remembered least chunk, and says so — unless the calibration's mixer reads longer than the
// start's prefill gives beside the calibration's GPU alone: one taken on a busier Mac, which calibrates again. --recalibrate-ane asks. A
// remembered calibration of no split leaves the GPU alone once the Neural Engine refused twice, calibrates again
// after one refusal, and one the plan no longer holds or whose least rows no function holds is calibrated again.
void testRemembered() {
  const std::array<ChunkTimings::Gpu, 2> gpu{{{512, 5.0}, {2048, 20.0}}};
  Fake fake;
  fake.stored = Calibration{9, 700, gpu};
  Served ran = served(fake);
  require(ran.outcome().kind == Kind::Split && !fake.timings && fake.prepared == std::pair(9u, false) &&
              !fake.remembered && fake.recalled && ran.started.start.plan &&
              ran.started.start.plan->breakdown().aneFfnBytes == fake.bytes(ran.started.start.maxAneUnits) &&
              has(ran.finished.log, "Neural Engine FFN split at share 0.26 for chunks of 700 rows or more, 2.5% from the "
                                    "GPU alone on the Neural Engine's part (set up as calibrated before, its mixer 0.0 "
                                    "ms a full chunk's layer, 2.0 now"),
          "the remembered calibration was not taken: " + ran.finished.log);
  // The mixer the start's prefill gives beside the record's GPU alone: a
  // recalled record's mixer within 10% of it serves, a longer one was taken
  // on a busier Mac and calibrates again.
  fake = {};
  fake.stored = Calibration{9, 700, gpu, 0, 0, 2.0};
  ran = served(fake);
  require(ran.outcome().kind == Kind::Split && !fake.timings && fake.prepared == std::pair(9u, false),
          "a calibration of no busier a Mac than now's was recalibrated");
  fake = {};
  fake.stored = Calibration{9, 700, gpu, 0, 0, 50.0};
  ran = served(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, true) &&
              fake.remembered && fake.remembered->aneUnits == 8 &&
              std::abs(fake.remembered->mixer - 2.0) < 1e-9 &&
              has(ran.finished.log, "calibration was taken on a busier Mac"),
          "a calibration of a busier Mac than now's was taken: " + ran.finished.log);
  // A record of no split: the Neural Engine's two refusals leave the GPU
  // alone, one refusal calibrates again counting it.
  fake = {};
  fake.stored = Calibration{.refusals = Calibration::kRefusals};
  ran = served(fake);
  require(ran.outcome().kind == Kind::Unavailable && !fake.timings && !fake.prepared &&
              has(ran.finished.log.empty() ? ran.started.log : ran.finished.log, "refused its split twice"),
          "a calibration of two refusals did not leave the GPU alone: " + ran.started.log + ran.finished.log);
  fake = {};
  fake.stored = Calibration{.refusals = 1};
  ran = served(fake);
  require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, true) &&
              fake.remembered && fake.remembered->refusals == 1,
          "a calibration of one refusal was not calibrated again counting it: " + ran.finished.log);
  // --recalibrate-ane takes nothing remembered.
  fake = {};
  fake.stored = Calibration{9, 700, gpu};
  ran = served(fake, {.recalibrate = true});
  require(ran.outcome().kind == Kind::Split && !fake.recalled && fake.timings == 1 &&
              fake.prepared == std::pair(8u, true),
          "--recalibrate-ane took the remembered calibration");
  for (const Calibration &unusable : {Calibration{34, 700, gpu}, Calibration{9, 511, gpu}, Calibration{9, 2049, gpu}}) {
    fake = {};
    fake.stored = unusable;
    ran = served(fake);
    require(ran.outcome().kind == Kind::Split && fake.timings == 1 && fake.prepared == std::pair(8u, true) &&
                fake.remembered && fake.remembered->aneUnits == 8,
            "a remembered calibration of " + std::to_string(unusable.aneUnits) + " units from " +
                std::to_string(unusable.minimumRows) + " rows was taken");
  }
}

// A given share runs the units nearest it, untimed, over the chunks given, and is not remembered; a given split
// of no units or of rows no function holds is refused.
void testGivenShare() {
  Fake fake;
  Started started = start(fake, {.given = AneFfnSetting::Given{0.2}});
  require(started.start.outcome.kind == Kind::Split && !fake.timings && fake.prepared == std::pair(7u, false) &&
              !fake.recalled && !fake.remembered && started.start.plan &&
              started.start.plan->breakdown().aneFfnBytes == fake.bytes(7) &&
              has(started.log, "Neural Engine FFN split at the given share 0.21 for chunks of 512 rows or more and "
                               "0.21 in two steps from 1024 rows, 2.5% from the GPU alone on the Neural Engine's part"),
          "the given share did not run as given: " + started.log);
  started = start(fake, {.given = AneFfnSetting::Given{0.2, 832}});
  require(started.start.outcome.kind == Kind::Split && has(started.log, "for chunks of 832 rows or more"),
          "the given least chunk did not run as given: " + started.log);
  for (const AneFfnSetting::Given given : {AneFfnSetting::Given{0.0}, AneFfnSetting::Given{1.0},
                                           AneFfnSetting::Given{0.2, 511}, AneFfnSetting::Given{0.2, 2049}})
    rejects([&] { static_cast<void>(start(fake, {.given = given})); }, "ANE FFN",
            "a given split of share " + std::to_string(given.share) + " from " + std::to_string(given.minimumRows) +
                " rows ran");
  // A dev tool's options: none calibrates, a share of 0 runs the GPU alone, and least rows need a share.
  const AneFfnSetting calibrated = AneFfnSetting::fromGiven(std::nullopt, std::nullopt),
                      off = AneFfnSetting::fromGiven(0.0, 832), given = AneFfnSetting::fromGiven(0.3, 832);
  require(calibrated.enabled && !calibrated.given && !off.enabled && given.enabled && given.given &&
              given.given->share == 0.3 && given.given->minimumRows == 832 &&
              AneFfnSetting::fromGiven(0.3, std::nullopt).given->minimumRows == ops::FfnSplit::kMinimumRows,
          "a dev tool's options did not give their setting");
  rejects([] { static_cast<void>(AneFfnSetting::fromGiven(std::nullopt, 832)); }, "without a share",
          "least rows without a share were taken");
}

} // namespace

int main() {
  try {
    testOff();
    testUnsupported();
    testRefusedContext();
    testRefusedAutomatic();
    testPending();
    testAutomaticContext();
    testDeterministicContext();
    testExplicitContext();
    testFailures();
    testNoGain();
    testRemembered();
    testGivenShare();
    std::cout << "ane ffn startup tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ane ffn startup tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
