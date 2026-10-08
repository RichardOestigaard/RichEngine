// DiffusionGemma host checks: the layout's diffusion schedule, the
// sampler's accept/exit/commit policy, the canvas page-table geometry and
// the packed manifest descriptor — all without a Metal device.

#include "TestChecks.hpp"
#include "model/DiffusionGemma.hpp"
#include "model/DiffusionSampler.hpp"
#include "model/ModelDescriptor.hpp"

#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

namespace {

using namespace richengine;
using richengine::test::require;

constexpr model::DiffusionGemmaLayout layout{};

void checkLayout() {
  require(layout.layers == 30 && layout.hiddenSize == 2816 &&
              layout.vocabularySize == 262144,
          "the diffusion trunk lost the Gemma 4 sizes");
  const model::DiffusionSchedule &schedule = layout.diffusion;
  require(schedule.valid() && schedule.canvasLength == 256 &&
              schedule.maxDenoisingSteps == 48,
          "the default canvas schedule is invalid");
  require(schedule.canvasLength % richengine::kv::kPageTokens == 0,
          "the canvas length is not page-aligned");
  require(std::fabs(schedule.temperature(48) - 0.8F) < 1e-6F &&
              std::fabs(schedule.temperature(24) - 0.6F) < 1e-6F &&
              std::fabs(schedule.temperature(0) - 0.4F) < 1e-6F,
          "the temperature ramp does not interpolate tMin..tMax");
  // The manifest's equality check is what validateDiffusionGemma runs.
  require(layout.layerMagic == model::Gemma4MoeLayout{}.layerMagic &&
              layout.headMagic == model::Gemma4MoeLayout{}.headMagic,
          "the packed file magics diverged from the trunk's");
  require(layout.stopTokens[0] == 1 && layout.stopTokens[1] == 106,
          "the diffusion stop tokens changed");
}

void checkAcceptMask() {
  model::DiffusionSampler sampler(layout.diffusion);
  // A zero-entropy canvas accepts every row in sorted order.
  std::vector<float> entropy(layout.diffusion.canvasLength, 0.0F);
  require(sampler.acceptMask(entropy) ==
              std::vector<bool>(layout.diffusion.canvasLength, true),
          "a zero-entropy canvas should accept every row");
  // The exclusive prefix rule: entropy 0.05, 0.05, 1.0, ... accepts the two
  // cheap rows (0.05 + 0.05 <= 0.1 at their ranks) and rejects the rest.
  entropy.assign(layout.diffusion.canvasLength, 0.05F);
  entropy[7] = 1.0F;
  entropy[200] = 0.02F;
  const std::vector<bool> mask = sampler.acceptMask(entropy);
  require(mask[200] && mask[7] == false,
          "the accept order ignores the ascending entropy sort");
  uint32_t accepted = 0;
  for (bool bit : mask) accepted += bit;
  // prefix before row 7's rank: 0.02 + 255*0.05 = 12.77 > 0.1 by then.
  require(accepted < layout.diffusion.canvasLength &&
              accepted > 0 && mask[0] && mask[1],
          "the entropy bound did not cap the accepted prefix");
}

void checkEarlyExit() {
  model::DiffusionSampler sampler(layout.diffusion);
  // The first step can never exit.
  require(!sampler.earlyExit({0.0F, true}),
          "the first step must not early-exit");
  // Stable argmax with low entropy exits on the second step.
  require(sampler.earlyExit({0.001F, true}),
          "a stable low-entropy step should exit");
  sampler = model::DiffusionSampler(layout.diffusion);
  require(!sampler.earlyExit({0.001F, true}), "first step");
  require(!sampler.earlyExit({0.5F, true}),
          "a stable step above the confidence bound must not exit");
  sampler = model::DiffusionSampler(layout.diffusion);
  static_cast<void>(sampler.earlyExit({0.001F, true}));
  require(!sampler.earlyExit({0.001F, false}),
          "an unstable canvas must not exit");
}

void checkCommit() {
  model::DiffusionSampler sampler(layout.diffusion);
  const uint32_t rows = layout.diffusion.canvasLength;
  std::vector<uint32_t> argmax(rows, 42);
  const std::array<uint32_t, 3> stops{1, 106, 50};
  argmax[10] = 106;
  argmax[11] = 999;
  const auto commit = sampler.commit(argmax, stops);
  require(commit.stop && commit.stopIndex == 10,
          "the commit lost the first stop token");
  require(commit.tokens.size() == rows && commit.tokens[10] == 106 &&
              commit.tokens[11] == layout.diffusion.padToken &&
              commit.tokens[255] == layout.diffusion.padToken,
          "the rows past the stop were not padded");
  // Emit through the stop inclusive; the pad rows stay KV-only.
  require(sampler.finished(commit, 11) && !sampler.finished(commit, 10),
          "finished() must gate on the emitted prefix");
  // No stop: the whole canvas commits, nothing finished.
  argmax.assign(rows, 42);
  const auto clean = sampler.commit(argmax, stops);
  require(!clean.stop && clean.stopIndex == rows &&
              !sampler.finished(clean, rows),
          "a clean canvas should not finish");
  // The budget caps the emitted prefix.
  require(sampler.emitBudget(10, 20) == 10 &&
              sampler.emitBudget(5000, 6000) == rows &&
              sampler.emitBudget(700, 700) == 0,
          "the emit budget ignores max_new_tokens");
  require(commit.stop && !sampler.finished(commit, 0), "no emit, no finish");
}

// A compact schedule for the driver checks: canvas of one page, a few
// steps, exit after one stable step below the confidence bound.
model::DiffusionSchedule smallSchedule(uint32_t steps = 6) {
  model::DiffusionSchedule schedule;
  schedule.canvasLength = richengine::kv::kPageTokens;
  schedule.maxDenoisingSteps = steps;
  schedule.entropyBound = 0.1F;
  schedule.confidenceThreshold = 0.005F;
  schedule.stabilityThreshold = 1;
  schedule.padToken = 0;
  return schedule;
}

const std::array<uint32_t, 3> kStops{1, 106, 50};

void checkChunkPlan() {
  const model::DiffusionSchedule schedule = smallSchedule(6);
  model::CanvasStepDriver driver(schedule, {4, false, false}, kStops);
  // Four steps per command: {6,5,4,3}, then {2,1} — and the schedule's
  // last step carries no commit tail when the flag is off.
  const auto first = driver.nextChunk();
  require(first.firstStep == 6 && first.count == 4 && !first.commitTail,
          "the first chunk did not take four descending steps");
  std::vector<uint32_t> argmax(schedule.canvasLength, 7);
  for (uint32_t slot = 0; slot < first.count; ++slot) {
    const auto outcome =
        driver.drain(slot, {0.5F, false}, argmax);
    require(!outcome.exited && !outcome.scheduleEnd,
            "a high-entropy unstable step must not exit");
  }
  require(!driver.finished(), "the schedule ended a chunk early");
  const auto second = driver.nextChunk();
  require(second.firstStep == 2 && second.count == 2 && !second.commitTail,
          "the tail chunk lost its trailing steps");
  static_cast<void>(driver.drain(0, {0.5F, false}, argmax));
  const auto last = driver.drain(1, {0.5F, false}, argmax);
  require(driver.finished() && !last.scheduleEnd,
          "the last scheduled step did not finish the canvas");
  require(driver.nextChunk().count == 0,
          "a finished driver must not plan more work");
  // The steps-per-command value clamps to the ring depth.
  model::CanvasStepDriver clamped(schedule, {9, false, false}, kStops);
  require(clamped.nextChunk().count == 4 &&
              model::CanvasStepDriver::kMaxStepsPerCommand == 4,
          "steps-per-command did not clamp to the ring size");
}

void checkExitDeadWork() {
  const model::DiffusionSchedule schedule = smallSchedule(6);
  model::CanvasStepDriver driver(schedule, {4, false, false}, kStops);
  const auto chunk = driver.nextChunk();
  require(chunk.firstStep == 6 && chunk.count == 4, "chunk plan");
  std::vector<uint32_t> argmax(schedule.canvasLength, 7);
  // Slot 0 is the first step: it can never exit. Slot 1 exits on the
  // kernel's stable flag; slots 2..3 were dead compute and drain stops.
  auto outcome = driver.drain(0, {0.001F, true}, argmax);
  require(!outcome.exited, "the first step must not early-exit");
  outcome = driver.drain(1, {0.001F, true}, argmax);
  require(outcome.exited && driver.finished(),
          "a stable low-entropy slot did not exit mid-chunk");
  bool threw = false;
  try {
    static_cast<void>(driver.drain(2, {0.001F, true}, argmax));
  } catch (const std::logic_error &) {
    threw = true;
  }
  require(threw, "draining a slot past the exit was allowed");
  // Out-of-order drains are rejected.
  model::CanvasStepDriver ordered(schedule, {2, false, false}, kStops);
  static_cast<void>(ordered.nextChunk());
  threw = false;
  try {
    static_cast<void>(ordered.drain(1, {0.5F, false}, argmax));
  } catch (const std::logic_error &) {
    threw = true;
  }
  require(threw, "an out-of-order ring drain was accepted");
}

void checkPrefixExit() {
  const model::DiffusionSchedule schedule = smallSchedule(6);
  const uint32_t rows = schedule.canvasLength;
  model::CanvasStepDriver driver(schedule, {4, true, false}, kStops);
  static_cast<void>(driver.nextChunk());
  // Step 1: a stop at index 5 with noise past it. Step 2: the prefix
  // [0..5] identical, the rows past the stop churn — the whole-canvas flag
  // is false, but the prefix exit must still fire.
  std::vector<uint32_t> argmax(rows, 42);
  argmax[5] = 106;
  argmax[20] = 900;
  auto outcome = driver.drain(0, {0.001F, false}, argmax);
  require(!outcome.exited && outcome.stopIndex == 5,
          "the first step must not exit; the stop scan is off");
  argmax[20] = 555; // dead region churns past the first stop
  outcome = driver.drain(1, {0.001F, false}, argmax);
  require(outcome.exited && outcome.argmaxStable &&
              outcome.stopIndex == 5,
          "a stable argmax prefix past its stop must exit");
  // The confidence bound still gates: the whole-canvas mean entropy stays
  // an AND condition (the kernel emits no prefix entropy).
  model::CanvasStepDriver hot(schedule, {4, true, false}, kStops);
  static_cast<void>(hot.nextChunk());
  argmax[20] = 900;
  static_cast<void>(hot.drain(0, {0.001F, false}, argmax));
  argmax[20] = 555;
  outcome = hot.drain(1, {0.5F, false}, argmax);
  require(!outcome.exited,
          "prefix exit ignored the confidence bound");
  // Prefix churn before the stop blocks the exit.
  model::CanvasStepDriver churn(schedule, {4, true, false}, kStops);
  static_cast<void>(churn.nextChunk());
  static_cast<void>(churn.drain(0, {0.001F, true}, argmax));
  argmax[2] = 77; // inside the committed prefix
  outcome = churn.drain(1, {0.001F, true}, argmax);
  require(!outcome.exited,
          "prefix exit fired though the committed prefix changed");
}

void checkPrefixEntropy() {
  const model::DiffusionSchedule schedule = smallSchedule(6);
  const uint32_t rows = schedule.canvasLength;
  // With the slot's per-row entropies the exit gates on the prefix mean:
  // noise past a stop is dead output and must not hold the canvas.
  model::CanvasStepDriver driver(schedule, {4, true, false}, kStops);
  static_cast<void>(driver.nextChunk());
  std::vector<uint32_t> argmax(rows, 42);
  argmax[5] = 106;
  argmax[20] = 900;
  std::vector<float> entropies(rows, 9.0F);
  std::fill(entropies.begin(), entropies.begin() + 5, 0.001F);
  static_cast<void>(
      driver.drain(0, {0.5F, false}, argmax,
                   {entropies.data(), rows}));
  argmax[20] = 555;
  const auto outcome =
      driver.drain(1, {0.5F, false}, argmax, {entropies.data(), rows});
  require(outcome.exited && outcome.argmaxStable,
          "a settled prefix must exit though the tail stays hot");
  // A hot prefix keeps the canvas alive even with the mean available.
  model::CanvasStepDriver cold(schedule, {4, true, false}, kStops);
  static_cast<void>(cold.nextChunk());
  std::fill(entropies.begin(), entropies.begin() + 5, 9.0F);
  static_cast<void>(
      cold.drain(0, {0.001F, false}, argmax, {entropies.data(), rows}));
  const auto blocked =
      cold.drain(1, {0.001F, false}, argmax, {entropies.data(), rows});
  require(!blocked.exited,
          "prefix exit fired though the prefix rows stayed hot");
  // No entropy span falls back to the kernel's whole-canvas mean.
  model::CanvasStepDriver blind(schedule, {4, true, false}, kStops);
  static_cast<void>(blind.nextChunk());
  static_cast<void>(blind.drain(0, {0.001F, false}, argmax));
  const auto gated = blind.drain(1, {0.5F, false}, argmax);
  require(!gated.exited,
          "the whole-canvas mean must still gate without entropies");
}

void checkStableFraction() {
  const model::DiffusionSchedule schedule = smallSchedule(6);
  const uint32_t rows = schedule.canvasLength;
  // Fraction-settled exit: the churning tail keeps the whole-canvas stats
  // hot, but a stable confident supermajority still ends the canvas.
  model::CanvasStepDriver driver(schedule, {4, false, false, 0.75F}, kStops);
  static_cast<void>(driver.nextChunk());
  std::vector<uint32_t> argmax(rows, 42);
  std::vector<float> entropies(rows, 9.0F);
  std::fill(entropies.begin(), entropies.begin() + 24, 0.001F);
  static_cast<void>(
      driver.drain(0, {0.5F, false}, argmax, {entropies.data(), rows}));
  // Eight rows churn (new argmax, hot entropy); the other 24 stay put.
  for (uint32_t i = 24; i < rows; ++i)
    argmax[i] = 1000 + i;
  const auto outcome =
      driver.drain(1, {2.5F, false}, argmax, {entropies.data(), rows});
  require(outcome.exited,
          "a settled supermajority must exit though the tail churns");
  // Below the fraction (10 of 32 churn) the canvas stays alive.
  model::CanvasStepDriver cold(schedule, {4, false, false, 0.75F}, kStops);
  static_cast<void>(cold.nextChunk());
  std::vector<uint32_t> first(rows, 42);
  static_cast<void>(
      cold.drain(0, {0.5F, false}, first, {entropies.data(), rows}));
  for (uint32_t i = 22; i < rows; ++i)
    first[i] = 1000 + i;
  const auto blocked =
      cold.drain(1, {0.5F, false}, first, {entropies.data(), rows});
  require(!blocked.exited,
          "fraction exit fired below the configured fraction");
  // Disabled keeps the strict whole-canvas gate.
  model::CanvasStepDriver strict(schedule, {4, false, false, 0.0F}, kStops);
  static_cast<void>(strict.nextChunk());
  std::vector<uint32_t> second(rows, 42);
  static_cast<void>(
      strict.drain(0, {0.5F, false}, second, {entropies.data(), rows}));
  for (uint32_t i = 24; i < rows; ++i)
    second[i] = 1000 + i;
  const auto held =
      strict.drain(1, {0.5F, false}, second, {entropies.data(), rows});
  require(!held.exited,
          "the fraction exit must stay off when disabled");
  // A churning stable set that stays hot must not exit on the fraction.
  model::CanvasStepDriver hot(schedule, {4, false, false, 0.75F}, kStops);
  static_cast<void>(hot.nextChunk());
  std::vector<uint32_t> third(rows, 42);
  static_cast<void>(
      hot.drain(0, {0.5F, false}, third, {entropies.data(), rows}));
  // The 24 stable rows stay hot this time; whole-canvas and stable mean
  // are both above the threshold.
  std::fill(entropies.begin(), entropies.begin() + 24, 9.0F);
  const auto stillHot =
      hot.drain(1, {0.5F, false}, third, {entropies.data(), rows});
  require(!stillHot.exited,
          "the fraction exit needs confident stable rows, not just counts");
  // Drift mode waives the confidence check: the same hot-but-settled canvas
  // commits its argmax.
  model::CanvasStepDriver drift(schedule, {4, true, false, 0.75F, true},
                                kStops);
  static_cast<void>(drift.nextChunk());
  static_cast<void>(
      drift.drain(0, {0.5F, false}, third, {entropies.data(), rows}));
  const auto drifted =
      drift.drain(1, {0.5F, false}, third, {entropies.data(), rows});
  require(drifted.exited,
          "drift exit commits a fraction-settled canvas regardless of entropy");
  // Drift prefix scope: a stop mid-canvas with a settled prefix exits even
  // though the dead tail churns the whole-canvas fraction below the bar.
  model::CanvasStepDriver driftPrefix(schedule,
                                      {4, true, false, 0.75F, true}, kStops);
  static_cast<void>(driftPrefix.nextChunk());
  std::vector<uint32_t> half(rows, 42);
  static_cast<void>(
      driftPrefix.drain(0, {0.5F, false}, half, {entropies.data(), rows}));
  half[5] = 106; // stop at row 5
  for (uint32_t i = 6; i < rows; ++i)
    half[i] = 3000 + i; // dead tail churns — never counts
  const auto deadTail =
      driftPrefix.drain(1, {0.5F, false}, half, {entropies.data(), rows});
  require(deadTail.exited,
          "drift exit must judge the live prefix, not the dead tail");
  // Maturity guard on the prefix path: a stop token appearing while the
  // canvas still churns must not commit — the prefix only exits once the
  // canvas is mostly settled.
  model::CanvasStepDriver early(schedule, {4, true, false, 0.75F}, kStops);
  static_cast<void>(early.nextChunk());
  std::vector<uint32_t> noise(rows, 42);
  std::fill(entropies.begin(), entropies.end(), 9.0F);
  static_cast<void>(
      early.drain(0, {0.5F, false}, noise, {entropies.data(), rows}));
  noise[5] = 106; // a stop token appears mid-denoise
  for (uint32_t i = 6; i < rows; ++i)
    noise[i] = 2000 + i; // the tail still churns
  const auto premature =
      early.drain(1, {0.5F, false}, noise, {entropies.data(), rows});
  require(!premature.exited,
          "a stop on a churning canvas must not commit early");
  // Once the whole canvas settles the same prefix exit fires.
  std::fill(entropies.begin(), entropies.begin() + 5, 0.001F);
  const auto settled =
      early.drain(2, {0.01F, false}, noise, {entropies.data(), rows});
  require(settled.exited,
          "the same prefix must exit once the canvas is mature");
}

void checkSpeculationArm() {
  const model::DiffusionSchedule schedule = [] {
    model::DiffusionSchedule s = smallSchedule(6);
    s.stabilityThreshold = 2; // one stable step arms, the second exits
    return s;
  }();
  model::CanvasStepDriver driver(schedule, {4, false, false}, kStops);
  static_cast<void>(driver.nextChunk());
  std::vector<uint32_t> argmax(schedule.canvasLength, 42);
  static_cast<void>(driver.drain(0, {0.001F, true}, argmax));
  auto outcome = driver.drain(1, {0.001F, true}, argmax);
  require(!outcome.exited && driver.speculationArmed(),
          "one stable step below the threshold must arm the speculation");
  outcome = driver.drain(2, {0.001F, true}, argmax);
  require(outcome.exited && !driver.speculationArmed(),
          "the second stable step must exit; the snapshot stays valid");
  // A churning step disarms: the provisional commit is stale.
  model::CanvasStepDriver stale(schedule, {4, false, false}, kStops);
  static_cast<void>(stale.nextChunk());
  static_cast<void>(stale.drain(0, {0.001F, true}, argmax));
  static_cast<void>(stale.drain(1, {0.001F, true}, argmax));
  argmax[3] = 999;
  outcome = stale.drain(2, {0.001F, false}, argmax);
  require(!outcome.exited && !stale.speculationArmed(),
          "a changed argmax must disarm the speculative prefill");
}

void checkCommitTail() {
  const model::DiffusionSchedule schedule = smallSchedule(3);
  model::CanvasStepDriver driver(schedule, {2, false, true}, kStops);
  static_cast<void>(driver.nextChunk());
  // An undrained chunk must not plan again.
  bool threw = false;
  try {
    static_cast<void>(driver.nextChunk());
  } catch (const std::logic_error &) {
    threw = true;
  }
  require(threw, "planning over an undrained stats ring was allowed");
  std::vector<uint32_t> warm(schedule.canvasLength, 5);
  static_cast<void>(driver.drain(0, {0.5F, false}, warm));
  static_cast<void>(driver.drain(1, {0.5F, false}, warm));
  const auto tail = driver.nextChunk();
  require(tail.commitTail && tail.firstStep == 1 && tail.count == 1,
          "the commit tail did not mark the last scheduled step");
  std::vector<uint32_t> argmax(schedule.canvasLength, 9);
  // The tail slot wrote no stats; its argmax is the commit.
  const auto outcome = driver.drain(0, {0.0F, false}, argmax);
  require(outcome.scheduleEnd && driver.finished() && !outcome.exited,
          "the commit tail must end the schedule without an exit");
}

void checkCanvasPages() {
  model::DiffusionSampler sampler(layout.diffusion);
  require(sampler.canvasPages() == 256 / richengine::kv::kPageTokens,
          "the canvas scratch page count is off");
  // A 33-token prefix covers two pages; the canvas appends eight scratch
  // entries, keeping the prefix's partial page for its straddling rows.
  const std::vector<RichKvPage> prefix{0x1000, 0x2000};
  const std::vector<RichKvPage> table =
      sampler.canvasPageTable(prefix, 0x4000);
  require(table.size() == prefix.size() + sampler.canvasPages() &&
              table[0] == 0x1000 && table[1] == 0x2000 &&
              table[2] == 0x4000 && table[9] == 0x4007,
          "the canvas page table does not append the scratch extent");
  bool threw = false;
  try {
    static_cast<void>(sampler.canvasPageTable(prefix, 1)); // index bits set
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "a misaligned scratch extent was accepted");
}

void writeFile(const std::filesystem::path &path, std::string_view text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out << text;
  require(bool(out), "unable to write the fixture");
}

void checkDescriptor() {
  const std::filesystem::path root =
      std::filesystem::temp_directory_path() /
      ("diffusiongemma-" + std::to_string(::getpid()));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  // The packed manifest: the shared format fields, the Gemma 4 target
  // declaration under its diffusion architecture, and the schedule block.
  writeFile(root / "manifest.json", R"json({
    "schema_version": 1,
    "model": "google/diffusiongemma-26B-A4B-it",
    "execution_geometry": {
      "draft_proposal_tokens": 7,
      "draft_query_rows": 8,
      "draft_sliding_window": 1024
    },
    "format": {
      "name": "richengine-packed-q4-diffusiongemma",
      "q4_bits": 4,
      "q4_group_size": 64,
      "q4_storage_n": 256,
      "section_alignment_bytes": 16384,
      "target_layer_magic": "GEMM0001",
      "draft_layer_magic": "MDFD0004",
      "vision_magic": "MDFV0001"
    },
    "target": {
      "architecture": "diffusion_gemma_text",
      "layers": 30,
      "hidden_size": 2816,
      "vocabulary_size": 262144,
      "num_attention_heads": 16,
      "num_key_value_heads": 8,
      "head_dim": 256,
      "global_num_key_value_heads": 2,
      "global_head_dim": 512,
      "sliding_window": 1024,
      "experts": 128,
      "experts_per_token": 8,
      "moe_intermediate_size": 704,
      "shared_expert_intermediate_size": 2112,
      "final_logit_softcapping": 30,
      "global_rope_theta": 1000000,
      "rope_theta": 10000,
      "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention"
      ]
    },
    "diffusion": {
      "canvas_length": 256,
      "max_denoising_steps": 48,
      "t_min": 0.4,
      "t_max": 0.8,
      "entropy_bound": 0.1,
      "confidence_threshold": 0.005,
      "stability_threshold": 1
    }
  })json");
  writeFile(root / "tokenizer" / "config.json", R"json({
    "model_type": "diffusion_gemma",
    "text_config": {
      "model_type": "diffusion_gemma_text",
      "hidden_size": 2816,
      "vocab_size": 262144,
      "max_position_embeddings": 131072
    }
  })json");

  const model::ModelDescriptor descriptor = model::inspectModelPackage(root);
  require(descriptor.name == "google/diffusiongemma-26B-A4B-it",
          "the descriptor lost the model name");
  const auto *diffusionLayout =
      std::get_if<model::DiffusionGemmaLayout>(&descriptor.target);
  require(diffusionLayout, "the descriptor did not pick the diffusion layout");
  require(diffusionLayout->diffusion.canvasLength == 256 &&
              diffusionLayout->diffusion.stabilityThreshold == 1,
          "the manifest's diffusion block did not reach the layout");
  require(descriptor.targetSource == model::TargetSource::Packed &&
              descriptor.draft.kind == model::DraftKind::Null &&
              descriptor.visionSource == model::VisionSource::None,
          "a diffusion package is a packed text-only, draftless target");
  require(descriptor.capabilities.vocabularySize == 262144 &&
              descriptor.targetKvLayout.valid(),
          "the descriptor's capabilities or KV layout are wrong");

  // A declared draft or a missing diffusion block must reject.
  writeFile(root / "tokenizer" / "config.json", R"json({
    "model_type": "diffusion_gemma",
    "text_config": {
      "model_type": "diffusion_gemma_text",
      "hidden_size": 2816,
      "vocab_size": 262144,
      "max_position_embeddings": 131072
    }
  })json");
  std::ifstream manifest(root / "manifest.json");
  std::string text((std::istreambuf_iterator<char>(manifest)),
                   std::istreambuf_iterator<char>());
  const size_t offset = text.find("\"diffusion\"");
  require(offset != std::string::npos, "fixture manifest lacks diffusion");
  const std::string original = text.substr(offset);
  // Draft declaration rejection.
  writeFile(root / "manifest.draft.json",
            text.substr(0, offset) +
                "\"draft\": {\"kind\": \"plain\"},\n" + original);
  std::filesystem::rename(root / "manifest.draft.json",
                          root / "manifest.json");
  bool threw = false;
  try {
    static_cast<void>(model::inspectModelPackage(root));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "a diffusion manifest with a draft was accepted");
  // A foreign target architecture rejects.
  const size_t architecture = text.find("diffusion_gemma_text");
  require(architecture != std::string::npos, "fixture lacks architecture");
  writeFile(root / "manifest.json",
            text.substr(0, architecture) + "gemma4_text" +
                text.substr(architecture + 20));
  threw = false;
  try {
    static_cast<void>(model::inspectModelPackage(root));
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "a wrong architecture was accepted");

  std::filesystem::remove_all(root, ec);
}

} // namespace

int main() {
  try {
    checkLayout();
    checkAcceptMask();
    checkEarlyExit();
    checkCommit();
    checkChunkPlan();
    checkExitDeadWork();
    checkPrefixExit();
    checkPrefixEntropy();
    checkStableFraction();
    checkSpeculationArm();
    checkCommitTail();
    checkCanvasPages();
    checkDescriptor();
    std::cout << "diffusion gemma: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
