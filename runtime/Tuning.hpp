#pragma once

// Central registry of the engine's RICHENGINE_* development/tuning knobs.
// Call sites take tuning().field instead of reading the environment: the
// snapshot is built once per process (function-local static), the same
// once-read semantics the scattered `static const` reads had. Flag
// conventions are preserved from Env.hpp — a "(presence)" comment means
// envFlag, "=1" means envFlagOn, a bare name means envUint parse-or-default.

#include "Env.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace richengine {

struct Tuning {
  // ops/MoE.cpp, ops/ExecutionPlans.cpp
  bool moeStats = false;      // RICHENGINE_MOE_STATS=1
  bool moePackedOff = false;  // RICHENGINE_MOE_PACKED_OFF (presence)
  uint32_t moeUnionCap = 0;   // RICHENGINE_MOE_UNION
  uint32_t moeTopK = 0;       // RICHENGINE_MOE_TOPK
  // ops/GDN.cpp, model/RuntimeArenas.mm
  uint32_t gdnChunked = 0;    // RICHENGINE_GDN_CHUNKED
  // ops/Linear.cpp, ops/LinearGguf.cpp
  bool ggufPackedOn = false;  // RICHENGINE_GGUF_PACKED_ON (presence)
  bool ggufPackedOff = false; // RICHENGINE_GGUF_PACKED_OFF (presence)
  // ops/Linear.cpp, model/RuntimeArenas.mm
  bool prefillFastInt8 = false; // RICHENGINE_PREFILL_FAST_INT8 (presence)
  // metal/MetalEncode.mm
  bool icbOff = false;          // RICHENGINE_ICB_OFF (presence)
  bool mtl4 = false;            // RICHENGINE_MTL4 (presence)
  bool preparedCacheOff = false; // RICHENGINE_PREPARED_CACHE_OFF (presence)
  bool opTimings = false;       // RICHENGINE_OP_TIMINGS=1
  // metal/CommandGraph.hpp
  bool patchableOff = false;    // RICHENGINE_PATCHABLE_OFF (presence)
  // model/TargetModelGemma.cpp
  bool gemmaSkipAttn = false;   // RICHENGINE_GEMMA_SKIP_ATTN (presence)
  bool gemmaSkipMoe = false;    // RICHENGINE_GEMMA_SKIP_MOE (presence)
  bool gegluOff = false;        // RICHENGINE_GEGLU_OFF (presence)
  // model/TargetModel.cpp, model/TargetModelGemma.cpp
  bool noFusedGate = false;     // RICHENGINE_NO_FUSED_GATE (presence)
  // ops/Canvas.cpp A/B gates: on by default, =0 selects the old kernel.
  bool canvasEmbedHist = true;  // RICHENGINE_CANVAS_EMBED_HIST
  bool canvasStatsFused = true; // RICHENGINE_CANVAS_STATS_FUSED
  // model/RuntimeImpl.hpp canvas policy (model/RuntimeDiffusion.mm consumes).
  uint32_t canvasStepsPerCmd = 4;   // RICHENGINE_CANVAS_STEPS_PER_CMD, clamped 1..4
  bool canvasPrefixExit = true;     // RICHENGINE_CANVAS_PREFIX_EXIT, =0 disables
  bool canvasCommitTail = true;     // RICHENGINE_CANVAS_COMMIT_TAIL, =0 disables
  bool canvasSpeculativePrefill = true; // RICHENGINE_CANVAS_SPECULATIVE_PREFILL, =0 disables
  std::string_view canvasProfile;   // RICHENGINE_CANVAS_PROFILE
  float canvasExitStable = 0.9F;    // RICHENGINE_CANVAS_EXIT_STABLE (else profile default)
  bool canvasDriftExit = false;     // RICHENGINE_CANVAS_EXIT_DRIFT=1 (else profile default)
  uint32_t canvasMaxSteps = 0;      // RICHENGINE_CANVAS_MAX_STEPS (else profile default)
  std::string_view canvasProbe;     // RICHENGINE_CANVAS_PROBE
  bool canvasTiming = false;        // RICHENGINE_CANVAS_TIMING (presence)
  // model/RuntimeImpl.hpp speculative-decode policy
  bool verifyTreeSet = false;       // RICHENGINE_VERIFY_TREE (presence)
  bool verifyTree = false;          // RICHENGINE_VERIFY_TREE=1
  bool adaptiveProposals = true;    // RICHENGINE_ADAPTIVE_PROPOSALS, =0 disables
  bool ngramPredraft = true;        // RICHENGINE_NGRAM_PREDRAFT, =0 disables
  bool draftBypass = true;          // RICHENGINE_DRAFT_BYPASS, =0 disables
  double draftBypassExpect = 0.15;  // RICHENGINE_DRAFT_BYPASS_EXPECT
  double ngramDraftExpect = 4.0;    // RICHENGINE_NGRAM_DRAFT_EXPECT
  uint32_t ngramWarmup = 8;         // RICHENGINE_NGRAM_WARMUP
  double ngramTreeMin = 1.0;        // RICHENGINE_NGRAM_TREE_MIN
  // model/RuntimeImpl.hpp debug gates
  bool treeDebug = false;           // RICHENGINE_TREE_DEBUG (presence)
  bool draftDebug = false;          // RICHENGINE_DRAFT_DEBUG (presence)
  bool draftConf = false;           // RICHENGINE_DRAFT_CONF (presence)
  std::string treeSkip;             // RICHENGINE_TREE_SKIP
  bool aneDebug = false;            // RICHENGINE_ANE_DEBUG (presence)
  bool ngramDebug = false;          // RICHENGINE_NGRAM_DEBUG (presence)
  uint32_t aneWaitMs = 3;           // RICHENGINE_ANE_WAIT_MS
  const char *aneMedusa = nullptr;  // RICHENGINE_ANE_MEDUSA path
  const char *anePredraft = nullptr; // RICHENGINE_ANE_PREDRAFT path
  bool diffusionAr = false;         // RICHENGINE_DIFFUSION_AR=1
  // model/DFlashDraft.cpp
  bool dflashPoolSet = false;       // RICHENGINE_DFLASH_POOL (presence)
  bool dflashPool = false;          // RICHENGINE_DFLASH_POOL=1
  // Registered for completeness; the call sites below live in engine/ or in
  // model/*.mm files outside the conversion scope and still read the
  // environment directly.
  bool submitAhead = false;         // RICHENGINE_SUBMIT_AHEAD — engine/Engine.cpp
  bool decodeTiming = false;        // RICHENGINE_DECODE_TIMING — model/Runtime.mm
  bool headFusedOff = false;        // RICHENGINE_HEAD_FUSED_OFF — model/RuntimeEncode.mm
  bool proposalCapSet = false;      // RICHENGINE_PROPOSAL_CAP (presence) — model/Runtime.mm
  uint32_t proposalCap = 0;         // RICHENGINE_PROPOSAL_CAP — model/Runtime.mm
  bool denyDebug = false;           // RICHENGINE_DENY_DEBUG — engine/
};

namespace detail {

inline Tuning makeTuning() {
  // On-by-default knobs: "0" is the only disabling value; any other set
  // value (including "1" and the empty string) leaves them on.
  const auto unlessZero = [](const char *name) {
    const char *value = std::getenv(name);
    return !(value && std::string_view(value) == "0");
  };
  const auto stringOrEmpty = [](const char *name) -> std::string_view {
    const char *value = std::getenv(name);
    return value ? std::string_view(value) : std::string_view{};
  };
  Tuning t;
  t.moeStats = envFlagOn("RICHENGINE_MOE_STATS");
  t.moePackedOff = envFlag("RICHENGINE_MOE_PACKED_OFF");
  t.moeUnionCap = envUint("RICHENGINE_MOE_UNION", 0);
  t.moeTopK = envUint("RICHENGINE_MOE_TOPK", 0);
  t.gdnChunked = envUint("RICHENGINE_GDN_CHUNKED", 0);
  t.ggufPackedOn = envFlag("RICHENGINE_GGUF_PACKED_ON");
  t.ggufPackedOff = envFlag("RICHENGINE_GGUF_PACKED_OFF");
  t.prefillFastInt8 = envFlag("RICHENGINE_PREFILL_FAST_INT8");
  t.icbOff = envFlag("RICHENGINE_ICB_OFF");
  t.mtl4 = envFlag("RICHENGINE_MTL4");
  t.preparedCacheOff = envFlag("RICHENGINE_PREPARED_CACHE_OFF");
  t.opTimings = envFlagOn("RICHENGINE_OP_TIMINGS");
  t.patchableOff = envFlag("RICHENGINE_PATCHABLE_OFF");
  t.gemmaSkipAttn = envFlag("RICHENGINE_GEMMA_SKIP_ATTN");
  t.gemmaSkipMoe = envFlag("RICHENGINE_GEMMA_SKIP_MOE");
  t.gegluOff = envFlag("RICHENGINE_GEGLU_OFF");
  t.noFusedGate = envFlag("RICHENGINE_NO_FUSED_GATE");
  t.canvasEmbedHist = unlessZero("RICHENGINE_CANVAS_EMBED_HIST");
  t.canvasStatsFused = unlessZero("RICHENGINE_CANVAS_STATS_FUSED");
  t.canvasStepsPerCmd =
      std::clamp(envUint("RICHENGINE_CANVAS_STEPS_PER_CMD", 4), 1u, 4u);
  t.canvasPrefixExit = unlessZero("RICHENGINE_CANVAS_PREFIX_EXIT");
  t.canvasCommitTail = unlessZero("RICHENGINE_CANVAS_COMMIT_TAIL");
  t.canvasSpeculativePrefill =
      unlessZero("RICHENGINE_CANVAS_SPECULATIVE_PREFILL");
  t.canvasProfile = stringOrEmpty("RICHENGINE_CANVAS_PROFILE");
  // The CANVAS_PROFILE bundle supplies each knob's default when the knob's
  // own variable is unset; set always wins over the profile.
  if (const char *value = std::getenv("RICHENGINE_CANVAS_EXIT_STABLE"))
    t.canvasExitStable = std::clamp(std::strtof(value, nullptr), 0.0F, 1.0F);
  else
    t.canvasExitStable = t.canvasProfile == "fast"     ? 0.6F
                         : t.canvasProfile == "balanced" ? 0.75F
                                                         : 0.9F;
  if (const char *value = std::getenv("RICHENGINE_CANVAS_EXIT_DRIFT"))
    t.canvasDriftExit = std::string_view(value) == "1";
  else
    t.canvasDriftExit =
        t.canvasProfile == "balanced" || t.canvasProfile == "fast";
  if (std::getenv("RICHENGINE_CANVAS_MAX_STEPS"))
    t.canvasMaxSteps = envUint("RICHENGINE_CANVAS_MAX_STEPS", 0);
  else
    t.canvasMaxSteps = t.canvasProfile == "fast"     ? 24u
                       : t.canvasProfile == "balanced" ? 32u
                                                       : 0u;
  t.canvasProbe = stringOrEmpty("RICHENGINE_CANVAS_PROBE");
  t.canvasTiming = envFlag("RICHENGINE_CANVAS_TIMING");
  t.verifyTreeSet = envFlag("RICHENGINE_VERIFY_TREE");
  t.verifyTree = envFlagOn("RICHENGINE_VERIFY_TREE");
  t.adaptiveProposals = unlessZero("RICHENGINE_ADAPTIVE_PROPOSALS");
  t.ngramPredraft = unlessZero("RICHENGINE_NGRAM_PREDRAFT");
  t.draftBypass = unlessZero("RICHENGINE_DRAFT_BYPASS");
  t.draftBypassExpect = envDouble("RICHENGINE_DRAFT_BYPASS_EXPECT", 0.15);
  t.ngramDraftExpect = envDouble("RICHENGINE_NGRAM_DRAFT_EXPECT", 4.0);
  t.ngramWarmup = envUint("RICHENGINE_NGRAM_WARMUP", 8);
  t.ngramTreeMin = envDouble("RICHENGINE_NGRAM_TREE_MIN", 1.0);
  t.treeDebug = envFlag("RICHENGINE_TREE_DEBUG");
  t.draftDebug = envFlag("RICHENGINE_DRAFT_DEBUG");
  t.draftConf = envFlag("RICHENGINE_DRAFT_CONF");
  if (const char *value = std::getenv("RICHENGINE_TREE_SKIP"))
    t.treeSkip = value;
  t.aneDebug = envFlag("RICHENGINE_ANE_DEBUG");
  t.ngramDebug = envFlag("RICHENGINE_NGRAM_DEBUG");
  t.aneWaitMs = envUint("RICHENGINE_ANE_WAIT_MS", 3);
  t.aneMedusa = std::getenv("RICHENGINE_ANE_MEDUSA");
  t.anePredraft = std::getenv("RICHENGINE_ANE_PREDRAFT");
  t.diffusionAr = envFlagOn("RICHENGINE_DIFFUSION_AR");
  t.dflashPoolSet = envFlag("RICHENGINE_DFLASH_POOL");
  t.dflashPool = envFlagOn("RICHENGINE_DFLASH_POOL");
  t.submitAhead = envFlag("RICHENGINE_SUBMIT_AHEAD");
  t.decodeTiming = envFlag("RICHENGINE_DECODE_TIMING");
  t.headFusedOff = envFlag("RICHENGINE_HEAD_FUSED_OFF");
  t.proposalCapSet = envFlag("RICHENGINE_PROPOSAL_CAP");
  t.proposalCap = envUint("RICHENGINE_PROPOSAL_CAP", 0);
  t.denyDebug = envFlag("RICHENGINE_DENY_DEBUG");
  return t;
}

} // namespace detail

// The process-wide snapshot, built on first call.
inline const Tuning &tuning() {
  static const Tuning snapshot = detail::makeTuning();
  return snapshot;
}

} // namespace richengine
