#pragma once

// The served model's resolved runtime policy: the family maker's ModelTuning
// defaults, each overlaid by an explicitly set RICHENGINE_* variable.
//
// A serving process binds exactly one model (Runtime::Impl installs the
// policy when the package loads, before any arena or op consults it), so a
// call site too deep for the descriptor — ops/MoE.cpp's union cap,
// ops/GDN.cpp's chunk factor — reads modelPolicy() instead of the raw
// environment. One model's defaults can then never leak into another's, and
// the process env remains the explicit override layer.
//
// Fields not under the model's control (debug gates, kernel ABI geometry)
// stay on tuning().

#include "Env.hpp"
#include "Tuning.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "model/ModelTuning.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace richengine::model {

struct ModelPolicy final {
  // ops/MoE.cpp
  uint32_t moeTopK = 0;      // RICHENGINE_MOE_TOPK
  uint32_t moeUnionCap = 0;  // RICHENGINE_MOE_UNION
  bool moePackedOff = false; // RICHENGINE_MOE_PACKED_OFF
  // ops/GDN.cpp, model/PrefillArena.mm: the WY/UT scan's chunk factor
  // (32/64/128; anything else disables).
  uint32_t gdnChunked = 0;   // RICHENGINE_GDN_CHUNKED
  // ops/Linear.cpp
  bool prefillFastInt8 = false; // RICHENGINE_PREFILL_FAST_INT8
  // model/TargetModel*.cpp
  bool gegluOff = false;    // RICHENGINE_GEGLU_OFF
  bool noFusedGate = false; // RICHENGINE_NO_FUSED_GATE
  // ops/DraftAttention.cpp: the draft kernel's split count, clamped to the
  // compile-time maximum.
  uint32_t draftAttentionSplits = RICHENGINE_DRAFT_ATTENTION_SPLITS;
  // model/DFlashTwoDraft.cpp
  bool dflashPool = false; // RICHENGINE_DFLASH_POOL=1
  // model/RuntimeImpl.hpp: the decode proposal ceiling; unset leaves the
  // draft checkpoint's trained block bound.
  std::optional<uint32_t> proposalCap; // RICHENGINE_PROPOSAL_CAP
  // metal/MetalEncode.mm: baked indirect dispatch off entirely.
  bool icbOff = false;                  // RICHENGINE_ICB_OFF
  // model/RuntimeEncodeDraft.mm: the in-command chained predraft.
  bool chainPredraft = false;           // RICHENGINE_CHAIN_PREDRAFT
  // model/RuntimeEncodeDraft.mm + Runtime.mm: the DSpark confidence-prune
  // threshold; 0 leaves the proposals unpruned.
  double draftConfMin = 0.0;            // RICHENGINE_DRAFT_CONF_MIN
  // model/RuntimeImpl.hpp adaptive decode gates: an explicitly set variable
  // wins a disable; otherwise the family maker's default applies.
  bool adaptiveProposals = false;  // RICHENGINE_ADAPTIVE_PROPOSALS
  bool draftBypass = true;        // RICHENGINE_DRAFT_BYPASS
  bool ngramPredraft = true;      // RICHENGINE_NGRAM_PREDRAFT
  bool gpuPredraft = true;        // RICHENGINE_GPU_PREDRAFT
  double draftBypassExpect = 0.15; // RICHENGINE_DRAFT_BYPASS_EXPECT
  double ngramDraftExpect = 4.0;   // RICHENGINE_NGRAM_DRAFT_EXPECT
  uint32_t ngramWarmup = 8;        // RICHENGINE_NGRAM_WARMUP
  double ngramTreeMin = 1.0;       // RICHENGINE_NGRAM_TREE_MIN
  // Cross-request corpus (model/NgramIndex.hpp Corpus): 0 off, 1 all
  // greedy lanes, 2 hinted lanes only; CORPUS_ALL includes the prompt.
  uint32_t ngramCorpus = 0;        // RICHENGINE_NGRAM_CORPUS
  bool ngramCorpusAll = false;     // RICHENGINE_NGRAM_CORPUS_ALL
  uint32_t ngramCorpusMargin = 5;  // RICHENGINE_NGRAM_CORPUS_MARGIN
  double treeAcceptMin = 0.0;      // RICHENGINE_TREE_ACCEPT_MIN
  double treeAcceptMax = 6.0;      // RICHENGINE_TREE_ACCEPT_MAX
  // ops/Canvas.cpp + model/RuntimeImpl.hpp canvas policy. The profile bundles
  // the exit/cap knobs; a knob's own variable still wins over it.
  std::string canvasProfile;      // RICHENGINE_CANVAS_PROFILE
  float canvasExitStable = 0.9F;  // RICHENGINE_CANVAS_EXIT_STABLE
  bool canvasDriftExit = false;   // RICHENGINE_CANVAS_EXIT_DRIFT
  uint32_t canvasMaxSteps = 0;    // RICHENGINE_CANVAS_MAX_STEPS
  // Canvas execution knobs outside the profile bundle; an explicit variable
  // wins, then the maker's default, then the shipped default.
  uint32_t canvasStepsPerCmd = 4;  // RICHENGINE_CANVAS_STEPS_PER_CMD (clamped 1..4)
  bool canvasPrefixExit = true;    // RICHENGINE_CANVAS_PREFIX_EXIT
  bool canvasCommitTail = true;    // RICHENGINE_CANVAS_COMMIT_TAIL
  bool canvasSpecPrefill = true;   // RICHENGINE_CANVAS_SPECULATIVE_PREFILL
  // model/RuntimeImpl.hpp: the constrained-decode (DFA) scope — "off",
  // "forced" or "full"; empty keeps the host path.
  std::string dfaMode;            // RICHENGINE_DFA_MODE
};

namespace detail {

[[nodiscard]] inline ModelPolicy resolveModelPolicy(const ModelTuning &maker) {
  const Tuning &env = tuning();
  ModelPolicy p;
  // An explicitly set variable wins; else the maker's declared default; else
  // the knob's shipped default (the env read's own fallback).
  p.moeTopK = envFlag("RICHENGINE_MOE_TOPK")
                  ? env.moeTopK
                  : maker.moeTopK.value_or(0);
  p.moeUnionCap = envFlag("RICHENGINE_MOE_UNION")
                      ? env.moeUnionCap
                      : maker.moeUnionCap.value_or(0);
  p.moePackedOff = envFlag("RICHENGINE_MOE_PACKED_OFF")
                       ? env.moePackedOff
                       : maker.moePackedOff.value_or(false);
  p.gdnChunked = envFlag("RICHENGINE_GDN_CHUNKED")
                     ? env.gdnChunked
                     : maker.gdnChunked.value_or(0);
  p.prefillFastInt8 = envFlag("RICHENGINE_PREFILL_FAST_INT8")
                          ? env.prefillFastInt8
                          : maker.prefillFastInt8.value_or(false);
  p.gegluOff = envFlag("RICHENGINE_GEGLU_OFF")
                   ? env.gegluOff
                   : maker.gegluOff.value_or(false);
  p.noFusedGate = envFlag("RICHENGINE_NO_FUSED_GATE")
                      ? env.noFusedGate
                      : maker.noFusedGate.value_or(false);
  p.draftAttentionSplits = std::clamp(
      envFlag("RICHENGINE_DRAFT_SPLITS")
          ? env.draftAttentionSplits
          : maker.draftAttentionSplits.value_or(RICHENGINE_DRAFT_ATTENTION_SPLITS),
      1u, RICHENGINE_DRAFT_SPLITS_MAXIMUM);
  p.dflashPool = envFlag("RICHENGINE_DFLASH_POOL")
                     ? env.dflashPool
                     : maker.dflashPool.value_or(false);
  p.proposalCap = env.proposalCapSet ? std::optional<uint32_t>(env.proposalCap)
                                     : maker.proposalCap;
  p.icbOff = envFlag("RICHENGINE_ICB_OFF") ? env.icbOff
                                         : maker.icbOff.value_or(false);
  p.chainPredraft = envFlag("RICHENGINE_CHAIN_PREDRAFT")
                        ? env.chainPredraft
                        : maker.chainPredraft.value_or(false);
  p.draftConfMin = env.draftConfMinSet ? env.draftConfMin
                                       : maker.draftConfMin.value_or(0.0);
  // unlessZero knobs: the variable only ever disables, so an explicit set
  // wins; otherwise the maker's default stands.
  p.adaptiveProposals = envFlag("RICHENGINE_ADAPTIVE_PROPOSALS")
                            ? env.adaptiveProposals
                            : maker.adaptiveProposals;
  p.draftBypass = envFlag("RICHENGINE_DRAFT_BYPASS") ? env.draftBypass
                                                   : maker.draftBypass;
  p.ngramPredraft = envFlag("RICHENGINE_NGRAM_PREDRAFT") ? env.ngramPredraft
                                                       : maker.ngramPredraft;
  p.gpuPredraft = envFlag("RICHENGINE_GPU_PREDRAFT") ? env.gpuPredraft
                                                   : maker.gpuPredraft;
  p.draftBypassExpect = env.draftBypassExpectSet ? env.draftBypassExpect
                                                 : maker.draftBypassExpect;
  p.ngramDraftExpect = env.ngramDraftExpectSet ? env.ngramDraftExpect
                                               : maker.ngramDraftExpect;
  p.ngramWarmup =
      env.ngramWarmupSet ? env.ngramWarmup : maker.ngramWarmup;
  p.ngramTreeMin =
      env.ngramTreeMinSet ? env.ngramTreeMin : maker.ngramTreeMin;
  p.ngramCorpus = env.ngramCorpus;
  p.ngramCorpusAll = env.ngramCorpusAll;
  p.ngramCorpusMargin = env.ngramCorpusMargin;
  p.treeAcceptMin =
      env.treeAcceptMinSet ? env.treeAcceptMin : maker.treeAcceptMin;
  p.treeAcceptMax =
      env.treeAcceptMaxSet ? env.treeAcceptMax : maker.treeAcceptMax;
  p.canvasProfile =
      env.canvasProfile.empty()
          ? maker.canvasProfile.value_or(std::string{})
          : std::string(env.canvasProfile);
  // The profile-derived knobs: an explicit variable wins, then the resolved
  // profile's bundle (the maker's or the environment's), then "paper".
  if (std::getenv("RICHENGINE_CANVAS_EXIT_STABLE"))
    p.canvasExitStable = env.canvasExitStable;
  else
    p.canvasExitStable = p.canvasProfile == "fast"     ? 0.6F
                         : p.canvasProfile == "balanced" ? 0.75F
                                                          : 0.9F;
  if (std::getenv("RICHENGINE_CANVAS_EXIT_DRIFT"))
    p.canvasDriftExit = env.canvasDriftExit;
  else
    p.canvasDriftExit =
        p.canvasProfile == "balanced" || p.canvasProfile == "fast";
  if (std::getenv("RICHENGINE_CANVAS_MAX_STEPS"))
    p.canvasMaxSteps = env.canvasMaxSteps;
  else
    p.canvasMaxSteps = p.canvasProfile == "fast"     ? 24u
                       : p.canvasProfile == "balanced" ? 32u
                                                       : 0u;
  p.canvasStepsPerCmd =
      std::getenv("RICHENGINE_CANVAS_STEPS_PER_CMD")
          ? env.canvasStepsPerCmd
          : std::clamp(maker.canvasStepsPerCmd.value_or(4u), 1u, 4u);
  p.canvasPrefixExit = std::getenv("RICHENGINE_CANVAS_PREFIX_EXIT")
                           ? env.canvasPrefixExit
                           : maker.canvasPrefixExit.value_or(true);
  p.canvasCommitTail = std::getenv("RICHENGINE_CANVAS_COMMIT_TAIL")
                           ? env.canvasCommitTail
                           : maker.canvasCommitTail.value_or(true);
  p.canvasSpecPrefill = std::getenv("RICHENGINE_CANVAS_SPECULATIVE_PREFILL")
                            ? env.canvasSpeculativePrefill
                            : maker.canvasSpeculativePrefill.value_or(true);
  p.dfaMode = env.dfaMode.empty() ? maker.dfaMode.value_or(std::string{})
                                : std::string(env.dfaMode);
  return p;
}

// The policy's process slot: initialized env-resolved with an empty maker,
// so a caller that never installs (inspection tools, tests) sees exactly
// what tuning() reads today.
[[nodiscard]] inline ModelPolicy &policySlot() {
  static ModelPolicy value = resolveModelPolicy(ModelTuning{});
  return value;
}

} // namespace detail

// The active policy. Installed when the model loads; before then it holds
// the env-resolved values of an empty maker (identical to reading tuning()).
[[nodiscard]] inline const ModelPolicy &modelPolicy() {
  return detail::policySlot();
}

// Runtime::Impl installs the loaded descriptor's ModelTuning once, before
// arenas or ops consult it.
inline void installModelPolicy(const ModelTuning &maker) {
  detail::policySlot() = detail::resolveModelPolicy(maker);
}

} // namespace richengine::model
