#pragma once

// Per-model runtime policy, set by the family makers (model/families/*.mm).
// This is the home of tunables a family can veto or flip without a code
// change elsewhere: geometry stays in the target/draft layouts, kernel-baked
// constants stay in metal/abi/ExecutionGeometry.h, and only runtime-legal
// values belong here. A field needs no manifest key: makers set it directly.
//
// The optional fields are the family's declared defaults for knobs whose
// call sites resolve through modelPolicy() (model/ModelPolicy.hpp): an
// explicitly set RICHENGINE_* variable still overrides, so the env wins a
// user override while the maker — not the process environment — states the
// model's own default. An unset optional keeps the knob's shipped default.

#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace richengine::model {

struct ModelTuning final {
  // How the model's draft comb-tree verify tables engage
  // (Runtime::Impl::treeDraftCapable). RICHENGINE_VERIFY_TREE overrides the
  // default either way when set.
  enum class TreeVerify : uint8_t {
    Off,      // the draft never emits a tree table
    OptIn,    // trees only when the env opts in
    On,       // trees by default
    Adaptive, // trees while every lane's acceptance EWMA sits in the band
  };
  TreeVerify treeVerify = TreeVerify::Off;

  // Adaptive admission band for the comb verify, in accepted-proposal EWMA
  // (Runtime::Impl::treeVerifyBatch): a lane joins a tree batch only while
  // its rolling accepted count stays inside [treeAcceptMin, treeAcceptMax].
  // Below the comb's own chain depth the chain's extra verify rows are
  // statistically unused anyway and the sibling leaves can rescue a rejected
  // position's runner-up; at saturation the deeper chain wins and at
  // near-zero the doubled row block rescues nothing. The upper default sits
  // one token under the comb's seven-proposal chain coverage.
  // RICHENGINE_TREE_ACCEPT_MIN / RICHENGINE_TREE_ACCEPT_MAX override.
  double treeAcceptMin = 0.0;
  double treeAcceptMax = 6.0;

  // Adaptive decode gates. Each family's value is a default a maker can
  // drop; the matching RICHENGINE_* knob still applies on top, so the env
  // always wins a disable.
  // Off by default: a saved verify row costs nearly nothing against a
  // full weight pass — the draft forward and the verify's fixed work
  // dominate the step — so shrinking toward the acceptance EWMA's +2
  // margin only loses tail tokens. Measured ~10 tok/s at B1 on the 27B
  // and ~23% on a 15-proposal draft (Ornith 9B, M5 Pro).
  // RICHENGINE_ADAPTIVE_PROPOSALS=1 restores it.
  bool adaptiveProposals = false; // RICHENGINE_ADAPTIVE_PROPOSALS
  bool draftBypass = true;       // RICHENGINE_DRAFT_BYPASS
  bool ngramPredraft = true;     // RICHENGINE_NGRAM_PREDRAFT
  // Next-step GPU draft encoded at a committed step's finalize: its forward
  // overlaps the host's result handling instead of the next command.
  bool gpuPredraft = true;       // RICHENGINE_GPU_PREDRAFT

  // Adaptive-gate thresholds. These are the family's tuned defaults; an
  // explicitly set RICHENGINE_* variable overrides each one (the env's own
  // literal default does not).
  double draftBypassExpect = 0.15; // break-even accepted-token EWMA
  double ngramDraftExpect = 4.0;   // per-lane expected acceptance to predraft
  uint32_t ngramWarmup = 8;        // rounds before a lane's EWMA is trusted
  double ngramTreeMin = 1.0;       // EWMA needed before a lane emits leaves

  // Maker-declared defaults resolved into ModelPolicy. Every one names the
  // RICHENGINE_* variable that overrides it.
  std::optional<uint32_t> moeTopK;        // RICHENGINE_MOE_TOPK
  std::optional<uint32_t> moeUnionCap;    // RICHENGINE_MOE_UNION
  std::optional<bool> moePackedOff;       // RICHENGINE_MOE_PACKED_OFF
  std::optional<uint32_t> gdnChunked;     // RICHENGINE_GDN_CHUNKED (32/64/128)
  std::optional<bool> prefillFastInt8;    // RICHENGINE_PREFILL_FAST_INT8
  std::optional<bool> gegluOff;           // RICHENGINE_GEGLU_OFF
  std::optional<bool> noFusedGate;        // RICHENGINE_NO_FUSED_GATE
  std::optional<uint32_t> draftAttentionSplits; // RICHENGINE_DRAFT_SPLITS
  std::optional<bool> dflashPool;         // RICHENGINE_DFLASH_POOL
  std::optional<uint32_t> proposalCap;    // RICHENGINE_PROPOSAL_CAP
  std::optional<bool> icbOff;             // RICHENGINE_ICB_OFF
  std::optional<bool> chainPredraft;      // RICHENGINE_CHAIN_PREDRAFT
  // The canvas bundle the family's diffusion model runs under
  // ("paper"|"balanced"|"fast"); RICHENGINE_CANVAS_PROFILE overrides.
  std::optional<std::string> canvasProfile;
  // The canvas execution knobs the profile does not bundle; each
  // RICHENGINE_CANVAS_* variable overrides its own knob when set.
  std::optional<uint32_t> canvasStepsPerCmd;          // RICHENGINE_CANVAS_STEPS_PER_CMD (clamped 1..4)
  std::optional<bool> canvasPrefixExit;               // RICHENGINE_CANVAS_PREFIX_EXIT
  std::optional<bool> canvasCommitTail;               // RICHENGINE_CANVAS_COMMIT_TAIL
  std::optional<bool> canvasSpeculativePrefill;       // RICHENGINE_CANVAS_SPECULATIVE_PREFILL
  // The constrained-decode (DFA) scope: "off" keeps every constraint on the
  // host path, "forced" adds on-device forced spans, "full" computes masks
  // and spans on the GPU. RICHENGINE_DFA_MODE overrides.
  std::optional<std::string> dfaMode;

  // A family's own verify-budget acceptance prior (model/VerifyBudget):
  // fitted on the family's draft, it replaces the upstream models' kPrior
  // as the acceptance model's start and shrink target — the budget's early
  // cycles then price verify rows by this draft's real confidence curve
  // instead of another model's. The sparse-MoE verify path alone consumes
  // it. Positions map to the compiled RICHENGINE_DRAFT_PROPOSAL_TOKENS.
  struct VerifyAcceptancePrior final {
    float slope = 0.0F;
    std::array<float, RICHENGINE_DRAFT_PROPOSAL_TOKENS> intercepts{};
  };
  std::optional<VerifyAcceptancePrior> verifyPrior;
};

} // namespace richengine::model
