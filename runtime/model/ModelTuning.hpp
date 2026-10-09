#pragma once

// Per-model runtime policy, set by the family makers (model/families/*.mm).
// This is the home of tunables a family can veto or flip without a code
// change elsewhere: geometry stays in the target/draft layouts, kernel-baked
// constants stay in metal/abi/ExecutionGeometry.h, and only runtime-legal
// values belong here. A field needs no manifest key: makers set it directly.

#include <cstdint>

namespace richengine::model {

struct ModelTuning final {
  // How the model's draft comb-tree verify tables engage
  // (Runtime::Impl::treeDraftCapable). RICHENGINE_VERIFY_TREE overrides the
  // default either way when set.
  enum class TreeVerify : uint8_t {
    Off,   // the draft never emits a tree table
    OptIn, // trees only when the env opts in
    On,    // trees by default
  };
  TreeVerify treeVerify = TreeVerify::Off;

  // Adaptive decode gates. Each family's value is a default a maker can
  // drop; the matching RICHENGINE_* knob still applies on top, so the env
  // always wins a disable.
  bool adaptiveProposals = true; // RICHENGINE_ADAPTIVE_PROPOSALS
  bool draftBypass = true;       // RICHENGINE_DRAFT_BYPASS
  bool ngramPredraft = true;     // RICHENGINE_NGRAM_PREDRAFT

  // Adaptive-gate thresholds. These are the family's tuned defaults; an
  // explicitly set RICHENGINE_* variable overrides each one (the env's own
  // literal default does not).
  double draftBypassExpect = 0.15; // break-even accepted-token EWMA
  double ngramDraftExpect = 4.0;   // per-lane expected acceptance to predraft
  uint32_t ngramWarmup = 8;        // rounds before a lane's EWMA is trusted
  double ngramTreeMin = 1.0;       // EWMA needed before a lane emits leaves
};

} // namespace richengine::model
