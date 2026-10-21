// The served model's ModelPolicy must resolve as: an explicitly set
// RICHENGINE_* variable, else the family maker's declared default, else the
// knob's shipped default — and a model swap must drop the previous model's
// defaults entirely. These run in one process, so the environment is set
// before the first tuning() snapshot builds.

#include "TestChecks.hpp"
#include "model/ModelPolicy.hpp"

#include <cstdlib>
#include <iostream>

namespace {

using richengine::model::ModelTuning;
using richengine::model::detail::resolveModelPolicy;
using richengine::model::installModelPolicy;
using richengine::model::modelPolicy;
using richengine::test::require;

} // namespace

int main() {
  // Before the tuning snapshot: two explicit overrides, everything else
  // unset so the maker defaults show through.
  setenv("RICHENGINE_MOE_UNION", "7", 1);
  setenv("RICHENGINE_DRAFT_BYPASS", "0", 1);
  for (const char *name :
       {"RICHENGINE_MOE_TOPK", "RICHENGINE_MOE_PACKED_OFF",
        "RICHENGINE_GDN_CHUNKED", "RICHENGINE_PREFILL_FAST_INT8",
        "RICHENGINE_GEGLU_OFF", "RICHENGINE_NO_FUSED_GATE",
        "RICHENGINE_DRAFT_SPLITS", "RICHENGINE_DFLASH_POOL",
        "RICHENGINE_PROPOSAL_CAP", "RICHENGINE_CANVAS_PROFILE",
        "RICHENGINE_ADAPTIVE_PROPOSALS", "RICHENGINE_NGRAM_PREDRAFT",
        "RICHENGINE_DRAFT_BYPASS_EXPECT", "RICHENGINE_NGRAM_DRAFT_EXPECT",
        "RICHENGINE_NGRAM_WARMUP", "RICHENGINE_NGRAM_TREE_MIN"})
    unsetenv(name);

  ModelTuning maker;
  maker.moeTopK = 3;
  maker.moeUnionCap = 12;       // env "7" wins
  maker.gdnChunked = 64;
  maker.prefillFastInt8 = true;
  maker.draftBypass = true;     // env "0" wins
  maker.ngramPredraft = false;
  maker.ngramDraftExpect = 2.5;
  maker.ngramWarmup = 32;
  maker.proposalCap = 5;
  maker.canvasProfile = "fast";

  const auto resolved = resolveModelPolicy(maker);
  require(resolved.moeUnionCap == 7, "env RICHENGINE_MOE_UNION lost to the maker");
  require(resolved.moeTopK == 3, "maker moeTopK did not apply");
  require(resolved.gdnChunked == 64, "maker gdnChunked did not apply");
  require(resolved.prefillFastInt8, "maker prefillFastInt8 did not apply");
  require(!resolved.draftBypass, "env RICHENGINE_DRAFT_BYPASS=0 lost to the maker");
  require(!resolved.ngramPredraft, "maker ngramPredraft did not apply");
  require(resolved.ngramDraftExpect == 2.5, "maker ngramDraftExpect did not apply");
  require(resolved.ngramWarmup == 32, "maker ngramWarmup did not apply");
  require(resolved.proposalCap && *resolved.proposalCap == 5,
          "maker proposalCap did not apply");
  require(resolved.canvasProfile == "fast", "maker canvasProfile did not apply");
  // The resolved profile derives the bundled knobs the env leaves unset.
  require(resolved.canvasExitStable == 0.6F,
          "fast profile did not derive canvasExitStable");
  require(resolved.canvasMaxSteps == 24,
          "fast profile did not derive canvasMaxSteps");

  // An empty maker falls back to the shipped defaults, env still winning.
  const auto empty = resolveModelPolicy(ModelTuning{});
  require(empty.moeUnionCap == 7, "env RICHENGINE_MOE_UNION lost on empty maker");
  require(empty.moeTopK == 0, "empty maker kept a stale moeTopK");
  require(!empty.prefillFastInt8, "empty maker kept a stale prefillFastInt8");
  require(!empty.draftBypass, "env RICHENGINE_DRAFT_BYPASS=0 lost on empty maker");
  require(empty.ngramPredraft, "empty maker lost the shipped ngramPredraft");
  require(empty.ngramWarmup == 8, "empty maker lost the shipped ngramWarmup");
  require(!empty.proposalCap, "empty maker kept a stale proposalCap");
  require(empty.canvasProfile.empty(),
          "empty maker kept a stale canvasProfile");
  require(empty.canvasExitStable == 0.9F,
          "empty maker kept the fast profile's canvasExitStable");

  // Installing a second model drops the first's defaults: no leak.
  installModelPolicy(maker);
  require(modelPolicy().moeTopK == 3, "installed maker policy did not apply");
  installModelPolicy(ModelTuning{});
  require(modelPolicy().moeTopK == 0,
          "model swap leaked the previous model's moeTopK");
  require(modelPolicy().ngramPredraft,
          "model swap leaked the previous model's ngramPredraft");
  require(!modelPolicy().proposalCap,
          "model swap leaked the previous model's proposalCap");

  std::cout << "model_policy_test: ok\n";
  return 0;
}
