#pragma once

// Central registry of the engine's RICHENGINE_* development/tuning knobs.
// Call sites take tuning().field instead of reading the environment: the
// snapshot is built once per process (function-local static), the same
// once-read semantics the scattered `static const` reads had. Flag
// conventions are preserved from Env.hpp — a "(presence)" comment means
// envFlag, "=1" means envFlagOn, a bare name means envUint parse-or-default.

#include "Env.hpp"
#include "metal/abi/ExecutionGeometry.h"

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
  // ops/MoE.cpp, ops/MoE.hpp
  bool moeGgufFused = false;      // RICHENGINE_MOE_GGUF_FUSED (presence)
  std::string_view moeGgufTile;   // RICHENGINE_MOE_GGUF_TILE=register|staged
  // ops/GDN.cpp, model/RuntimeArenas.mm
  uint32_t gdnChunked = 0;    // RICHENGINE_GDN_CHUNKED
  // ops/Linear.cpp, ops/LinearGguf.cpp
  bool ggufPackedOn = false;  // RICHENGINE_GGUF_PACKED_ON (presence)
  bool ggufPackedOff = false; // RICHENGINE_GGUF_PACKED_OFF (presence)
  // ops/Linear.cpp, model/RuntimeArenas.mm
  bool prefillFastInt8 = false; // RICHENGINE_PREFILL_FAST_INT8 (presence)
  // metal/MetalEncode.mm
  bool icbOff = false;          // RICHENGINE_ICB_OFF (presence)
  bool mtl4 = false;            // RICHENGINE_MTL4_OFF disables (default on)
  // Baked spans replay while another command is in flight: the parameter
  // arena and ICB of the submission's parity bank are built on first use, so
  // a backend that never overlaps pays nothing for the second bank. Off
  // restores the GPU-idle-only replay.
  bool spanBankOff = false;     // RICHENGINE_SPAN_BANK_OFF (presence)
  // A baked span barriers a command only where its bytes overlap an earlier
  // command's since the last barrier. Real spans are chains (N-1 of N
  // barriers needed), so this is opt-in until a graph with independent
  // branches shows a win.
  bool spanBarriers = false;    // RICHENGINE_SPAN_BARRIERS (presence)
  // Event-bearing commands on the Metal 4 encoder (queue-level signal/wait
  // between the split's commits). Off keeps the Metal 3 encoder.
  bool mtl4Events = false;      // RICHENGINE_MTL4_EVENTS (presence)
  bool preparedCacheOff = false; // RICHENGINE_PREPARED_CACHE_OFF (presence)
  bool opTimings = false;       // RICHENGINE_OP_TIMINGS=1
  bool gpuSpans = false;        // RICHENGINE_GPU_SPANS (presence)
  // ops/PagedAttention.cpp: the kv4/g6 16-row chain verify splits its two
  // fused-row passes across two threadgroups (the _r2 kernels), halving the
  // tile's threadgroup footprint so two groups share a core. Measured 1.40x
  // at 4K history, 1.36x at 32K, 1.24x at 131K on the q8 attention harness.
  // Off restores the single-group split.
  bool verifySplitR2Off = false; // RICHENGINE_VERIFY_SPLIT_R2_OFF (presence)
  // model/RuntimeEncodeVerify.mm: the verify round's GPU-DFA mask pass
  // launches through an indirect grid written on-device by
  // decode_dfa_mask_grid, which also consumes the lanes' GPU forced spans.
  bool gpuDecodeGrid = false;   // RICHENGINE_GPU_DECODE_GRID_OFF disables (default on)
  // model/RuntimeEncodeVerify.mm: decode_step_pack mirrors each lane's
  // acceptance output into a device params block the next step's
  // verify-input pass binds for its anchor and live rows — the encode
  // carries no host-readback dependency on the previous step.
  bool deviceVerifyParams = false; // RICHENGINE_DEVICE_VERIFY_PARAMS_OFF disables (default on)
  // model/RuntimeImpl.hpp: a Full-mode constrained batch whose lanes all
  // carry a GPU DFA and owe no host-pushed forced span encodes draft,
  // verify and commit as one command — decode_dfa_span_apply moves the
  // forced-span patch on-device and the WaitingMask stage never opens.
  // Unset keeps the three-command split.
  bool dfaMerge = false;          // RICHENGINE_DFA_MERGE_OFF disables (default on)
  // model/Runtime.mm: a greedy DFA lane whose pending GPU forced span
  // covers its whole proposal budget takes every live proposal row from
  // the span, so when every lane of a batch qualifies the draft embedding
  // and forward encode nothing the verify stage reads and are skipped.
  bool dfaSpanSkipDraft = false;  // RICHENGINE_DFA_SPAN_SKIP_DRAFT_OFF disables (default on)
  // ops/Embedding.cpp: a presence flag restores the pre-anchor-fill verify
  // routing (every row live), the A/B kill switch for the anchor-fill path.
  bool verifyAllLive = false;   // RICHENGINE_VERIFY_ALL_LIVE (presence)
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
  // ops/LfmConv.cpp: the chain verify mixer writes the out-projection's
  // Table16 operand itself (verify_lfm_conv_v4_table16) when the consumer's
  // plan wants it — a register-tile out_proj then skips its
  // decode_linear_gguf_prepare dispatch per conv layer. Measured −0.2…1.9%
  // decode on LFM2.5-24B; bitwise-identical.
  bool convFuse = false;            // RICHENGINE_CONV_FUSE_OFF disables (default on)
  // model/RuntimeImpl.hpp speculative-decode policy
  bool verifyTreeSet = false;       // RICHENGINE_VERIFY_TREE (presence)
  bool verifyTree = false;          // RICHENGINE_VERIFY_TREE=1
  bool adaptiveProposals = true;    // RICHENGINE_ADAPTIVE_PROPOSALS, =0 disables
  bool ngramPredraft = true;        // RICHENGINE_NGRAM_PREDRAFT, =0 disables
  bool draftBypass = true;          // RICHENGINE_DRAFT_BYPASS, =0 disables
  bool gpuPredraft = true;          // RICHENGINE_GPU_PREDRAFT, =0 disables
  // GPU-DFA constraint scope (model/RuntimeImpl.hpp): "full" computes the
  // lane's masks and forced spans on the GPU, "forced" keeps the host mask
  // round trip and only derives forced spans and the automaton state
  // on-device, "off" drops every DFA. Anything else is "full".
  std::string_view dfaMode;         // RICHENGINE_DFA_MODE
  double draftBypassExpect = 0.15;  // RICHENGINE_DRAFT_BYPASS_EXPECT
  double ngramDraftExpect = 4.0;    // RICHENGINE_NGRAM_DRAFT_EXPECT
  // A DSpark draft's confidence head scores each emitted proposal when
  // set; positions whose head sigmoid falls under it cut the lane's
  // proposal suffix before verify (draft_conf_score + VerifyRowsParams's
  // conf_min). 0 disables.
  double draftConfMin = 0.0;        // RICHENGINE_DRAFT_CONF_MIN
  uint32_t ngramWarmup = 8;         // RICHENGINE_NGRAM_WARMUP
  double ngramTreeMin = 1.0;        // RICHENGINE_NGRAM_TREE_MIN
  double treeAcceptMin = 0.0;       // RICHENGINE_TREE_ACCEPT_MIN
  double treeAcceptMax = 6.0;       // RICHENGINE_TREE_ACCEPT_MAX
  // Presence flags for the threshold knobs above: a family's ModelTuning
  // value is the default and an explicitly set variable overrides it.
  bool draftBypassExpectSet = false;
  bool ngramDraftExpectSet = false;
  bool ngramWarmupSet = false;
  bool ngramTreeMinSet = false;
  bool treeAcceptMinSet = false;
  bool treeAcceptMaxSet = false;
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
  // model/DFlashTwoDraft.cpp
  bool dflashPoolSet = false;       // RICHENGINE_DFLASH_POOL (presence)
  bool dflashPool = false;          // RICHENGINE_DFLASH_POOL=1
  // Registered for completeness; the call sites below live in engine/ or in
  // model/*.mm files outside the conversion scope and still read the
  // environment directly.
  bool submitAhead = false;         // RICHENGINE_SUBMIT_AHEAD — engine/Engine.cpp
  bool decodeRing2 = false;         // RICHENGINE_DECODE_RING2_OFF disables (default on) — engine/Engine.cpp
  bool decodeTiming = false;        // RICHENGINE_DECODE_TIMING — model/Runtime.mm
  bool headFusedOff = false;        // RICHENGINE_HEAD_FUSED_OFF — model/RuntimeEncode.mm
  bool proposalCapSet = false;      // RICHENGINE_PROPOSAL_CAP (presence) — model/Runtime.mm
  uint32_t proposalCap = 0;         // RICHENGINE_PROPOSAL_CAP — model/Runtime.mm
  bool denyDebug = false;           // RICHENGINE_DENY_DEBUG — engine/
  // RICHENGINE_SYSTEMONE_WINDOW — server/frontend.py caps /v1/systemone's
  // in-flight questions below the preparation bound; the runtime never
  // reads it.
  // ops/LinearGguf.cpp, model/TargetModel.cpp: an all-score batch's head
  // dispatches only the quant tiles holding a wanted logit column.
  bool scoreGather = false;         // RICHENGINE_SCORE_GATHER=1
  // metal/kernels/shared/gguf_score.metal: the same restriction as one
  // gather dispatch over the wanted columns, the fused variant folding the
  // final RMS norm into it.
  bool scoreTensor = false;         // RICHENGINE_SCORE_TENSOR=1
  bool scoreFused = false;          // RICHENGINE_SCORE_FUSED=1
  // ops/LinearGguf.cpp: GGUF decode tiles take their packed-operand
  // (tensor-operand) variants for the formats that have one — the
  // SCORE_GATHER head's dispatch among them. RICHENGINE_GGUF_PACKED_ON's
  // enablement scoped to the score pathway's kernels.
  bool scoreTensorPrefill = false;  // RICHENGINE_SCORE_TENSOR_PREFILL=1
  // Kernel geometry the registry can reshape at process start
  // (ExecutionGeometry.h's table). The draft split count reaches
  // draft.metal as the Metal function constant
  // RICHENGINE_DRAFT_SPLITS_FUNCTION_CONSTANT (MetalEncode.mm) — workspace,
  // dispatch and kernel read the same snapshot. The verify/prefill maxima
  // only cap downward: workspaces stay sized for the compile-time maximum
  // and the shaders' contract bounds stay the baked ones.
  uint32_t draftAttentionSplits =
      RICHENGINE_DRAFT_ATTENTION_SPLITS;  // RICHENGINE_DRAFT_SPLITS
  uint32_t verifySplitsMax =
      RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS;  // RICHENGINE_VERIFY_SPLITS_MAX
  uint32_t prefillSplitsMax =
      RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS;  // RICHENGINE_PREFILL_SPLITS_MAX
  // The install's measured affine Linear plans, written into tuning.json by
  // the autotune's kernel pass (install/autotune.py) as
  // "{{output, input}, rows, LinearPhase::X, LinearEpilogue::Y}="
  // "{LinearTile::Z, groups, LinearSimdgroups::W, splits}" entries separated
  // by ';' — tune-kernels' report text verbatim. ops/DeviceTuning.cpp
  // parses it once and lets an entry outrank the shipped measured tables.
  const char *linearPlans = nullptr;  // RICHENGINE_LINEAR_PLANS
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
  t.moeGgufFused = envFlag("RICHENGINE_MOE_GGUF_FUSED");
  t.moeGgufTile = stringOrEmpty("RICHENGINE_MOE_GGUF_TILE");
  t.gdnChunked = envUint("RICHENGINE_GDN_CHUNKED", 0);
  t.ggufPackedOn = envFlag("RICHENGINE_GGUF_PACKED_ON");
  t.ggufPackedOff = envFlag("RICHENGINE_GGUF_PACKED_OFF");
  t.prefillFastInt8 = envFlag("RICHENGINE_PREFILL_FAST_INT8");
  t.icbOff = envFlag("RICHENGINE_ICB_OFF");
  // A/B-measured winners default on; the *_OFF flag is the escape hatch.
  t.mtl4 = !envFlag("RICHENGINE_MTL4_OFF");
  t.spanBankOff = envFlag("RICHENGINE_SPAN_BANK_OFF");
  t.spanBarriers = envFlag("RICHENGINE_SPAN_BARRIERS");
  t.mtl4Events = envFlag("RICHENGINE_MTL4_EVENTS");
  t.preparedCacheOff = envFlag("RICHENGINE_PREPARED_CACHE_OFF");
  t.verifySplitR2Off = envFlag("RICHENGINE_VERIFY_SPLIT_R2_OFF");
  t.opTimings = envFlagOn("RICHENGINE_OP_TIMINGS");
  t.gpuSpans = envFlag("RICHENGINE_GPU_SPANS");
  t.gpuDecodeGrid = !envFlag("RICHENGINE_GPU_DECODE_GRID_OFF");
  t.deviceVerifyParams = !envFlag("RICHENGINE_DEVICE_VERIFY_PARAMS_OFF");
  t.dfaMerge = !envFlag("RICHENGINE_DFA_MERGE_OFF");
  t.dfaSpanSkipDraft = !envFlag("RICHENGINE_DFA_SPAN_SKIP_DRAFT_OFF");
  t.verifyAllLive = envFlag("RICHENGINE_VERIFY_ALL_LIVE");
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
  t.convFuse = !envFlag("RICHENGINE_CONV_FUSE_OFF");
  t.verifyTreeSet = envFlag("RICHENGINE_VERIFY_TREE");
  t.verifyTree = envFlagOn("RICHENGINE_VERIFY_TREE");
  t.adaptiveProposals = unlessZero("RICHENGINE_ADAPTIVE_PROPOSALS");
  t.ngramPredraft = unlessZero("RICHENGINE_NGRAM_PREDRAFT");
  t.draftBypass = unlessZero("RICHENGINE_DRAFT_BYPASS");
  t.gpuPredraft = unlessZero("RICHENGINE_GPU_PREDRAFT");
  t.dfaMode = stringOrEmpty("RICHENGINE_DFA_MODE");
  t.draftBypassExpectSet = envFlag("RICHENGINE_DRAFT_BYPASS_EXPECT");
  t.draftBypassExpect = envDouble("RICHENGINE_DRAFT_BYPASS_EXPECT", 0.15);
  t.ngramDraftExpectSet = envFlag("RICHENGINE_NGRAM_DRAFT_EXPECT");
  t.ngramDraftExpect = envDouble("RICHENGINE_NGRAM_DRAFT_EXPECT", 4.0);
  t.draftConfMin = envDouble("RICHENGINE_DRAFT_CONF_MIN", 0.0);
  t.ngramWarmupSet = envFlag("RICHENGINE_NGRAM_WARMUP");
  t.ngramWarmup = envUint("RICHENGINE_NGRAM_WARMUP", 8);
  t.ngramTreeMinSet = envFlag("RICHENGINE_NGRAM_TREE_MIN");
  t.ngramTreeMin = envDouble("RICHENGINE_NGRAM_TREE_MIN", 1.0);
  t.treeAcceptMinSet = envFlag("RICHENGINE_TREE_ACCEPT_MIN");
  t.treeAcceptMin = envDouble("RICHENGINE_TREE_ACCEPT_MIN", 0.0);
  t.treeAcceptMaxSet = envFlag("RICHENGINE_TREE_ACCEPT_MAX");
  t.treeAcceptMax = envDouble("RICHENGINE_TREE_ACCEPT_MAX", 6.0);
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
  t.decodeRing2 = !envFlag("RICHENGINE_DECODE_RING2_OFF");
  t.decodeTiming = envFlag("RICHENGINE_DECODE_TIMING");
  t.headFusedOff = envFlag("RICHENGINE_HEAD_FUSED_OFF");
  t.proposalCapSet = envFlag("RICHENGINE_PROPOSAL_CAP");
  t.proposalCap = envUint("RICHENGINE_PROPOSAL_CAP", 0);
  t.denyDebug = envFlag("RICHENGINE_DENY_DEBUG");
  t.scoreGather = envFlagOn("RICHENGINE_SCORE_GATHER");
  t.scoreTensor = envFlagOn("RICHENGINE_SCORE_TENSOR");
  t.scoreFused = envFlagOn("RICHENGINE_SCORE_FUSED");
  t.scoreTensorPrefill = envFlagOn("RICHENGINE_SCORE_TENSOR_PREFILL");
  t.draftAttentionSplits = std::clamp(
      envUint("RICHENGINE_DRAFT_SPLITS", RICHENGINE_DRAFT_ATTENTION_SPLITS),
      1u, RICHENGINE_DRAFT_SPLITS_MAXIMUM);
  t.verifySplitsMax = std::clamp(
      envUint("RICHENGINE_VERIFY_SPLITS_MAX",
              RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS),
      1u, RICHENGINE_VERIFY_ATTENTION_MAXIMUM_SPLITS);
  t.prefillSplitsMax = std::clamp(
      envUint("RICHENGINE_PREFILL_SPLITS_MAX",
              RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS),
      1u, RICHENGINE_PREFILL_ATTENTION_MAXIMUM_SPLITS);
  t.linearPlans = std::getenv("RICHENGINE_LINEAR_PLANS");
  return t;
}

} // namespace detail

// The process-wide snapshot, built on first call.
inline const Tuning &tuning() {
  static const Tuning snapshot = detail::makeTuning();
  return snapshot;
}

} // namespace richengine
