"""The autotune knob framework: what a family offers the sweep to try.

A Knob is one RICHENGINE_* environment variable plus the candidate values the
sweep measures; the engine reads these once at process start, so the sweep
spawns one engine per candidate. A knob applies only when every gate on it
matches the TuneContext — model type, draft kind, target format, GPU family
and memory bandwidth — so a family file lists everything it could ever tune
and the sweep keeps the subset this chip and this install can reach. Most
knobs are the engine's disabled-by-default paths (the pool selector, tree
verify for GPU drafts, the chunked GDN scan, packed GGUF, Metal 4): cheaper
to win by trying than to reason about, and the sweep settles each on
measurement.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TuneContext:
    """What the sweep knows about this install on this machine: enough to
    drop knobs that cannot run (a GGUF knob on an MLX install, a conv knob
    on a target without one) or that a weaker bandwidth class never wins."""

    # The family name the assembly's record states (FAMILIES' key); "" when
    # the install names a family the table does not know.
    family: str
    chip: str
    # MTLGPUFamilyAppleN; 0 when the chip's family is unknown — family-gated
    # knobs then stay out.
    gpu_family: int
    # The chip's unified-memory bandwidth in GB/s; 0 when unknown.
    bandwidth_gbps: int
    # The family's signature model_type (qwen3_5_text, lfm2, gemma4_text…).
    model_type: str
    # The installed target's format (mlx, gguf, packed-gemma4, …).
    target_format: str
    # The draft kind the descriptor loads: dflash2, dflash, dspark or none.
    draft_kind: str
    # Whether the target is a block-diffusion model (DiffusionGemma).
    diffusion: bool


@dataclass(frozen=True)
class Knob:
    """One sweepable engine switch.

    env — the RICHENGINE_* variable the runtime's Tuning registry reads.
    values — the candidates to measure, each as the variable's text; the
      sweep keeps whichever measured fastest and writes it into the model's
      tuning file.
    reason — one line on what the path changes, for the sweep's log.
    Gates (all must hold): model_types, draft_kinds, formats restrict to
    matching installs; min_gpu_family requires MTLGPUFamilyAppleN — an
    unknown family (0) stays out since the path may not exist; diffusion
    restricts to (True) or from (False) canvas targets;
    min_bandwidth_gbps holds the knob back on chips whose unified-memory
    bandwidth cannot feed the path — an unknown bandwidth (0) is unproven,
    not excluded, and measurement settles it.
    quality_sensitive knobs trade output quality for speed (canvas step
    caps, exit thresholds); the sweep leaves them out unless asked.
    ordered knobs list their candidates closest-to-default first; when the
    first loses beyond the sweep's prune depth, the sweep assumes the rest
    deviate more and lose worse — set it only where the knob's effect is
    monotone in its deviation from the default.
    """

    env: str
    values: tuple[str, ...]
    reason: str
    model_types: tuple[str, ...] = ()
    draft_kinds: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    min_gpu_family: int = 0
    min_bandwidth_gbps: int = 0
    diffusion: bool | None = None
    quality_sensitive: bool = False
    ordered: bool = False

    def applies(self, context: TuneContext) -> bool:
        return (
            (not self.model_types or context.model_type in self.model_types)
            and (not self.draft_kinds or context.draft_kind in self.draft_kinds)
            and (not self.formats or context.target_format in self.formats)
            and self.min_gpu_family <= context.gpu_family
            and (
                not context.bandwidth_gbps
                or self.min_bandwidth_gbps <= context.bandwidth_gbps
            )
            and (self.diffusion is None or self.diffusion == context.diffusion)
        )


@dataclass(frozen=True)
class TunableKnobs:
    """One model family's sweep table."""

    family: str
    knobs: tuple[Knob, ...]


def eligible(tuning: TunableKnobs, context: TuneContext) -> tuple[Knob, ...]:
    """The family's knobs this install on this chip can actually run."""
    return tuple(knob for knob in tuning.knobs if knob.applies(context))


# The knobs every drafted family shares; family files mix these with their
# own. "0" candidates are the off-by-default paths' counterpart — the sweep
# proves an on-by-default gate pays on this chip too.
VERIFY_TREE = Knob(
    "RICHENGINE_VERIFY_TREE",
    ("1", "0"),
    "comb-tree verify of draft proposals; off-by-default on GPU drafts",
)
ADAPTIVE_PROPOSALS = Knob(
    "RICHENGINE_ADAPTIVE_PROPOSALS",
    ("0",),
    "per-row proposal width vs the fixed rows; on by default",
    draft_kinds=("dflash2", "dflash", "dspark"),
)
DRAFT_BYPASS = Knob(
    "RICHENGINE_DRAFT_BYPASS",
    ("0",),
    "anchor-only decode when the draft's acceptance EWMA is low",
    draft_kinds=("dflash2", "dflash", "dspark"),
)
PREFILL_FAST_INT8 = Knob(
    "RICHENGINE_PREFILL_FAST_INT8",
    ("1",),
    "the split int8 prefill operand path; off by default",
)
MTL4 = Knob(
    "RICHENGINE_MTL4",
    ("1",),
    "the Metal 4 command-graph path; off by default",
    min_gpu_family=10,
)
GGUF_PACKED = Knob(
    "RICHENGINE_GGUF_PACKED_ON",
    ("1",),
    "packed GGUF decode operands; off by default",
    formats=("gguf",),
)
DFLASH_POOL = Knob(
    "RICHENGINE_DFLASH_POOL",
    ("1",),
    "the 128-slot pool selector over the merged top-16; off by default",
    draft_kinds=("dflash2",),
    min_bandwidth_gbps=200,
)
GDN_CHUNKED = Knob(
    "RICHENGINE_GDN_CHUNKED",
    ("32", "64", "128"),
    "the WY/UT chunked prefill scan vs the serial one; off by default",
    model_types=("qwen3_5_text", "qwen3_5_moe_text", "qwen3_6_moe_text"),
)
# The n-gram predraft's gates — the only proposer of a draft-less family.
NGRAM_PREDRAFT = Knob(
    "RICHENGINE_NGRAM_PREDRAFT",
    ("0",),
    "the learning-free n-gram predraft; on by default",
)
NGRAM_DRAFT_EXPECT = Knob(
    "RICHENGINE_NGRAM_DRAFT_EXPECT",
    ("2.0", "6.0"),
    "expected acceptance a lane must total to go predrafted",
)
NGRAM_WARMUP = Knob(
    "RICHENGINE_NGRAM_WARMUP",
    ("4", "16"),
    "rounds before a lane's acceptance EWMA is trusted",
)
NGRAM_TREE_MIN = Knob(
    "RICHENGINE_NGRAM_TREE_MIN",
    ("0.5", "2.0"),
    "EWMA a lane needs before its table emits sibling leaves",
)
# MoE decode knobs: the routed-expert union cap is off (0) by default —
# the widest cap is closest to it, so candidates run 16 → 8 → 4 and a
# deep loss on 16 skips the tighter caps outright.
MOE_UNION = Knob(
    "RICHENGINE_MOE_UNION",
    ("16", "8", "4"),
    "routed-expert union cap per decode step; off by default",
    ordered=True,
)
# The fused greedy head is on by default; off is the reference path.
HEAD_FUSED = Knob(
    "RICHENGINE_HEAD_FUSED_OFF",
    ("1",),
    "the fused greedy argmax head vs the logits-then-sample path",
)
# The Metal-encode escape hatches: each presence flag drops an on-by-default
# optimization back to its plain path, so "1" is the only candidate — the
# sweep proves the fast path pays on this chip or keeps the plain one.
SUBMIT_AHEAD = Knob(
    "RICHENGINE_SUBMIT_AHEAD",
    ("1",),
    "a second prefill command overlapping the running one; off by default",
)
ICB_OFF = Knob(
    "RICHENGINE_ICB_OFF",
    ("1",),
    "direct dispatch vs the baked indirect command buffer; ICB on by default",
)
PREPARED_CACHE_OFF = Knob(
    "RICHENGINE_PREPARED_CACHE_OFF",
    ("1",),
    "full command prepare vs the prepared-command cache; cache on by default",
)
PATCHABLE_OFF = Knob(
    "RICHENGINE_PATCHABLE_OFF",
    ("1",),
    "non-bakeable spans vs the patchable-payload replay; on by default",
)
NO_FUSED_GATE = Knob(
    "RICHENGINE_NO_FUSED_GATE",
    ("1",),
    "the two-pass verify reduce vs the fused out-projection fold; on by default",
)
# MoE decode: the packed expert path is on by default; MoE tables mix this in.
MOE_PACKED_OFF = Knob(
    "RICHENGINE_MOE_PACKED_OFF",
    ("1",),
    "the unpacked expert path vs the packed one; packed on by default",
)
# Draft geometry in the proposal budget, not the kernels: a cap below the
# smallest trained block (7 proposals — the 27B's DFlash2 and MiniCPM5's
# DSpark ship 8- and 7-row blocks) cuts live verify rows, and the bypass
# EWMA moves the anchor-only break-even. The kernel-geometry constants of
# metal/abi/ExecutionGeometry.h — attention splits, tile rows, shard counts —
# are compile-time defines baked into the metallib; no environment variable
# reaches them, so the sweep cannot offer them.
# Candidates closest to the trained limit first: 6 deviates less than 4,
# so a deep loss on 6 means 4 — further from the default — loses worse.
PROPOSAL_CAP = Knob(
    "RICHENGINE_PROPOSAL_CAP",
    ("6", "4"),
    "per-step proposal cap vs the draft's trained block limit",
    draft_kinds=("dflash2", "dflash", "dspark"),
    ordered=True,
)
DRAFT_BYPASS_EXPECT = Knob(
    "RICHENGINE_DRAFT_BYPASS_EXPECT",
    ("0.08", "0.30"),
    "the acceptance EWMA below which the batch bypasses the draft",
    draft_kinds=("dflash2", "dflash", "dspark"),
)
