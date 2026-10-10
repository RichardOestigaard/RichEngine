"""Tunable knobs of the Gemma 4 families: the 26B-A4B packed MoE on a plain
DFlash draft, and DiffusionGemma, whose canvas loop has its own sweep."""

from .base import (
    ADAPTIVE_PROPOSALS,
    ANE_WAIT,
    DRAFT_BYPASS,
    DRAFT_BYPASS_EXPECT,
    DRAFT_SPLITS,
    HEAD_FUSED,
    ICB_OFF,
    MOE_PACKED_OFF,
    MOE_UNION,
    MTL4,
    NGRAM_DRAFT_EXPECT,
    NGRAM_TREE_MIN,
    NGRAM_WARMUP,
    NO_FUSED_GATE,
    PATCHABLE_OFF,
    PREFILL_FAST_INT8,
    PREPARED_CACHE_OFF,
    PROPOSAL_CAP,
    SUBMIT_AHEAD,
    VERIFY_TREE,
    Knob,
    TunableKnobs,
)

_AUTOREGRESSIVE = (
    "Gemma4-26B-A4B",
    (
        VERIFY_TREE,
        PREFILL_FAST_INT8,
        ADAPTIVE_PROPOSALS,
        DRAFT_BYPASS,
        DRAFT_BYPASS_EXPECT,
        PROPOSAL_CAP,
        DRAFT_SPLITS,
        HEAD_FUSED,
        MOE_UNION,
        MOE_PACKED_OFF,
        MTL4,
        SUBMIT_AHEAD,
        ICB_OFF,
        PREPARED_CACHE_OFF,
        PATCHABLE_OFF,
        NO_FUSED_GATE,
        NGRAM_DRAFT_EXPECT,
        NGRAM_WARMUP,
        NGRAM_TREE_MIN,
        ANE_WAIT,
    ),
)

# The canvas knobs: the step-chunking, the denoise budget and the exit
# policy — every one a speed/quality trade only measurement settles.
_CANVAS = (
    Knob(
        "RICHENGINE_CANVAS_PROFILE",
        ("fast", "balanced"),
        "the bundled speed/quality preset; 'paper' is the default",
        diffusion=True,
        quality_sensitive=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_STEPS_PER_CMD",
        ("1", "4"),
        "denoise steps encoded per command buffer",
        diffusion=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_MAX_STEPS",
        ("24", "32"),
        "the denoising schedule cap; 0 keeps the manifest's 48",
        diffusion=True,
        quality_sensitive=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_EXIT_STABLE",
        ("0.6", "0.75"),
        "the fraction-settled early exit's share",
        diffusion=True,
        quality_sensitive=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_EXIT_DRIFT",
        ("1",),
        "fraction-settled exit without the confidence check",
        diffusion=True,
        quality_sensitive=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_SPECULATIVE_PREFILL",
        ("0",),
        "the commit prefill encoded ahead of the formal exit",
        diffusion=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_COMMIT_TAIL",
        ("0",),
        "the last scheduled step's argmax-direct commit",
        diffusion=True,
    ),
    # The on-by-default fused canvas kernels; "0" selects the reference path.
    Knob(
        "RICHENGINE_CANVAS_EMBED_HIST",
        ("0",),
        "the fused embed-histogram kernel vs the unfused path; on by default",
        diffusion=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_STATS_FUSED",
        ("0",),
        "the fused canvas-stats kernel vs the unfused path; on by default",
        diffusion=True,
    ),
    Knob(
        "RICHENGINE_CANVAS_PREFIX_EXIT",
        ("0",),
        "the prefix-settled early exit vs the full schedule; on by default",
        diffusion=True,
        quality_sensitive=True,
    ),
)

TUNABLE = (
    TunableKnobs(*_AUTOREGRESSIVE),
    TunableKnobs("DiffusionGemma-26B-A4B", _CANVAS),
)
