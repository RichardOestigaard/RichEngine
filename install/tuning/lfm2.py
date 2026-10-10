"""Tunable knobs of the LFM2.5 families: the conv-hybrid 2.6B and the
conv/MoE 8B-A1B, both on DSpark drafts. No GDN scan — their recurrent slots
are the double-gated short convolutions."""

from .base import (
    ADAPTIVE_PROPOSALS,
    ANE_WAIT,
    DRAFT_BYPASS,
    DRAFT_BYPASS_EXPECT,
    DRAFT_SPLITS,
    GGUF_PACKED,
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
    TunableKnobs,
)

_COMMON = (
    VERIFY_TREE,
    PREFILL_FAST_INT8,
    ADAPTIVE_PROPOSALS,
    DRAFT_BYPASS,
    DRAFT_BYPASS_EXPECT,
    PROPOSAL_CAP,
    DRAFT_SPLITS,
    HEAD_FUSED,
    MTL4,
    GGUF_PACKED,
    SUBMIT_AHEAD,
    ICB_OFF,
    PREPARED_CACHE_OFF,
    PATCHABLE_OFF,
    NO_FUSED_GATE,
    NGRAM_DRAFT_EXPECT,
    NGRAM_WARMUP,
    NGRAM_TREE_MIN,
    ANE_WAIT,
)

TUNABLE = (
    TunableKnobs("LFM2.5-2.6B", _COMMON),
    TunableKnobs("LFM2.5-8B-A1B", _COMMON + (MOE_UNION, MOE_PACKED_OFF)),
)
