"""Tunable knobs of the Granite 4.2 dense targets: draft-less, so the
n-gram predraft is their only proposer and its gates are the sweep. Tree
verify defaults on for them — the sweep measures whether it pays."""

from .base import (
    ANE_WAIT,
    HEAD_FUSED,
    ICB_OFF,
    MTL4,
    NGRAM_DRAFT_EXPECT,
    NGRAM_PREDRAFT,
    NGRAM_TREE_MIN,
    NGRAM_WARMUP,
    NO_FUSED_GATE,
    PATCHABLE_OFF,
    PREFILL_FAST_INT8,
    PREPARED_CACHE_OFF,
    SUBMIT_AHEAD,
    Knob,
    TunableKnobs,
)

# Both dense targets share the sweep: the n-gram gates, the prefetched
# prefill path and the Metal-encode escape hatches.
_COMMON = (
    # Their tree verify defaults on; "0" is the candidate.
    Knob(
        "RICHENGINE_VERIFY_TREE",
        ("0",),
        "the n-gram comb tree vs the bare chain; on by default",
    ),
    NGRAM_PREDRAFT,
    NGRAM_DRAFT_EXPECT,
    NGRAM_WARMUP,
    NGRAM_TREE_MIN,
    ANE_WAIT,
    PREFILL_FAST_INT8,
    HEAD_FUSED,
    MTL4,
    SUBMIT_AHEAD,
    ICB_OFF,
    PREPARED_CACHE_OFF,
    PATCHABLE_OFF,
    NO_FUSED_GATE,
)

TUNABLE = (
    TunableKnobs("Granite-4.2-3B", _COMMON),
    TunableKnobs("Granite-4.2-8B", _COMMON),
)
