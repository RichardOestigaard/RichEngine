"""Tunable knobs of the Qwen families: the GDN-hybrid 27B and the GDN+MoE
35B-A3B, both on DFlash2 drafts."""

from .base import (
    ADAPTIVE_PROPOSALS,
    DFLASH_POOL,
    DRAFT_BYPASS,
    DRAFT_BYPASS_EXPECT,
    GDN_CHUNKED,
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

# The shared Qwen sweep: tree verify, the DFlash2 pool selector, the chunked
# GDN scan, and the adaptive gates. The MoE target adds the union cap; the
# GGUF install of the 35B adds the packed operand path.
_COMMON = (
    VERIFY_TREE,
    DFLASH_POOL,
    GDN_CHUNKED,
    PREFILL_FAST_INT8,
    ADAPTIVE_PROPOSALS,
    DRAFT_BYPASS,
    DRAFT_BYPASS_EXPECT,
    PROPOSAL_CAP,
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
)

TUNABLE = (
    TunableKnobs("Qwen3.8-27B", _COMMON),
    # Bonsai 2 is a ternary Qwen3.8-27B on the same DFlash2 draft; its GGUF
    # target makes the packed-operand path the interesting one.
    TunableKnobs("Bonsai-2-27B", _COMMON),
    TunableKnobs("Qwen3.6-35B-A3B", _COMMON + (MOE_UNION, MOE_PACKED_OFF)),
)
