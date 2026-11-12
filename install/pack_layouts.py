"""The per-family pack geometry install/pack.py packs to.

The packer is generic; what a family owns is data in its
install/families/<name>.json "pack" section — the target and draft layout
fields (the Layout/DraftLayout dataclasses here), the diffusion schedule
and the package's execution geometry — which pack_layout()/draft_layout()
materialize. The packed layer files' section names and type tags below are
the format's, shared: the read order of runtime/model/Gemma4Moe.cpp
readLayer and DFlashV1Draft.cpp. The format magics, the tile geometry and
the packing itself stay in pack.py.
"""

from __future__ import annotations

from dataclasses import dataclass


def pack_spec(family):
    """The family's "pack" section, or a ModelError when it declares none."""
    from . import models

    if family.pack is None:
        raise models.ModelError(f"{family.name} declares no packed layout")
    return family.pack


def pack_layout(family):
    """The family's packed target geometry, per its file's pack.layout."""
    spec = pack_spec(family)
    # qwen4exp's pack.layout states Qwen4ExpLayout's fields; every other
    # family's states Layout's.
    layout_type = (
        Qwen4ExpLayout
        if dict(family.signature).get("model_type") == "qwen4_exp_text"
        else Layout
    )
    return layout_type(**spec["layout"])


def draft_layout(family) -> "DraftLayout | DFlash2Layout | None":
    """The family's packed draft geometry, or None for a draft-less pack:
    a DraftLayout for the plain DFlash draft, a DFlash2Layout for DFlash2
    (the spec's "kind" names which)."""
    spec = pack_spec(family).get("draft_layout")
    return draft_layout_from_spec(spec) if spec else None


def draft_layout_from_spec(spec) -> "DraftLayout | DFlash2Layout | None":
    """A pack section's draft_layout dict as its layout dataclass; the
    spec's "kind" picks DFlash2Layout ("dflash2"), DraftLayout otherwise."""
    from . import models

    if not spec:
        return None
    spec = dict(spec)
    kind = spec.pop("kind", "dflash")
    if kind == "dflash2":
        return DFlash2Layout(**spec)
    if kind != "dflash":
        raise models.ModelError(f"unknown packed draft kind: {kind}")
    return DraftLayout(**spec)


def diffusion_schedule(family):
    """The family's denoising schedule, or None for an autoregressive pack."""
    return pack_spec(family).get("diffusion")


def execution_geometry(family, draft: "DraftLayout | None") -> dict:
    """The manifest's execution_geometry: the family's declared values with
    the draft sliding window the draft's signature states (its trained
    window, under the runtime's draft ring capacity)."""
    geometry = dict(pack_spec(family)["execution_geometry"])
    if family.draft is not None:
        geometry["draft_sliding_window"] = dict(family.draft.signature)[
            "sliding_window"
        ]
    return geometry


@dataclass(frozen=True)
class Layout:
    """The Gemma 4 26B-A4B geometry (Gemma4MoeLayout). Tests may shrink it,
    keeping the same rules; the runtime layout's own asserts apply."""

    layers: int = 30
    hidden: int = 2816
    vocabulary: int = 262144
    query_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 256
    global_kv_heads: int = 2
    global_head_dim: int = 512
    global_period: int = 6
    experts: int = 128
    experts_per_token: int = 8
    expert_intermediate: int = 704
    packed_expert_width: int = 768
    shared_intermediate: int = 2112
    packed_shared_width: int = 2304
    sliding_window: int = 1024
    rotary_theta: int = 10000
    global_rotary_theta: int = 1000000
    logit_softcap: float = 30.0
    max_position_embeddings: int = 262144

    def is_global(self, layer: int) -> bool:
        return layer % self.global_period == self.global_period - 1

    def head_dim_at(self, layer: int) -> int:
        return self.global_head_dim if self.is_global(layer) else self.head_dim

    def kv_heads_at(self, layer: int) -> int:
        return self.global_kv_heads if self.is_global(layer) else self.kv_heads

    def packed_width_at(self, layer: int) -> int:
        # q rows + k rows + (the locals only) v rows.
        values = 1 if self.is_global(layer) else 2
        return (self.query_heads + values * self.kv_heads_at(layer)) * (
            self.head_dim_at(layer)
        )

    def attention_width_at(self, layer: int) -> int:
        return self.query_heads * self.head_dim_at(layer)

    def layer_types(self):
        return [
            "full_attention" if self.is_global(layer) else "sliding_attention"
            for layer in range(self.layers)
        ]


@dataclass(frozen=True)
class DraftLayout:
    """The z-lab DFlash plain-transformer draft's geometry
    (DFlashDraftLayout over DraftKind::DFlashV1). causal_layers is the bitmask
    ModelDescriptor builds from layer_types: a set bit marks sliding
    attention."""

    layers: int = 5
    hidden: int = 2816
    kv_heads: int = 8
    head_dim: int = 128
    query_heads: int = 32
    intermediate: int = 5632
    rotary_theta: int = 1000000
    block_size: int = 16
    sliding_window: int = 2048
    target_hidden: int = 6 * 2816
    causal_layers: int = 0b01111

    @property
    def attention(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def qkv(self) -> int:
        return (self.query_heads + 2 * self.kv_heads) * self.head_dim


@dataclass(frozen=True)
class DFlash2Layout:
    """The DFlash2 draft's packed geometry (DFlashTwoDraft.cpp readDraft,
    DraftKind::DFlashTwo): the dynamic-convolution draft whose two dynamic
    projections per layer are hidden/4 rows — off the affine 256-row tile —
    and pack as F32 GGUF segments instead. attention and qkv derive from the
    head pattern as DraftLayout's do; dynamic is hidden // 4, which
    requireLayout enforces."""

    layers: int = 3
    hidden: int = 2560
    vocabulary: int = 248320
    dynamic: int = 640
    intermediate: int = 5120
    selector_rank: int = 256
    kv_heads: int = 8
    head_dim: int = 128
    query_heads: int = 32
    rotary_theta: float = 1_000_000.0
    block_size: int = 8
    target_hidden: int = 10240
    sliding_window: int = 2048

    @property
    def attention(self) -> int:
        return self.query_heads * self.head_dim

    @property
    def qkv(self) -> int:
        return (self.query_heads + 2 * self.kv_heads) * self.head_dim

    @property
    def context_kv(self) -> int:
        return self.qkv - self.attention


# The packed draft's files, in the order DFlashV1Draft.cpp reads them
# (DraftCheckpoint.cpp dflashV1LayerImage/dflashV1ModelImage).
DRAFT_LAYER_SECTIONS = (
    "input-layernorm",
    "attention-qkv",
    "query-norm",
    "key-norm",
    "attention-output",
    "post-attention-norm",
    "mlp-gate",
    "mlp-up",
    "mlp-down",
)

# The packed file's per-layer type tag: isFullAttentionLayer names the global
# layers (PackedTargetFiles::layer).
LAYER_TYPE_LOCAL, LAYER_TYPE_GLOBAL = 0, 1
HEAD_LAYER_TYPE = 2
# DiffusionGemma's extra GEMM0002 files' type tags: the self-conditioning
# stack and the encoder's per-layer scalars.
SELF_CONDITIONING_TYPE = 3
ENCODER_SCALARS_TYPE = 4

# The section names of a packed layer file, in write order: the read order of
# Gemma4Moe.cpp readLayer, listed for the tests.
LAYER_SECTIONS = (
    "input-norm",
    "attention-input",
    "query-norm",
    "key-norm",
    "attention-output",
    "post-attention-norm",
    "pre-ffn-norm",
    "pre-ffn-norm-routed",
    "router-scale",
    "router-weights",
    "per-expert-scale",
    "experts-gate",
    "experts-up",
    "experts-down",
    "shared-expert-gate",
    "shared-expert-up",
    "shared-expert-down",
    "post-ffn-norm-shared",
    "post-ffn-norm-routed",
    "post-ffn-norm",
    "layer-scalar",
)


@dataclass(frozen=True)
class Qwen4ExpLayout:
    """Qwen3.8-Flash-Next's packed target geometry, the fields of
    qwen4ExpTargetLayout (runtime/model/Qwen4Exp.hpp) a pack declares: the
    GDN/full-attention hybrid's widths, the sparse MoE, the
    hyper-connected residual, the QSA indexer and the PLE layer. The file
    magics (MDFQ0001 layer, MDFQ0002 head, MDFE0001 embedding, MDFQ0003
    ple-table) and the compiled kernel cores are the format's and the
    runtime's, not the family's data."""

    maximum_context_tokens: int = 262144
    layers: int = 48
    hidden_size: int = 2560
    vocabulary_size: int = 248320
    packed_gdn_width: int = 16512
    packed_full_width: int = 13312
    convolution_dimension: int = 10240
    convolution_taps: int = 4
    gdn_key_heads: int = 16
    gdn_value_heads: int = 48
    gdn_head_dimension: int = 128
    attention_width: int = 6144
    attention_query_heads: int = 24
    attention_kv_heads: int = 2
    attention_head_dimension: int = 256
    # The full-attention layers' packed QKV rows interleave a gate row per
    # query head ([q|gate]).
    attention_query_stride: int = 2
    attention_qk_norm: bool = True
    rotary_pairs: int = 32
    rotary_theta: float = 10_000_000.0
    full_attention_period: int = 4
    mask_token: int = 248077
    stop_tokens: tuple = (248044, 248046)
    capture_layers: tuple = (3, 17, 31, 47)
    experts: int = 512
    experts_per_token: int = 10
    expert_intermediate_size: int = 640
    packed_expert_width: int = 768
    moe_shared_expert: bool = True
    hyper_connections: int = 4
    hyper_rank: int = 320
    indexer_heads: int = 4
    indexer_head_dimension: int = 128
    indexer_top_blocks: int = 512
    indexer_block_tokens: int = 4
    ple_layer: int = 1
    ple_ngram: int = 3
    ple_heads_per_ngram: int = 8
    ple_head_dimension: int = 160
    ple_convolution_taps: int = 4
    ple_eos_token: int = 248044
    ple_image_token: int = 248056

    def is_full_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_period == 0

    @property
    def ple_heads(self) -> int:
        """The PLE table's hashed heads: (ngram - 1) orders x heads each."""
        return (self.ple_ngram - 1) * self.ple_heads_per_ngram

    def layer_types(self):
        # The packed manifest's names for the two mixers (the qwen3_5_moe
        # convention of ModelDescriptor.mm's requireLayerTypes).
        return [
            "attention" if self.is_full_attention(layer) else "gdn"
            for layer in range(self.layers)
        ]
