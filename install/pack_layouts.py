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


def pack_layout(family) -> "Layout":
    """The family's packed target geometry, per its file's pack.layout."""
    return Layout(**pack_spec(family)["layout"])


def draft_layout(family) -> "DraftLayout | None":
    """The family's packed draft geometry, or None for a draft-less pack."""
    spec = pack_spec(family).get("draft_layout")
    return DraftLayout(**spec) if spec else None


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
