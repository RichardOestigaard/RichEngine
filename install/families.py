"""The model families RichEngine serves: each architecture's signature and the
DFlash2 draft trained for it.

A target is identified by its own configuration, never by its repository's
name: an MLX config.json states it, and gguf.model_config derives the same
fields from a GGUF header. Legacy RichEngine packages pack these same layouts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import models


@dataclass(frozen=True)
class Draft:
    # The repository of the DFlash2 checkpoint trained for the family, as its
    # release publishes it: config.json and BF16 safetensors. Installations
    # follow its default branch as they follow the target's.
    repo: str
    # The config.json fields, dotted into its objects, that the native draft
    # inspection requires, with the values it requires: a commit stating
    # others is not installed, so it never replaces a draft that loads.
    signature: tuple[tuple[str, object], ...]

    @property
    def layers(self):
        return dict(self.signature)["num_hidden_layers"]


# The fields every DFlash2 draft the runtime loads states alike.
DFLASH2 = (
    ("architectures", ("DFlash2DraftModel",)),
    ("sliding_window", 2048),
    ("is_causal", False),
    ("attention_bias", False),
    ("tie_word_embeddings", False),
    ("rms_norm_eps", 1e-6),
    ("hidden_act", "silu"),
    ("rope_parameters.rope_type", "default"),
    ("rope_parameters.rope_theta", 10000000),
    ("dflash_config.block_size", 8),
    ("dflash_config.conv_group_size", 16),
    ("dflash_config.conv_kernel_size", 2),
    ("dflash_config.selector_top_k", 16),
)

# The fields every plain transformer draft (a DFlash release, architecture
# DFlashDraftModel) the runtime loads states alike: a causal qwen3 checkpoint
# of five sliding-attention layers and one full-attention layer.
DFLASH = (
    ("architectures", ("DFlashDraftModel",)),
    ("model_type", "qwen3"),
    ("sliding_window", 4096),
    ("attention_bias", False),
    ("tie_word_embeddings", False),
    ("rms_norm_eps", 1e-6),
    ("hidden_act", "silu"),
    ("rope_parameters.rope_type", "default"),
    ("rope_parameters.rope_theta", 10000000),
    ("layer_types", ("sliding_attention",) * 5 + ("full_attention",)),
    ("dflash_config.block_size", 16),
)

# The Gemma 4 dialect of a DFlash release (z-lab's DFlashDraftModel): a
# causal qwen3 checkpoint of four sliding-attention layers and one
# full-attention layer, which states its block size flat and its rope base
# at the config's root rather than under rope_parameters.
DFLASH_GEMMA4 = (
    ("architectures", ("DFlashDraftModel",)),
    ("model_type", "qwen3"),
    ("sliding_window", 2048),
    ("attention_bias", False),
    ("tie_word_embeddings", False),
    ("rms_norm_eps", 1e-6),
    ("hidden_act", "silu"),
    ("rope_theta", 1000000),
    ("layer_types", ("sliding_attention",) * 4 + ("full_attention",)),
    ("block_size", 16),
    ("final_logit_softcapping", 30.0),
)


# The fields every DSpark draft the runtime loads states alike: a qwen3
# checkpoint of five full-attention layers with a Markov head and a
# confidence head. The two published DSpark dialects spell the rest
# differently: MiniCPM5's states the dflash fields flat at the config's
# root, LFM2.5's nests them under dflash_config, so those fields sit in
# each draft's own signature.
DSPARK = (
    ("model_type", "qwen3"),
    ("hidden_act", "silu"),
    ("num_hidden_layers", 5),
    ("layer_types", ("full_attention",) * 5),
    ("markov_rank", 256),
    ("markov_head_type", "vanilla"),
    ("enable_confidence_head", True),
)


@dataclass(frozen=True)
class FamilyDefaults:
    """The family's install policy the signature cannot state: the packed
    target format install/pack.py writes for it (one of pack.TARGET_FORMATS,
    None for a family served from its MLX or GGUF checkpoint) and the
    quantization its target ships. A family's Draft names the suggested
    draft checkpoint."""

    packed_format: str | None = None
    quant: str = "q4"


@dataclass(frozen=True)
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header,
    # including every one the native source model inspection requires.
    signature: tuple[tuple[str, object], ...]
    # None for a family with no GPU draft; the n-gram predraft proposes
    # for it.
    draft: Draft | None = None
    # Whether the runtime serves this family's vision tower; a family without
    # one is installed text-only whatever --language-only says.
    vision: bool = True
    # The (field, value) pairs the config must not state: a field naming
    # another architecture's value disqualifies the family however it was
    # found, strict signature or name. A field the config omits contradicts
    # nothing.
    forbids: tuple[tuple[str, object], ...] = ()
    defaults: FamilyDefaults = FamilyDefaults()


FAMILIES = (
    ModelFamily(
        "Qwen3.8-27B",
        (
            ("model_type", "qwen3_5_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 5120),
            ("num_hidden_layers", 64),
            ("vocab_size", 248320),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
        ),
        Draft(
            "incoai/Qwen3.8-27B-DFlash2",
            DFLASH2
            + (
                ("num_hidden_layers", 5),
                ("hidden_size", 5120),
                ("vocab_size", 248320),
                ("intermediate_size", 17408),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.selector_rank", 256),
                ("dflash_config.mask_token_id", 248070),
                ("dflash_config.target_layer_ids", (5, 19, 33, 47, 61)),
            ),
        ),
    ),
    # Prism ML's ternary Bonsai 2 is a Qwen3.8-27B (the GGUF's qwen35
    # header states the same fields); its name alone selects it, so the
    # signature's fields must not distinguish it. Its draft was continued
    # on the ternary target (ProCreations/Ternary-Bonsai-2-27B-DFlash2,
    # DFlash2DraftModel with the Qwen3.8 signature's field values).
    ModelFamily(
        "Bonsai-2-27B",
        (
            ("model_type", "qwen3_5_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 5120),
            ("num_hidden_layers", 64),
            ("vocab_size", 248320),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
            # A field Qwen3.8-27B's signature omits keeps the otherwise
            # identical geometry unambiguous: the longer signature wins.
            ("partial_rotary_factor", 0.25),
        ),
        Draft(
            "ProCreations/Ternary-Bonsai-2-27B-DFlash2",
            DFLASH2
            + (
                ("num_hidden_layers", 5),
                ("hidden_size", 5120),
                ("vocab_size", 248320),
                ("intermediate_size", 17408),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.selector_rank", 256),
                ("dflash_config.mask_token_id", 248070),
                ("dflash_config.target_layer_ids", (5, 19, 33, 47, 61)),
            ),
        ),
        # Qwen3.8 keeps its MTP head: its GGUF declares 65 blocks, one a
        # nextn layer gguf.model_config subtracts, and Bonsai's ternary
        # conversion drops it. A config stating the layer is the base
        # model whatever its repository or file is named.
        forbids=(("num_nextn_predict_layers", 1),),
    ),
    ModelFamily(
        "Ornith-1.5-9B",
        (
            ("model_type", "qwen3_5_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 4096),
            ("num_hidden_layers", 32),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
        ),
        Draft(
            "ornith-ai/Ornith-1.5-9B-DFlash",
            DFLASH
            + (
                ("num_hidden_layers", 6),
                ("num_target_layers", 32),
                ("hidden_size", 4096),
                ("vocab_size", 248320),
                ("intermediate_size", 12288),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.mask_token_id", 248077),
                ("dflash_config.target_layer_ids", (1, 5, 9, 13, 17, 21, 25, 29)),
            ),
        ),
        # Ornith's released weights carry a vision tower the runtime does
        # not serve.
        vision=False,
    ),
    ModelFamily(
        "Qwen3.6-35B-A3B",
        (
            ("model_type", "qwen3_5_moe_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2048),
            ("num_hidden_layers", 40),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
        ),
        Draft(
            "incoai/Qwen3.6-35B-A3B-DFlash2",
            DFLASH2
            + (
                ("num_hidden_layers", 6),
                ("hidden_size", 2048),
                ("vocab_size", 248320),
                ("intermediate_size", 6144),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.selector_rank", 256),
                ("dflash_config.mask_token_id", 248077),
                ("dflash_config.target_layer_ids", (1, 6, 11, 16, 22, 27, 32, 37)),
            ),
        ),
    ),
    ModelFamily(
        "Ornith-1.5-35B-A3B",
        (
            # Ornith 1.5's 35B-A3B shares Qwen3.6-35B-A3B's geometry; its
            # fine-tune dropped the router's auxiliary loss, which the base
            # config states as 0.001. family_for resolves the more specific
            # signature.
            ("model_type", "qwen3_5_moe_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2048),
            ("num_hidden_layers", 40),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
            ("router_aux_loss_coef", 0.0),
        ),
        Draft(
            "ornith-ai/Ornith-1.5-35B-A3B-DFlash",
            DFLASH
            + (
                ("num_hidden_layers", 6),
                ("num_target_layers", 40),
                ("hidden_size", 2048),
                ("vocab_size", 248320),
                ("intermediate_size", 6144),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.mask_token_id", 248077),
                ("dflash_config.target_layer_ids", (1, 6, 11, 16, 22, 27, 32, 37)),
            ),
        ),
        vision=False,
    ),
    ModelFamily(
        "MiniCPM5-2B",
        (
            ("model_type", "llama"),
            ("max_position_embeddings", 131072),
            ("hidden_size", 2048),
            ("num_hidden_layers", 42),
            ("vocab_size", 130560),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 128),
            ("intermediate_size", 6144),
        ),
        Draft(
            "openbmb/MiniCPM5-2B-DSpark",
            DSPARK
            + (
                ("architectures", ("Qwen3DSparkModel",)),
                ("hidden_size", 2048),
                ("vocab_size", 130560),
                ("intermediate_size", 6144),
                ("num_attention_heads", 16),
                ("num_key_value_heads", 2),
                ("head_dim", 128),
                ("num_target_layers", 42),
                ("block_size", 7),
                ("max_position_embeddings", 131072),
                ("rms_norm_eps", 1e-6),
                ("attention_bias", False),
                ("tie_word_embeddings", False),
                ("attention_mode", "gqa"),
                ("mask_token_id", 75982),
                ("target_layer_ids", (1, 10, 20, 30, 39)),
                ("projector_type", "dspark"),
                ("confidence_head_alpha", 1.0),
                ("confidence_head_with_markov", True),
            ),
        ),
        vision=False,
    ),
    ModelFamily(
        "LFM2.5-2.6B",
        (
            ("model_type", "lfm2"),
            ("hidden_size", 2048),
            ("num_hidden_layers", 30),
            ("vocab_size", 128000),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 8),
            ("intermediate_size", 10752),
            ("tie_word_embeddings", True),
            # Eight full-attention layers (2, 5, 9, 13, 17, 21, 24, 27)
            # among the shortconv layers, as config.json's layer_types
            # states them and the GGUF's per-layer KV head count implies.
            (
                "layer_types",
                tuple(
                    "full_attention"
                    if index in {2, 5, 9, 13, 17, 21, 24, 27}
                    else "conv"
                    for index in range(30)
                ),
            ),
        ),
        Draft(
            "LiquidAI/LFM2.5-2.6B-DSpark",
            DSPARK
            + (
                ("architectures", ("Lfm2DSparkDraftModel",)),
                ("hidden_size", 2048),
                ("vocab_size", 128000),
                ("intermediate_size", 6144),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 64),
                ("block_size", 9),
                ("max_position_embeddings", 128000),
                ("rms_norm_eps", 1e-5),
                ("rope_theta", 10000000.0),
                ("rope_is_neox_style", False),
                ("dflash_config.mask_token_id", 125017),
                ("dflash_config.target_layer_ids", (2, 9, 17, 21, 27)),
                ("dflash_config.num_target_layers", 30),
            ),
        ),
        vision=False,
    ),
    ModelFamily(
        "LFM2.5-8B-A1B",
        (
            ("model_type", "lfm2_moe"),
            ("hidden_size", 2048),
            ("num_hidden_layers", 24),
            ("vocab_size", 128000),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 8),
            ("intermediate_size", 7168),
            ("moe_intermediate_size", 1792),
            ("num_experts", 32),
            ("num_experts_per_tok", 4),
            ("tie_word_embeddings", True),
            # Six full-attention layers (2, 6, 10, 14, 18, 21) among the
            # shortconv layers, as config.json's layer_types states them
            # and the GGUF's per-layer KV head count implies.
            (
                "layer_types",
                tuple(
                    "full_attention" if index in {2, 6, 10, 14, 18, 21} else "conv"
                    for index in range(24)
                ),
            ),
        ),
        Draft(
            "LiquidAI/LFM2.5-8B-A1B-DSpark",
            DSPARK
            + (
                ("architectures", ("Lfm2DSparkDraftModel",)),
                ("hidden_size", 2048),
                ("vocab_size", 128000),
                ("intermediate_size", 6144),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 64),
                ("block_size", 9),
                ("max_position_embeddings", 128000),
                ("rms_norm_eps", 1e-5),
                ("rope_theta", 5000000.0),
                ("rope_is_neox_style", False),
                ("dflash_config.mask_token_id", 125017),
                ("dflash_config.target_layer_ids", (2, 6, 10, 14, 18)),
                ("dflash_config.num_target_layers", 24),
            ),
        ),
        vision=False,
    ),
    ModelFamily(
        "Granite-4.2-3B",
        (
            ("model_type", "granite"),
            ("hidden_size", 2560),
            ("num_hidden_layers", 40),
            ("vocab_size", 100352),
            ("num_attention_heads", 40),
            ("num_key_value_heads", 8),
            ("head_dim", 64),
            ("intermediate_size", 8192),
            ("max_position_embeddings", 131072),
            ("rope_theta", 10000000.0),
            ("attention_multiplier", 0.015625),
        ),
        None,
        vision=False,
    ),
    ModelFamily(
        "Granite-4.2-8B",
        (
            ("model_type", "granite"),
            ("hidden_size", 4096),
            ("num_hidden_layers", 40),
            ("vocab_size", 100352),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("intermediate_size", 12800),
            ("max_position_embeddings", 131072),
            ("rope_theta", 10000000.0),
            ("attention_multiplier", 0.0078125),
        ),
        None,
        vision=False,
    ),
    ModelFamily(
        "Gemma4-26B-A4B",
        (
            ("model_type", "gemma4_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2816),
            ("num_hidden_layers", 30),
            ("vocab_size", 262144),
            # The local (sliding-attention) layers' heads; the global
            # layers' wider heads and fewer KV heads are stated apart.
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("num_global_key_value_heads", 2),
            ("head_dim", 256),
            ("global_head_dim", 512),
            ("intermediate_size", 2112),
            ("num_experts", 128),
            ("top_k_experts", 8),
            ("moe_intermediate_size", 704),
            ("enable_moe_block", True),
            ("sliding_window", 1024),
            ("num_kv_shared_layers", 0),
            ("attention_k_eq_v", True),
            ("final_logit_softcapping", 30.0),
            ("tie_word_embeddings", True),
            # A global layer every sixth (5, 11, 17, 23, 29), as
            # config.json's layer_types states them and the GGUF's
            # sliding_window_pattern implies.
            (
                "layer_types",
                tuple(
                    "full_attention" if index % 6 == 5 else "sliding_attention"
                    for index in range(30)
                ),
            ),
        ),
        Draft(
            "z-lab/gemma-4-26B-A4B-it-DFlash",
            DFLASH_GEMMA4
            + (
                ("num_hidden_layers", 5),
                ("num_target_layers", 30),
                ("hidden_size", 2816),
                ("vocab_size", 262144),
                ("intermediate_size", 5632),
                ("num_attention_heads", 32),
                ("num_key_value_heads", 8),
                ("head_dim", 128),
                ("dflash_config.mask_token_id", 4),
                ("dflash_config.target_layer_ids", (1, 6, 11, 17, 22, 27)),
            ),
        ),
        # Gemma 4's released weights carry a vision tower the runtime does
        # not serve.
        vision=False,
        defaults=FamilyDefaults(packed_format="packed-gemma4"),
    ),
    ModelFamily(
        "DiffusionGemma-26B-A4B",
        (
            ("model_type", "diffusion_gemma_text"),
            ("max_position_embeddings", 262144),
            ("hidden_size", 2816),
            ("num_hidden_layers", 30),
            ("vocab_size", 262144),
            # The local (sliding-attention) layers' heads; the global
            # layers' wider heads and fewer KV heads are stated apart.
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("num_global_key_value_heads", 2),
            ("head_dim", 256),
            ("global_head_dim", 512),
            ("intermediate_size", 2112),
            ("num_experts", 128),
            ("top_k_experts", 8),
            ("moe_intermediate_size", 704),
            ("sliding_window", 1024),
            # DiffusionGemma's text_config states bidirectional attention
            # where Gemma 4's states attention_k_eq_v.
            ("use_bidirectional_attention", "vision"),
            ("final_logit_softcapping", 30.0),
            ("tie_word_embeddings", True),
            # A global layer every sixth (5, 11, 17, 23, 29), as
            # config.json's layer_types states them.
            (
                "layer_types",
                tuple(
                    "full_attention" if index % 6 == 5 else "sliding_attention"
                    for index in range(30)
                ),
            ),
        ),
        # DiffusionGemma has no draft; the packed manifest keeps the Null
        # predraft's magic.
        draft=None,
        # Its encoder's vision tower is not served.
        vision=False,
        defaults=FamilyDefaults(packed_format="packed-diffusiongemma"),
    ),
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def _names(family, hint):
    """Whether the hint names the family: its name bounded by characters a
    repository or file name separates tokens with, never inside another
    word (NotBonsai-2-27B names no Bonsai)."""
    pattern = r"(?<![0-9a-z])" + re.escape(family.name.lower()) + r"(?![0-9a-z])"
    return re.search(pattern, hint.lower()) is not None


def family_for(config, name=None):
    """The one family whose architecture the target's config states.

    name is an optional hint such as a repository or GGUF name. It supplies
    the fields a config omits: a family it names wins over a signature that
    matches only because the config omits the deciding field (a GGUF states
    no router_aux_loss_coef), and never over a strict match of its own
    length or longer — a stated discriminator beats the name. A named
    family must still not contradict any field the config states, and a
    field a family forbids disqualifies it however it was found."""
    text = config.get("text_config") if isinstance(config, dict) else None
    if not isinstance(text, dict):
        # A text-only model (a llama or lfm2 checkpoint) states the same
        # fields flat at its config's root instead of under text_config.
        if isinstance(config, dict) and isinstance(config.get("model_type"), str):
            text = config
        else:
            raise models.ModelError("upstream configuration has no text_config")

    def stated(key, expected, missing_ok):
        value = text.get(key, expected if missing_ok else None)
        if isinstance(value, list):
            value = tuple(value)
        return value == expected

    def forbidden(family):
        # A stated field the family forbids is a contradiction whichever
        # path found it; a field the config omits contradicts nothing.
        return any(
            key in text and stated(key, value, False) for key, value in family.forbids
        )

    matches = [
        f
        for f in FAMILIES
        if not forbidden(f) and all(stated(k, v, False) for k, v in f.signature)
    ]
    if name:
        named = [
            f
            for f in FAMILIES
            if _names(f, name)
            and not forbidden(f)
            and all(stated(k, v, True) for k, v in f.signature)
        ]
        if named and max(len(f.signature) for f in named) > max(
            (len(f.signature) for f in matches), default=0
        ):
            matches = named
    if not matches:
        keys = sorted({key for family in FAMILIES for key, _ in family.signature})
        found = ", ".join(f"{key}={text.get(key)}" for key in keys if key in text)
        raise models.ModelError(
            f"no supported model has this architecture ({found}); "
            f"supported: {', '.join(f.name for f in FAMILIES)}"
        )
    # A family whose signature extends another's (Ornith-1.5-35B-A3B states
    # Qwen3.6-35B-A3B's geometry plus its own router loss) matches beside it;
    # the signature asking the most states the architecture most exactly.
    best = max(matches, key=lambda family: len(family.signature))
    if sum(len(f.signature) == len(best.signature) for f in matches) > 1:
        raise models.ModelError("this architecture matches no single supported model")
    return best
