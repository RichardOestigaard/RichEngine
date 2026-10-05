"""The model families Splash serves: each architecture's signature and the
DFlash2 draft trained for it.

A target is identified by its own configuration, never by its repository's
name: an MLX config.json states it, and gguf.model_config derives the same
fields from a GGUF header. Legacy Splash packages pack these same layouts.
"""

from __future__ import annotations

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
class ModelFamily:
    name: str
    # The text_config fields that identify the architecture, as an MLX config
    # states them and as gguf.model_config derives them from a GGUF header,
    # including every one the native source model inspection requires.
    signature: tuple[tuple[str, object], ...]
    draft: Draft
    # Whether the runtime serves this family's vision tower; a family without
    # one is installed text-only whatever --language-only says.
    vision: bool = True


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
            ("layer_types", tuple(
                "full_attention" if index in {2, 5, 9, 13, 17, 21, 24, 27} else "conv"
                for index in range(30)
            )),
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
            ("layer_types", tuple(
                "full_attention" if index in {2, 6, 10, 14, 18, 21} else "conv"
                for index in range(24)
            )),
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
)


def named(name):
    """The family called name, or None."""
    return next((family for family in FAMILIES if family.name == name), None)


def family_for(config, name=None):
    """The one family whose architecture the target's config states.

    name is an optional hint such as a repository or GGUF name: a family it
    names wins over a signature that matches only because the config omits
    the deciding field (a GGUF states no router_aux_loss_coef). A named
    family must still not contradict any field the config states."""
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

    matches = [
        f for f in FAMILIES if all(stated(k, v, False) for k, v in f.signature)
    ]
    if name:
        named = [
            f
            for f in FAMILIES
            if f.name.lower() in name.lower()
            and all(stated(k, v, True) for k, v in f.signature)
        ]
        if named:
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
