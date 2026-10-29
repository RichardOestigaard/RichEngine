"""Read GGUF metadata without mapping or decoding any weight tensors.

The supported tokenizer profile describes algorithms; vocabulary, merge ranks,
special-token IDs and the chat template always come from the selected GGUF.

This file is the metadata adapter, identified by the SHA-256 of its bytes:
assembly._metadata_key keys each derived metadata entry by that hash, the
tokenizers version and the size and digest of each source GGUF. After any
edit to this file, a comment included, preparing an installed GGUF model
reports that the adapter changed and derives its metadata again.
"""

from __future__ import annotations

import collections
import contextlib
import struct
from dataclasses import dataclass
from pathlib import Path

from . import layout, models
from .models import ModelError

# The files derived_files derives from a target GGUF, by assembly path.
DERIVED_FILES = (
    "config.json",
    layout.TOKENIZER + "/tokenizer.json",
    layout.TOKENIZER + "/tokenizer_config.json",
    layout.TOKENIZER + "/chat_template.jinja",
)


class Metadata:
    """Bounded little-endian GGUF v3 metadata reader."""

    # The version, and the bound on each of the tensor and key counts, that
    # the native reader (runtime/model/GgufFile.cpp) accepts.
    VERSION = 3
    MAX_FIELDS = 16384
    MAX_BYTES = 128 * 1024 * 1024
    # The bound on one array's elements: a vocabulary or its merges.
    MAX_ITEMS = 1_000_000
    # GGML_MAX_DIMS, the highest tensor rank the native reader
    # (runtime/model/GgufFile.cpp) accepts.
    MAX_DIMENSIONS = 4
    # GGUF value types (gguf_type): a string, an array of one other type, and
    # the struct format of each scalar type.
    STRING = 8
    ARRAY = 9
    SCALAR_FORMATS = {
        0: "B",
        1: "b",
        2: "H",
        3: "h",
        4: "I",
        5: "i",
        6: "f",
        7: "?",
        10: "Q",
        11: "q",
        12: "d",
    }

    def __init__(self, source, *, tensors=False):
        """Read a local path, or a binary stream at the start of the file such
        as a Hub range reader, which its caller closes. tensors also reads
        each tensor's type."""
        self.values = {}
        self.tensors = {}
        self.consumed = 0
        with (
            Path(source).open("rb")
            if isinstance(source, (str, Path))
            else contextlib.nullcontext(source)
        ) as self.stream:
            if self.read(4) != b"GGUF" or self.scalar("I") != self.VERSION:
                raise ModelError("unsupported GGUF header (expected little-endian v3)")
            tensor_count = self.scalar("Q")
            key_count = self.scalar("Q")
            if key_count > self.MAX_FIELDS or tensor_count > self.MAX_FIELDS:
                raise ModelError("GGUF metadata has too many fields")
            for _ in range(key_count):
                key = self.string()
                if key in self.values:
                    raise ModelError("duplicate GGUF metadata key: " + key)
                self.values[key] = self.value(self.scalar("I"))
            for _ in range(tensor_count if tensors else 0):
                name = self.string()
                dimensions = self.scalar("I")
                if dimensions > self.MAX_DIMENSIONS:
                    raise ModelError("invalid GGUF tensor rank: " + name)
                self.read(dimensions * struct.calcsize("<Q"))  # Its extents.
                kind = self.scalar("I")
                self.scalar("Q")  # Data offset; the payload is never read here.
                if name in self.tensors:
                    raise ModelError("duplicate GGUF tensor: " + name)
                self.tensors[name] = kind

    def read(self, size):
        if size > self.MAX_BYTES - self.consumed:
            raise ModelError("GGUF metadata exceeds the size limit")
        data = self.stream.read(size)
        if len(data) != size:
            raise ModelError("truncated GGUF metadata")
        self.consumed += size
        return data

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize("<" + fmt)))[0]

    def string(self):
        try:
            return self.read(self.scalar("Q")).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ModelError("invalid UTF-8 in GGUF metadata") from error

    def value(self, value_type):
        if value_type == self.STRING:
            return self.string()
        if value_type == self.ARRAY:
            element, count = self.scalar("I"), self.scalar("Q")
            if (
                element not in (*self.SCALAR_FORMATS, self.STRING)
                or count > self.MAX_ITEMS
            ):
                raise ModelError("unsupported or oversized GGUF metadata array")
            return [self.value(element) for _ in range(count)]
        if value_type not in self.SCALAR_FORMATS:
            raise ModelError("unknown GGUF metadata type")
        return self.scalar(self.SCALAR_FORMATS[value_type])

    def require(self, key, kind):
        value = self.values.get(key)
        if type(value) is not kind:
            raise ModelError("missing or invalid GGUF metadata: " + key)
        return value

    def positive(self, key):
        value = self.require(key, int)
        if value <= 0:
            raise ModelError("GGUF metadata must be positive: " + key)
        return value


# The qwen35 profile extends Qwen2's letter class to include combining marks.
# Its NFC normalizer and byte-level BPE match the original Qwen tokenizer.
QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# The minicpm5 profile splits digit runs of at most three out before the
# GPT-2 pattern, as MiniCPM5's published tokenizer does, and normalizes
# nothing.
MINICPM5_PATTERNS = (
    r"\p{N}{1,3}",
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
)
# The lfm2 profile llama.cpp declares for Liquid's LFM2.5 tokenizer.
LFM2_PATTERNS = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+",
)
# Each tokenizer.ggml.pre profile RichEngine can rebuild: its Split patterns in
# application order and whether the input is NFC-normalized first.
# The granite-docling profile is the original GPT-2 pattern, unchanged.
GRANITE_DOCLING_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+|\s+(?!\S)"
)
TOKENIZER_PROFILES = {
    "qwen35": ((QWEN35_PATTERN,), True),
    "minicpm5": (MINICPM5_PATTERNS, False),
    "lfm2": (LFM2_PATTERNS, False),
    "granite-docling": ((GRANITE_DOCLING_PATTERN,), False),
}
# The GGUF token types (llama_token_type) a byte-level BPE vocabulary uses:
# control tokens are special added tokens, user-defined ones added tokens
# that are not special, and unused ones fill the vocabulary to its size. A
# SentencePiece (gemma4) vocabulary also holds unknown and byte-fallback
# tokens, which stay ordinary vocabulary entries.
NORMAL_TOKEN = 1
UNKNOWN_TOKEN = 2
CONTROL_TOKEN = 3
USER_DEFINED_TOKEN = 4
UNUSED_TOKEN = 5
BYTE_TOKEN = 6


def _bpe_tokenizer(metadata, tokens, types):
    """The byte-level BPE backend of a gpt2-model GGUF vocabulary, rebuilt as
    the profiles TOKENIZER_PROFILES names describe it."""
    from tokenizers import (
        Regex,
        Tokenizer,
        decoders,
        normalizers,
        pre_tokenizers,
        processors,
    )
    from tokenizers import models as token_models

    pre = metadata.require("tokenizer.ggml.pre", str)
    if pre not in TOKENIZER_PROFILES:
        raise ModelError(
            "unsupported GGUF tokenizer profile: "
            + pre
            + f" (supported: {', '.join(sorted(TOKENIZER_PROFILES))})"
        )
    patterns, normalize = TOKENIZER_PROFILES[pre]
    merges = metadata.require("tokenizer.ggml.merges", list)
    if not set(pre_tokenizers.ByteLevel.alphabet()) <= set(tokens):
        raise ModelError("GGUF byte-level tokenizer is missing byte tokens")
    pairs = []
    for merge in merges:
        if not isinstance(merge, str) or len(parts := merge.split(" ")) != 2:
            raise ModelError("invalid GGUF BPE merge")
        pairs.append(tuple(parts))
    try:
        backend = Tokenizer(
            token_models.BPE({t: i for i, t in enumerate(tokens)}, pairs)
        )
    except Exception as error:
        raise ModelError("invalid GGUF BPE vocabulary or merges") from error
    if normalize:
        backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            *(
                pre_tokenizers.Split(Regex(pattern), behavior="isolated")
                for pattern in patterns
            ),
            pre_tokenizers.ByteLevel(
                add_prefix_space=False, use_regex=False, trim_offsets=False
            ),
        ]
    )
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=False, use_regex=False, trim_offsets=False
    )
    backend.decoder = decoders.ByteLevel()
    return backend


def _unigram_tokenizer(metadata, tokens, types):
    """The SentencePiece unigram backend of a gemma4-model GGUF vocabulary:
    its tokens and scores as the upstream tokenizer.json publishes them, with
    SentencePiece's metaspace encoding and byte fallback. The GGUF carries no
    precompiled normalizer, so none is applied — the vocabulary's literal ▁
    pieces segment what Metaspace writes."""
    from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers import models as token_models

    scores = metadata.require("tokenizer.ggml.scores", list)
    if len(scores) != len(tokens) or not all(
        type(score) in (int, float) for score in scores
    ):
        raise ModelError("invalid GGUF unigram vocabulary scores")
    unknown = metadata.values.get("tokenizer.ggml.unknown_token_id")
    if unknown is not None and not (
        type(unknown) is int and 0 <= unknown < len(tokens)
    ):
        raise ModelError("invalid GGUF metadata: tokenizer.ggml.unknown_token_id")
    try:
        backend = Tokenizer(
            token_models.Unigram(
                [(token, float(score)) for token, score in zip(tokens, scores)],
                unk_id=unknown,
                byte_fallback=BYTE_TOKEN in types,
            )
        )
    except Exception as error:
        raise ModelError("invalid GGUF unigram vocabulary") from error
    backend.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )
    backend.decoder = decoders.Metaspace(
        replacement="▁", prepend_scheme="always", split=True
    )
    if metadata.values.get("tokenizer.ggml.add_bos_token") is True:
        # SentencePiece prepends BOS; the BPE path refuses it (byte-level
        # profiles state their own), but a gemma4 vocabulary expects it.
        bos = metadata.values["tokenizer.ggml.bos_token_id"]
        backend.post_processor = processors.TemplateProcessing(
            single=f"{tokens[bos]} $A",
            special_tokens=[(tokens[bos], bos)],
        )
    return backend


def derived_files(target, vision=None):
    """DERIVED_FILES' contents, derived from a target GGUF's metadata and its
    vision projector's (local paths)."""
    metadata = Metadata(target)
    config = model_config(metadata, None if vision is None else Metadata(vision))
    return {"config.json": models.json_bytes(config), **tokenizer_files(metadata)}


def tokenizer_files(metadata):
    from tokenizers import AddedToken

    model = metadata.require("tokenizer.ggml.model", str)
    if model not in ("gpt2", "gemma4"):
        raise ModelError(
            f"unsupported GGUF tokenizer model: {model}; expected gpt2 or gemma4"
        )
    tokens = metadata.require("tokenizer.ggml.tokens", list)
    types = metadata.require("tokenizer.ggml.token_type", list)
    # The llama_token_type values each supported model's vocabulary may hold:
    # a gemma4 (SentencePiece) vocabulary also uses unknown and byte tokens.
    allowed = (NORMAL_TOKEN, CONTROL_TOKEN, USER_DEFINED_TOKEN, UNUSED_TOKEN) + (
        (UNKNOWN_TOKEN, BYTE_TOKEN) if model == "gemma4" else ()
    )
    if (
        not tokens
        or not all(isinstance(t, str) and t for t in tokens)
        or len(set(tokens)) != len(tokens)
        or len(types) != len(tokens)
        or any(type(t) is not int or t not in allowed for t in types)
    ):
        raise ModelError("invalid or unsupported GGUF tokenizer vocabulary")
    bos = metadata.values.get("tokenizer.ggml.bos_token_id")
    for key in ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"):
        automatic = metadata.values.get(key, False)
        if automatic is not False and not (
            model == "gemma4"
            and key == "tokenizer.ggml.add_bos_token"
            and automatic is True
            and type(bos) is int
            and 0 <= bos < len(tokens)
        ):
            raise ModelError("GGUF automatic BOS/EOS insertion is unsupported: " + key)
    if model == "gpt2":
        backend = _bpe_tokenizer(metadata, tokens, types)
    else:
        backend = _unigram_tokenizer(metadata, tokens, types)
    # The index of each added token, and whether it is special.
    added = {
        index: kind == CONTROL_TOKEN
        for index, kind in enumerate(types)
        if kind in (CONTROL_TOKEN, USER_DEFINED_TOKEN)
    }
    backend.add_tokens(
        [
            AddedToken(tokens[index], normalized=False, special=is_special)
            for index, is_special in added.items()
        ]
    )

    def special_token(name, *, optional=False):
        key = "tokenizer.ggml." + name + "_token_id"
        if optional and key not in metadata.values:
            return None
        index = metadata.require(key, int)
        if not 0 <= index < len(tokens) or types[index] != CONTROL_TOKEN:
            raise ModelError("invalid GGUF special token: " + key)
        return tokens[index]

    # A GGUF that states only named templates (d1-omni ships the systemone
    # decision template, no default) derives it under its own name — a
    # decision model has no chat template to ship under chat_template's.
    template = metadata.values.get("tokenizer.chat_template")
    names = metadata.values.get("tokenizer.chat_templates")
    template_file = "chat_template.jinja"
    if template is None and isinstance(names, list) and names:
        template = metadata.values.get(
            "tokenizer.chat_template." + str(names[0])
        )
        template_file = str(names[0]) + "_template.jinja"
    if type(template) is not str:
        raise ModelError("missing or invalid GGUF metadata: tokenizer.chat_template")
    if not template.strip():
        raise ModelError("GGUF chat template is empty")
    config = {
        "tokenizer_class": "TokenizersBackend",
        "model_max_length": metadata.positive(
            text_architecture(metadata) + ".context_length"
        ),
        "clean_up_tokenization_spaces": False,
        "bos_token": special_token("bos", optional=True),
        "eos_token": special_token("eos"),
        "pad_token": special_token("padding", optional=True),
        "added_tokens_decoder": {
            str(index): {
                "content": tokens[index],
                "special": is_special,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
            }
            for index, is_special in added.items()
        },
    }
    return {
        layout.TOKENIZER + "/tokenizer.json": backend.to_str().encode(),
        layout.TOKENIZER + "/tokenizer_config.json": models.json_bytes(config),
        layout.TOKENIZER + "/" + template_file: template.encode(),
    }


# The GGUF text architectures RichEngine serves: a registry where each arch
# owns its derivation — how the shared field map reads its head width and
# KV head count, which extra config fields it states (config), and which
# tensors the native loader reads of it (tensors). Adding an architecture
# is a new row and its handlers, not edits inside another's branches.
@dataclass(frozen=True)
class GgufArch:
    # The model_type the derived text_config states.
    model_type: str
    # The metadata key of the local (sliding) layers' head width; gemma4's
    # key_length is its global layers' width and key_length_swa the local.
    head_dim_key: str = "attention.key_length"
    # head_count_kv arrives as a per-layer list (lfm2's conv layers state 0,
    # gemma4's two attention kinds state their own count).
    kv_per_layer: bool = False
    # A per-layer kv list reduces to its maximum (gemma4's per-kind split is
    # validated in its fields handler).
    kv_max: bool = False
    # The arch states no key_length; the head width is hidden/heads.
    head_dim_derived: bool = False
    # feed_forward_length states the config's intermediate_size.
    feed_forward: bool = False
    # (text, metadata) -> None: the arch's own config fields.
    fields: "callable" = None
    # (metadata, layers) -> tensor map: the arch's loaded_tensors.
    tensors: "callable" = None


def _granite_fields(arch, text, metadata):
    # The runtime's config check reads rope_theta and the fixed softmax
    # multiplier.
    theta = metadata.values.get(arch + ".rope.freq_base")
    if type(theta) is not float or theta <= 0:
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".rope.freq_base"
        )
    text["rope_theta"] = theta
    scale = metadata.values.get(arch + ".attention.scale")
    if type(scale) is not float or scale <= 0:
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".attention.scale"
        )
    text["attention_multiplier"] = scale
    # Only the attention multiplier reaches the kernels. A granite that
    # also scales embeddings, residuals or logits (the 3.x series) would
    # generate silently wrong output, so refuse it here.
    for other in ("embedding_scale", "residual_scale", "logit_scale"):
        value = metadata.values.get(arch + "." + other)
        if value is not None and (type(value) is not float or value != 1.0):
            raise ModelError(
                "unsupported GGUF metadata: " + arch + "." + other + " must be 1.0"
            )


def _lfm2_fields(arch, text, metadata):
    if arch == "lfm2moe":
        text["num_experts"] = metadata.positive(arch + ".expert_count")
        text["num_experts_per_tok"] = metadata.positive(arch + ".expert_used_count")
        text["moe_intermediate_size"] = metadata.positive(
            arch + ".expert_feed_forward_length"
        )
        text["num_dense_layers"] = metadata.positive(
            arch + ".leading_dense_block_count"
        )
    # The per-layer KV head count names the conv (0) and full-attention
    # layers, as the config's layer_types does.
    kv = metadata.values.get(arch + ".attention.head_count_kv")
    if isinstance(kv, list):
        text["layer_types"] = [
            "conv" if count == 0 else "full_attention" for count in kv
        ]
        # Attention layers whose counts differ state each layer's own.
        if len({count for count in kv if count}) > 1:
            text["layer_kv_heads"] = kv
    # The d1-omni decision model is encoder-only: bidirectional attention,
    # no causal decode.
    if metadata.values.get(arch + ".decision.type") == "lfm2-d1-omni":
        text["encoder"] = True
    text["conv_dim"] = text["hidden_size"]
    text["conv_L_cache"] = metadata.positive(arch + ".shortconv.l_cache")


def _lfm3_fields(arch, text, metadata):
    layers = text["num_hidden_layers"]
    # The per-layer KV head count names the conv layers (0); among the
    # attention layers, the sliding-window pattern names the local ones, as
    # the config's layer_types does.
    kv = metadata.values.get(arch + ".attention.head_count_kv")
    pattern = metadata.values.get(arch + ".attention.sliding_window_pattern")
    if (
        not isinstance(kv, list)
        or len(kv) < layers
        or not all(type(count) is int and count >= 0 for count in kv[:layers])
        or not isinstance(pattern, list)
        or len(pattern) < layers
        or not all(type(local) is bool for local in pattern[:layers])
        or any(pattern[layer] and not kv[layer] for layer in range(layers))
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: "
            + arch
            + ".attention.head_count_kv or .attention.sliding_window_pattern"
        )
    text["layer_types"] = [
        (
            "conv"
            if kv[layer] == 0
            else "sliding_attention" if pattern[layer] else "full_attention"
        )
        for layer in range(layers)
    ]
    # Every attention layer is K == V.
    if metadata.values.get(arch + ".attention.value_length") != metadata.values.get(
        arch + ".attention.key_length"
    ):
        raise ModelError(
            "unsupported GGUF metadata: "
            + arch
            + ".attention.key_length and its value_length differ"
        )
    text["conv_L_cache"] = metadata.positive(arch + ".shortconv.l_cache")
    text["sliding_window"] = metadata.positive(arch + ".attention.sliding_window")
    text["use_sandwich_norm"] = True
    text["tie_word_embeddings"] = True
    epsilon = metadata.values.get(arch + ".attention.layer_norm_rms_epsilon")
    if type(epsilon) is not float or epsilon <= 0:
        raise ModelError(
            "missing or invalid GGUF metadata: "
            + arch
            + ".attention.layer_norm_rms_epsilon"
        )
    # The GGUF stores the F32 rounding of config.json's norm_eps; emit the
    # config's literal when the stored value is its image, else the value
    # itself (a family signature or the runtime's layout check refuses it).
    text["norm_eps"] = (
        1e-5 if abs(epsilon - 1e-5) <= 1e-5 * 1e-6 else epsilon
    )
    # The GGUF states the two rope bases flat, where config.json nests them
    # under rope_parameters by layer kind; emit config.json's shape. The
    # global layers' partial rotary covers the first half of each head.
    bases = {}
    for kind, key in (
        ("full_attention", "rope.freq_base"),
        ("sliding_attention", "rope.freq_base_swa"),
    ):
        theta = metadata.values.get(arch + "." + key)
        if type(theta) is not float or theta <= 0:
            raise ModelError(
                "missing or invalid GGUF metadata: " + arch + "." + key
            )
        bases[kind] = theta
    text["rope_parameters"] = {
        "full_attention": {
            "partial_rotary_factor": 0.5,
            "rope_theta": bases["full_attention"],
            "rope_type": "default",
        },
        "sliding_attention": {
            "rope_theta": bases["sliding_attention"],
            "rope_type": "default",
        },
    }


def _qwen35moe_fields(arch, text, metadata):
    text.update(
        num_experts=metadata.positive(arch + ".expert_count"),
        num_experts_per_tok=metadata.positive(arch + ".expert_used_count"),
    )


def _gemma4_fields(arch, text, metadata):
    layers = text["num_hidden_layers"]
    # The sliding-window pattern names each layer's attention kind, as
    # the config's layer_types does: True marks the local layers.
    pattern = metadata.values.get(arch + ".attention.sliding_window_pattern")
    if (
        not isinstance(pattern, list)
        or len(pattern) < layers
        or not all(type(swa) is bool for swa in pattern[:layers])
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: "
            + arch
            + ".attention.sliding_window_pattern"
        )
    text["layer_types"] = [
        "sliding_attention" if swa else "full_attention" for swa in pattern[:layers]
    ]
    # head_count_kv is per layer: the local layers' count is the
    # config's num_key_value_heads, the global layers' its
    # num_global_key_value_heads. Each kind states one count.
    kv = metadata.values.get(arch + ".attention.head_count_kv")
    if (
        not isinstance(kv, list)
        or len(kv) < layers
        or not all(type(count) is int and count > 0 for count in kv[:layers])
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".attention.head_count_kv"
        )
    local = {kv[i] for i in range(layers) if pattern[i]}
    wide = {kv[i] for i in range(layers) if not pattern[i]}
    if len(local) != 1 or len(wide) != 1:
        raise ModelError(
            "unsupported GGUF metadata: "
            + arch
            + ".attention.head_count_kv differs between layers of a kind"
        )
    text["num_key_value_heads"] = local.pop()
    text["num_global_key_value_heads"] = wide.pop()
    text["global_head_dim"] = metadata.positive(arch + ".attention.key_length")
    # Gemma 4 requires K == V head widths on both layer kinds
    # (attention_k_eq_v); refuse a GGUF stating otherwise.
    for swa in ("", "_swa"):
        key = arch + ".attention.key_length" + swa
        value = metadata.values.get(
            arch + ".attention.value_length" + swa,
            metadata.values.get(key),
        )
        if value != metadata.values.get(key):
            raise ModelError(
                "unsupported GGUF metadata: " + key + " and its value_length differ"
            )
    text["attention_k_eq_v"] = True
    text["sliding_window"] = metadata.positive(arch + ".attention.sliding_window")
    shared = metadata.values.get(arch + ".attention.shared_kv_layers", 0)
    if type(shared) is not int or shared < 0:
        raise ModelError(
            "invalid GGUF metadata: " + arch + ".attention.shared_kv_layers"
        )
    text["num_kv_shared_layers"] = shared
    # Per-layer token inputs (gemma4's per_layer_token_embd path) are
    # unsupported; the field must state them absent.
    per_layer = metadata.values.get(arch + ".embedding_length_per_layer_input", 0)
    if type(per_layer) is not int or per_layer != 0:
        raise ModelError(
            "unsupported GGUF metadata: "
            + arch
            + ".embedding_length_per_layer_input must be 0"
        )
    text.update(
        num_experts=metadata.positive(arch + ".expert_count"),
        top_k_experts=metadata.positive(arch + ".expert_used_count"),
        moe_intermediate_size=metadata.positive(
            arch + ".expert_feed_forward_length"
        ),
        enable_moe_block=True,
        tie_word_embeddings=True,
    )
    softcap = metadata.values.get(arch + ".final_logit_softcapping")
    if type(softcap) is not float or softcap <= 0:
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".final_logit_softcapping"
        )
    text["final_logit_softcapping"] = softcap
    # The GGUF states the two rope bases flat, where config.json nests
    # them under rope_parameters by layer kind; emit config.json's shape.
    bases = {}
    for kind, key in (
        ("full_attention", "rope.freq_base"),
        ("sliding_attention", "rope.freq_base_swa"),
    ):
        theta = metadata.values.get(arch + "." + key)
        if type(theta) is not float or theta <= 0:
            raise ModelError(
                "missing or invalid GGUF metadata: " + arch + "." + key
            )
        bases[kind] = theta
    text["rope_parameters"] = {
        "full_attention": {
            "rope_theta": bases["full_attention"],
            "rope_type": "proportional",
            # The global layers' p-RoPE rotates the first 128 of 512
            # head dimensions; GGUF states no factor, the checkpoint's
            # config.json does.
            "partial_rotary_factor": 0.25,
        },
        "sliding_attention": {
            "rope_theta": bases["sliding_attention"],
            "rope_type": "default",
        },
    }


def text_architecture(metadata):
    arch = metadata.require("general.architecture", str)
    if arch not in GGUF_ARCHS:
        raise ModelError("unsupported GGUF model architecture: " + arch)
    return arch


# Architectures whose checkpoints install packed (install/pack.py), never
# from a GGUF: deriving just their model_type reaches pack.packable's
# refusal, which names the safetensors checkpoint to install instead.
PACKED_ONLY_ARCHS = {"diffusiongemma": "diffusion_gemma_text"}


def model_config(metadata, vision=None):
    arch = metadata.require("general.architecture", str)
    if arch in PACKED_ONLY_ARCHS:
        return {
            "model_type": PACKED_ONLY_ARCHS[arch].removesuffix("_text"),
            "text_config": {"model_type": PACKED_ONLY_ARCHS[arch]},
        }
    arch = text_architecture(metadata)
    entry = GGUF_ARCHS[arch]
    fields = {
        "hidden_size": "embedding_length",
        "max_position_embeddings": "context_length",
        "num_attention_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "head_dim": entry.head_dim_key,
    }
    text = {}
    for name, key in fields.items():
        value = metadata.values.get(arch + "." + key)
        if isinstance(value, list):
            # Only the architectures whose layers differ in kind state
            # head_count_kv per layer: lfm2's conv layers state 0, gemma4's
            # two attention kinds state their own count (separated in its
            # fields handler). Any other array is a geometry the runtime
            # does not serve.
            if (
                key != "attention.head_count_kv"
                or not entry.kv_per_layer
                or not all(type(item) is int and item >= 0 for item in value)
                or not value
            ):
                raise ModelError("invalid GGUF metadata array: " + arch + "." + key)
            if entry.kv_max:
                value = max(value)
            else:
                counts = collections.Counter(item for item in value if item)
                if not counts:
                    raise ModelError(
                        "invalid GGUF metadata array: " + arch + "." + key
                    )
                # Attention layers may disagree on their count — d1-omni's
                # last layers widen. The arch's fields handler emits the
                # per-layer list; the flat count states its mode.
                value = counts.most_common(1)[0][0]
        elif value is None and name == "head_dim" and entry.head_dim_derived:
            # lfm2 and granite state no key_length; their head width divides
            # the hidden size evenly across the attention heads.
            continue
        if type(value) is not int or value <= 0:
            raise ModelError("missing or invalid GGUF metadata: " + arch + "." + key)
        text[name] = value
    if "head_dim" not in text:
        if text["hidden_size"] % text["num_attention_heads"]:
            raise ModelError(
                "missing or invalid GGUF metadata: " + arch + ".attention.key_length"
            )
        text["head_dim"] = text["hidden_size"] // text["num_attention_heads"]
    text.update(
        num_hidden_layers=loaded_layers(metadata, arch),
        model_type=entry.model_type,
        vocab_size=len(metadata.require("tokenizer.ggml.tokens", list)),
    )
    decision_type = metadata.values.get(arch + ".decision.type") or metadata.values.get(
        "general.decision_type"
    )
    if isinstance(decision_type, str):
        text["decision_type"] = decision_type
    if entry.feed_forward:
        width = metadata.values.get(arch + ".feed_forward_length")
        if isinstance(width, list):
            # A per-layer list — d1-omni narrows its last layers. The flat
            # intermediate_size states the mode, the per-layer list the
            # actual widths.
            if not width or not all(type(w) is int and w > 0 for w in width):
                raise ModelError(
                    "invalid GGUF metadata array: " + arch + ".feed_forward_length"
                )
            if len(set(width)) != 1:
                text["layer_intermediate_sizes"] = width
            width = collections.Counter(width).most_common(1)[0][0]
        if type(width) is not int or width <= 0:
            raise ModelError(
                "missing or invalid GGUF metadata: " + arch + ".feed_forward_length"
            )
        text["intermediate_size"] = width
    if entry.fields is not None:
        entry.fields(arch, text, metadata)
    for name in ("eos", "bos", "padding"):
        value = metadata.values.get(f"tokenizer.ggml.{name}_token_id")
        if value is not None and (type(value) is not int or value < 0):
            raise ModelError(f"invalid GGUF metadata: tokenizer.ggml.{name}_token_id")
        if type(value) is int:
            text[name + "_token_id"] = value
    config = {
        "model_type": entry.model_type.removesuffix("_text"),
        "text_config": text,
    }
    if vision is not None:
        config["vision_config"] = vision_config(vision)
        audio = audio_config(vision)
        if audio is not None:
            config["audio_config"] = audio
        speech = speech_config(vision)
        if speech is not None:
            config["speech_config"] = speech
    return config


def audio_config(vision):
    """The omni mmproj's lfm3a audio encoder section, or None for a mmproj
    that declares none: 17 hybrid conv/attention blocks at hidden 768
    feeding the LayerNorm + GELU(erf) projector into the text hidden. The
    GGUF's clip.audio.feed_forward_length (1792) is authoritative — the HF
    config's audio_encoder_config states 2560 for a different module."""
    if vision.values.get("clip.audio.projector_type") != "lfm3a":
        return None
    if vision.values.get("clip.has_audio_encoder") is not True:
        raise ModelError("unsupported GGUF audio architecture")
    epsilon = vision.require("clip.audio.attention.layer_norm_epsilon", float)
    if not 0 < epsilon <= 1e-5:
        raise ModelError("unsupported GGUF audio norm epsilon")
    fields = {
        "depth": "block_count",
        "hidden_size": "embedding_length",
        "num_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "intermediate_size": "feed_forward_length",
        "out_hidden_size": "projection_dim",
        "n_mel": "num_mel_bins",
        "sliding_window": "window_size",
    }
    result = {
        name: vision.positive("clip.audio." + key) for name, key in fields.items()
    }
    if result["hidden_size"] % result["num_heads"] or (
        result["num_heads"] % result["num_key_value_heads"]
    ):
        raise ModelError("invalid GGUF audio head geometry")
    result.update(
        projector_type="lfm3a",
        head_dim=result["hidden_size"] // result["num_heads"],
        # 16 kHz mono, n_fft/win 400, hop 160, 8 mel frames per LM token.
        sampling_rate=16000,
        n_fft=400,
        win_length=400,
        hop_length=160,
        encoder_stride=8,
    )
    return result


def speech_config(vision):
    """The omni mmproj's lfm3a_gen speech decoder section, or None for a
    mmproj that declares none. The runtime's gen loader validates the
    section's tensors and metadata itself; the record only names the
    projector."""
    if vision.values.get("clip.gen.audio.projector_type") != "lfm3a_gen":
        return None
    epsilon = vision.require("clip.gen.audio.attention.layer_norm_epsilon", float)
    if not 0 < epsilon <= 1e-5:
        raise ModelError("unsupported GGUF speech norm epsilon")
    return {"projector_type": "lfm3a_gen"}


def loaded_layers(metadata, arch):
    """The target's layer count without its MTP layers, which are never
    loaded."""
    layers = metadata.positive(arch + ".block_count")
    mtp = metadata.values.get(arch + ".nextn_predict_layers", 0)
    if type(mtp) is not int or not 0 <= mtp < layers:
        raise ModelError("invalid GGUF MTP layer count")
    return layers - mtp


def vision_config(vision):
    if vision.require("general.architecture", str) != "clip":
        raise ModelError("unsupported GGUF vision architecture")
    # clip.projector_type is llama.cpp's single-modality key; a
    # mixed-modality mmproj (LFM3-Omni carries vision and audio towers in
    # one file) states the modality key instead.
    projector = vision.values.get("clip.projector_type") or vision.values.get(
        "clip.vision.projector_type"
    )
    if projector == "lfm3vl":
        return _lfm3vl_vision_config(vision)
    if (
        projector != "qwen3vl_merger"
        or vision.require("clip.use_gelu", bool) is not True
    ):
        raise ModelError("unsupported GGUF vision architecture")
    fields = {
        "depth": "block_count",
        "hidden_size": "embedding_length",
        "num_heads": "attention.head_count",
        "intermediate_size": "feed_forward_length",
        "out_hidden_size": "projection_dim",
        "patch_size": "patch_size",
        "spatial_merge_size": "spatial_merge_size",
    }
    result = {
        name: vision.positive("clip.vision." + key) for name, key in fields.items()
    }
    image_size = vision.positive("clip.vision.image_size")
    if image_size % result["patch_size"]:
        raise ModelError("invalid GGUF vision position grid")
    if any(vision.require("clip.vision.is_deepstack_layers", list)):
        raise ModelError("GGUF vision deepstack layers are unsupported")
    # qwen3vl_merger uses RGB patches with two temporal slices; the native loader
    # independently checks both patch tensors and every other weight shape.
    result.update(
        projector_type="qwen3vl_merger",
        temporal_patch_size=2,
        in_channels=3,
        num_position_embeddings=(image_size // result["patch_size"]) ** 2,
        hidden_act="gelu_pytorch_tanh",
        deepstack_visual_indexes=[],
    )
    return result


def _lfm3vl_vision_config(vision):
    """The LFM3-Omni vision tower's clip section: a fourteen-block hybrid of
    depthwise-conv and bidirectional GQA blocks, silu-gated FFNs, axial 2D
    rope, and a LayerNorm + GELU(erf) 2x2-merge projector. Its mmproj quantizes
    the blocks (Q8_0 with F32 norms and conv kernels, F16 patch embeddings),
    which select_vision admits for this projector alone."""
    if (
        vision.require("clip.use_silu", bool) is not True
        or vision.values.get("clip.use_gelu") is True
        or vision.positive("clip.vision.spatial_merge_size") != 2
        or vision.require("clip.vision.image_resize_algo", str) != "bicubic"
    ):
        raise ModelError("unsupported GGUF vision architecture")
    epsilon = vision.require("clip.vision.attention.layer_norm_epsilon", float)
    if not 0 < epsilon <= 1e-5:
        raise ModelError("unsupported GGUF vision norm epsilon")
    fields = {
        "depth": "block_count",
        "hidden_size": "embedding_length",
        "num_heads": "attention.head_count",
        "num_key_value_heads": "attention.head_count_kv",
        "intermediate_size": "feed_forward_length",
        "out_hidden_size": "projection_dim",
        "patch_size": "patch_size",
        "spatial_merge_size": "spatial_merge_size",
    }
    result = {
        name: vision.positive("clip.vision." + key) for name, key in fields.items()
    }
    if result["num_heads"] % result["num_key_value_heads"]:
        raise ModelError("invalid GGUF vision head grouping")
    # RGB patches with two temporal slices, as the v.patch_embd pair states;
    # the tower carries no learned position table (axial rope instead).
    result.update(
        projector_type="lfm3vl",
        temporal_patch_size=2,
        in_channels=3,
        hidden_act="silu",
        min_pixels=vision.positive("clip.vision.image_min_pixels"),
        max_pixels=vision.positive("clip.vision.image_max_pixels"),
    )
    return result


def processor_config(vision):
    config = vision_config(vision)
    return {
        "patch_size": config["patch_size"],
        "temporal_patch_size": config["temporal_patch_size"],
        "merge_size": config["spatial_merge_size"],
        "image_mean": vision.require("clip.vision.image_mean", list),
        "image_std": vision.require("clip.vision.image_std", list),
    }


# GGML tensor type names, and the quantized formats the native loader reads
# (runtime/metal/abi/QuantFormat.h, checked by the tests).
TENSOR_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    29: "IQ1_M",
    30: "BF16",
    39: "MXFP4",
    142: "PQ2_0",
}
QUANTIZED_TYPES = {
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_NL",
    "IQ4_XS",
    "MXFP4",
    "PQ2_0",
}
# The token rows the embedding kernels gather (gguf_embedding_format in
# runtime/metal/abi/Gguf.h).
EMBEDDING_TYPES = {
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "Q4_0",
    "Q4_1",
    "IQ3_S",
    "IQ4_NL",
    "IQ4_XS",
    "PQ2_0",
}
# The tensors the native loader reads from a target, and the types it accepts
# for each (runtime/model/GgufImage.cpp): quantized projections; F32 norms,
# small GDN vectors and MoE routers, which llama.cpp keeps unquantized and
# which run unrounded; GDN alpha and beta, both of one type: a quantized
# format, F32 or BF16, which preparation widens to the F32 values it equals.
F32 = {"F32"}
MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "output_norm.weight": F32,
    "output.weight": QUANTIZED_TYPES,
}
LAYER_TENSORS = {"attn_norm.weight": F32, "post_attention_norm.weight": F32}
ATTENTION_TENSORS = {
    "attn_q.weight": QUANTIZED_TYPES,
    "attn_k.weight": QUANTIZED_TYPES,
    "attn_v.weight": QUANTIZED_TYPES,
    "attn_q_norm.weight": F32,
    "attn_k_norm.weight": F32,
    "attn_output.weight": QUANTIZED_TYPES,
}
GDN_TENSORS = {
    "attn_qkv.weight": QUANTIZED_TYPES,
    "attn_gate.weight": QUANTIZED_TYPES,
    "ssm_alpha.weight": QUANTIZED_TYPES | {"F32", "BF16"},
    "ssm_beta.weight": QUANTIZED_TYPES | {"F32", "BF16"},
    "ssm_conv1d.weight": F32,
    "ssm_a": F32,
    "ssm_dt.bias": F32,
    "ssm_norm.weight": F32,
    "ssm_out.weight": QUANTIZED_TYPES,
}
DENSE_TENSORS = {
    "ffn_gate.weight": QUANTIZED_TYPES,
    "ffn_up.weight": QUANTIZED_TYPES,
    "ffn_down.weight": QUANTIZED_TYPES,
}
MOE_TENSORS = {
    "ffn_gate_inp.weight": F32,
    "ffn_gate_exps.weight": QUANTIZED_TYPES,
    "ffn_up_exps.weight": QUANTIZED_TYPES,
    "ffn_down_exps.weight": QUANTIZED_TYPES,
    "ffn_gate_inp_shexp.weight": F32,
    "ffn_gate_shexp.weight": QUANTIZED_TYPES,
    "ffn_up_shexp.weight": QUANTIZED_TYPES,
    "ffn_down_shexp.weight": QUANTIZED_TYPES,
}


# The names llama.cpp writes for a plain llama model (MiniCPM5): FFN norms
# rather than the qwen35 post_attention_norm, and no QK norm.
LLAMA_LAYER_TENSORS = {"attn_norm.weight": F32, "ffn_norm.weight": F32}
LLAMA_ATTENTION_TENSORS = {
    "attn_q.weight": QUANTIZED_TYPES,
    "attn_k.weight": QUANTIZED_TYPES,
    "attn_v.weight": QUANTIZED_TYPES,
    "attn_output.weight": QUANTIZED_TYPES,
}
# The hybrid lfm2 target: its attention layers add QK norms, its conv layers
# run the shortconv triple, and tied embeddings leave token_embd_norm where
# an untied model's output head stands.
LFM2_ATTENTION_TENSORS = {
    **LLAMA_ATTENTION_TENSORS,
    "attn_q_norm.weight": F32,
    "attn_k_norm.weight": F32,
}
LFM2_CONV_TENSORS = {
    "shortconv.conv.weight": F32,
    "shortconv.in_proj.weight": QUANTIZED_TYPES | F32,
    "shortconv.out_proj.weight": QUANTIZED_TYPES | F32,
}
LFM2_MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "token_embd_norm.weight": F32,
}
# The d1-omni decision blocks (the trailing lfm2.decision.block_count
# layers): LayerNorms with bias, a fused biased QKV and an ungated biased
# FFN — a different tensor set than the causal blocks state.
LFM2_DECISION_TENSORS = {
    "attn_norm.bias": F32,
    "attn_qkv.weight": QUANTIZED_TYPES,
    "attn_qkv.bias": F32,
    "attn_output.weight": QUANTIZED_TYPES,
    "attn_output.bias": F32,
    "ffn_norm.bias": F32,
    "ffn_up.weight": QUANTIZED_TYPES,
    "ffn_up.bias": F32,
    "ffn_down.weight": QUANTIZED_TYPES,
    "ffn_down.bias": F32,
}
# d1-omni's model-level extras: token-type embeddings and the cls pooling
# head the decision readout runs.
LFM2_OMNI_MODEL_TENSORS = {
    "token_types.weight": F32,
    "cls.weight": QUANTIZED_TYPES | F32,
    "cls.bias": F32,
    "cls.norm.weight": F32,
    "cls.norm.bias": F32,
    "cls.output.weight": F32,
    "cls.output.bias": F32,
}
# The tri-hybrid lfm3 target (LFM3, d1-2B): its attention layers carry QK
# norms and the narrow per-head gate projection, its conv layers lfm2's
# shortconv triple, and every layer the sandwich norms — a post norm after
# the mixer and one after the FFN, beside the tied embedding's output_norm.
LFM3_ATTENTION_TENSORS = {
    "attn_q.weight": QUANTIZED_TYPES,
    "attn_k.weight": QUANTIZED_TYPES,
    "attn_v.weight": QUANTIZED_TYPES,
    "attn_q_norm.weight": F32,
    "attn_k_norm.weight": F32,
    "attn_output.weight": QUANTIZED_TYPES,
    "attn_gate.weight": QUANTIZED_TYPES,
}
LFM3_LAYER_TENSORS = {
    "attn_norm.weight": F32,
    "post_attention_norm.weight": F32,
    "ffn_norm.weight": F32,
    "post_ffw_norm.weight": F32,
}
LFM3_MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "output_norm.weight": F32,
}
# The lfm2moe MoE blocks: sigmoid router with an expert bias, no shared
# experts, dense FFNs only on the leading dense layers.
LFM2MOE_TENSORS = {
    "ffn_gate_inp.weight": F32,
    "exp_probs_b.bias": F32,
    "ffn_gate_exps.weight": QUANTIZED_TYPES,
    "ffn_up_exps.weight": QUANTIZED_TYPES,
    "ffn_down_exps.weight": QUANTIZED_TYPES,
}


# The gemma4 target (llama.cpp's tensor names): the shared expert rides the
# dense FFN tensors, the MoE block adds its router (weights plus the learned
# input scale at ffn_gate_inp.scale), its second pre-FFN norm and dual
# post-FFN norms, and every layer ends in the scalar layer_output_scale.
# Global layers have no V projection (k_eq_v); expert gate/up may be one
# fused tensor (ffn_gate_up_exps) or split (ffn_gate_exps, ffn_up_exps).
GEMMA4_MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "output_norm.weight": F32,
}
GEMMA4_LAYER_TENSORS = {
    "attn_norm.weight": F32,
    "attn_q.weight": QUANTIZED_TYPES,
    "attn_k.weight": QUANTIZED_TYPES,
    "attn_q_norm.weight": F32,
    "attn_k_norm.weight": F32,
    "attn_output.weight": QUANTIZED_TYPES,
    "post_attention_norm.weight": F32,
    "ffn_norm.weight": F32,
    "ffn_gate.weight": QUANTIZED_TYPES,
    "ffn_up.weight": QUANTIZED_TYPES,
    "ffn_down.weight": QUANTIZED_TYPES,
    "ffn_gate_inp.weight": F32,
    "ffn_gate_inp.scale": F32,
    "pre_ffw_norm_2.weight": F32,
    "post_ffw_norm.weight": F32,
    "post_ffw_norm_1.weight": F32,
    "post_ffw_norm_2.weight": F32,
    "ffn_down_exps.weight": QUANTIZED_TYPES,
    "layer_output_scale.weight": F32,
}


def _gemma4_tensors(metadata, layers):
    arch = "gemma4"
    pattern = metadata.values.get(arch + ".attention.sliding_window_pattern")
    if (
        not isinstance(pattern, list)
        or len(pattern) < layers
        or not all(type(swa) is bool for swa in pattern[:layers])
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: "
            + arch
            + ".attention.sliding_window_pattern"
        )
    # The fused gate_up experts tensor and the split pair are
    # alternatives; require whichever the file states (or the fused one
    # when the header was read without its tensor table).
    fused = not metadata.tensors or any(
        name.endswith("ffn_gate_up_exps.weight") for name in metadata.tensors
    )
    tensors = dict(GEMMA4_MODEL_TENSORS)
    for layer in range(layers):
        expected = dict(GEMMA4_LAYER_TENSORS)
        if pattern[layer]:
            expected["attn_v.weight"] = QUANTIZED_TYPES
        if fused:
            expected["ffn_gate_up_exps.weight"] = QUANTIZED_TYPES
        else:
            expected["ffn_gate_exps.weight"] = QUANTIZED_TYPES
            expected["ffn_up_exps.weight"] = QUANTIZED_TYPES
        for name, types in expected.items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def _dense_tensors(metadata, layers):
    tensors = dict(MODEL_TENSORS)
    for layer in range(layers):
        for name, types in (
            LLAMA_LAYER_TENSORS | LLAMA_ATTENTION_TENSORS | DENSE_TENSORS
        ).items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def _lfm3_tensors(metadata, layers):
    arch = text_architecture(metadata)
    kv = metadata.values.get(arch + ".attention.head_count_kv")
    if (
        not isinstance(kv, list)
        or len(kv) < layers
        or not all(type(count) is int and count >= 0 for count in kv[:layers])
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".attention.head_count_kv"
        )
    tensors = dict(LFM3_MODEL_TENSORS)
    for layer in range(layers):
        mixer = LFM2_CONV_TENSORS if kv[layer] == 0 else LFM3_ATTENTION_TENSORS
        for name, types in (LFM3_LAYER_TENSORS | mixer | DENSE_TENSORS).items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def _lfm2_tensors(metadata, layers):
    arch = text_architecture(metadata)
    kv = metadata.values.get(arch + ".attention.head_count_kv")
    if (
        not isinstance(kv, list)
        or len(kv) < layers
        or not all(type(count) is int and count >= 0 for count in kv[:layers])
    ):
        raise ModelError(
            "missing or invalid GGUF metadata: " + arch + ".attention.head_count_kv"
        )
    tensors = dict(LFM2_MODEL_TENSORS)
    # d1-omni's trailing blocks are bespoke decision layers, and the model
    # carries its token-type table and cls head beside the embeddings.
    decision_blocks = 0
    if metadata.values.get(arch + ".decision.type") == "lfm2-d1-omni":
        decision_blocks = metadata.positive(arch + ".decision.block_count")
        if decision_blocks > layers:
            raise ModelError(
                "invalid GGUF metadata: " + arch + ".decision.block_count"
            )
        tensors.update(LFM2_OMNI_MODEL_TENSORS)
    dense_layers = (
        metadata.positive(arch + ".leading_dense_block_count")
        if arch == "lfm2moe"
        else layers
    )
    for layer in range(layers):
        if layer >= layers - decision_blocks:
            expected = LLAMA_LAYER_TENSORS | LFM2_DECISION_TENSORS
        else:
            mixer = (
                LFM2_CONV_TENSORS if kv[layer] == 0 else LFM2_ATTENTION_TENSORS
            )
            ffn = DENSE_TENSORS if layer < dense_layers else LFM2MOE_TENSORS
            expected = LLAMA_LAYER_TENSORS | mixer | ffn
        for name, types in expected.items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


def _gdn_tensors(metadata, layers):
    """qwen35 and qwen35moe: every full_attention_interval-th layer is full
    attention, the others GDN."""
    arch = text_architecture(metadata)
    period = metadata.positive(arch + ".full_attention_interval")
    ffn = MOE_TENSORS if arch == "qwen35moe" else DENSE_TENSORS
    tensors = dict(MODEL_TENSORS)
    for layer in range(layers):
        mixer = ATTENTION_TENSORS if (layer + 1) % period == 0 else GDN_TENSORS
        for name, types in (LAYER_TENSORS | mixer | ffn).items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors


GGUF_ARCHS = {
    "qwen35": GgufArch("qwen3_5_text", tensors=_gdn_tensors),
    "qwen35moe": GgufArch(
        "qwen3_5_moe_text", fields=_qwen35moe_fields, tensors=_gdn_tensors
    ),
    "llama": GgufArch("llama", feed_forward=True, tensors=_dense_tensors),
    "granite": GgufArch(
        "granite",
        head_dim_derived=True,
        feed_forward=True,
        fields=_granite_fields,
        tensors=_dense_tensors,
    ),
    "lfm2": GgufArch(
        "lfm2",
        kv_per_layer=True,
        head_dim_derived=True,
        feed_forward=True,
        fields=_lfm2_fields,
        tensors=_lfm2_tensors,
    ),
    "lfm2moe": GgufArch(
        "lfm2_moe",
        kv_per_layer=True,
        head_dim_derived=True,
        feed_forward=True,
        fields=_lfm2_fields,
        tensors=_lfm2_tensors,
    ),
    "lfm3": GgufArch(
        "lfm3",
        kv_per_layer=True,
        feed_forward=True,
        fields=_lfm3_fields,
        tensors=_lfm3_tensors,
    ),
    "gemma4": GgufArch(
        "gemma4_text",
        head_dim_key="attention.key_length_swa",
        kv_per_layer=True,
        kv_max=True,
        feed_forward=True,
        fields=_gemma4_fields,
        tensors=_gemma4_tensors,
    ),
}


def loaded_tensors(metadata):
    """Each tensor the native loader reads from the target that metadata
    describes, with the types it accepts. MTP layers are not loaded; every
    full_attention_interval-th qwen35 layer is full attention, the others
    GDN; a llama is all attention, and an lfm2's per-layer KV head count
    names its conv (0) and attention layers. A gemma4's sliding-window
    pattern names its local layers, which alone carry attn_v."""
    arch = text_architecture(metadata)
    return GGUF_ARCHS[arch].tensors(metadata, loaded_layers(metadata, arch))


# Prism ML's input rotation of a dense target: the parameters and keys
# GgufFile::readRotation (runtime/model/GgufFile.cpp) accepts;
# gguf::planImages (runtime/model/GgufImage.cpp, requireRotation) refuses MoE
# targets and ungrouped GDN value heads and checks the names.
ROTATION_PREFIX = "prism.hadamard."
ROTATION = {
    "version": 1,
    "block_size": 1024,
    "transform": "normalized-sylvester-walsh-hadamard",
    "axis": "input-last-dimension",
    "sign_mode": "explicit",
    "gdn_v_grouped": True,
}
ROTATION_ARRAYS = {"weight_names", "inverse_weight_names", "sign_widths", "sign_values"}


def require_rotation(metadata):
    """Reject a rotated target (Prism ML's GGUFs) whose rotation the native
    loader does not run: a MoE target, other rotation parameters or a
    rotation key it does not know."""
    keys = {
        key[len(ROTATION_PREFIX) :]
        for key in metadata.values
        if key.startswith(ROTATION_PREFIX)
    }
    if not keys:
        return
    if (
        text_architecture(metadata) != "qwen35"
        or not keys <= ROTATION.keys() | ROTATION_ARRAYS
        or any(
            type(metadata.values.get(ROTATION_PREFIX + key)) is not type(value)
            or metadata.values.get(ROTATION_PREFIX + key) != value
            for key, value in ROTATION.items()
        )
    ):
        raise ModelError(
            "this GGUF's input rotation is not one RichEngine runs; choose another variant"
        )


def require_loadable(metadata):
    """Reject a target the native loader cannot read, from its header alone,
    so an unusable file is never downloaded: every tensor it reads must be
    present, with a type it accepts for that tensor."""
    require_rotation(metadata)
    unsupported = collections.Counter()
    for name, types in loaded_tensors(metadata).items():
        kind = metadata.tensors.get(name)
        found = "missing" if kind is None else TENSOR_TYPES.get(kind, f"type {kind}")
        if found not in types:
            unsupported[
                name.split(".", 2)[-1] if name.startswith("blk.") else name, found
            ] += 1
        # The two GDN input gates run as one segment of their shared format.
        if name.endswith(".ssm_beta.weight") and kind != metadata.tensors.get(
            name.replace("ssm_beta", "ssm_alpha")
        ):
            unsupported[
                "ssm_alpha.weight and ssm_beta.weight", "of different types"
            ] += 1
    if unsupported:
        listed = ", ".join(
            f"{tensor} {found} ({count} {'tensor' if count == 1 else 'tensors'})"
            for (tensor, found), count in sorted(unsupported.items())
        )
        raise ModelError(
            f"this GGUF stores tensors RichEngine cannot load: {listed}; choose another variant"
        )
