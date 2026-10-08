"""Install-side weight packing: build a packed runtime package from an
upstream safetensors checkpoint.

The packed format is the only target source the runtime loads for Gemma 4
(GgufTarget and the affine planner have no gemma4 tensor map), so the
installer packs the HF checkpoint itself: BF16 weights are quantized to the
runtime's affine 4-bit/group-64 format and written as the aligned section
files runtime/model/WeightStore.cpp reads (weightFileHeader + 16 KiB-aligned
sections), under a manifest.json that inspectModelPackage validates like a
published RichEngine package.

The section order of each layer file is the read order of
runtime/model/Gemma4Moe.cpp readLayer; a layer file holds:

  input-norm, attention-input (fused QKV, no V rows on global layers),
  query-norm, key-norm, attention-output, post-attention-norm,
  pre-ffn-norm, router-scale, router-weights, per-expert-scale,
  experts-gate, experts-up, experts-down (each expert's 704 rows padded to
  the packed 768), shared-expert-gate, shared-expert-up,
  shared-expert-down (the 2112-wide GeGLU padded to 2304),
  post-ffn-norm-shared, post-ffn-norm-routed, layer-scalar.

numpy is a declared installer dependency; safetensors payloads are read
through it. The quantization below is a vectorized port of
runtime/model/AffinePreparation.cpp's quantizeGroup, which matches MLX's
4-bit affine rounding.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from . import models
from .layout import DRAFT, PACK_STAGING, PACKAGE_MANIFEST, PACKED, TARGET, TOKENIZER

# The format constants the packed readers fix (runtime/model/WeightLayout.hpp,
# WeightStore.cpp): the file magic strings, the 16-byte header, the 16 KiB
# section alignment and the 4-bit/64-group/256-row tile geometry.
ALIGNMENT = 16384
Q4_GROUP = 64
Q4_TILE_ROWS = 256
LAYER_MAGIC = b"GEMM0001"
HEAD_MAGIC = b"GEMM0002"
EMBEDDING_MAGIC = b"MDFE0001"
# The Null draft's magic (validateCommonFormat checks it even though a
# package without a draft loads no draft weights).
DRAFT_MAGIC = "MDFD0004"
# The packed plain-transformer draft's magic (kDFlashV1DraftMagic,
# runtime/model/DFlashV1Draft.hpp): a manifest whose `draft` object declares
# DFlashDraftModel must carry it in format.draft_layer_magic.
DRAFT_LAYER_MAGIC = b"MDFP0005"
VISION_MAGIC = "MDFV0001"
FORMAT_NAME = "richengine-packed-q4-gemma4"
DIFFUSION_FORMAT_NAME = "richengine-packed-q4-diffusiongemma"
# The Target.format upstream.py uses for a checkpoint the installer packs.
TARGET_FORMAT = "packed-gemma4"
DIFFUSION_TARGET_FORMAT = "packed-diffusiongemma"
PackedFormat = Literal["packed-gemma4", "packed-diffusiongemma"]
TARGET_FORMATS: tuple[PackedFormat, ...] = (TARGET_FORMAT, DIFFUSION_TARGET_FORMAT)
# The text model_types the installer packs, to their package format name
# and target format.
PACKABLE = {
    "gemma4_text": (FORMAT_NAME, TARGET_FORMAT),
    "diffusion_gemma_text": (DIFFUSION_FORMAT_NAME, DIFFUSION_TARGET_FORMAT),
}
# DiffusionGemma's denoising schedule, as its generation_config.json and the
# block-diffusion scheduler's defaults state it.
DIFFUSION = {
    "canvas_length": 256,
    "max_denoising_steps": 48,
    "t_min": 0.4,
    "t_max": 0.8,
    "entropy_bound": 0.1,
    "confidence_threshold": 0.005,
    "stability_threshold": 1,
}


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


GEMMA4 = Layout()


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


GEMMA4_DRAFT = DraftLayout()

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


def _header(magic: bytes, layer: int, kind: int) -> bytes:
    if len(magic) != 8:
        raise models.ModelError("a weight file magic is eight bytes")
    return struct.pack("<8sII", magic, layer, kind)


def _aligned(offset: int) -> int:
    return (offset + ALIGNMENT - 1) & ~(ALIGNMENT - 1)


def write_packed_file(path: Path, magic: bytes, layer: int, kind: int, sections):
    """A packed weight file: the 16-byte header, then each section's bytes at
    the next 16 KiB boundary, the file padded to one (WeightFile::section and
    finish() require both)."""
    offset = 16
    with path.open("wb") as stream:
        stream.write(_header(magic, layer, kind))
        for section in sections:
            if not len(section):
                raise models.ModelError("a packed section must not be empty")
            pad = _aligned(offset) - offset
            if pad:
                stream.write(b"\0" * pad)
            offset = _aligned(offset) + len(section)
            stream.write(section)
        stream.write(b"\0" * (_aligned(offset) - offset))


def _round_away(values):
    """std::round: halves away from zero (np.round is half-to-even)."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def _to_bf16_bits(values):
    """The BF16 nearest each finite fp32 value, ties to even."""
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _from_bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _quantize_rows(weights: np.ndarray):
    """Quantize an [n, k] fp32 matrix of whole 64-groups into the three
    tile-ordered planes (codes, scales, biases) of a packed affine
    projection: quantizeGroup's rounding, and each plane arranged as
    [n/256][groups][256] tiles of its group unit (AffinePreparation's
    writeQuantizedStep)."""
    n, k = weights.shape
    if not n or not k or n % Q4_TILE_ROWS or k % Q4_GROUP:
        raise models.ModelError(
            "packed projection dimensions must be positive, "
            "256-row and 64-column aligned"
        )
    groups = weights.reshape(n, k // Q4_GROUP, Q4_GROUP)
    if not np.isfinite(groups).all():
        raise models.ModelError("non-finite weight in a BF16 projection")
    minimum = groups.min(axis=-1)
    maximum = np.maximum(groups.max(axis=-1), 0.0)
    minimum_edge = np.abs(minimum) > np.abs(maximum)
    step = np.maximum((maximum - minimum) / 15.0, 1e-7)
    step = np.where(minimum_edge, step, -step).astype(np.float32)
    edge = np.where(minimum_edge, minimum, maximum)
    q0 = _round_away(edge / step)
    nonzero = q0 != 0.0
    step = np.where(nonzero, edge / np.where(nonzero, q0, 1.0), step).astype(np.float32)
    offset = np.where(nonzero, edge, 0.0).astype(np.float32)
    codes = np.clip(
        _round_away((groups - offset[..., None]) / step[..., None]), 0.0, 15.0
    ).astype(np.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    planes = (
        packed.reshape(n, k // Q4_GROUP, Q4_GROUP // 2),
        _to_bf16_bits(step)[..., None],
        _to_bf16_bits(offset)[..., None],
    )
    # Each plane into [n/256][groups][256] tiles.
    return [
        plane.reshape(n // Q4_TILE_ROWS, Q4_TILE_ROWS, k // Q4_GROUP, -1)
        .transpose(0, 2, 1, 3)
        .tobytes()
        for plane in planes
    ]


def pack_projection(weights: np.ndarray, out: int, width: int) -> bytes:
    """The codes|scales|biases section of an [out, width] affine projection,
    the source matrix zero-padded to those packed dimensions."""
    padded = np.zeros((out, width), np.float32)
    padded[: weights.shape[0], : weights.shape[1]] = weights
    return b"".join(_quantize_rows(padded))


def pack_experts(weights: np.ndarray, out: int, width: int) -> bytes:
    """One [experts, n, k] slab as the packed format stores it: each expert's
    [codes|scales|biases] in turn (readAffineExpertProjection's stride)."""
    if weights.shape[0] == 0 or weights.shape[1] > out or weights.shape[2] > width:
        raise models.ModelError("expert slab does not fit the packed geometry")
    return b"".join(
        pack_projection(weights[e], out, width) for e in range(len(weights))
    )


def _bf16_bytes(values: np.ndarray) -> bytes:
    """values as little-endian BF16: exact when they are BF16 already."""
    if values.dtype == np.uint16:
        return values.tobytes()
    return _to_bf16_bits(values.astype(np.float32)).tobytes()


class Safetensors:
    """One shard's tensors, read on demand through a memory map."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= 1 << 20:
                raise models.ModelError(f"invalid safetensors header in {path.name}")
            try:
                header = json.loads(stream.read(size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise models.ModelError(
                    f"invalid safetensors header in {path.name}"
                ) from error
            self.base = 8 + size
        if not isinstance(header, dict):
            raise models.ModelError(f"invalid safetensors header in {path.name}")
        self.entries = {k: v for k, v in header.items() if k != "__metadata__"}
        self.raw = np.memmap(path, dtype=np.uint8, mode="r")

    def info(self, name: str):
        entry = self.entries.get(name)
        if entry is None:
            raise models.ModelError(f"{self.path.name} has no tensor {name}")
        dtype, shape, (start, end) = (
            entry["dtype"],
            entry["shape"],
            entry["data_offsets"],
        )
        if dtype == "BF16" or dtype == "F16":
            dtype, width = np.uint16, 2
        elif dtype == "F32":
            dtype, width = np.float32, 4
        else:
            raise models.ModelError(f"unsupported tensor dtype {dtype} for {name}")
        if int(np.prod(shape)) * width != end - start:
            raise models.ModelError(f"tensor size mismatch: {name}")
        return dtype, shape, start, end

    def read(self, name: str) -> np.ndarray:
        """The tensor as fp32, and its raw BF16 bytes when it stores them."""
        dtype, shape, start, end = self.info(name)
        data = self.raw[self.base + start : self.base + end].view(dtype).reshape(shape)
        if dtype == np.float32:
            return data.astype(np.float32)
        if dtype == np.uint16 and self.entries[name]["dtype"] == "BF16":
            return _from_bf16(data)
        # F16 widens through fp32.
        return data.view(np.float16).astype(np.float32)

    def read_bf16(self, name: str) -> np.ndarray:
        """The raw BF16 payload; an F32 tensor is converted, anything else is
        refused: norms and routing tensors ship BF16 in the checkpoint."""
        _dtype, shape, start, end = self.info(name)
        data = self.raw[self.base + start : self.base + end]
        if self.entries[name]["dtype"] == "BF16":
            return np.array(data.view(np.uint16)).reshape(shape)
        if self.entries[name]["dtype"] == "F32":
            return _to_bf16_bits(data.view(np.float32)).reshape(shape)
        raise models.ModelError(f"tensor {name} must be BF16 or F32")


class Checkpoint:
    """The target checkpoint's shards, resolving the text model's tensor names
    under whatever prefix the repository publishes."""

    PREFIXES = (
        "",
        "model.",
        "model.language_model.",
        "model.decoder.",
        "model.decoder.language_model.",
        "language_model.",
    )

    def __init__(self, files):
        """files: the downloaded repository paths of every safetensors shard."""
        self.shards = [Safetensors(path) for path in files]
        self.names = {name for shard in self.shards for name in shard.entries}

    def resolve(self, suffix: str) -> str:
        for prefix in self.PREFIXES:
            if (name := prefix + suffix) in self.names:
                return name
        raise models.ModelError(f"the checkpoint has no tensor {suffix}")

    def read(self, suffix: str) -> np.ndarray:
        name = self.resolve(suffix)
        shard = next(s for s in self.shards if name in s.entries)
        return shard.read(name)

    def read_bf16(self, suffix: str) -> np.ndarray:
        name = self.resolve(suffix)
        shard = next(s for s in self.shards if name in s.entries)
        return shard.read_bf16(name)


def _norm(values: np.ndarray, width: int) -> bytes:
    if values.shape != (width,):
        raise models.ModelError(f"a norm must be {width} wide")
    return _bf16_bytes(values)


def _layer_sections(target: Checkpoint, layout: Layout, layer: int):
    """One layer file's sections in readLayer order (LAYER_SECTIONS)."""
    prefix = f"layers.{layer}."
    hidden = layout.hidden
    head_dim = layout.head_dim_at(layer)
    width = layout.packed_width_at(layer)
    query = target.read(prefix + "self_attn.q_proj.weight").astype(np.float32)
    key = target.read(prefix + "self_attn.k_proj.weight").astype(np.float32)
    qkv = [query, key]
    if not layout.is_global(layer):
        # k_eq_v: global layers store no V projection.
        qkv.append(target.read(prefix + "self_attn.v_proj.weight").astype(np.float32))
    fused = np.concatenate(qkv)
    if fused.shape != (layout.packed_width_at(layer), hidden):
        raise models.ModelError(f"layer {layer} QKV is not {width} x {hidden}")
    # Batched-expert tensors are plain Parameters: no .weight suffix.
    gate_up = target.read(prefix + "experts.gate_up_proj").astype(np.float32)
    down = target.read(prefix + "experts.down_proj").astype(np.float32)
    half = layout.expert_intermediate
    if gate_up.shape != (layout.experts, 2 * half, hidden):
        raise models.ModelError(f"layer {layer} expert gate_up shape is off")
    if down.shape != (layout.experts, hidden, half):
        raise models.ModelError(f"layer {layer} expert down shape is off")
    scalar = target.read(prefix + "layer_scalar")
    scalar32 = np.asarray(scalar, np.float32).reshape(1)
    sections = [
        _norm(target.read_bf16(prefix + "input_layernorm.weight"), hidden),
        pack_projection(fused, width, hidden),
        _norm(target.read_bf16(prefix + "self_attn.q_norm.weight"), head_dim),
        _norm(target.read_bf16(prefix + "self_attn.k_norm.weight"), head_dim),
        pack_projection(
            target.read(prefix + "self_attn.o_proj.weight").astype(np.float32),
            hidden,
            layout.attention_width_at(layer),
        ),
        _norm(target.read_bf16(prefix + "post_attention_layernorm.weight"), hidden),
        # One pre-FFN norm feeds the shared expert; the routed experts take
        # the checkpoint's second pre-FFN norm.
        _norm(target.read_bf16(prefix + "pre_feedforward_layernorm.weight"), hidden),
        _norm(target.read_bf16(prefix + "pre_feedforward_layernorm_2.weight"), hidden),
        _norm(target.read_bf16(prefix + "router.scale"), hidden),
        _bf16_bytes(target.read_bf16(prefix + "router.proj.weight").reshape(-1)),
        np.asarray(target.read(prefix + "router.per_expert_scale"), np.float32)
        .reshape(-1)
        .tobytes(),
        # gate rows 0..704 and up rows 704..1407 of each expert's fused
        # gate_up, each padded to the packed expert width.
        pack_experts(gate_up[:, :half, :], layout.packed_expert_width, hidden),
        pack_experts(gate_up[:, half:, :], layout.packed_expert_width, hidden),
        pack_experts(down, hidden, layout.packed_expert_width),
        pack_projection(
            target.read(prefix + "mlp.gate_proj.weight").astype(np.float32),
            layout.packed_shared_width,
            hidden,
        ),
        pack_projection(
            target.read(prefix + "mlp.up_proj.weight").astype(np.float32),
            layout.packed_shared_width,
            hidden,
        ),
        pack_projection(
            target.read(prefix + "mlp.down_proj.weight").astype(np.float32),
            hidden,
            layout.packed_shared_width,
        ),
        _norm(target.read_bf16(prefix + "post_feedforward_layernorm_1.weight"), hidden),
        _norm(target.read_bf16(prefix + "post_feedforward_layernorm_2.weight"), hidden),
        _norm(target.read_bf16(prefix + "post_feedforward_layernorm.weight"), hidden),
        scalar32.tobytes(),
    ]
    if len(sections) != len(LAYER_SECTIONS):
        raise models.ModelError("packed layer section count changed")
    return sections


def _draft_layer_sections(draft: Checkpoint, layout: DraftLayout, layer: int):
    """One draft layer file's sections, in dflashV1LayerImage's read order
    (DRAFT_LAYER_SECTIONS)."""
    prefix = f"layers.{layer}."
    attention = prefix + "self_attn."
    fused = np.concatenate(
        [
            draft.read(attention + name + ".weight").astype(np.float32)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
    )
    if fused.shape != (layout.qkv, layout.hidden):
        raise models.ModelError(f"draft layer {layer} QKV is not the packed shape")
    return [
        _norm(draft.read_bf16(prefix + "input_layernorm.weight"), layout.hidden),
        pack_projection(fused, layout.qkv, layout.hidden),
        _norm(draft.read_bf16(attention + "q_norm.weight"), layout.head_dim),
        _norm(draft.read_bf16(attention + "k_norm.weight"), layout.head_dim),
        pack_projection(
            draft.read(attention + "o_proj.weight").astype(np.float32),
            layout.hidden,
            layout.attention,
        ),
        _norm(
            draft.read_bf16(prefix + "post_attention_layernorm.weight"),
            layout.hidden,
        ),
        pack_projection(
            draft.read(prefix + "mlp.gate_proj.weight").astype(np.float32),
            layout.intermediate,
            layout.hidden,
        ),
        pack_projection(
            draft.read(prefix + "mlp.up_proj.weight").astype(np.float32),
            layout.intermediate,
            layout.hidden,
        ),
        pack_projection(
            draft.read(prefix + "mlp.down_proj.weight").astype(np.float32),
            layout.hidden,
            layout.intermediate,
        ),
    ]


def pack_draft(draft: Checkpoint, out_dir: Path, layout: DraftLayout):
    """The packed plain DFlash draft: draft/layer-N.bin and draft/model.bin
    (PackedDFlashV1DraftFiles' reads)."""
    for layer in range(layout.layers):
        write_packed_file(
            out_dir / f"layer-{layer}.bin",
            DRAFT_LAYER_MAGIC,
            layer,
            0,
            _draft_layer_sections(draft, layout, layer),
        )
    write_packed_file(
        out_dir / "model.bin",
        DRAFT_LAYER_MAGIC,
        layout.layers,
        1,
        [
            pack_projection(
                draft.read("fc.weight").astype(np.float32),
                layout.hidden,
                layout.target_hidden,
            ),
            _norm(draft.read_bf16("hidden_norm.weight"), layout.hidden),
            _norm(draft.read_bf16("norm.weight"), layout.hidden),
        ],
    )


def pack_diffusion_extras(checkpoint: Checkpoint, layout: Layout, target_dir: Path):
    """DiffusionGemma's extra packed files: the self-conditioning stack (a
    pre-norm, then the 2112-wide GeGLU's gate, up and down at the shared
    expert's packed padding) and the encoder's per-layer scalars, one fp32
    per layer in layer order."""
    prefix = "self_conditioning."
    write_packed_file(
        target_dir / "self_conditioning.bin",
        HEAD_MAGIC,
        0,
        SELF_CONDITIONING_TYPE,
        [
            _norm(checkpoint.read_bf16(prefix + "pre_norm.weight"), layout.hidden),
            pack_projection(
                checkpoint.read(prefix + "gate_proj.weight").astype(np.float32),
                layout.packed_shared_width,
                layout.hidden,
            ),
            pack_projection(
                checkpoint.read(prefix + "up_proj.weight").astype(np.float32),
                layout.packed_shared_width,
                layout.hidden,
            ),
            pack_projection(
                checkpoint.read(prefix + "down_proj.weight").astype(np.float32),
                layout.hidden,
                layout.packed_shared_width,
            ),
        ],
    )
    scalars = np.concatenate(
        [
            np.asarray(
                checkpoint.read(
                    f"model.encoder.language_model.layers.{layer}.layer_scalar"
                ),
                np.float32,
            ).reshape(-1)
            for layer in range(layout.layers)
        ]
    ).astype(np.float32)
    if scalars.shape != (layout.layers,):
        raise models.ModelError("the encoder's layer scalars are not one per layer")
    write_packed_file(
        target_dir / "encoder_scalars.bin",
        HEAD_MAGIC,
        0,
        ENCODER_SCALARS_TYPE,
        [scalars.tobytes()],
    )


def package_manifest(
    model,
    family_name,
    layout: Layout,
    sources,
    artifacts,
    draft: DraftLayout | None = None,
    *,
    format_name: str = FORMAT_NAME,
    architecture: str,
    diffusion=None,
):
    """manifest.json of a built package, as inspectModelPackage validates it
    (validateGemma4 / validateNewPackedFormat)."""
    return {
        "model": model,
        "schema_version": 1,
        "family": family_name,
        "built_by": "install/pack.py",
        "sources": sources,
        "format": {
            "name": format_name,
            "q4_bits": 4,
            "q4_group_size": Q4_GROUP,
            "q4_storage_n": Q4_TILE_ROWS,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": LAYER_MAGIC.decode(),
            "draft_layer_magic": (
                DRAFT_LAYER_MAGIC.decode() if draft is not None else DRAFT_MAGIC
            ),
            "vision_magic": VISION_MAGIC,
        },
        "execution_geometry": {
            "draft_proposal_tokens": 15,
            "draft_query_rows": 16,
            # The DFlash draft's trained window (family signature), under the
            # runtime's draft ring capacity.
            "draft_sliding_window": 2048,
        },
        TARGET: {
            "architecture": architecture,
            "layers": layout.layers,
            "hidden_size": layout.hidden,
            "vocabulary_size": layout.vocabulary,
            "num_attention_heads": layout.query_heads,
            "num_key_value_heads": layout.kv_heads,
            "head_dim": layout.head_dim,
            "global_num_key_value_heads": layout.global_kv_heads,
            "global_head_dim": layout.global_head_dim,
            "sliding_window": layout.sliding_window,
            "experts": layout.experts,
            "experts_per_token": layout.experts_per_token,
            "moe_intermediate_size": layout.expert_intermediate,
            "shared_expert_intermediate_size": layout.shared_intermediate,
            "final_logit_softcapping": layout.logit_softcap,
            "global_rope_theta": layout.global_rotary_theta,
            "rope_theta": layout.rotary_theta,
            "layer_types": layout.layer_types(),
        },
        **(
            {
                # applyDeclaredDraft's fields for DraftKind::DFlashV1 (the
                # packed descriptor stays Null when no draft is declared).
                DRAFT: {
                    "architecture": "DFlashDraftModel",
                    "num_attention_heads": draft.query_heads,
                    "num_key_value_heads": draft.kv_heads,
                    "head_dim": draft.head_dim,
                    "layers": draft.layers,
                    "hidden_size": draft.hidden,
                    "intermediate_size": draft.intermediate,
                    "rope_theta": draft.rotary_theta,
                    "block_size": draft.block_size,
                    "causal_layers": draft.causal_layers,
                    "sliding_window": draft.sliding_window,
                }
            }
            if draft is not None
            else {}
        ),
        # DiffusionGemma's denoising schedule (generation_config.json plus
        # the scheduler's defaults); a packed gemma4 carries none.
        **({"diffusion": dict(diffusion)} if diffusion is not None else {}),
        "artifacts": artifacts,
    }


def _text_type(config):
    """The model_type the config's text model states, or None."""
    text = (
        config.get("text_config")
        if isinstance(config.get("text_config"), dict)
        else config
    )
    return text.get("model_type") if isinstance(text, dict) else None


def packable(config) -> bool:
    """Whether this upstream config is a packed-only target."""
    return _text_type(config) in PACKABLE


def target_format(config):
    """The packed package's format name and Target.format of a packable
    config's kind (PACKABLE)."""
    return PACKABLE[_text_type(config)]


def is_built(link: Path) -> bool:
    """Whether the selection link names a locally packed package (a directory
    with a manifest.json of this format), not a Hub package snapshot."""
    path = link / PACKAGE_MANIFEST
    if not path.exists():
        return False
    try:
        manifest = models.read_json(path)
    except models.ModelError:
        return False
    format_ = manifest.get("format")
    return (
        isinstance(format_, dict)
        and format_.get("name") in {FORMAT_NAME, DIFFUSION_FORMAT_NAME}
        and isinstance(manifest.get("sources"), dict)
    )


def manifest_of(link: Path):
    return models.read_json(link / PACKAGE_MANIFEST)


def _artifact(path: Path, root: Path):
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": models.sha256(path),
    }


def _package_key(sources) -> str:
    """The .packed entry of these sources and this packer build."""
    identity = dict(sources)
    identity["adapter"] = models.sha256(Path(__file__))
    return hashlib.sha256(models.json_bytes(identity)).hexdigest()


def verify(link: Path, *, full=False):
    """The built package at a selection link: its manifest names the packed
    format and every artifact's size (with full, its content) matches."""
    manifest = manifest_of(link)
    records = manifest.get("artifacts")
    if not is_built(link) or not isinstance(records, list) or not records:
        raise models.ModelError("not a locally packed model package")
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size", "sha256"}
            or not models.is_safe_path(record["path"])
            or type(record["size"]) is not int
            or record["size"] <= 0
            or not models.is_hex_digest(record["sha256"], 64)
        ):
            raise models.ModelError("packed package manifest has an invalid artifact")
        path = link / record["path"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise models.ModelError(
                f"installed artifact has the wrong size: {record['path']}"
            )
        if record["path"].endswith(".bin") and record["size"] % ALIGNMENT:
            raise models.ModelError(
                f"installed packed file is unaligned: {record['path']}"
            )
        if full and models.sha256(path) != record["sha256"]:
            raise models.ModelError(
                f"installed artifact checksum changed: {record['path']}"
            )
    return manifest


def _copy(source: Path, stage: Path, name: str):
    destination = stage / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def build_package(
    stage: Path,
    model: str,
    family_name: str,
    config,
    weights,
    tokenizer_files,
    draft_files,
    sources,
    *,
    layout: Layout = GEMMA4,
    draft_layout: DraftLayout | None = GEMMA4_DRAFT,
):
    """Write the packed package into stage: manifest.json, target/*.bin from
    the checkpoint shards in weights (path per repository name), the
    tokenizer files and the draft's packed layer files."""
    stage.mkdir(parents=True, exist_ok=True)
    target_dir = stage / TARGET
    target_dir.mkdir()
    checkpoint = Checkpoint(
        [weights[name] for name in sorted(weights) if name.endswith(".safetensors")]
    )
    embedding = checkpoint.read("embed_tokens.weight").astype(np.float32)
    if embedding.shape != (layout.vocabulary, layout.hidden):
        raise models.ModelError("the token embedding shape does not match the layout")
    # readAffineEmbedding reads three separately aligned sections where a
    # projection's planes share one; the logits head takes the same planes
    # joined (tied embeddings quantize identically).
    planes = _quantize_rows(embedding)
    write_packed_file(
        target_dir / "embedding.bin",
        EMBEDDING_MAGIC,
        layout.vocabulary,
        layout.hidden,
        planes,
    )
    write_packed_file(
        target_dir / "head.bin",
        HEAD_MAGIC,
        layout.layers,
        HEAD_LAYER_TYPE,
        [
            _norm(checkpoint.read_bf16("norm.weight"), layout.hidden),
            b"".join(planes),
        ],
    )
    for layer in range(layout.layers):
        write_packed_file(
            target_dir / f"layer-{layer}.bin",
            LAYER_MAGIC,
            layer,
            LAYER_TYPE_GLOBAL if layout.is_global(layer) else LAYER_TYPE_LOCAL,
            _layer_sections(checkpoint, layout, layer),
        )
    model_type = _text_type(config)
    diffusion = model_type == "diffusion_gemma_text"
    if diffusion:
        pack_diffusion_extras(checkpoint, layout, target_dir)
    for name, source in tokenizer_files.items():
        _copy(source, stage / TOKENIZER, name)
    _write_tokenizer_config(stage, config)
    # Pack the plain DFlash draft's checkpoint (PackedDFlashV1DraftFiles reads
    # draft/layer-N.bin and draft/model.bin). A package without one keeps
    # the Null draft's n-gram predraft.
    draft = None
    draft_weights = [
        path for name, path in draft_files.items() if name.endswith(".safetensors")
    ]
    if draft_weights:
        draft = draft_layout or GEMMA4_DRAFT
        draft_dir = stage / DRAFT
        draft_dir.mkdir()
        pack_draft(Checkpoint(draft_weights), draft_dir, draft)
        for name, source in draft_files.items():
            if name.endswith("config.json"):
                _copy(source, stage / DRAFT, name.removeprefix(DRAFT + "/"))
    written = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = package_manifest(
        model,
        family_name,
        layout,
        sources,
        [_artifact(path, stage) for path in written],
        draft,
        format_name=PACKABLE.get(model_type, (FORMAT_NAME,))[0],
        architecture=model_type if model_type in PACKABLE else "gemma4_text",
        diffusion=DIFFUSION if diffusion else None,
    )
    (stage / PACKAGE_MANIFEST).write_bytes(models.json_bytes(manifest))
    return manifest


def _write_tokenizer_config(stage: Path, config):
    """tokenizer/config.json: the text configuration the descriptor's
    validateTokenizer reads (nested under text_config, like an MLX target)."""
    text = (
        config.get("text_config")
        if isinstance(config.get("text_config"), dict)
        else config
    )
    _stage_write(
        stage / TOKENIZER / "config.json", models.json_bytes({"text_config": text})
    )


def _stage_write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def install(selection, family, config, repo, target_files, draft_repo, draft_files):
    """Pack the downloaded target into a package under the models root and
    link the selection to it, pinning the source snapshots. target_files and
    draft_files map repository/assembly names to local paths."""
    from . import hub

    weights = {
        name: path
        for name, path in target_files.items()
        if name.endswith(".safetensors")
    }
    tokenizer = {
        name: path
        for name, path in target_files.items()
        # upstream.TOKENIZER_FILES' set, repeated: upstream imports this
        # module.
        if name
        in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "added_tokens.json",
            "special_tokens_map.json",
        )
    }
    sources = {TARGET: repo.identity()}
    if draft_repo is not None:
        sources[DRAFT] = draft_repo.identity()
    models_root = selection.models_root
    destination = models_root / PACKED / _package_key(sources)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=PACK_STAGING, dir=destination.parent))
    try:
        manifest = build_package(
            stage,
            selection.model,
            family.name,
            config,
            weights,
            tokenizer,
            draft_files,
            sources,
        )
        if destination.exists():
            shutil.rmtree(destination)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    pins = []
    for files, source_repo in (
        (target_files, repo),
        (draft_files, draft_repo),
    ):
        if source_repo is None or source_repo.revision is None:
            continue
        snapshot = next(
            (snap for path in files.values() if (snap := hub.snapshot_of(str(path)))),
            None,
        )
        if snapshot is not None:
            pins.append(hub.pin(snapshot, source_repo.name, selection.link))
    models.link_selection(selection.link, destination)
    verify(selection.link)
    hub.retire_other_pins(pins)
    return manifest
