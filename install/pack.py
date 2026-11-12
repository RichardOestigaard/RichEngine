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
from pathlib import Path
from typing import Literal

import numpy as np

from . import models
from .layout import DRAFT, PACK_STAGING, PACKAGE_MANIFEST, PACKED, TARGET, TOKENIZER
from .pack_layouts import (
    ENCODER_SCALARS_TYPE,
    HEAD_LAYER_TYPE,
    LAYER_SECTIONS,
    LAYER_TYPE_GLOBAL,
    LAYER_TYPE_LOCAL,
    SELF_CONDITIONING_TYPE,
    DFlash2Layout,
    DraftLayout,
    Layout,
    Qwen4ExpLayout,
    draft_layout_from_spec,
    execution_geometry,
)
from .pack_layouts import (
    draft_layout as family_draft_layout,
)

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
QWEN4_FORMAT_NAME = "richengine-packed-gguf-qwen4exp"
# The Target.format upstream.py uses for a checkpoint the installer packs.
TARGET_FORMAT = "packed-gemma4"
DIFFUSION_TARGET_FORMAT = "packed-diffusiongemma"
QWEN4_TARGET_FORMAT = "packed-qwen4exp"
PackedFormat = Literal["packed-gemma4", "packed-diffusiongemma", "packed-qwen4exp"]
TARGET_FORMATS: tuple[PackedFormat, ...] = (
    TARGET_FORMAT,
    DIFFUSION_TARGET_FORMAT,
    QWEN4_TARGET_FORMAT,
)
# The text model_types the installer packs, to their package format name
# and target format.
PACKABLE = {
    "gemma4_text": (FORMAT_NAME, TARGET_FORMAT),
    "diffusion_gemma_text": (DIFFUSION_FORMAT_NAME, DIFFUSION_TARGET_FORMAT),
    "qwen4_exp_text": (QWEN4_FORMAT_NAME, QWEN4_TARGET_FORMAT),
}


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
        if dtype in ("BF16", "F16"):
            dtype, width = np.uint16, 2
        elif dtype == "F32":
            dtype, width = np.float32, 4
        elif dtype in ("F64",):
            dtype, width = np.float64, 8
        elif dtype in ("I64", "U64"):
            dtype, width = np.int64 if dtype == "I64" else np.uint64, 8
        elif dtype in ("I32", "U32"):
            dtype, width = np.int32 if dtype == "I32" else np.uint32, 4
        elif dtype in ("I16", "U16"):
            dtype, width = np.int16 if dtype == "I16" else np.uint16, 2
        elif dtype in ("I8", "U8", "BOOL"):
            dtype, width = np.int8 if dtype == "I8" else np.uint8, 1
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

    def read_int(self, name: str) -> np.ndarray:
        """An integer tensor as int64 (the PLE hash's buffers)."""
        data = self.raw[self.base + self.info(name)[2] : self.base + self.info(name)[3]]
        dtype = self.info(name)[0]
        if np.issubdtype(dtype, np.integer) or dtype == np.uint8:
            return data.view(dtype).reshape(self.info(name)[1]).astype(np.int64)
        raise models.ModelError(f"tensor {name} is not an integer buffer")


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

    def read_int(self, suffix: str) -> np.ndarray:
        name = self.resolve(suffix)
        shard = next(s for s in self.shards if name in s.entries)
        return shard.read_int(name)

    def has(self, suffix: str) -> bool:
        return any(prefix + suffix in self.names for prefix in self.PREFIXES)

    def read_sharded(self, suffix: str) -> np.ndarray:
        """A tensor stored whole, or `suffix`'s shard_N pieces concatenated
        along dim 0 (the PLE n-gram table's split_ngram_parts export)."""
        try:
            return self.read(suffix)
        except models.ModelError:
            pass
        pieces = []
        # HF's sharded export names the pieces <stem>.shard_N.weight.
        stem = suffix[: -len(".weight")] if suffix.endswith(".weight") else suffix
        for prefix in self.PREFIXES:
            base = prefix + stem
            shards = sorted(
                (
                    name
                    for name in self.names
                    if name.startswith(base + ".shard_") and name.endswith(".weight")
                ),
                key=lambda name: int(name.rsplit(".shard_", 1)[1].split(".")[0]),
            )
            if shards:
                pieces = [self._read_name(name) for name in shards]
                break
        if not pieces:
            raise models.ModelError(f"the checkpoint has no tensor {suffix}")
        return np.concatenate(pieces, axis=0)

    def _read_name(self, name: str) -> np.ndarray:
        shard = next(s for s in self.shards if name in s.entries)
        return shard.read(name)


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


# ============================ dflash2 ============================
# The packed DFlash2 draft (DraftKind::DFlashTwo, runtime/model/
# DFlashTwoDraft.cpp readDraft): draft/layer-N.bin and draft/model.bin under
# the MDFD0004 magic. Each layer file's sections are readDraft's order —
# the input norm, the attention convolution's bf16 base kernel, the
# attention dynamic projection, the fused QKV, the per-head norms, the
# output projection, the post norm, then the MLP's convolution, dynamic,
# gate, up and down. model.bin holds the context projection, the two
# norms, the selector projection and the predecessor/successor codebooks.
#
# The two dynamic projections per layer are hidden/4 rows — 640 at hidden
# 2560 — which the affine 256-row tile cannot hold, so they pack as F32
# GGUF segments (a GgufTensorDescriptor and its fp32 rows, the gguf_float
# decode path); every other projection keeps the affine Q4 planes, which
# validateQ4Layout accepts. readDraft reads the dynamics through
# readBlockProjection for exactly this reason.


def _dflash2_convolution(draft: Checkpoint, prefix: str, hidden: int) -> bytes:
    """A dynamic convolution's bf16 base kernel: the raw [2, 2, hidden]
    section readDraft takes as one section."""
    kernel = draft.read_bf16(prefix + "base_kernel").reshape(-1)
    if kernel.size != 4 * hidden:
        raise models.ModelError(f"{prefix}base_kernel is not 2x2x{hidden}")
    return _bf16_bytes(kernel)


def _dflash2_dynamic(draft: Checkpoint, name: str, layout: DFlash2Layout):
    """A dynamic projection's F32 GGUF segment: [dynamic, hidden] fp32 rows
    behind their descriptor."""
    weights = draft.read(name + ".weight").astype(np.float32)
    if weights.shape != (layout.dynamic, layout.hidden):
        raise models.ModelError(f"{name} is not {layout.dynamic} x {layout.hidden}")
    return _f32_segment(weights)


def _dflash2_layer_sections(draft: Checkpoint, layout: DFlash2Layout, layer: int):
    """One DFlash2 draft layer file's sections, in readDraft's read order."""
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
    output = draft.read(attention + "o_proj.weight").astype(np.float32)
    if output.shape != (layout.hidden, layout.attention):
        raise models.ModelError(f"draft layer {layer} output projection is off")
    return [
        _norm(draft.read_bf16(prefix + "input_layernorm.weight"), layout.hidden),
        _dflash2_convolution(draft, prefix + "attention_conv.", layout.hidden),
        *_dflash2_dynamic(draft, prefix + "attention_conv.kernel_projection", layout),
        pack_projection(fused, layout.qkv, layout.hidden),
        _bf16_bytes(draft.read_bf16(attention + "q_norm.weight").reshape(-1)),
        _bf16_bytes(draft.read_bf16(attention + "k_norm.weight").reshape(-1)),
        pack_projection(output, layout.hidden, layout.attention),
        _norm(
            draft.read_bf16(prefix + "post_attention_layernorm.weight"),
            layout.hidden,
        ),
        _dflash2_convolution(draft, prefix + "mlp_conv.", layout.hidden),
        *_dflash2_dynamic(draft, prefix + "mlp_conv.kernel_projection", layout),
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


def _dflash2_codebook(draft: Checkpoint, name: str, layout: DFlash2Layout) -> bytes:
    """A selector codebook's raw bf16 [vocabulary, selector_rank] rows."""
    table = draft.read_bf16(name).reshape(-1)
    if table.size != layout.vocabulary * layout.selector_rank:
        raise models.ModelError(f"{name} is not the packed codebook shape")
    return _bf16_bytes(table)


def pack_draft2(draft: Checkpoint, out_dir: Path, layout: DFlash2Layout):
    """The packed DFlash2 draft: draft/layer-N.bin and draft/model.bin
    (PackedDraftFiles' reads, DFlashTwoDraft.cpp)."""
    if layout.dynamic * 4 != layout.hidden:
        raise models.ModelError("a DFlash2 draft's dynamic width is hidden / 4")
    for layer in range(layout.layers):
        write_packed_file(
            out_dir / f"layer-{layer}.bin",
            DRAFT_MAGIC.encode(),
            layer,
            0,
            _dflash2_layer_sections(draft, layout, layer),
        )
    selector = "candidate_selector."
    write_packed_file(
        out_dir / "model.bin",
        DRAFT_MAGIC.encode(),
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
            pack_projection(
                draft.read(selector + "hidden_projection.weight").astype(np.float32),
                layout.selector_rank,
                layout.hidden,
            ),
            _dflash2_codebook(draft, selector + "predecessor_codebook", layout),
            _dflash2_codebook(draft, selector + "successor_codebook", layout),
        ],
    )


def _dflash2_manifest_draft(layout: DFlash2Layout) -> dict:
    """The manifest's draft declaration of a packed DFlash2 draft: the
    geometry applyManifestDraftKind's family layout fixes, stated so the
    package self-describes (validateQwen4Exp re-checks it)."""
    return {
        "architecture": "DFlash2DraftModel",
        "layers": layout.layers,
        "hidden_size": layout.hidden,
        "vocabulary_size": layout.vocabulary,
        "intermediate_size": layout.intermediate,
        "num_attention_heads": layout.query_heads,
        "num_key_value_heads": layout.kv_heads,
        "head_dim": layout.head_dim,
        "rope_theta": layout.rotary_theta,
        "block_size": layout.block_size,
        "selector_rank": layout.selector_rank,
        "dynamic_size": layout.dynamic,
        "sliding_window": layout.sliding_window,
        "target_hidden_size": layout.target_hidden,
    }


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


# ============================ qwen4exp ============================
# Qwen3.8-Flash-Next (model_type qwen4_exp_text, GGUF architecture
# qwen4exp): the packed-only target runtime/model/Qwen4Exp.cpp loads. Its
# files carry the MDFQ/MDFE magics and, unlike the Gemma formats' affine Q4
# planes, every projection is a native GGUF block tensor behind the 64-byte
# GgufTensorDescriptor of model/GgufImageLayout.hpp: a descriptor section,
# then the format's plane0/plane1/meta sections, each 16 KiB aligned. A
# safetensors checkpoint holds only floats, so the packer quantizes to Q8_0
# (GGUF type 8: an fp16 scale and 32 int8 codes per 32 columns) — a format
# every native kernel path reads, including the embedding gathers and the
# block MoE slabs — and keeps F32 segments where the loader or the MoE
# contract requires them (the router and shared scalar gate, the
# hyper-connection low-rank projections, the GDN alpha/beta rows, the QSA
# indexer and every norm).

QWEN4_LAYER_MAGIC = b"MDFQ0001"
QWEN4_HEAD_MAGIC = b"MDFQ0002"
QWEN4_TABLE_MAGIC = b"MDFQ0003"
GGML_F32 = 0
GGML_Q8_0 = 8
# Q8_0's QuantFormat row (metal/abi/QuantFormat.h): 32 elements a block,
# 34 native block bytes, 32 plane0 and 2 meta bytes per row group of 32.
_Q8O_BLOCK = 32
_Q8O_PLANE0 = 32
_Q8O_META = 2


# The lane-owned chunk order of a group's 32 elements
# (metal/abi/QuantFormat.h): element e sits at slot quant_slot(e); the
# projection planes store the group's bytes in slot order.
def _quant_slot(element: int) -> int:
    return 8 * ((element >> 2) & 3) + 4 * (element >> 4) + (element & 3)


_Q8O_SLOT_ORDER = np.argsort([_quant_slot(e) for e in range(_Q8O_BLOCK)])


def _q8o_blocks(weights: np.ndarray):
    """An [n, k] fp32 matrix (k a multiple of 32) as Q8_0: per 32-column
    group an fp16 scale and int8 codes, ggml's absmax/127 quantization."""
    n, k = weights.shape
    groups = weights.reshape(n, k // _Q8O_BLOCK, _Q8O_BLOCK).astype(np.float32)
    if not np.isfinite(groups).all():
        raise models.ModelError("non-finite weight in a Q8_0 projection")
    amax = np.abs(groups).max(axis=-1)
    scale = amax / 127.0
    inverse = np.where(scale > 0, 1.0 / np.where(scale > 0, scale, 1.0), 0.0)
    codes = np.clip(_round_away(groups * inverse[..., None]), -127.0, 127.0).astype(
        np.int8
    )
    return scale.astype(np.float16), codes.reshape(n, k)


def _q8o_native_rows(weights: np.ndarray) -> bytes:
    """The native GGUF row plane of an [n, k] tensor: per row, per group of
    32 columns the block {fp16 d, int8 qs[32]} in element order — what
    ggufRowBytes counts and the embedding/PLE gathers decode."""
    n, k = weights.shape
    if k % _Q8O_BLOCK:
        raise models.ModelError("a native Q8_0 row needs 32-column groups")
    scales, codes = _q8o_blocks(weights)
    groups = k // _Q8O_BLOCK
    rows = np.empty((n, groups, 2 + _Q8O_BLOCK), np.uint8)
    rows[:, :, :2] = (
        scales.view(np.uint16)[..., None].view(np.uint8).reshape(n, groups, 2)
    )
    rows[:, :, 2:] = codes.reshape(n, groups, _Q8O_BLOCK).view(np.uint8)
    return rows.reshape(n, groups * (2 + _Q8O_BLOCK)).tobytes()


def _q8o_planes(weights: np.ndarray):
    """The repacked tile-major planes of an [n, k] tensor in Q8_0, rows
    padded to whole 256-row tiles: plane0 [tiles][k/32][256][32]
    slot-ordered codes, no plane1, meta [tiles][k/32][256][2] fp16 scales.
    Returns (plane0, plane1, meta, padded_rows)."""
    n, k = weights.shape
    if k % _Q8O_BLOCK:
        raise models.ModelError("a tiled Q8_0 tensor needs 32-column groups")
    groups = k // _Q8O_BLOCK
    tiles = (n + Q4_TILE_ROWS - 1) // Q4_TILE_ROWS
    rows = tiles * Q4_TILE_ROWS
    padded = np.zeros((rows, k), np.float32)
    padded[:n] = weights
    scales, codes = _q8o_blocks(padded)
    plane0 = (
        codes.reshape(rows, groups, _Q8O_BLOCK)[:, :, _Q8O_SLOT_ORDER]
        .reshape(tiles, Q4_TILE_ROWS, groups, _Q8O_BLOCK)
        .transpose(0, 2, 1, 3)
    )
    meta = (
        scales.reshape(rows, groups)
        .reshape(tiles, Q4_TILE_ROWS, groups)
        .transpose(0, 2, 1)
    )
    return (
        plane0.view(np.uint8).tobytes(),
        b"",
        meta.view(np.uint16).tobytes(),
        rows,
    )


def _gguf_descriptor(kind: int, rows: int, columns: int, planes) -> bytes:
    """The 64-byte little-endian GgufTensorDescriptor. planes is None for an
    F32 tensor (rows of fp32 in plane0) or (plane0, plane1, meta) byte
    counts of a Q8_0 one."""
    if planes is None:
        p0 = p1 = meta_bytes = meta_groups = 0
        plane0, plane1, meta = rows * columns * 4, 0, 0
    else:
        plane0, plane1, meta = planes
        p0, p1, meta_bytes, meta_groups = _Q8O_PLANE0, 0, _Q8O_META, 1
    return struct.pack(
        "<8I4Q",
        kind,
        rows,
        columns,
        p0,
        p1,
        meta_bytes,
        meta_groups,
        0,
        plane0,
        plane1,
        meta,
        0,
    )


def _f32_descriptor(rows: int, columns: int) -> bytes:
    return _gguf_descriptor(GGML_F32, rows, columns, None)


def _q8o_descriptor(rows: int, columns: int, plane0: int, meta: int) -> bytes:
    return _gguf_descriptor(GGML_Q8_0, rows, columns, (plane0, 0, meta))


def _f32_segment(weights: np.ndarray) -> list:
    """A descriptor + fp32 rows section pair of an [rows, columns] float
    tensor — the segments the loader reads unrounded."""
    weights = np.ascontiguousarray(weights, np.float32)
    rows, columns = weights.shape
    return [
        _f32_descriptor(rows, columns),
        weights.tobytes(),
    ]


def _q8o_segment(weights: np.ndarray) -> list:
    """A descriptor + plane0 + meta section triple of an [rows, columns]
    tensor in Q8_0. The descriptor states the logical rows and columns and
    the planes' true byte counts; the runtime's tile-alignment check wants
    rows and columns in 256-column multiples, which every projection but
    the 640-wide expert intermediates satisfies."""
    plane0, _plane1, meta, _padded = _q8o_planes(weights)
    rows, columns = weights.shape
    return [
        _q8o_descriptor(rows, columns, len(plane0), len(meta)),
        plane0,
        meta,
    ]


def _raw_f32(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, np.float32).tobytes()


def _narrow_bf16(values: np.ndarray) -> bytes:
    """values as little-endian BF16 (the GDN convolution and time bias the
    kernels read narrowed)."""
    return _to_bf16_bits(np.ascontiguousarray(values, np.float32)).tobytes()


def _qwen4_hyper_sections(
    checkpoint: Checkpoint, prefix: str, inject: bool, layout
) -> list:
    """One hyper-connection mix's sections (readHyper): the streams' F32
    norm, the low-rank down/up F32 projections and — when the mix injects —
    the injection weights as raw F32."""
    streams = layout.hyper_connections * layout.hidden_size
    rank = layout.hyper_rank
    norm = checkpoint.read(prefix + "hc_norm.weight")
    down = checkpoint.read(prefix + "input_mix_weight_down.weight")
    up = checkpoint.read(prefix + "input_mix_weight_up.weight")
    if (
        norm.shape != (streams,)
        or down.shape != (rank, streams)
        or up.shape != (streams, rank)
    ):
        raise models.ModelError(f"{prefix}hyper-connection shapes are off")
    sections = [
        _raw_f32(norm),
        *_f32_segment(down),
        *_f32_segment(up),
    ]
    if inject:
        inject_w = checkpoint.read(prefix + "block_inject_weight.weight")
        if inject_w.shape != (layout.hyper_connections, streams):
            raise models.ModelError(f"{prefix}block_inject_weight shape is off")
        sections.append(_raw_f32(inject_w))
    return sections


def pack_qwen4exp_layer_sections(checkpoint: Checkpoint, layout, layer: int):
    """One qwen4exp layer file's sections, in loadQwen4ExpWeights' read
    order: mixer hyper, mixer, the QSA indexer on full-attention layers,
    FFN hyper, the MoE block and — on the PLE layer — the PLE tensors."""
    prefix = f"layers.{layer}."
    sections = []
    # The mix before the mixer: hc_attn_*.
    sections += _qwen4_hyper_sections(
        checkpoint, prefix + "attn_hyper_connection.", True, layout
    )
    if layout.is_full_attention(layer):
        sections += _qwen4_attention_sections(checkpoint, prefix, layout)
    else:
        sections += _qwen4_gdn_sections(checkpoint, prefix, layout)
    if layout.is_full_attention(layer):
        sections += _qwen4_indexer_sections(checkpoint, prefix, layout)
    # The mix before the FFN: mlp_hyper_connection / hc_ffn_*.
    sections += _qwen4_hyper_sections(
        checkpoint, prefix + "mlp_hyper_connection.", True, layout
    )
    sections += _qwen4_moe_sections(checkpoint, prefix, layout)
    if layer == layout.ple_layer:
        sections += _qwen4_ple_sections(checkpoint, prefix, layout)
    return sections


def _qwen4_attention_sections(checkpoint, prefix, layout):
    """A full-attention layer's mixer sections: the fused attn-q/k/v
    segments ([q|gate] rows per head, then k and v), the QK norms and the
    output projection."""
    hidden = layout.hidden_size
    head_dim = layout.attention_head_dimension
    query_rows = layout.attention_query_stride * layout.attention_width
    kv_rows = layout.attention_kv_heads * head_dim
    q = checkpoint.read(prefix + "self_attn.q_proj.weight")
    k = checkpoint.read(prefix + "self_attn.k_proj.weight")
    v = checkpoint.read(prefix + "self_attn.v_proj.weight")
    if (
        q.shape != (query_rows, hidden)
        or k.shape != (kv_rows, hidden)
        or v.shape != (kv_rows, hidden)
    ):
        raise models.ModelError(
            f"{prefix}self_attn projections do not match the packed QKV"
        )
    sections = []
    for weight in (q, k, v):
        sections += _q8o_segment(weight)
    sections.append(
        _raw_f32(checkpoint.read(prefix + "self_attn.q_norm.weight").reshape(-1))
    )
    sections.append(
        _raw_f32(checkpoint.read(prefix + "self_attn.k_norm.weight").reshape(-1))
    )
    sections += _q8o_segment(checkpoint.read(prefix + "self_attn.o_proj.weight"))
    return sections


def _qwen4_gdn_sections(checkpoint, prefix, layout):
    """A GDN layer's mixer sections: the fused qkv/z/ab projections, the
    depthwise convolution (bf16), the per-value-head decay (f32) and time
    bias (bf16), the gated norm and the output projection whose input
    columns are the kernels' tiled value-head order."""
    hidden = layout.hidden_size
    heads = layout.gdn_value_heads
    conv_dim = layout.convolution_dimension
    taps = layout.convolution_taps
    value_dim = heads * layout.gdn_head_dimension
    qkv = checkpoint.read(prefix + "linear_attn.in_proj_qkv.weight")
    z = checkpoint.read(prefix + "linear_attn.in_proj_z.weight")
    beta = checkpoint.read(prefix + "linear_attn.in_proj_b.weight")
    alpha = checkpoint.read(prefix + "linear_attn.in_proj_a.weight")
    conv = checkpoint.read(prefix + "linear_attn.conv1d.weight")
    a_log = checkpoint.read(prefix + "linear_attn.A_log")
    dt_bias = checkpoint.read(prefix + "linear_attn.dt_bias")
    norm = checkpoint.read(prefix + "linear_attn.norm.weight")
    out = checkpoint.read(prefix + "linear_attn.out_proj.weight")
    if qkv.shape != (conv_dim, hidden) or z.shape != (value_dim, hidden):
        raise models.ModelError(f"{prefix}linear_attn qkv/z shapes are off")
    if beta.shape != (heads, hidden) or alpha.shape != (heads, hidden):
        raise models.ModelError(f"{prefix}linear_attn a/b shapes are off")
    if conv.shape != (conv_dim, 1, taps):
        raise models.ModelError(f"{prefix}linear_attn conv1d shape is off")
    # The kernels' tiled value-head order for the out projection's inputs:
    # target head block t reads source head block s(t), the inverse of
    # gdn_output_head(h) = (h % headsPerKey) * keyHeads + h / headsPerKey.
    heads_per_key = heads // layout.gdn_key_heads
    source = np.array(
        [
            (t % layout.gdn_key_heads) * heads_per_key + t // layout.gdn_key_heads
            for t in range(heads)
        ]
    )
    head_dim = layout.gdn_head_dimension
    out_perm = out.reshape(out.shape[0], heads, head_dim)[:, source, :]
    out_perm = out_perm.reshape(out.shape[0], heads * head_dim)
    decay = -np.exp(a_log.astype(np.float32)).astype(np.float32)
    if a_log.size != heads or dt_bias.size != heads:
        raise models.ModelError(f"{prefix}linear_attn decay shapes are off")
    # gdn-ab fills the fused width's tail: beta and alpha rows, then zero
    # rows to packed_gdn_width (the fused reader wants the three segments
    # to cover it exactly).
    ab_rows = layout.packed_gdn_width - conv_dim - value_dim
    ab = np.zeros((ab_rows, hidden), np.float32)
    ab[:heads] = beta
    ab[heads : 2 * heads] = alpha
    sections = []
    sections += _q8o_segment(qkv)
    sections += _q8o_segment(z)
    sections += _f32_segment(ab)
    sections.append(_narrow_bf16(conv.reshape(conv_dim, taps)))
    sections.append(_raw_f32(decay))
    sections.append(_narrow_bf16(dt_bias.reshape(-1)))
    sections.append(_raw_f32(norm.reshape(-1)))
    sections += _q8o_segment(out_perm)
    return sections


def _qwen4_indexer_sections(checkpoint, prefix, layout):
    """A full-attention layer's QSA indexer: the fused query/key projection
    split into its two F32 segments and the per-head norms."""
    hidden = layout.hidden_size
    dim = layout.indexer_head_dimension
    query = layout.indexer_heads * dim
    fused = checkpoint.read(prefix + "self_attn.indexer.index_qk_proj.weight")
    if fused.shape != (query + dim, hidden):
        raise models.ModelError(f"{prefix}indexer index_qk_proj shape is off")
    sections = []
    sections += _f32_segment(fused[:query])
    sections += _f32_segment(fused[query:])
    sections.append(
        _raw_f32(
            checkpoint.read(prefix + "self_attn.indexer.q_layernorm.weight").reshape(-1)
        )
    )
    sections.append(
        _raw_f32(
            checkpoint.read(prefix + "self_attn.indexer.k_layernorm.weight").reshape(-1)
        )
    )
    return sections


def _qwen4_moe_sections(checkpoint, prefix, layout):
    """The MoE block's sections (readMoeBlock): the F32 router, the routed
    experts' gate/up/down slabs, the shared expert's and its scalar gate."""
    hidden = layout.hidden_size
    width = layout.expert_intermediate_size
    packed = layout.packed_expert_width
    experts = layout.experts
    gate_up = checkpoint.read(prefix + "mlp.experts.gate_up_proj")
    down = checkpoint.read(prefix + "mlp.experts.down_proj")
    if gate_up.shape != (experts, 2 * width, hidden):
        raise models.ModelError(f"{prefix}mlp.experts.gate_up_proj shape is off")
    if down.shape != (experts, hidden, width):
        raise models.ModelError(f"{prefix}mlp.experts.down_proj shape is off")
    router = checkpoint.read(prefix + "mlp.gate.weight")
    if router.shape != (experts, hidden):
        raise models.ModelError(f"{prefix}mlp.gate shape is off")
    # The packed expert slabs tile on 256-row/column boundaries the 640-wide
    # intermediate misses, so it pads to packed_expert_width with zeros —
    # dead gate/up rows emit zeros the down projection multiplies away
    # (TargetLayout.packedExpertWidth).
    pad = packed - width
    if pad < 0:
        raise models.ModelError(f"{prefix}packed expert width is under the logical one")
    sections = [*_f32_segment(router.astype(np.float32))]
    # Expert-major slabs: gate rows [0, width), up rows [width, 2w) of each
    # expert's fused gate_up, then the down projections.
    sections += _q8o_segment_experts(
        _pad_axis(gate_up[:, :width, :], 1, pad), packed, hidden
    )
    sections += _q8o_segment_experts(
        _pad_axis(gate_up[:, width:, :], 1, pad), packed, hidden
    )
    sections += _q8o_segment_experts(_pad_axis(down, 2, pad), hidden, packed)
    shared_gate = checkpoint.read(prefix + "mlp.shared_expert.gate_proj.weight")
    shared_up = checkpoint.read(prefix + "mlp.shared_expert.up_proj.weight")
    shared_down = checkpoint.read(prefix + "mlp.shared_expert.down_proj.weight")
    if shared_gate.shape != (width, hidden) or shared_up.shape != (width, hidden):
        raise models.ModelError(f"{prefix}shared expert gate/up shape is off")
    if shared_down.shape != (hidden, width):
        raise models.ModelError(f"{prefix}shared expert down shape is off")
    sections += _q8o_segment(_pad_axis(shared_gate, 0, pad))
    sections += _q8o_segment(_pad_axis(shared_up, 0, pad))
    sections += _q8o_segment(_pad_axis(shared_down, 1, pad))
    sections += _f32_segment(
        checkpoint.read(prefix + "mlp.shared_expert_gate.weight").astype(np.float32)
    )
    return sections


def _pad_axis(weights: np.ndarray, axis: int, pad: int) -> np.ndarray:
    """weights with `pad` zero rows appended along `axis`."""
    if not pad:
        return weights
    widths = [(0, 0)] * weights.ndim
    widths[axis] = (0, pad)
    return np.pad(weights, widths)


def _q8o_segment_experts(weights: np.ndarray, rows: int, columns: int) -> list:
    """One routed-expert slab as a single Q8_0 segment of experts * rows
    rows: every expert's [rows, columns] tile-major planes concatenated in
    expert order (moe_gguf_segment's e * output_size / 256 tiles)."""
    experts = weights.shape[0]
    flat = weights.reshape(experts * rows, columns)
    plane0, _plane1, meta, _padded = _q8o_planes(flat)
    return [
        _q8o_descriptor(experts * rows, columns, len(plane0), len(meta)),
        plane0,
        meta,
    ]


def _qwen4_ple_sections(checkpoint, prefix, layout):
    """The PLE layer's sections after its FFN (readPle): the gathered
    rows' key and value projections, the three stream-width norms and the
    dilated convolution as raw F32."""
    hidden = layout.hidden_size
    streams = layout.hyper_connections * hidden
    ple_width = layout.ple_heads * layout.ple_head_dimension
    key = checkpoint.read(prefix + "ple.key_proj.weight")
    value = checkpoint.read(prefix + "ple.value_proj.weight")
    if key.shape != (streams, ple_width) or value.shape != (hidden, ple_width):
        raise models.ModelError(f"{prefix}ple key/value shapes are off")
    conv = checkpoint.read(prefix + "ple.conv1d.weight")
    if conv.shape != (streams, 1, layout.ple_convolution_taps):
        raise models.ModelError(f"{prefix}ple conv1d shape is off")
    sections = []
    sections += _q8o_segment(key)
    sections += _q8o_segment(value)
    for name in ("norm_key", "norm_query", "norm_conv"):
        sections.append(
            _raw_f32(checkpoint.read(prefix + f"ple.{name}.weight").reshape(-1))
        )
    sections.append(_raw_f32(conv.reshape(streams, layout.ple_convolution_taps)))
    return sections


def _qwen4_head_sections(checkpoint, layout):
    """head.bin's sections: the final hyper-connection mix without
    injection, then the logits projection (the token embedding when the
    checkpoint ties them)."""
    sections = _qwen4_hyper_sections(
        checkpoint, "hyper_connection_mixer.", False, layout
    )
    lm_head = (
        checkpoint.read_sharded("lm_head.weight")
        if checkpoint.has("lm_head.weight")
        else checkpoint.read_sharded("embed_tokens.weight")
    )
    hidden = layout.hidden_size
    vocab = layout.vocabulary_size
    padded = (vocab + Q4_TILE_ROWS - 1) // Q4_TILE_ROWS * Q4_TILE_ROWS
    if lm_head.shape != (vocab, hidden):
        raise models.ModelError("lm_head shape does not match the layout")
    if padded != vocab:
        lm_head = np.concatenate(
            [lm_head, np.zeros((padded - vocab, hidden), np.float32)], axis=0
        )
    sections += _q8o_segment(lm_head.astype(np.float32))
    return sections


def _qwen4_embedding_sections(checkpoint, layout):
    """embedding.bin's sections: the native-descriptor token table."""
    embedding = checkpoint.read_sharded("embed_tokens.weight")
    hidden = layout.hidden_size
    vocab = layout.vocabulary_size
    if embedding.shape != (vocab, hidden):
        raise models.ModelError("embed_tokens shape does not match the layout")
    rows = _q8o_native_rows(embedding.astype(np.float32))
    return [
        _gguf_descriptor(GGML_Q8_0, vocab, hidden, (len(rows), 0, 0)),
        rows,
    ]


_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_MASK64 = (1 << 64) - 1
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, int(value**0.5) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def _qwen4_ple_hash(checkpoint, config, layout, table_rows: int):
    """The Qwen4PleHeader's 176 bytes and checks: the n-gram multipliers,
    each head's table rows and the eos/image tokens. The checkpoint's PLE
    buffers state them; a checkpoint without them recomputes them from the
    config (seed, ngram_vocab_size_base) the way
    Qwen4ExpTextNGramEmbedding does."""
    heads = layout.ple_heads
    eos = layout.ple_eos_token
    image = layout.ple_image_token
    ple_prefix = f"layers.{layout.ple_layer}.ple.ple_embedding."
    try:
        multipliers = [
            int(v)
            for v in checkpoint.read_int(ple_prefix + "layer_multipliers").reshape(-1)[
                : layout.ple_ngram
            ]
        ]
        vocab_sizes = [
            int(v)
            for v in checkpoint.read_int(
                ple_prefix + "ngram_heads_vocab_sizes"
            ).reshape(-1)
        ]
        head_offsets = [
            int(v)
            for v in checkpoint.read_int(ple_prefix + "ngram_heads_offsets").reshape(-1)
        ]
    except models.ModelError:
        text = (
            config.get("text_config")
            if isinstance(config.get("text_config"), dict)
            else config
        )
        seed = int(text.get("seed", 1234))
        base = int(text.get("ngram_vocab_size_base", 20_000_000))
        unigram = int(text.get("vocab_size", layout.vocabulary_size))
        # HF: ple_layer_ids.index(layer_idx + 1); a config without the list
        # has no PLE layer, so the first index is the only consistent one.
        ple_layer_ids = text.get("ple_layer_ids") or []
        ple_layer_index = (
            ple_layer_ids.index(layout.ple_layer + 1)
            if (layout.ple_layer + 1) in ple_layer_ids
            else 0
        )
        multiplier_max = ((1 << 63) - 1) // max(unigram, 1)
        half_bound = max(1, multiplier_max // 2)
        base_seed = seed + _PRIME_1 * ple_layer_index
        multipliers = [
            2
            * (
                _splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64)
                % half_bound
            )
            + 1
            for i in range(layout.ple_ngram)
        ]
        head_offsets = []
        vocab_sizes = []
        total = 0
        for head in range(heads):
            global_head = ple_layer_index * heads + head
            size = _find_nth_prime_after(base - 1, global_head + 1)
            vocab_sizes.append(size)
            head_offsets.append(total)
            total += size
    if (
        len(multipliers) < layout.ple_ngram
        or len(vocab_sizes) != heads
        or len(head_offsets) != heads
    ):
        raise models.ModelError("the PLE hash buffers do not match the layout")
    for offset, size in zip(head_offsets, vocab_sizes):
        if not size or offset + size > table_rows:
            raise models.ModelError("a PLE head's rows lie outside the table")
    header = struct.pack(
        "<4I Q 3Q",
        eos,
        image,
        heads,
        0,
        table_rows,
        *[_mask64(v) for v in (list(multipliers) + [0, 0, 0])[:3]],
    )
    if len(header) != 48:
        raise models.ModelError("the PLE hash header prefix is off")
    # The runtime's Qwen4PleHeader is a fixed 176 bytes: the heads array
    # holds kQwen4PleMaxNgram's worth of offset/vocabulary pairs (32 words
    # for ngram 3), zero-padded past this layout's head count.
    head_words = (list(head_offsets) + list(vocab_sizes) + [0] * 64)[:32]
    header += struct.pack("<32I", *head_words)
    if len(header) != 176:
        raise models.ModelError("the PLE hash header is not 176 bytes")
    return header


def _mask64(value: int) -> int:
    return value & _MASK64


def _qwen4_table_sections(checkpoint, config, layout):
    """ple-table.bin's sections: the PLE hash header, the native Q8_0
    descriptor of the n-gram table and its row plane."""
    table = checkpoint.read_sharded(
        f"layers.{layout.ple_layer}.ple.ple_embedding.ngram_embedding.weight"
    )
    head_dim = layout.ple_head_dimension
    if table.ndim != 2 or table.shape[1] != head_dim:
        raise models.ModelError("the PLE n-gram table's row width is off")
    rows = table.shape[0]
    header = _qwen4_ple_hash(checkpoint, config, layout, rows)
    plane = _q8o_native_rows(table.astype(np.float32))
    return [
        header,
        _gguf_descriptor(GGML_Q8_0, rows, head_dim, (len(plane), 0, 0)),
        plane,
    ]


def _qwen4_manifest(
    model,
    family_name,
    layout,
    sources,
    artifacts,
    geometry,
    serving,
    signature,
    draft=None,
):
    """manifest.json of a qwen4exp package: the family's geometry under
    target/, the packed format's magics and — when the package ships one —
    the DFlash2 draft's declaration (inspectModelPackage's
    richengine-packed-gguf-qwen4exp branch binds it)."""
    return {
        "model": model,
        "schema_version": 1,
        "family": family_name,
        **({"serving": dict(serving)} if serving else {}),
        **(
            {"signature": {k: _jsonable(v) for k, v in signature.items()}}
            if signature
            else {}
        ),
        "built_by": "install/pack.py",
        "sources": sources,
        "format": {
            "name": QWEN4_FORMAT_NAME,
            "q4_bits": 4,
            "q4_group_size": Q4_GROUP,
            "q4_storage_n": Q4_TILE_ROWS,
            "section_alignment_bytes": ALIGNMENT,
            "target_layer_magic": QWEN4_LAYER_MAGIC.decode(),
            # MDFD0004 is both the DFlash2 draft's magic and the Null
            # draft's; the draft/ payload and the manifest's draft object
            # say which a package carries.
            "draft_layer_magic": DRAFT_MAGIC,
            "vision_magic": VISION_MAGIC,
        },
        **(
            {"draft": _dflash2_manifest_draft(draft)}
            if isinstance(draft, DFlash2Layout)
            else {}
        ),
        "execution_geometry": geometry,
        TARGET: {
            "architecture": "qwen4exp",
            "layers": layout.layers,
            "hidden_size": layout.hidden_size,
            "vocabulary_size": layout.vocabulary_size,
            "packed_gdn_width": layout.packed_gdn_width,
            "packed_full_width": layout.packed_full_width,
            "convolution_dimension": layout.convolution_dimension,
            "convolution_taps": layout.convolution_taps,
            "gdn_key_heads": layout.gdn_key_heads,
            "gdn_value_heads": layout.gdn_value_heads,
            "gdn_head_dimension": layout.gdn_head_dimension,
            "attention_width": layout.attention_width,
            "attention_query_heads": layout.attention_query_heads,
            "attention_kv_heads": layout.attention_kv_heads,
            "attention_head_dimension": layout.attention_head_dimension,
            "attention_query_stride": layout.attention_query_stride,
            "full_attention_period": layout.full_attention_period,
            "experts": layout.experts,
            "experts_per_token": layout.experts_per_token,
            "expert_intermediate_size": layout.expert_intermediate_size,
            "hyper_connections": layout.hyper_connections,
            "hyper_rank": layout.hyper_rank,
            "indexer_heads": layout.indexer_heads,
            "indexer_head_dimension": layout.indexer_head_dimension,
            "indexer_top_blocks": layout.indexer_top_blocks,
            "indexer_block_tokens": layout.indexer_block_tokens,
            "ple_layer": layout.ple_layer,
            "ple_ngram": layout.ple_ngram,
            "ple_heads_per_ngram": layout.ple_heads_per_ngram,
            "ple_head_dimension": layout.ple_head_dimension,
            "ple_convolution_taps": layout.ple_convolution_taps,
            "layer_types": layout.layer_types(),
        },
        "artifacts": artifacts,
    }


def _build_qwen4exp_package(
    stage: Path,
    model: str,
    family_name: str,
    config,
    weights,
    tokenizer_files,
    draft_files,
    sources,
    *,
    layout,
    draft_layout=None,
    serving=None,
    signature=None,
    geometry=None,
):
    """Write the qwen4exp packed package into stage."""
    from .pack_layouts import Qwen4ExpLayout

    if not isinstance(layout, Qwen4ExpLayout):
        raise models.ModelError("qwen4exp packing needs a Qwen4ExpLayout")
    stage.mkdir(parents=True, exist_ok=True)
    target_dir = stage / TARGET
    target_dir.mkdir()
    checkpoint = Checkpoint(
        [weights[name] for name in sorted(weights) if name.endswith(".safetensors")]
    )
    for layer in range(layout.layers):
        write_packed_file(
            target_dir / f"layer-{layer}.bin",
            QWEN4_LAYER_MAGIC,
            layer,
            1 if layout.is_full_attention(layer) else 0,
            pack_qwen4exp_layer_sections(checkpoint, layout, layer),
        )
    write_packed_file(
        target_dir / "head.bin",
        QWEN4_HEAD_MAGIC,
        layout.layers,
        2,
        _qwen4_head_sections(checkpoint, layout),
    )
    write_packed_file(
        target_dir / "embedding.bin",
        EMBEDDING_MAGIC,
        layout.vocabulary_size,
        layout.hidden_size,
        _qwen4_embedding_sections(checkpoint, layout),
    )
    write_packed_file(
        target_dir / "ple-table.bin",
        QWEN4_TABLE_MAGIC,
        0,
        0,
        _qwen4_table_sections(checkpoint, config, layout),
    )
    for name, source in tokenizer_files.items():
        _copy(source, stage / TOKENIZER, name)
    _write_tokenizer_config(stage, config)
    # The DFlash2 draft: draft/layer-N.bin and draft/model.bin under
    # MDFD0004, packed the way DFlashTwoDraft.cpp's readDraft reads them.
    # Without the safetensors the package carries no draft payload — its
    # manifest's missing draft object binds the Null draft.
    draft = None
    draft_weights = [
        path for name, path in draft_files.items() if name.endswith(".safetensors")
    ]
    if draft_weights:
        if not isinstance(draft_layout, DFlash2Layout):
            raise models.ModelError(
                "packing a qwen4exp draft needs a DFlash2 draft layout"
            )
        draft = draft_layout
        draft_dir = stage / DRAFT
        draft_dir.mkdir()
        pack_draft2(Checkpoint(draft_weights), draft_dir, draft)
        for name, source in draft_files.items():
            if name.endswith("config.json"):
                _copy(source, draft_dir, name.removeprefix(DRAFT + "/"))
    written = sorted(p for p in stage.rglob("*") if p.is_file())
    manifest = _qwen4_manifest(
        model,
        family_name,
        layout,
        sources,
        [_artifact(path, stage) for path in written],
        geometry or {},
        serving,
        signature,
        draft,
    )
    (stage / PACKAGE_MANIFEST).write_bytes(models.json_bytes(manifest))
    return manifest


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
    geometry: dict,
    diffusion=None,
    serving=None,
    signature=None,
):
    """manifest.json of a built package, as inspectModelPackage validates it
    (validateGemma4 / validateNewPackedFormat)."""
    return {
        "model": model,
        "schema_version": 1,
        "family": family_name,
        # The family data the server reads at load, recorded so the package
        # serves the same spellings even if the family file changes later.
        **({"serving": dict(serving)} if serving else {}),
        **(
            {"signature": {k: _jsonable(v) for k, v in signature.items()}}
            if signature
            else {}
        ),
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
        # The family's declared geometry: the draft sliding window is the
        # draft signature's trained window (execution_geometry()).
        "execution_geometry": geometry,
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


def _jsonable(value):
    """A signature value as manifest JSON states it: tuples as lists."""
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


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
        and format_.get("name")
        in {FORMAT_NAME, DIFFUSION_FORMAT_NAME, QWEN4_FORMAT_NAME}
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
    layout: Layout | Qwen4ExpLayout | None = None,
    draft_layout: DraftLayout | DFlash2Layout | None = None,
    pack: dict | None = None,
    serving: dict | None = None,
    signature: dict | None = None,
):
    """Write the packed package into stage: manifest.json, target/*.bin from
    the checkpoint shards in weights (path per repository name), the
    tokenizer files and the draft's packed layer files.

    The geometry is the family's: pass its pack section (pack=family.pack)
    or the layouts directly (tests). serving and signature are the family's
    serving contract and identity signature, recorded into the manifest so
    the package self-describes (server/serving.py reads it at load)."""
    if _text_type(config) == "qwen4_exp_text":
        # qwen4exp is packed-only: its own writer, layout and manifest.
        if pack is not None and "layout" in pack:
            qwen4_layout = Qwen4ExpLayout(**pack["layout"])
        elif isinstance(layout, Qwen4ExpLayout):
            qwen4_layout = layout
        else:
            raise models.ModelError(
                "qwen4exp packing needs the family's Qwen4ExpLayout"
            )
        spec_draft = (pack or {}).get("draft_layout")
        qwen4_draft = (
            draft_layout_from_spec(spec_draft)
            if spec_draft
            else draft_layout
            if isinstance(draft_layout, DFlash2Layout)
            else None
        )
        return _build_qwen4exp_package(
            stage,
            model,
            family_name,
            config,
            weights,
            tokenizer_files,
            draft_files,
            sources,
            layout=qwen4_layout,
            draft_layout=qwen4_draft,
            serving=serving,
            signature=signature,
            geometry=(pack or {}).get("execution_geometry"),
        )
    if pack is not None:
        layout = Layout(**pack["layout"])
        draft_layout = draft_layout_from_spec(pack.get("draft_layout"))
        if draft_layout is not None and not isinstance(draft_layout, DraftLayout):
            raise models.ModelError(
                "this family's packed draft is not a plain DFlash draft"
            )
    if layout is None:
        raise models.ModelError("build_package needs the family's pack geometry")
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
        if draft_layout is None:
            raise models.ModelError(
                "the family's pack declares no draft layout for its draft files"
            )
        draft = draft_layout
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
        geometry=(
            (pack or {}).get("execution_geometry")
            or {
                "draft_proposal_tokens": 15,
                "draft_query_rows": 16,
                "draft_sliding_window": draft.sliding_window if draft else 2048,
            }
        ),
        diffusion=(pack or {}).get("diffusion") if diffusion else None,
        serving=serving,
        signature=signature,
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
    # The family file declares the pack: its format must be one this packer
    # writes, and the model_type's PACKABLE entry must agree.
    model_type = _text_type(config)
    if (
        family.defaults.packed_format is None
        or family.defaults.packed_format not in TARGET_FORMATS
        or PACKABLE.get(model_type, (None, None))[1] != family.defaults.packed_format
    ):
        raise models.ModelError(
            f"{family.name} declares no packed format for {model_type}"
        )
    spec = dict(family.pack)
    spec["execution_geometry"] = execution_geometry(family, family_draft_layout(family))
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
            pack=spec,
            serving=family.serving,
            signature=dict(family.signature),
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
