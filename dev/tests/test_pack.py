"""The Gemma 4 install-side packer (install/pack.py): the packed file layout,
the affine quantization and the section order readLayer reads, exercised on
a synthetic safetensors checkpoint at a shrunken layout."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from install import pack, pack_layouts


def bf16(values):
    """float32 values as BF16 payload bytes."""
    array = np.asarray(values, np.float32)
    bits = array.view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def write_safetensors(path, tensors):
    """A safetensors file of the (dtype, shape, payload) tensors."""
    header, offset, payload = {}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.append(data)
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(payload))
    return path


def f32_bytes(values):
    return np.asarray(values, np.float32).tobytes()


LAYOUT = pack_layouts.Layout(
    layers=2,
    hidden=256,
    vocabulary=512,
    query_heads=2,
    kv_heads=1,
    head_dim=128,
    global_kv_heads=1,
    global_head_dim=256,
    global_period=2,  # layer 1 is global
    experts=4,
    experts_per_token=2,
    expert_intermediate=64,
    packed_expert_width=256,
    shared_intermediate=128,
    packed_shared_width=256,
)


DRAFT_LAYOUT = pack_layouts.DraftLayout(
    layers=1,
    hidden=256,
    kv_heads=2,
    head_dim=128,
    query_heads=4,
    intermediate=512,
    target_hidden=256,
    causal_layers=0,
)


def _fields(obj):
    import dataclasses

    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


PACK_SPEC = {
    "layout": _fields(LAYOUT),
    "draft_layout": _fields(DRAFT_LAYOUT),
    "execution_geometry": {
        "draft_proposal_tokens": 15,
        "draft_query_rows": 16,
        "draft_sliding_window": DRAFT_LAYOUT.sliding_window,
    },
}

DIFFUSION_SPEC = {
    **PACK_SPEC,
    "draft_layout": None,
    # DiffusionGemma's denoising schedule, as its generation_config.json
    # and the block-diffusion scheduler's defaults state it.
    "diffusion": {
        "canvas_length": 256,
        "max_denoising_steps": 48,
        "t_min": 0.4,
        "t_max": 0.8,
        "entropy_bound": 0.1,
        "confidence_threshold": 0.005,
        "stability_threshold": 1,
    },
}


def synthetic_draft_checkpoint(root):
    """A draft safetensors shard at DRAFT_LAYOUT's shapes, the tensor names
    DraftCheckpoint.cpp's plain images read."""
    rng = np.random.default_rng(11)
    hidden = DRAFT_LAYOUT.hidden
    kv = DRAFT_LAYOUT.kv_heads * DRAFT_LAYOUT.head_dim
    tensors = {
        "fc.weight": (
            "BF16",
            (hidden, DRAFT_LAYOUT.target_hidden),
            bf16(rng.normal(0, 0.05, (hidden, DRAFT_LAYOUT.target_hidden))),
        ),
        "hidden_norm.weight": ("BF16", (hidden,), bf16(rng.normal(1, 0.1, hidden))),
        "norm.weight": ("BF16", (hidden,), bf16(rng.normal(1, 0.1, hidden))),
    }
    for layer in range(DRAFT_LAYOUT.layers):
        prefix = f"layers.{layer}."
        tensors.update(
            {
                f"{prefix}input_layernorm.weight": (
                    "BF16",
                    (hidden,),
                    bf16(rng.normal(1, 0.1, hidden)),
                ),
                f"{prefix}self_attn.q_proj.weight": (
                    "BF16",
                    (DRAFT_LAYOUT.attention, hidden),
                    bf16(rng.normal(0, 0.05, (DRAFT_LAYOUT.attention, hidden))),
                ),
                f"{prefix}self_attn.k_proj.weight": (
                    "BF16",
                    (kv, hidden),
                    bf16(rng.normal(0, 0.05, (kv, hidden))),
                ),
                f"{prefix}self_attn.v_proj.weight": (
                    "BF16",
                    (kv, hidden),
                    bf16(rng.normal(0, 0.05, (kv, hidden))),
                ),
                f"{prefix}self_attn.q_norm.weight": (
                    "BF16",
                    (DRAFT_LAYOUT.head_dim,),
                    bf16(rng.normal(1, 0.1, DRAFT_LAYOUT.head_dim)),
                ),
                f"{prefix}self_attn.k_norm.weight": (
                    "BF16",
                    (DRAFT_LAYOUT.head_dim,),
                    bf16(rng.normal(1, 0.1, DRAFT_LAYOUT.head_dim)),
                ),
                f"{prefix}self_attn.o_proj.weight": (
                    "BF16",
                    (hidden, DRAFT_LAYOUT.attention),
                    bf16(rng.normal(0, 0.05, (hidden, DRAFT_LAYOUT.attention))),
                ),
                f"{prefix}post_attention_layernorm.weight": (
                    "BF16",
                    (hidden,),
                    bf16(rng.normal(1, 0.1, hidden)),
                ),
                f"{prefix}mlp.gate_proj.weight": (
                    "BF16",
                    (DRAFT_LAYOUT.intermediate, hidden),
                    bf16(rng.normal(0, 0.05, (DRAFT_LAYOUT.intermediate, hidden))),
                ),
                f"{prefix}mlp.up_proj.weight": (
                    "BF16",
                    (DRAFT_LAYOUT.intermediate, hidden),
                    bf16(rng.normal(0, 0.05, (DRAFT_LAYOUT.intermediate, hidden))),
                ),
                f"{prefix}mlp.down_proj.weight": (
                    "BF16",
                    (hidden, DRAFT_LAYOUT.intermediate),
                    bf16(rng.normal(0, 0.05, (hidden, DRAFT_LAYOUT.intermediate))),
                ),
            }
        )
    return write_safetensors(root / "draft.safetensors", tensors)


def synthetic_checkpoint(root, base="model."):
    """A safetensors shard holding every tensor the packer reads, random but
    deterministic, at LAYOUT's shapes. base is the tensor-name prefix the
    checkpoint publishes (model. for Gemma 4, model.decoder. for
    DiffusionGemma)."""
    rng = np.random.default_rng(7)
    hidden, vocab = LAYOUT.hidden, LAYOUT.vocabulary
    tensors = {
        f"{base}embed_tokens.weight": (
            "BF16",
            (vocab, hidden),
            bf16(rng.normal(0, 0.02, (vocab, hidden))),
        ),
        f"{base}norm.weight": ("BF16", (hidden,), bf16(rng.normal(1, 0.1, hidden))),
    }
    for layer in range(LAYOUT.layers):
        prefix = f"{base}layers.{layer}."
        head_dim = LAYOUT.head_dim_at(layer)
        kv_heads = LAYOUT.kv_heads_at(layer)
        global_ = LAYOUT.is_global(layer)
        attention = {
            f"{prefix}self_attn.q_proj.weight": (2 * head_dim, hidden),
            f"{prefix}self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
            f"{prefix}self_attn.o_proj.weight": (hidden, 2 * head_dim),
            f"{prefix}self_attn.q_norm.weight": (head_dim,),
            f"{prefix}self_attn.k_norm.weight": (head_dim,),
        }
        if not global_:
            attention[f"{prefix}self_attn.v_proj.weight"] = (
                kv_heads * head_dim,
                hidden,
            )
        for name, shape in attention.items():
            tensors[name] = ("BF16", shape, bf16(rng.normal(0, 0.05, shape)))
        for name in (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "pre_feedforward_layernorm_2.weight",
            "post_feedforward_layernorm.weight",
            "post_feedforward_layernorm_1.weight",
            "post_feedforward_layernorm_2.weight",
        ):
            tensors[prefix + name] = (
                "BF16",
                (hidden,),
                bf16(rng.normal(1, 0.1, hidden)),
            )
        e, inter = LAYOUT.experts, LAYOUT.expert_intermediate
        tensors.update(
            {
                f"{prefix}router.scale": (
                    "BF16",
                    (hidden,),
                    bf16(rng.normal(1, 0.1, hidden)),
                ),
                f"{prefix}router.proj.weight": (
                    "BF16",
                    (e, hidden),
                    bf16(rng.normal(0, 0.05, (e, hidden))),
                ),
                f"{prefix}router.per_expert_scale": (
                    "F32",
                    (e,),
                    f32_bytes(rng.normal(1, 0.1, e)),
                ),
                f"{prefix}experts.gate_up_proj": (
                    "BF16",
                    (e, 2 * inter, hidden),
                    bf16(rng.normal(0, 0.05, (e, 2 * inter, hidden))),
                ),
                f"{prefix}experts.down_proj": (
                    "BF16",
                    (e, hidden, inter),
                    bf16(rng.normal(0, 0.05, (e, hidden, inter))),
                ),
                f"{prefix}mlp.gate_proj.weight": (
                    "BF16",
                    (LAYOUT.shared_intermediate, hidden),
                    bf16(rng.normal(0, 0.05, (LAYOUT.shared_intermediate, hidden))),
                ),
                f"{prefix}mlp.up_proj.weight": (
                    "BF16",
                    (LAYOUT.shared_intermediate, hidden),
                    bf16(rng.normal(0, 0.05, (LAYOUT.shared_intermediate, hidden))),
                ),
                f"{prefix}mlp.down_proj.weight": (
                    "BF16",
                    (hidden, LAYOUT.shared_intermediate),
                    bf16(rng.normal(0, 0.05, (hidden, LAYOUT.shared_intermediate))),
                ),
                f"{prefix}layer_scalar": ("F32", (1,), f32_bytes([0.5])),
            }
        )
    return write_safetensors(root / "model.safetensors", tensors)


def synthetic_diffusion_checkpoint(root):
    """DiffusionGemma's checkpoint at LAYOUT's shapes: the decoder's tensors
    prefixed model.decoder., its self-conditioning stack, and the encoder's
    per-layer scalars."""
    shard = synthetic_checkpoint(root, base="model.decoder.")
    rng = np.random.default_rng(13)
    hidden = LAYOUT.hidden
    tensors = {
        f"model.decoder.self_conditioning.{name}": (
            "BF16",
            shape,
            bf16(rng.normal(0, 0.05, shape)),
        )
        for name, shape in (
            ("pre_norm.weight", (hidden,)),
            ("gate_proj.weight", (LAYOUT.shared_intermediate, hidden)),
            ("up_proj.weight", (LAYOUT.shared_intermediate, hidden)),
            ("down_proj.weight", (hidden, LAYOUT.shared_intermediate)),
        )
    }
    for layer in range(LAYOUT.layers):
        tensors[f"model.encoder.language_model.layers.{layer}.layer_scalar"] = (
            "F32",
            (1,),
            f32_bytes([layer + 0.5]),
        )
    return write_safetensors(root / "diffusion.safetensors", tensors), shard


DIFFUSION_CONFIG = {
    "model_type": "diffusion_gemma",
    "canvas_length": 256,
    "text_config": {"model_type": "diffusion_gemma_text"},
}


def quantized_bytes(n, k):
    return n * k // 2 + 2 * (n * k // 32)


class PackTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_packed_file_header_and_section_alignment(self):
        path = self.root / "layer-0.bin"
        pack.write_packed_file(path, b"GEMM0001", 3, 1, [b"xy", b"zzz"])
        data = path.read_bytes()
        self.assertEqual(data[:16], struct.pack("<8sII", b"GEMM0001", 3, 1))
        self.assertFalse(len(data) % pack.ALIGNMENT)
        # First section at the first boundary after the 16-byte header.
        self.assertEqual(data[pack.ALIGNMENT : pack.ALIGNMENT + 2], b"xy")
        self.assertEqual(data[2 * pack.ALIGNMENT : 2 * pack.ALIGNMENT + 3], b"zzz")

    def test_quantization_matches_the_affine_writer(self):
        # A 256x64 tile of weights, then a constant row (degenerate group) and
        # a negative-only row exercise minimumEdge, q0 and the pad-zero rule.
        rng = np.random.default_rng(3)
        weights = rng.normal(0, 0.3, (256, 64)).astype(np.float32)
        weights[4] = 0.5
        weights[9] = -rng.uniform(0.1, 0.2, 64).astype(np.float32)
        section = pack.pack_projection(weights, 256, 64)
        self.assertEqual(len(section), quantized_bytes(256, 64))
        codes, scales, biases = (
            section[: 256 * 32],
            section[256 * 32 : 256 * 32 + 256 * 2],
            section[256 * 32 + 256 * 2 :],
        )

        # Decode row r group 0: codes[r*32], scales[r*2:r*2+2] (one group).
        def dequant(row):
            c = np.frombuffer(codes, np.uint8)[row * 32 : row * 32 + 32]
            codes64 = np.empty(64, np.uint8)
            codes64[0::2] = c & 15
            codes64[1::2] = c >> 4
            s = np.frombuffer(scales, np.uint16)[row]
            b = np.frombuffer(biases, np.uint16)[row]
            scale = (np.array([s], np.uint32) << 16).view(np.float32)[0]
            bias = (np.array([b], np.uint32) << 16).view(np.float32)[0]
            return codes64, scale, bias

        n, s, b = dequant(4)
        # Constant row: the codes share one value and dequant back to ~0.5.
        self.assertTrue((n == n[0]).all())
        self.assertAlmostEqual(s * n[0] + b, 0.5, places=2)

    def test_layer_sections_follow_read_layer_order(self):
        shard = synthetic_checkpoint(self.root)
        checkpoint = pack.Checkpoint([shard])
        sections = pack._layer_sections(checkpoint, LAYOUT, 0)
        self.assertEqual(len(sections), len(pack_layouts.LAYER_SECTIONS))
        hidden, width, e, packed = 256, LAYOUT.packed_width_at(0), 4, 256
        sizes = [
            hidden * 2,  # input-norm
            quantized_bytes(width, hidden),  # attention-input (q+k+v)
            128 * 2,  # query-norm
            128 * 2,  # key-norm
            quantized_bytes(hidden, 2 * 128),  # attention-output
            hidden * 2,  # post-attention-norm
            hidden * 2,  # pre-ffn-norm
            hidden * 2,  # pre-ffn-norm-routed
            hidden * 2,  # router-scale
            e * hidden * 2,  # router-weights
            e * 4,  # per-expert-scale
            e * quantized_bytes(packed, hidden),  # experts-gate (704->768 pad)
            e * quantized_bytes(packed, hidden),  # experts-up
            e * quantized_bytes(hidden, packed),  # experts-down
            quantized_bytes(256, hidden),  # shared-expert-gate (padded 256)
            quantized_bytes(256, hidden),  # shared-expert-up
            quantized_bytes(hidden, 256),  # shared-expert-down
            hidden * 2,  # post-ffn-norm-shared
            hidden * 2,  # post-ffn-norm-routed
            hidden * 2,  # post-ffn-norm
            4,  # layer-scalar
        ]
        self.assertEqual([len(section) for section in sections], sizes)

    def test_global_layer_packs_no_v_projection(self):
        shard = synthetic_checkpoint(self.root)
        checkpoint = pack.Checkpoint([shard])
        global_ = pack._layer_sections(checkpoint, LAYOUT, 1)
        # A global layer's fused QKV is q+k only at the wider head dim.
        self.assertEqual(
            len(global_[1]), quantized_bytes(LAYOUT.packed_width_at(1), 256)
        )
        self.assertEqual(
            len(global_[2]), 256 * 2
        )  # query-norm at head_dim 256? no: global head_dim
        self.assertEqual(
            len(global_[4]), quantized_bytes(256, LAYOUT.attention_width_at(1))
        )  # o_proj [hidden, query heads * global head dim]
        self.assertEqual(struct.unpack("<f", global_[-1])[0], 0.5)

    def test_build_package_writes_a_validatable_manifest(self):
        shard = synthetic_checkpoint(self.root)
        draft = synthetic_draft_checkpoint(self.root)
        stage = self.root / "pkg"
        manifest = pack.build_package(
            stage,
            "owner/gemma4-test",
            "Gemma4-26B-A4B",
            {"text_config": {"model_type": "gemma4_text"}},
            {"model.safetensors": shard},
            {"tokenizer.json": shard, "tokenizer_config.json": shard},
            {"draft/config.json": shard, "draft/model.safetensors": draft},
            {"target": {"repo": "owner/gemma4-test", "revision": "a" * 40}},
            pack=PACK_SPEC,
        )
        self.assertEqual(manifest["format"]["name"], pack.FORMAT_NAME)
        self.assertEqual(manifest["format"]["target_layer_magic"], "GEMM0001")
        self.assertEqual(manifest["format"]["draft_layer_magic"], "MDFP0005")
        self.assertEqual(manifest["target"]["architecture"], "gemma4_text")
        declared = manifest["draft"]
        self.assertEqual(declared["architecture"], "DFlashDraftModel")
        self.assertEqual(declared["layers"], DRAFT_LAYOUT.layers)
        self.assertEqual(declared["hidden_size"], DRAFT_LAYOUT.hidden)
        self.assertEqual(declared["block_size"], DRAFT_LAYOUT.block_size)
        self.assertTrue((stage / "draft" / "model.bin").is_file())
        self.assertTrue((stage / "draft" / "layer-0.bin").is_file())
        # No declared draft: the Null magic and no draft object.
        bare = pack.build_package(
            self.root / "pkg2",
            "owner/gemma4-test",
            "Gemma4-26B-A4B",
            {"text_config": {"model_type": "gemma4_text"}},
            {"model.safetensors": shard},
            {"tokenizer.json": shard},
            {},
            {"target": {"repo": "owner/gemma4-test", "revision": "a" * 40}},
            pack={**PACK_SPEC, "draft_layout": None},
        )
        self.assertNotIn("draft", bare)
        self.assertEqual(bare["format"]["draft_layer_magic"], "MDFD0004")
        self.assertEqual(
            manifest["target"]["layer_types"],
            ["sliding_attention", "full_attention"],
        )
        for name in (
            "target/embedding.bin",
            "target/head.bin",
            "target/layer-0.bin",
            "target/layer-1.bin",
            "draft/config.json",
            "draft/model.bin",
            "draft/layer-0.bin",
            "tokenizer/config.json",
        ):
            self.assertTrue((stage / name).exists(), name)
        # Headers: embedding carries vocab/hidden, the head the layer count.
        embedding = stage / "target/embedding.bin"
        self.assertEqual(
            embedding.read_bytes()[:16],
            struct.pack("<8sII", b"MDFE0001", 512, 256),
        )
        # readAffineEmbedding reads three aligned sections: 512x256 codes,
        # then scales and biases, each at the next 16 KiB boundary.
        self.assertEqual(embedding.stat().st_size, 114688)
        self.assertEqual(
            (stage / "target/head.bin").read_bytes()[:16],
            struct.pack("<8sII", b"GEMM0002", 2, 2),
        )
        self.assertEqual(
            (stage / "target/layer-1.bin").read_bytes()[:16],
            struct.pack("<8sII", b"GEMM0001", 1, 1),
        )
        pack.verify(stage)

    def test_diffusion_checkpoint_is_packable(self):
        self.assertTrue(pack.packable(DIFFUSION_CONFIG))
        self.assertEqual(
            pack.target_format(DIFFUSION_CONFIG),
            (pack.DIFFUSION_FORMAT_NAME, pack.DIFFUSION_TARGET_FORMAT),
        )
        self.assertTrue(pack.packable({"text_config": {"model_type": "gemma4_text"}}))
        self.assertFalse(pack.packable({"text_config": {"model_type": "qwen3"}}))

    def test_diffusion_package_writes_the_extra_files(self):
        extras, shard = synthetic_diffusion_checkpoint(self.root)
        stage = self.root / "pkg"
        manifest = pack.build_package(
            stage,
            "owner/diffusiongemma-test",
            "DiffusionGemma-26B-A4B",
            DIFFUSION_CONFIG,
            {"model.safetensors": extras, "other.safetensors": shard},
            {"tokenizer.json": shard},
            {},
            {"target": {"repo": "owner/diffusiongemma-test", "revision": "a" * 40}},
            pack=DIFFUSION_SPEC,
        )
        self.assertEqual(manifest["format"]["name"], pack.DIFFUSION_FORMAT_NAME)
        self.assertEqual(manifest["format"]["draft_layer_magic"], "MDFD0004")
        self.assertNotIn("draft", manifest)
        self.assertEqual(manifest["target"]["architecture"], "diffusion_gemma_text")
        self.assertEqual(
            manifest["diffusion"],
            {
                "canvas_length": 256,
                "max_denoising_steps": 48,
                "t_min": 0.4,
                "t_max": 0.8,
                "entropy_bound": 0.1,
                "confidence_threshold": 0.005,
                "stability_threshold": 1,
            },
        )
        # self_conditioning.bin: pre-norm bf16, then gate|up|down q4 at the
        # shared expert's packed width.
        sc = stage / "target/self_conditioning.bin"
        self.assertEqual(
            sc.read_bytes()[:16],
            struct.pack("<8sII", b"GEMM0002", 0, 3),
        )
        data = sc.read_bytes()
        offset = pack.ALIGNMENT
        hidden = LAYOUT.hidden
        self.assertEqual(len(data[offset : offset + hidden * 2]), hidden * 2)
        offset += pack.ALIGNMENT
        gate = quantized_bytes(LAYOUT.packed_shared_width, hidden)
        self.assertEqual(len(data[offset : offset + gate]), gate)
        offset += pack.ALIGNMENT
        self.assertEqual(len(data[offset : offset + gate]), gate)
        offset += pack.ALIGNMENT
        down = quantized_bytes(hidden, LAYOUT.packed_shared_width)
        self.assertEqual(len(data[offset : offset + down]), down)
        # encoder_scalars.bin: one fp32 per layer, in layer order.
        scalars = stage / "target/encoder_scalars.bin"
        self.assertEqual(
            scalars.read_bytes()[:16],
            struct.pack("<8sII", b"GEMM0002", 0, 4),
        )
        values = np.frombuffer(
            scalars.read_bytes()[pack.ALIGNMENT :], np.float32, LAYOUT.layers
        )
        self.assertEqual(list(values), [layer + 0.5 for layer in range(LAYOUT.layers)])
        pack.verify(stage)


# A shrunken qwen4exp layout: every packed width stays a whole number of
# 32-column Q8_0 groups and every row count a whole 256-row tile where the
# runtime's tiled descriptor check applies. The PLE table's row width is a
# multiple of 32 for its native rows.
QWEN4_LAYOUT = pack_layouts.Qwen4ExpLayout(
    maximum_context_tokens=1024,
    layers=8,
    hidden_size=64,
    vocabulary_size=512,
    packed_gdn_width=112,  # conv 64 + z 32 + ab 16
    packed_full_width=80,  # stride 2 * attn 32 + 2 * kv 1 * 8
    convolution_dimension=64,
    convolution_taps=4,
    gdn_key_heads=2,
    gdn_value_heads=4,
    gdn_head_dimension=8,
    attention_width=32,
    attention_query_heads=4,
    attention_kv_heads=1,
    attention_head_dimension=8,
    attention_query_stride=2,
    full_attention_period=4,
    experts=4,
    experts_per_token=2,
    expert_intermediate_size=32,
    hyper_connections=2,
    hyper_rank=8,
    indexer_heads=2,
    indexer_head_dimension=8,
    indexer_top_blocks=8,
    indexer_block_tokens=4,
    ple_layer=1,
    ple_ngram=3,
    ple_heads_per_ngram=1,
    ple_head_dimension=32,
    ple_convolution_taps=4,
    ple_eos_token=100,
    ple_image_token=200,
)

QWEN4_CONFIG = {
    "text_config": {
        "model_type": "qwen4_exp_text",
        "vocab_size": QWEN4_LAYOUT.vocabulary_size,
        "hidden_size": QWEN4_LAYOUT.hidden_size,
    }
}


def synthetic_qwen4_checkpoint(root):
    """A safetensors shard at QWEN4_LAYOUT's shapes, the tensor names
    modeling_qwen4_exp.py's state dict publishes (the PLE n-gram table as
    its .shard_N.weight pieces and the hash buffers as I64)."""
    rng = np.random.default_rng(17)
    hidden = QWEN4_LAYOUT.hidden_size
    streams = QWEN4_LAYOUT.hyper_connections * hidden
    width = QWEN4_LAYOUT.expert_intermediate_size
    experts = QWEN4_LAYOUT.experts
    tensors = {
        "embed_tokens.weight": (
            "F32",
            (QWEN4_LAYOUT.vocabulary_size, hidden),
            f32_bytes(rng.normal(0, 0.05, (QWEN4_LAYOUT.vocabulary_size, hidden))),
        ),
        "lm_head.weight": (
            "F32",
            (QWEN4_LAYOUT.vocabulary_size, hidden),
            f32_bytes(rng.normal(0, 0.05, (QWEN4_LAYOUT.vocabulary_size, hidden))),
        ),
        "hyper_connection_mixer.hc_norm.weight": (
            "F32",
            (streams,),
            f32_bytes(rng.normal(1, 0.1, streams)),
        ),
        "hyper_connection_mixer.input_mix_weight_down.weight": (
            "F32",
            (QWEN4_LAYOUT.hyper_rank, streams),
            f32_bytes(rng.normal(0, 0.05, (QWEN4_LAYOUT.hyper_rank, streams))),
        ),
        "hyper_connection_mixer.input_mix_weight_up.weight": (
            "F32",
            (streams, QWEN4_LAYOUT.hyper_rank),
            f32_bytes(rng.normal(0, 0.05, (streams, QWEN4_LAYOUT.hyper_rank))),
        ),
    }
    for layer in range(QWEN4_LAYOUT.layers):
        prefix = f"layers.{layer}."
        for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
            for name, shape in (
                ("hc_norm.weight", (streams,)),
                ("input_mix_weight_down.weight", (QWEN4_LAYOUT.hyper_rank, streams)),
                ("input_mix_weight_up.weight", (streams, QWEN4_LAYOUT.hyper_rank)),
                (
                    "block_inject_weight.weight",
                    (QWEN4_LAYOUT.hyper_connections, streams),
                ),
            ):
                tensors[prefix + hc + "." + name] = (
                    "F32",
                    shape,
                    f32_bytes(rng.normal(0, 0.05, shape)),
                )
        if QWEN4_LAYOUT.is_full_attention(layer):
            for name, shape in (
                ("self_attn.q_proj.weight", (64, hidden)),
                ("self_attn.k_proj.weight", (8, hidden)),
                ("self_attn.v_proj.weight", (8, hidden)),
                ("self_attn.q_norm.weight", (8,)),
                ("self_attn.k_norm.weight", (8,)),
                ("self_attn.o_proj.weight", (hidden, 32)),
                ("self_attn.indexer.index_qk_proj.weight", (24, hidden)),
                ("self_attn.indexer.q_layernorm.weight", (8,)),
                ("self_attn.indexer.k_layernorm.weight", (8,)),
            ):
                tensors[prefix + name] = (
                    "F32",
                    shape,
                    f32_bytes(rng.normal(0, 0.05, shape)),
                )
        else:
            for name, shape in (
                ("linear_attn.in_proj_qkv.weight", (64, hidden)),
                ("linear_attn.in_proj_z.weight", (32, hidden)),
                ("linear_attn.in_proj_b.weight", (4, hidden)),
                ("linear_attn.in_proj_a.weight", (4, hidden)),
                ("linear_attn.conv1d.weight", (64, 1, 4)),
                ("linear_attn.A_log", (4,)),
                ("linear_attn.dt_bias", (4,)),
                ("linear_attn.norm.weight", (8,)),
                ("linear_attn.out_proj.weight", (hidden, 32)),
            ):
                tensors[prefix + name] = (
                    "F32",
                    shape,
                    f32_bytes(rng.normal(0, 0.05, shape)),
                )
        for name, shape in (
            ("mlp.gate.weight", (experts, hidden)),
            ("mlp.experts.gate_up_proj", (experts, 2 * width, hidden)),
            ("mlp.experts.down_proj", (experts, hidden, width)),
            ("mlp.shared_expert.gate_proj.weight", (width, hidden)),
            ("mlp.shared_expert.up_proj.weight", (width, hidden)),
            ("mlp.shared_expert.down_proj.weight", (hidden, width)),
            ("mlp.shared_expert_gate.weight", (1, hidden)),
        ):
            tensors[prefix + name] = (
                "F32",
                shape,
                f32_bytes(rng.normal(0, 0.05, shape)),
            )
    ple = f"layers.{QWEN4_LAYOUT.ple_layer}.ple."
    ple_embed = QWEN4_LAYOUT.ple_heads * QWEN4_LAYOUT.ple_head_dimension
    for name, shape in (
        ("key_proj.weight", (streams, ple_embed)),
        ("value_proj.weight", (hidden, ple_embed)),
        ("norm_key.weight", (streams,)),
        ("norm_query.weight", (streams,)),
        ("norm_conv.weight", (streams,)),
        ("conv1d.weight", (streams, 1, 4)),
    ):
        tensors[ple + name] = (
            "F32",
            shape,
            f32_bytes(rng.normal(0, 0.05, shape)),
        )
    pe = ple + "ple_embedding."
    for name, values in (
        ("layer_multipliers", np.array([11, 22, 33], np.int64)),
        ("ngram_heads_vocab_sizes", np.array([50, 60], np.int64)),
        ("ngram_heads_offsets", np.array([0, 50], np.int64)),
    ):
        tensors[pe + name] = ("I64", tuple(values.shape), values.tobytes())
    for shard in range(2):
        tensors[f"{pe}ngram_embedding.shard_{shard}.weight"] = (
            "F32",
            (150, QWEN4_LAYOUT.ple_head_dimension),
            f32_bytes(rng.normal(0, 0.05, (150, QWEN4_LAYOUT.ple_head_dimension))),
        )
    return write_safetensors(root / "model.safetensors", tensors)


class Qwen4ExpPackTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_qwen4exp_is_packable(self):
        self.assertTrue(pack.packable(QWEN4_CONFIG))
        self.assertEqual(
            pack.target_format(QWEN4_CONFIG),
            (pack.QWEN4_FORMAT_NAME, pack.QWEN4_TARGET_FORMAT),
        )
        self.assertFalse(pack.packable({"text_config": {"model_type": "other"}}))

    def test_q8o_planes_pad_rows_to_tiles(self):
        rng = np.random.default_rng(5)
        weights = rng.normal(0, 0.5, (300, 64)).astype(np.float32)
        plane0, plane1, meta, rows = pack._q8o_planes(weights)
        # 300 rows pad to 512: 2 tiles x 2 groups x 256 rows x 32 bytes.
        self.assertEqual(rows, 512)
        self.assertEqual(len(plane0), 2 * 2 * 256 * 32)
        self.assertEqual(len(meta), 2 * 2 * 256 * 2)
        self.assertEqual(plane1, b"")

    def test_gdn_fused_width_covers_the_layout(self):
        shard = synthetic_qwen4_checkpoint(self.root)
        checkpoint = pack.Checkpoint([shard])
        sections = pack.pack_qwen4exp_layer_sections(checkpoint, QWEN4_LAYOUT, 0)
        # mixer hyper 6, gdn 15, ffn hyper 6, moe 22 = 49 sections.
        self.assertEqual(len(sections), 49)
        # The gdn-qkv/z/ab segments follow the mixer hyper's six: their
        # descriptor row counts sum to packed_gdn_width (64 + 32 + 16).
        rows = [struct.unpack("<8I4Q", sections[index][:64])[1] for index in (6, 9, 12)]
        self.assertEqual(rows, [64, 32, 16])
        self.assertEqual(sum(rows), QWEN4_LAYOUT.packed_gdn_width)

    def test_qwen4exp_package_writes_a_validatable_manifest(self):
        shard = synthetic_qwen4_checkpoint(self.root)
        stage = self.root / "pkg"
        pack_spec = {
            "layout": _fields(QWEN4_LAYOUT),
            "execution_geometry": {
                "draft_proposal_tokens": 7,
                "draft_query_rows": 8,
            },
        }
        manifest = pack.build_package(
            stage,
            "owner/qwen4-test",
            "Qwen3.8-Flash-Next",
            QWEN4_CONFIG,
            {"model.safetensors": shard},
            {"tokenizer.json": shard},
            {},
            {"target": {"repo": "owner/qwen4-test", "revision": "a" * 40}},
            pack=pack_spec,
        )
        self.assertEqual(manifest["format"]["name"], pack.QWEN4_FORMAT_NAME)
        self.assertEqual(manifest["format"]["target_layer_magic"], "MDFQ0001")
        self.assertEqual(manifest["target"]["architecture"], "qwen4exp")
        self.assertEqual(manifest["target"]["layers"], QWEN4_LAYOUT.layers)
        self.assertNotIn("draft", manifest)
        for name in (
            "target/embedding.bin",
            "target/head.bin",
            "target/ple-table.bin",
            "target/layer-0.bin",
            "target/layer-7.bin",
            "tokenizer/config.json",
        ):
            self.assertTrue((stage / name).exists(), name)
        # File magics: layer MDFQ0001, head MDFQ0002, embedding MDFE0001,
        # table MDFQ0003; the PLE header is 176 bytes of hash metadata.
        self.assertEqual((stage / "target/layer-0.bin").read_bytes()[:8], b"MDFQ0001")
        self.assertEqual((stage / "target/head.bin").read_bytes()[:8], b"MDFQ0002")
        self.assertEqual((stage / "target/embedding.bin").read_bytes()[:8], b"MDFE0001")
        table = (stage / "target/ple-table.bin").read_bytes()
        self.assertEqual(table[:8], b"MDFQ0003")
        pack.verify(stage, full=True)
        # is_built recognizes the qwen4exp manifest.
        link = self.root / "link"
        link.mkdir()
        (link / "manifest.json").write_bytes((stage / "manifest.json").read_bytes())
        self.assertTrue(pack.is_built(link))


# A shrunken DFlash2 draft layout at the real draft's proportions: hidden
# 256 with the 64-wide dynamic projections (hidden/4, off the affine
# 256-row tile so they pack as F32 segments), the head pattern's fused
# QKV and attention widths and a 256-row selector rank.
DFLASH2_LAYOUT = pack_layouts.DFlash2Layout(
    layers=2,
    hidden=256,
    vocabulary=1024,
    dynamic=64,
    intermediate=512,
    selector_rank=256,
    kv_heads=2,
    head_dim=64,
    query_heads=4,
    rotary_theta=1_000_000.0,
    block_size=8,
    target_hidden=512,
    sliding_window=2048,
)


def synthetic_dflash2_checkpoint(root):
    """A DFlash2 draft safetensors shard at DFLASH2_LAYOUT's shapes, the
    tensor names DraftCheckpoint.cpp's dflash2 layer/model images read."""
    rng = np.random.default_rng(23)
    hidden = DFLASH2_LAYOUT.hidden
    kv = DFLASH2_LAYOUT.kv_heads * DFLASH2_LAYOUT.head_dim
    tensors = {
        "fc.weight": (
            "BF16",
            (hidden, DFLASH2_LAYOUT.target_hidden),
            bf16(rng.normal(0, 0.05, (hidden, DFLASH2_LAYOUT.target_hidden))),
        ),
        "hidden_norm.weight": ("BF16", (hidden,), bf16(rng.normal(1, 0.1, hidden))),
        "norm.weight": ("BF16", (hidden,), bf16(rng.normal(1, 0.1, hidden))),
        "candidate_selector.hidden_projection.weight": (
            "BF16",
            (DFLASH2_LAYOUT.selector_rank, hidden),
            bf16(rng.normal(0, 0.05, (DFLASH2_LAYOUT.selector_rank, hidden))),
        ),
        "candidate_selector.predecessor_codebook": (
            "BF16",
            (DFLASH2_LAYOUT.vocabulary, DFLASH2_LAYOUT.selector_rank),
            bf16(
                rng.normal(
                    0,
                    0.05,
                    (DFLASH2_LAYOUT.vocabulary, DFLASH2_LAYOUT.selector_rank),
                )
            ),
        ),
        "candidate_selector.successor_codebook": (
            "BF16",
            (DFLASH2_LAYOUT.vocabulary, DFLASH2_LAYOUT.selector_rank),
            bf16(
                rng.normal(
                    0,
                    0.05,
                    (DFLASH2_LAYOUT.vocabulary, DFLASH2_LAYOUT.selector_rank),
                )
            ),
        ),
    }
    for layer in range(DFLASH2_LAYOUT.layers):
        prefix = f"layers.{layer}."
        attention = prefix + "self_attn."
        tensors.update(
            {
                f"{prefix}input_layernorm.weight": (
                    "BF16",
                    (hidden,),
                    bf16(rng.normal(1, 0.1, hidden)),
                ),
                f"{prefix}attention_conv.base_kernel": (
                    "BF16",
                    (2, 2, hidden),
                    bf16(rng.normal(0, 0.05, (2, 2, hidden))),
                ),
                f"{prefix}attention_conv.kernel_projection.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.dynamic, hidden),
                    bf16(rng.normal(0, 0.05, (DFLASH2_LAYOUT.dynamic, hidden))),
                ),
                f"{attention}q_proj.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.attention, hidden),
                    bf16(rng.normal(0, 0.05, (DFLASH2_LAYOUT.attention, hidden))),
                ),
                f"{attention}k_proj.weight": (
                    "BF16",
                    (kv, hidden),
                    bf16(rng.normal(0, 0.05, (kv, hidden))),
                ),
                f"{attention}v_proj.weight": (
                    "BF16",
                    (kv, hidden),
                    bf16(rng.normal(0, 0.05, (kv, hidden))),
                ),
                f"{attention}q_norm.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.head_dim,),
                    bf16(rng.normal(1, 0.1, DFLASH2_LAYOUT.head_dim)),
                ),
                f"{attention}k_norm.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.head_dim,),
                    bf16(rng.normal(1, 0.1, DFLASH2_LAYOUT.head_dim)),
                ),
                f"{attention}o_proj.weight": (
                    "BF16",
                    (hidden, DFLASH2_LAYOUT.attention),
                    bf16(rng.normal(0, 0.05, (hidden, DFLASH2_LAYOUT.attention))),
                ),
                f"{prefix}post_attention_layernorm.weight": (
                    "BF16",
                    (hidden,),
                    bf16(rng.normal(1, 0.1, hidden)),
                ),
                f"{prefix}mlp_conv.base_kernel": (
                    "BF16",
                    (2, 2, hidden),
                    bf16(rng.normal(0, 0.05, (2, 2, hidden))),
                ),
                f"{prefix}mlp_conv.kernel_projection.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.dynamic, hidden),
                    bf16(rng.normal(0, 0.05, (DFLASH2_LAYOUT.dynamic, hidden))),
                ),
                f"{prefix}mlp.gate_proj.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.intermediate, hidden),
                    bf16(
                        rng.normal(
                            0,
                            0.05,
                            (DFLASH2_LAYOUT.intermediate, hidden),
                        )
                    ),
                ),
                f"{prefix}mlp.up_proj.weight": (
                    "BF16",
                    (DFLASH2_LAYOUT.intermediate, hidden),
                    bf16(
                        rng.normal(
                            0,
                            0.05,
                            (DFLASH2_LAYOUT.intermediate, hidden),
                        )
                    ),
                ),
                f"{prefix}mlp.down_proj.weight": (
                    "BF16",
                    (hidden, DFLASH2_LAYOUT.intermediate),
                    bf16(
                        rng.normal(
                            0,
                            0.05,
                            (hidden, DFLASH2_LAYOUT.intermediate),
                        )
                    ),
                ),
            }
        )
    return write_safetensors(root / "draft.safetensors", tensors)


class DFlash2PackTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def test_dflash2_layer_sections_follow_read_order(self):
        shard = synthetic_dflash2_checkpoint(self.root)
        checkpoint = pack.Checkpoint([shard])
        sections = pack._dflash2_layer_sections(checkpoint, DFLASH2_LAYOUT, 0)
        # readDraft's order: the two dynamic projections each take a GGUF
        # descriptor section and an fp32 rows section, the rest one each.
        self.assertEqual(len(sections), 15)
        hidden = DFLASH2_LAYOUT.hidden
        dynamic = DFLASH2_LAYOUT.dynamic
        sizes = [
            hidden * 2,  # input-norm
            4 * hidden * 2,  # attention-convolution
            64,  # attention-dynamic descriptor
            dynamic * hidden * 4,  # attention-dynamic fp32 rows
            quantized_bytes(DFLASH2_LAYOUT.qkv, hidden),  # qkv
            DFLASH2_LAYOUT.head_dim * 2,  # query-norm
            DFLASH2_LAYOUT.head_dim * 2,  # key-norm
            quantized_bytes(hidden, DFLASH2_LAYOUT.attention),  # output
            hidden * 2,  # post-attention-norm
            4 * hidden * 2,  # mlp-convolution
            64,  # mlp-dynamic descriptor
            dynamic * hidden * 4,  # mlp-dynamic fp32 rows
            quantized_bytes(DFLASH2_LAYOUT.intermediate, hidden),  # gate
            quantized_bytes(DFLASH2_LAYOUT.intermediate, hidden),  # up
            quantized_bytes(hidden, DFLASH2_LAYOUT.intermediate),  # down
        ]
        self.assertEqual([len(section) for section in sections], sizes)
        # The dynamic descriptors are F32 GGUF tensors of the dynamic
        # width by hidden — the shape the affine 256-row tile cannot hold.
        for index in (2, 10):
            kind, rows, columns = struct.unpack("<8I4Q", sections[index])[:3]
            self.assertEqual(kind, 0)  # ggml::kF32
            self.assertEqual((rows, columns), (dynamic, hidden))

    def test_pack_draft2_writes_mdfd_files(self):
        shard = synthetic_dflash2_checkpoint(self.root)
        out = self.root / "draft"
        out.mkdir()
        pack.pack_draft2(pack.Checkpoint([shard]), out, DFLASH2_LAYOUT)
        for layer in range(DFLASH2_LAYOUT.layers):
            data = (out / f"layer-{layer}.bin").read_bytes()
            self.assertEqual(
                data[:16],
                struct.pack("<8sII", b"MDFD0004", layer, 0),
            )
            self.assertFalse(len(data) % pack.ALIGNMENT)
        model = (out / "model.bin").read_bytes()
        self.assertEqual(
            model[:16],
            struct.pack("<8sII", b"MDFD0004", DFLASH2_LAYOUT.layers, 1),
        )
        self.assertFalse(len(model) % pack.ALIGNMENT)

    def test_qwen4exp_package_packs_the_dflash2_draft(self):
        shard = synthetic_qwen4_checkpoint(self.root)
        (self.root / "draftsrc").mkdir()
        draft = synthetic_dflash2_checkpoint(self.root / "draftsrc")
        stage = self.root / "pkg"
        pack_spec = {
            "layout": _fields(QWEN4_LAYOUT),
            "draft_layout": {
                **_fields(DFLASH2_LAYOUT),
                "kind": "dflash2",
            },
            "execution_geometry": {
                "draft_proposal_tokens": 7,
                "draft_query_rows": 8,
                "draft_sliding_window": DFLASH2_LAYOUT.sliding_window,
            },
        }
        manifest = pack.build_package(
            stage,
            "owner/qwen4-test",
            "Qwen3.8-Flash-Next",
            QWEN4_CONFIG,
            {"model.safetensors": shard},
            {"tokenizer.json": shard},
            {
                "draft/config.json": shard,
                "draft/model.safetensors": draft,
            },
            {"target": {"repo": "owner/qwen4-test", "revision": "a" * 40}},
            pack=pack_spec,
        )
        self.assertEqual(manifest["format"]["draft_layer_magic"], "MDFD0004")
        declared = manifest["draft"]
        self.assertEqual(declared["architecture"], "DFlash2DraftModel")
        self.assertEqual(declared["layers"], DFLASH2_LAYOUT.layers)
        self.assertEqual(declared["hidden_size"], DFLASH2_LAYOUT.hidden)
        self.assertEqual(declared["dynamic_size"], DFLASH2_LAYOUT.dynamic)
        self.assertEqual(declared["selector_rank"], DFLASH2_LAYOUT.selector_rank)
        self.assertEqual(declared["block_size"], DFLASH2_LAYOUT.block_size)
        self.assertEqual(declared["target_hidden_size"], DFLASH2_LAYOUT.target_hidden)
        paths = {record["path"] for record in manifest["artifacts"]}
        self.assertIn("draft/model.bin", paths)
        self.assertIn("draft/layer-0.bin", paths)
        self.assertIn("draft/layer-1.bin", paths)
        self.assertTrue((stage / "draft" / "config.json").is_file())
        pack.verify(stage, full=True)


if __name__ == "__main__":
    unittest.main()
