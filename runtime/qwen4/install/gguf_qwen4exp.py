"""Staged install-side support for the qwen4exp GGUF architecture
(Qwen3.8-Flash-Next today, the Qwen 4 family at release), ported from
upstream's install/gguf.py and adapted to this tree's GgufArch registry.

Not imported anywhere: install/gguf.py stays untouched until the runtime
serves the family. To integrate, splice the tensor sets, _qwen4exp_fields
and _qwen4exp_tensors below into install/gguf.py ahead of GGUF_ARCHS (they
reference its QUANTIZED_TYPES, EMBEDDING_TYPES and F32) and add the row

    "qwen4exp": GgufArch(
        "qwen4_exp_text", fields=_qwen4exp_fields, tensors=_qwen4exp_tensors
    ),

to GGUF_ARCHS. Upstream derivation also emitted no num_experts_per_tok for
qwen4exp; the fields handler emits it like _qwen35moe_fields so the same
signature fields can identify either registry's config.
"""

# Qwen3.8-Flash-Next (qwen4exp): hyper-connections replace every norm, so a
# layer has none of its own and the model no output norm; its full-attention
# layers carry the QSA indexer, whose projections are float tensors; its PLE
# layer carries the n-gram embedding's projections, norms and convolution,
# and the model the embedding's table.
QWEN4_FORMAT_TYPES = QUANTIZED_TYPES | {"BF16", "F16"}
QWEN4_NORM_TYPES = {"F32", "BF16", "F16"}
QWEN4_FLOAT_PROJECTION_TYPES = {"F32", "BF16"}
QWEN4_FLOAT_ROW_TYPES = {"F32", "BF16", "Q8_0"}

QWEN4_MODEL_TENSORS = {
    "token_embd.weight": EMBEDDING_TYPES,
    "per_layer_token_embd.weight": EMBEDDING_TYPES,
    "output.weight": QWEN4_FORMAT_TYPES,
    "output_hc_norm.weight": QWEN4_NORM_TYPES,
    "output_hc_down.weight": QWEN4_FORMAT_TYPES,
    "output_hc_up.weight": QWEN4_FORMAT_TYPES,
}
QWEN4_LAYER_TENSORS = {
    f"hc_{part}_{name}.weight": types
    for part in ("attn", "ffn")
    for name, types in (
        ("norm", QWEN4_NORM_TYPES),
        ("down", QWEN4_FORMAT_TYPES),
        ("up", QWEN4_FORMAT_TYPES),
        ("inject", QWEN4_FLOAT_ROW_TYPES),
    )
}
QWEN4_INDEXER_TENSORS = {
    "indexer.q_proj.weight": QWEN4_FLOAT_PROJECTION_TYPES,
    "indexer.k_proj.weight": QWEN4_FLOAT_PROJECTION_TYPES,
    "indexer.q_norm.weight": QWEN4_NORM_TYPES,
    "indexer.k_norm.weight": QWEN4_NORM_TYPES,
}
QWEN4_PLE_TENSORS = {
    "ple_key.weight": QWEN4_FORMAT_TYPES,
    "ple_value.weight": QWEN4_FORMAT_TYPES,
    "ple_norm_key.weight": QWEN4_NORM_TYPES,
    "ple_norm_query.weight": QWEN4_NORM_TYPES,
    "ple_norm_conv.weight": QWEN4_NORM_TYPES,
    "ple_conv1d.weight": QWEN4_NORM_TYPES,
}
# The fused gate_up experts tensor and the split pair are alternatives
# (dev's MOE_TENSORS names the split pair; gemma4's handler already accepts
# the fused one).
QWEN4_FUSED_MOE_TENSORS = {
    name: types
    for name, types in MOE_TENSORS.items()
    if name not in ("ffn_gate_exps.weight", "ffn_up_exps.weight")
} | {"ffn_gate_up_exps.weight": QWEN4_FORMAT_TYPES}

# A full-attention layer's entry of attention.compress_ratios, where a GDN
# layer's is 0; and the one layer ple.layers names, which reads the PLE
# tensors. Both are Qwen4ExpLayout's indexerBlockTokens and pleLayer
# (runtime/qwen4/model/Qwen4Exp.hpp).
QWEN4_QSA_BLOCK = 4
QWEN4_PLE_LAYER = 1


def _qwen4exp_fields(arch, text, metadata):
    text.update(
        num_experts=metadata.positive(arch + ".expert_count"),
        num_experts_per_tok=metadata.positive(arch + ".expert_used_count"),
    )
    # The metadata arrays the engine's model check never sees: with QSA,
    # attention.compress_ratios, from its first entry the QSA block for each
    # loaded full-attention layer and 0 for each other; with a PLE layer,
    # ple.layers, that layer alone. One error names every mismatch
    # (upstream's require_arrays).
    layers = text["num_hidden_layers"]
    period = metadata.positive(arch + ".full_attention_interval")
    ratios = [
        QWEN4_QSA_BLOCK if (layer + 1) % period == 0 else 0
        for layer in range(layers)
    ]
    mismatched = []
    found = metadata.values.get(arch + ".attention.compress_ratios")
    if type(found) is not list or found[: len(ratios)] != ratios:
        mismatched.append("attention.compress_ratios")
    if metadata.values.get(arch + ".ple.layers") != [QWEN4_PLE_LAYER]:
        mismatched.append("ple.layers")
    if mismatched:
        raise ModelError(
            "GGUF metadata does not match the target: " + ", ".join(mismatched)
        )


def _qwen4exp_tensors(metadata, layers):
    """qwen4exp: every full_attention_interval-th layer is full attention
    with the QSA indexer, the others GDN; the FFN is a sparse MoE, fused or
    split; the PLE layer reads the n-gram embedding's tensors too."""
    arch = text_architecture(metadata)
    period = metadata.positive(arch + ".full_attention_interval")
    tensors = dict(QWEN4_MODEL_TENSORS)
    for layer in range(layers):
        mixer = (
            ATTENTION_TENSORS | QWEN4_INDEXER_TENSORS
            if (layer + 1) % period == 0
            else GDN_TENSORS
        )
        ffn = (
            QWEN4_FUSED_MOE_TENSORS
            if f"blk.{layer}.ffn_gate_up_exps.weight" in metadata.tensors
            else MOE_TENSORS
        )
        parts = QWEN4_LAYER_TENSORS | mixer | ffn
        if layer == QWEN4_PLE_LAYER:
            parts |= QWEN4_PLE_TENSORS
        for name, types in parts.items():
            tensors[f"blk.{layer}.{name}"] = types
    return tensors
