#!/usr/bin/env python3
"""Generate a multi-function CoreAI .aimodel asset for RichEngine's dense FFN
split. One named entrypoint per supported intermediate-channel count; weights
are runtime inputs (never baked) so the engine stages them per layer.
Requires `pip install coreai-torch` on Python <=3.13.

Quantization contracts (--quant):
  pc  : int8 [A,H] weights + fp16 [A] per-channel scales (coreai::dequantize)
  b32 : int8 [A,H] weights + fp16 [A,H/32] block scales per 32 inputs
        (coreai::constexpr_blockwise_shift_scale, lowers to
        coreai.blockwise_shift_scale — despite the torch-op name, runtime
        inputs are accepted: the lowering just forwards operands)
  si4 : int4 [A,H] packed weights + fp16 [A,H/32] block scales. The
        runtime binds ScalarType.int4 on A*H/2 bytes. Verified on-device
        (probes /tmp/cai_tests, 2026-10-27): descriptors report
        `gate_q [A,H] int4`, eval is at parity with int8 on the ANE path.

Usage: gen_coreai_ffn.py --hidden 2048 --anes 4608,5120,5632 \
         --quant si4 -o out.aimodel
"""

import argparse
from pathlib import Path

import torch
import coreai_torch
import coreai_torch._compression.custom_layers  # registers coreai:: ops
import coreai_torch.converter as _converter


def _patch_runtime_inputs() -> None:
    """Skip inject_subbyte_tensors: it assumes quantized operands live in
    the state_dict (constants). Our weights are function inputs, whose
    input_spec.target is None — the pass crashes with AttributeError, and
    for an all-runtime-input graph it has nothing to do anyway."""
    _converter.inject_subbyte_tensors = lambda program: program


_BLOCK = 32


def make_ffn(hidden: int, ane: int, quant: str):
    if quant == "pc":
        class Ffn(torch.nn.Module):
            def forward(self, x, gate_q, gate_s, up_q, up_s, down_q, down_s):
                dq = torch.ops.coreai.dequantize
                g = dq(gate_q, gate_s, axis=0)
                u = dq(up_q, up_s, axis=0)
                d = dq(down_q, down_s, axis=0)
                h = torch.nn.functional.silu(x @ g.t()) * (x @ u.t())
                return h @ d.t()
    else:
        class Ffn(torch.nn.Module):
            def forward(self, x, gate_q, gate_s, up_q, up_s, down_q, down_s):
                dq = torch.ops.coreai.constexpr_blockwise_shift_scale
                kw = {"input_dtype": torch.int4} if quant == "si4" else {}
                g = dq(gate_q, gate_s, **kw)
                u = dq(up_q, up_s, **kw)
                d = dq(down_q, down_s, **kw)
                h = torch.nn.functional.silu(x @ g.t()) * (x @ u.t())
                return h @ d.t()

    rows = torch.export.Dim("rows", min=1, max=2048)
    if quant == "si4":
        qtype = torch.int4
        qfill = torch.empty  # zeros() hits `fill_cpu` unimplemented for Int4
    elif quant == "fp8":
        qtype = torch.float8_e4m3fn
        qfill = torch.zeros
    else:
        qtype = torch.int8
        qfill = torch.zeros
    if quant == "pc":
        sshape = lambda rows_, cols_: (rows_,)
    else:
        sshape = lambda rows_, cols_: (rows_, cols_ // _BLOCK)
    example = (
        torch.zeros(2, hidden, dtype=torch.float16),
        qfill(ane, hidden, dtype=qtype), torch.zeros(*sshape(ane, hidden), dtype=torch.float16),
        qfill(ane, hidden, dtype=qtype), torch.zeros(*sshape(ane, hidden), dtype=torch.float16),
        qfill(hidden, ane, dtype=qtype), torch.zeros(*sshape(hidden, ane), dtype=torch.float16),
    )
    ep = torch.export.export(
        Ffn(),
        example,
        dynamic_shapes=({0: rows},) + (None,) * 6,
    )
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--anes", type=str, required=True,
                   help="comma-separated intermediate-channel counts")
    p.add_argument("--quant", choices=["pc", "b32", "si4", "fp8"], default="b32",
                   help="b32 verified in-engine (1.4%% vs pc 1.9%%); si4 fails "
                        "verify on affine-Q4 sources (~15%%, double quantization); "
                        "fp8 = e4m3 data + fp16 block scales (DeepSeek-style)")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    _patch_runtime_inputs()
    conv = coreai_torch.TorchConverter()
    for ane in (int(a) for a in args.anes.split(",")):
        conv.add_exported_program(
            make_ffn(args.hidden, ane, args.quant),
            input_names=(
                "x", "gate_q", "gate_s", "up_q", "up_s", "down_q", "down_s"
            ),
            output_names=("y",),
            entrypoint_name=f"ffn_{args.hidden}_{ane}",
        )
        print(f"added ffn_{args.hidden}_{ane}")
    conv.to_coreai().save_asset(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
