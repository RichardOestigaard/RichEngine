#!/usr/bin/env python3
"""Generate a multi-function CoreAI .aimodel asset for RichEngine's dense FFN
split. One named entrypoint per supported intermediate-channel count; weights
are runtime int8/fp16 inputs (never baked) so the engine stages them per
layer. Requires `pip install coreai-torch` on Python <=3.13.

Usage: gen_coreai_ffn.py --hidden 2048 --anes 4608,5120,5632 -o out.aimodel
"""

import argparse
from pathlib import Path

import torch
import coreai_torch
import coreai_torch._compression.custom_layers  # registers coreai:: ops


def make_ffn(hidden: int, ane: int):
    class Ffn(torch.nn.Module):
        def forward(self, x, gate_q, gate_s, up_q, up_s, down_q, down_s):
            dq = torch.ops.coreai.dequantize
            g = dq(gate_q, gate_s, axis=0)
            u = dq(up_q, up_s, axis=0)
            d = dq(down_q, down_s, axis=0)
            h = torch.nn.functional.silu(x @ g.t()) * (x @ u.t())
            return h @ d.t()

    rows = torch.export.Dim("rows", min=1, max=2048)
    example = (
        torch.zeros(2, hidden, dtype=torch.float16),
        torch.zeros(ane, hidden, dtype=torch.int8),
        torch.zeros(ane, dtype=torch.float16),
        torch.zeros(ane, hidden, dtype=torch.int8),
        torch.zeros(ane, dtype=torch.float16),
        torch.zeros(hidden, ane, dtype=torch.int8),
        torch.zeros(hidden, dtype=torch.float16),
    )
    ep = torch.export.export(
        Ffn(),
        example,
        dynamic_shapes=(
            {0: rows},
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden", type=int, required=True)
    p.add_argument("--anes", type=str, required=True,
                   help="comma-separated intermediate-channel counts")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    conv = coreai_torch.TorchConverter()
    for ane in (int(a) for a in args.anes.split(",")):
        conv.add_exported_program(
            make_ffn(args.hidden, ane),
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
