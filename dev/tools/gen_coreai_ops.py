#!/usr/bin/env python3
"""Author the gather-bound CoreAI functions the engine's op set cannot
express well — the cases where a metal4_kernel beats composed ops
(probe-verified 2026-10-27: fused MoE gather-GEMV ~4x index_select+bmm):
requires `pip install coreai-torch` on Python <=3.13.

  moe_gemv : out[r] = x[r] @ w[idx[r]] — per-token expert gather inside the
             dot loop; no 16 MB gathered-weight materialization.
  kv_index : z[r] = kv[idx[r]] — a paged-KV / latent-index gather where the
             cache is a function STATE (states: bind persists across
             encodes; in-graph writes via index_put land on-device).
  sampler  : out[r] = argmax(logits[r] / temp) — fused sampling; extend the
             kernel body with top-k/min-p/RNG as needed (seed arrives as a
             tensor input: scalar args bake at compile time).

Kernel indexing: tensor params are `metal::dextents<int, rank>`;
`x[gid]` for rank-1, `array<int,N>` for rank-N with the INNERMOST dim
first — kv[array<int,2>({col, row})]. Dynamic result_shapes and runtime
grids segfault the macOS 27 runtime: keep grids/result_shapes static and
bound the work inside the kernel body.

Usage: gen_coreai_ops.py -o out.aimodel
"""

import argparse
from pathlib import Path

import torch
import coreai_torch
import coreai_torch._compression.custom_layers
import coreai_torch.converter as _converter

_converter.inject_subbyte_tensors = lambda p: p  # runtime-input graphs crash it

from coreai_torch import MetalParameter, TorchMetalKernel

GID = MetalParameter("gid", "uint", "thread_position_in_grid")
TI, K_DIM, DO, E = 8, 2048, 512, 64
KV_ROWS, KV_DIM = 4096, 128
VOCAB = 128256


def ref_moe(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.zeros(TI, DO, dtype=x.dtype)


k_moe = TorchMetalKernel(
    "moe_gemv", ["x", "idx", "w"], ["out"],
    "uint r = gid / %d, c = gid %% %d;\n"
    "  int e = idx[array<int,1>({int(r)})];\n"
    "  float a = 0.0;\n"
    "  for (int d = 0; d < %d; d++)\n"
    "    a += float(x[array<int,2>({d,int(r)})]) * float(w[array<int,3>({int(c),d,e})]);\n"
    "  out[array<int,2>({int(c),int(r)})] = T(a);" % (DO, DO, K_DIM),
    ref_moe, [GID], template_dtypes={"x": "T"})


class MoeFused(torch.nn.Module):
    def forward(self, x, idx, w):
        return k_moe(x, idx, w, threads_per_grid=(TI * DO, 1, 1),
                     threads_per_thread_group=(256, 1, 1), result_shapes=[[TI, DO]])


def ref_index(kv: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.zeros(TI, KV_DIM, dtype=kv.dtype)


k_index = TorchMetalKernel(
    "kv_index", ["kv", "idx"], ["out"],
    "uint r = gid / %d, c = gid %% %d;\n"
    "  int p = idx[array<int,1>({int(r)})];\n"
    "  out[array<int,2>({int(c),int(r)})] = kv[array<int,2>({int(c),p})];" % (KV_DIM, KV_DIM),
    ref_index, [GID])


class KvIndex(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kv", torch.zeros(KV_ROWS, KV_DIM, dtype=torch.float16))

    def forward(self, idx, vals):
        # in-graph state write then a kernel read of the updated rows —
        # the on-device KV-growth + gather composition.
        self.kv.index_copy_(0, idx.to(torch.long), vals)
        return k_index(self.kv, idx, threads_per_grid=(TI * KV_DIM, 1, 1),
                       threads_per_thread_group=(256, 1, 1), result_shapes=[[TI, KV_DIM]])


def ref_sample(logits: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
    return torch.zeros(1, dtype=torch.int32)


k_sample = TorchMetalKernel(
    "sampler", ["logits", "temp"], ["out"],
    "float t = float(temp[0]);\n"
    "  float best = -INFINITY; int arg = 0;\n"
    "  for (int v = int(gid); v < %d; v += 256) {\n"
    "    float s = float(logits[array<int,2>({v,0})]) / t;\n"
    "    if (s > best) { best = s; arg = v; }\n"
    "  }\n"
    "  threadgroup float bs[256]; threadgroup int ba[256];\n"
    "  bs[gid] = best; ba[gid] = arg;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (gid == 0) {\n"
    "    float m = bs[0]; int a = ba[0];\n"
    "    for (int i = 1; i < 256; i++) if (bs[i] > m) { m = bs[i]; a = ba[i]; }\n"
    "    out[0] = a;\n"
    "  }" % VOCAB,
    ref_sample, [GID])


class Sampler(torch.nn.Module):
    def forward(self, logits, temp):
        return k_sample(logits, temp, threads_per_grid=(256, 1, 1),
                        threads_per_thread_group=(256, 1, 1), result_shapes=[[1]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    conv = coreai_torch.TorchConverter()
    conv.register_custom_kernels([k_moe, k_index, k_sample])

    conv.add_exported_program(
        torch.export.export(
            MoeFused(),
            (torch.zeros(TI, K_DIM, dtype=torch.float16),
             torch.zeros(TI, dtype=torch.int32),
             torch.zeros(E, K_DIM, DO, dtype=torch.float16)),
        ).run_decompositions(coreai_torch.get_decomp_table()),
        input_names=("x", "idx", "w"), output_names=("out",), entrypoint_name="moe_gemv")

    conv.add_exported_program(
        torch.export.export(
            KvIndex(),
            (torch.zeros(TI, dtype=torch.int32),
             torch.zeros(TI, KV_DIM, dtype=torch.float16)),
        ).run_decompositions(coreai_torch.get_decomp_table()),
        input_names=("idx", "vals"), output_names=("out",),
        state_names=("kv",), entrypoint_name="kv_index")

    conv.add_exported_program(
        torch.export.export(
            Sampler(),
            (torch.zeros(1, VOCAB, dtype=torch.float16),
             torch.ones(1, dtype=torch.float16)),
        ).run_decompositions(coreai_torch.get_decomp_table()),
        input_names=("logits", "temp"), output_names=("out",), entrypoint_name="sampler")

    conv.to_coreai().save_asset(args.output)
    print(f"wrote {args.output}: moe_gemv, kv_index, sampler")


if __name__ == "__main__":
    main()
