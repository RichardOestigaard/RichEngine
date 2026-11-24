#!/usr/bin/env python3
"""Convert a Nemotron MTP draft GGUF (the mtp-*.gguf the ggml-org target
repository ships) into the safetensors + config.json draft directory
RichEngine's nemotron MTP draft loads: every tensor dequantized to BF16 in
HF row order ([out, in]), named mtp.*.

Streaming: the source is mmap'd and each tensor writes in row chunks, so
peak memory stays near one GGUF block range, not the file.

Usage: gguf_mtp_to_safetensors.py MTP.gguf OUT_DIR
"""

from __future__ import annotations

import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

TENSOR_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
                8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16"}

RENAME = {
    "blk.52.attn_norm.weight": "mtp.attn_norm.weight",
    "blk.52.attn_q.weight": "mtp.self_attn.q_proj.weight",
    "blk.52.attn_k.weight": "mtp.self_attn.k_proj.weight",
    "blk.52.attn_v.weight": "mtp.self_attn.v_proj.weight",
    "blk.52.attn_output.weight": "mtp.self_attn.o_proj.weight",
    "blk.52.post_attention_norm.weight": "mtp.post_attention_norm.weight",
    "blk.52.exp_probs_b.bias": "mtp.moe.exp_probs_b.bias",
    "blk.52.ffn_gate_inp.weight": "mtp.moe.gate_inp.weight",
    "blk.52.ffn_up_exps.weight": "mtp.moe.up_exps.weight",
    "blk.52.ffn_down_exps.weight": "mtp.moe.down_exps.weight",
    "blk.52.ffn_up_shexp.weight": "mtp.moe.up_shexp.weight",
    "blk.52.ffn_down_shexp.weight": "mtp.moe.down_shexp.weight",
    "blk.52.nextn.eh_proj.weight": "mtp.nextn.eh_proj.weight",
    "blk.52.nextn.enorm.weight": "mtp.nextn.enorm.weight",
    "blk.52.nextn.hnorm.weight": "mtp.nextn.hnorm.weight",
    "blk.52.nextn.shared_head_norm.weight": "mtp.nextn.shared_head_norm.weight",
    "output_norm.weight": "mtp.output_norm.weight",
}

CHUNK_ROWS = 4096


def read_str(buf, off):
    n = struct.unpack_from("<Q", buf, off)[0]
    return buf[off + 8:off + 8 + n].decode(), off + 8 + n


def read_value(buf, off, kind):
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
           7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    if kind in fmt:
        size = struct.calcsize(fmt[kind])
        return struct.unpack_from(fmt[kind], buf, off)[0], off + size
    if kind == 8:
        return read_str(buf, off)
    if kind == 9:
        etype = struct.unpack_from("<I", buf, off)[0]
        count = struct.unpack_from("<Q", buf, off + 4)[0]
        off += 12
        for _ in range(count):
            _, off = read_value(buf, off, etype)
        return None, off
    raise ValueError(f"metadata type {kind}")


def parse_header(buf):
    assert buf[:4] == b"GGUF"
    _, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    off = 24
    for _ in range(n_kv):
        _, off = read_str(buf, off)
        kind = struct.unpack_from("<I", buf, off)[0]
        _, off = read_value(buf, off + 4, kind)
    tensors = {}
    for _ in range(n_tensors):
        name, off = read_str(buf, off)
        ndim = struct.unpack_from("<I", buf, off)[0]
        off += 4
        dims = [struct.unpack_from("<Q", buf, off + 8 * i)[0] for i in range(ndim)]
        off += 8 * ndim
        kind, toff = struct.unpack_from("<IQ", buf, off)
        off += 12
        tensors[name] = (dims, kind, toff)
    return tensors, (off + 31) & ~31


def to_bf16(f32):
    u = f32.astype(np.float32, copy=False).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def tensor_bytes(dims, kind):
    """GGUF storage bytes for the whole tensor."""
    n = 1
    for d in dims:
        n *= d
    if kind == 0:
        return n * 4
    if kind == 30:
        return n * 2
    if kind == 2:
        return n // 32 * 18
    if kind == 8:
        return n // 32 * 34
    raise ValueError(f"tensor type {TENSOR_TYPES.get(kind, kind)} unsupported")


def dequant_chunk(buf, dims, kind, base, row0, row1):
    """Rows [row0,row1) of the dims[::-1]-shaped tensor as float32.

    dims[::-1] is [out..., in]: rows are contiguous spans of the trailing
    (in) dimension — one expert's matrices for slabs are contiguous too.
    """
    cols = dims[0]  # innermost ggml dim = stored row width
    n = (row1 - row0) * cols
    if kind == 0:
        off = base + row0 * cols * 4
        return np.frombuffer(buf, np.float32, n, off).astype(np.float32)
    if kind == 30:
        off = base + row0 * cols * 2
        u = np.frombuffer(buf, np.uint16, n, off).astype(np.uint32)
        return (u << 16).view(np.float32)
    blocks_per_row = cols // 32
    if kind == 2:
        off = base + row0 * blocks_per_row * 18
        rows = np.frombuffer(buf, np.uint8,
                             (row1 - row0) * blocks_per_row * 18, off)
        rows = rows.reshape(-1, 18)
        scales = rows[:, :2].copy().view(np.float16).astype(np.float32)
        q = rows[:, 2:]
        lo = (q & 0x0F).astype(np.int8) - 8
        hi = (q.astype(np.int8) >> 4) - 8
        vals = np.concatenate([lo, hi], axis=1).reshape(-1, 32)
        return (vals * scales).reshape(-1)
    if kind == 8:
        off = base + row0 * blocks_per_row * 34
        rows = np.frombuffer(buf, np.uint8,
                             (row1 - row0) * blocks_per_row * 34, off)
        rows = rows.reshape(-1, 34)
        scales = rows[:, :2].copy().view(np.float16).astype(np.float32)
        vals = rows[:, 2:].copy().view(np.int8).astype(np.float32)
        return (vals * scales).reshape(-1)
    raise ValueError(f"tensor type {TENSOR_TYPES.get(kind, kind)} unsupported")


def main():
    src = open(sys.argv[1], "rb")
    buf = mmap.mmap(src.fileno(), 0, access=mmap.ACCESS_READ)
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    tensors, data_start = parse_header(buf)

    header = {}
    offset = 0
    for name, (dims, kind, _) in sorted(tensors.items()):
        if name not in RENAME:
            continue
        shape = dims[::-1]
        size = 2
        for d in shape:
            size *= d
        header[RENAME[name]] = {"dtype": "BF16", "shape": shape,
                                "data_offsets": [offset, offset + size]}
        offset += size

    sink = open(out / "model.safetensors", "wb")
    meta = json.dumps(header).encode()
    sink.write(struct.pack("<Q", len(meta)) + meta)

    for name, (dims, kind, toff) in sorted(tensors.items()):
        if name not in RENAME:
            print(f"  skip {name}")
            continue
        shape = dims[::-1]
        rows = 1
        for d in shape[:-1]:
            rows *= d
        base = data_start + toff
        for row0 in range(0, rows, CHUNK_ROWS):
            f32 = dequant_chunk(buf, dims, kind, base, row0,
                                min(row0 + CHUNK_ROWS, rows))
            sink.write(to_bf16(f32).tobytes())
        print(f"  {name} {dims} {TENSOR_TYPES.get(kind, kind)}"
              f" -> {RENAME[name]} {shape}")
    sink.close()
    print(f"wrote {out / 'model.safetensors'}")


if __name__ == "__main__":
    main()
