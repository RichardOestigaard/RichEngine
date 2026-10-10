"""Copy a GGUF with the MoE expert tensors (ffn_*_exps.weight) re-quantized
to Q3_K — the deepest expert format RichEngine's staged tile reads that a
plain encoder can write (the IQ lattices need importance matrices).

Q3_K packs 256 elements into 110 bytes (~3.44 bits per element against
MXFP4's ~4.25 and Q4_K's ~4.5): 32 hmask bytes, 64 bytes of low 2 bits,
12 bytes of sixteen 6-bit scales, half d. Decodes as
d * (sc - 32) * (c2 - (hb1 ? 0 : 4)) per element, the layout
dev/tests/engine/GgufFormatReference.hpp's case Q3K reads.

Source tensors may be F32/F16/BF16 (via gguf_to_mxfp4's tensor_data) or
Q4_K/Q6_K/Q8_0, dequantized first — so the installed Q4_K_M file can be the
source. Every other tensor's bytes are copied through unchanged.

Usage: gguf_experts_q3k.py SOURCE.gguf DEST.gguf
"""

import struct
import sys

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/dev/", 1)[0])
from dev.convert import gguf_to_mxfp4 as base

F32, F16, BF16, Q8_0, Q3_K, Q4_K, Q6_K = 0, 1, 30, 8, 11, 12, 14


def f16(a):
    return a.copy().view("<f2").astype(np.float32)


def dequant_q4k(raw, count):
    """rows of 144-byte Q4_K blocks -> count float32, llama.cpp's mapping."""
    b = raw.reshape(-1, 144)
    d, dmin = f16(b[:, :2]), f16(b[:, 2:4])
    sc, qs = b[:, 4:16], b[:, 16:]
    out = np.empty((b.shape[0], 256), np.float32)
    for j in range(8):
        i0, i1 = (j, j + 4) if j < 4 else (j - 4, j + 4)
        if j < 4:
            s, m = sc[:, j] & 63, sc[:, j + 4] & 63
        else:
            s = (sc[:, j + 4] & 15) | ((sc[:, j - 4] >> 6) << 4)
            m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        q = qs[:, 32 * (j // 2) : 32 * (j // 2) + 32]
        out[:, 32 * j : 32 * j + 32] = (
            d[:, None] * s[:, None] * ((q >> (4 * (j % 2))) & 15)
            - dmin[:, None] * m[:, None]
        )
    return out.reshape(-1)[:count]


def dequant_q6k(raw, count):
    """rows of 210-byte Q6_K blocks -> count float32."""
    b = raw.reshape(-1, 210)
    ql, qh, scales, d = (
        b[:, :128],
        b[:, 128:192],
        b[:, 192:208].view(np.int8),
        f16(b[:, 208:]),
    )
    out = np.empty((b.shape[0], 256), np.float32)
    for j in range(8):
        n, r = j // 4, j % 4
        l4 = (
            ql[:, 64 * n + 32 * (r & 1) : 64 * n + 32 * (r & 1) + 32] >> (4 * (r >> 1))
        ) & 15
        h2 = (qh[:, 32 * n : 32 * n + 32] >> (2 * r)) & 3
        q6 = (l4 | (h2 << 4)).astype(np.int32) - 32
        sc = scales[:, 8 * n + 2 * r : 8 * n + 2 * r + 2]
        for half in range(2):
            out[:, 32 * j + 16 * half : 32 * j + 16 * half + 16] = (
                d[:, None] * sc[:, half : half + 1] * q6[:, 16 * half : 16 * half + 16]
            )
    return out.reshape(-1)[:count]


def dequant_q8_0(raw, count):
    b = raw.reshape(-1, 34)
    return (f16(b[:, :2])[:, None] * b[:, 2:].view(np.int8)).reshape(-1)[:count]


def tensor_floats(raw, info, data_start):
    count = int(np.prod(info["dims"]))
    begin = data_start + info["offset"]
    if info["type"] in (F32, F16, BF16):
        return base.tensor_data(raw, info, data_start)
    size = {Q8_0: 34, Q4_K: 144, Q6_K: 210}.get(info["type"])
    if size is None:
        raise ValueError(f"unreadable tensor type {info['type']} for {info['name']}")
    assert count % 256 == 0, f"{info['name']} is not a multiple of 256 elements"
    seg = np.frombuffer(raw, dtype=np.uint8, count=count // 256 * size, offset=begin)
    fn = {Q8_0: dequant_q8_0, Q4_K: dequant_q4k, Q6_K: dequant_q6k}[info["type"]]
    return fn(seg, count)


def quantize_q3k(w):
    """w multiple-of-256 float32 -> 110-byte Q3_K blocks."""
    g = w.reshape(-1, 256).astype(np.float64)
    nb = g.shape[0]
    subs = g.reshape(nb, 16, 16)
    m = np.abs(subs).max(axis=2)  # [nb][16] sub-block amaxes
    d = m.max(axis=1) / (4.0 * 32.0)
    safe = np.where(d > 0, d, 1.0)[:, None]
    sc = np.clip(np.round(m / 4.0 / safe), -32, 31)
    ds = np.clip(np.round(m / 4.0 / safe), -32, 31)[:, :, None]
    dd = np.where(d > 0, d, 0.0)[:, None, None]
    q = np.where(dd * ds != 0, np.round(subs / (dd * ds)), 0.0)
    q = np.clip(q, -4, 3).astype(np.int64)
    c2 = (q & 3).astype(np.uint8).reshape(nb, 256)
    hb1 = (q >= 0).reshape(nb, 8, 32)
    u = (sc + 32).astype(np.uint8)  # [nb][16] 6-bit scale fields

    hmask = np.zeros((nb, 32), np.uint8)
    for j in range(8):
        hmask |= hb1[:, j, :].astype(np.uint8) << j
    qs = np.zeros((nb, 64), np.uint8)
    for j in range(8):
        n, jj = j // 4, j % 4
        qs[:, 32 * n : 32 * n + 32] |= c2[:, 32 * j : 32 * j + 32] << (2 * jj)
    scales = np.zeros((nb, 12), np.uint8)
    for b in range(4):
        scales[:, b] = (u[:, b] & 15) | ((u[:, b + 8] & 15) << 4)
        scales[:, b + 4] = (u[:, b + 4] & 15) | ((u[:, b + 12] & 15) << 4)
        scales[:, b + 8] = (
            ((u[:, b] >> 4) & 3)
            | (((u[:, b + 4] >> 4) & 3) << 2)
            | (((u[:, b + 8] >> 4) & 3) << 4)
            | (((u[:, b + 12] >> 4) & 3) << 6)
        )
    block = np.empty((nb, 110), np.uint8)
    block[:, :32], block[:, 32:96], block[:, 96:108] = hmask, qs, scales
    block[:, 108:] = d.astype("<f2").view(np.uint8).reshape(nb, 2)
    return block.tobytes()


def convert(source, dest):
    raw, kvs, infos, data_start, alignment = base.parse(source)
    out_infos = []
    blob = bytearray()
    cursor = 0
    for info in infos:
        name, dims = info["name"], info["dims"]
        count = int(np.prod(dims))
        begin = data_start + info["offset"]
        if name.endswith("_exps.weight") and count % 256 == 0:
            new_type, data = Q3_K, quantize_q3k(tensor_floats(raw, info, data_start))
        else:
            size = {F32: 4, F16: 2, BF16: 2, Q8_0: 34, Q4_K: 144, Q6_K: 210}.get(
                info["type"]
            )
            if size is None:
                raise ValueError(f"cannot copy type {info['type']} of {name}")
            n = count // 256 * size if size > 8 else count * size
            new_type, data = info["type"], raw[begin : begin + n].tobytes()
        pad = (alignment - cursor % alignment) % alignment
        blob += b"\0" * pad
        cursor += pad
        out_infos.append((name, dims, new_type, cursor))
        blob += data
        cursor += len(data)
        print(f"{name}: {info['type']} -> {new_type} ({len(data)} bytes)", flush=True)

    head = [struct.pack("<4sIQQ", b"GGUF", 3, len(out_infos), len(kvs))]
    for key, vtype, value in kvs:
        head.append(
            struct.pack("<Q", len(key))
            + key.encode()
            + struct.pack("<I", vtype)
            + value
        )
    info_bytes = [
        struct.pack("<Q", len(name))
        + name.encode()
        + struct.pack("<I", len(dims))
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<IQ", ttype, offset)
        for name, dims, ttype, offset in out_infos
    ]
    pos = sum(len(h) for h in head) + sum(len(b) for b in info_bytes)
    data_begin = (pos + alignment - 1) // alignment * alignment
    with open(dest, "wb") as f:
        for h in head:
            f.write(h)
        for b in info_bytes:
            f.write(b)
        f.write(b"\0" * (data_begin - f.tell()))
        f.write(blob)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
