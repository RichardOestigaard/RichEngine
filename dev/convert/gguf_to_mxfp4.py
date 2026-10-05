"""Convert an F16/BF16 GGUF to an MXFP4-target GGUF RichEngine can serve.

Rewrites every tensor whose slot accepts a quantized format
(install/gguf.py's loaded_tensors) as GGML type 39 (MXFP4): 17-byte blocks
of one E8M0 group scale and 32 E2M1 nibbles, elements 0-15 the low nibbles
of qs and 16-31 the high (the layout GgufFormatReference.hpp reads).
token_embd.weight becomes Q8_0 (embedding formats exclude MXFP4); every
other tensor is widened to F32, which the norm, conv and router slots
accept, or copied through when already F32.

Usage: gguf_to_mxfp4.py SOURCE.gguf DEST.gguf
"""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from install import gguf as install_gguf

F32, F16, BF16, Q8_0, MXFP4 = 0, 1, 30, 8, 39

# E2M1 magnitudes and the boundaries rounding to them (midpoints).
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MID = (E2M1[:-1] + E2M1[1:]) / 2

SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_kv(raw, off):
    """A key/value entry's key, value type and raw serialized value bytes."""
    (klen,) = struct.unpack_from("<Q", raw, off)
    off += 8
    key = raw[off : off + klen].tobytes().decode()
    off += klen
    (vtype,) = struct.unpack_from("<I", raw, off)
    off += 4
    start = off
    if vtype == 8:  # string
        (vlen,) = struct.unpack_from("<Q", raw, off)
        off += 8 + vlen
    elif vtype == 9:  # array
        etype, count = struct.unpack_from("<IQ", raw, off)
        off += 12
        if etype == 8:  # array of strings
            for _ in range(count):
                (slen,) = struct.unpack_from("<Q", raw, off)
                off += 8 + slen
        else:
            off += count * SCALAR_SIZES[etype]
    else:
        off += SCALAR_SIZES[vtype]
    return key, vtype, raw[start:off].tobytes(), off


def parse(path):
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    assert raw[:4].tobytes() == b"GGUF", "not a GGUF"
    version, ntensors, nkv = struct.unpack_from("<IQQ", raw, 4)
    assert version == 3
    off = 24
    kvs = []
    for _ in range(nkv):
        key, vtype, value, off = read_kv(raw, off)
        kvs.append((key, vtype, value))
    infos = []
    for _ in range(ntensors):
        (nlen,) = struct.unpack_from("<Q", raw, off)
        off += 8
        name = raw[off : off + nlen].tobytes().decode()
        off += nlen
        (ndim,) = struct.unpack_from("<I", raw, off)
        off += 4
        dims = list(struct.unpack_from(f"<{ndim}Q", raw, off))
        off += 8 * ndim
        ttype, toff = struct.unpack_from("<IQ", raw, off)
        off += 12
        infos.append({"name": name, "dims": dims, "type": ttype, "offset": toff})
    alignment = 32
    for key, _, value in kvs:
        if key == "general.alignment":
            (alignment,) = struct.unpack("<I", value)
    data_start = (off + alignment - 1) // alignment * alignment
    return raw, kvs, infos, data_start, alignment


def tensor_data(raw, info, data_start):
    count = int(np.prod(info["dims"]))
    begin = data_start + info["offset"]
    if info["type"] == F32:
        return np.frombuffer(raw, dtype="<f4", count=count, offset=begin).copy()
    if info["type"] == F16:
        return (
            np.frombuffer(raw, dtype="<f2", count=count, offset=begin)
            .astype(np.float32)
        )
    if info["type"] == BF16:
        u = np.frombuffer(raw, dtype="<u2", count=count, offset=begin)
        return (u.astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"unreadable tensor type {info['type']} for {info['name']}")


def quantize_mxfp4(w):
    """w [.., K] float32 -> 17-byte blocks per 32 elements, K % 32 == 0."""
    w = w.reshape(-1, 32).astype(np.float64)
    amax = np.abs(w).max(axis=1)
    # Group exponent: try the floor and ceil of log2(amax/6), keep the lower
    # squared error per group.
    with np.errstate(divide="ignore"):
        e0 = np.floor(np.log2(amax / 6.0))
    e0 = np.where(np.isfinite(e0), e0, -127.0)
    best_e, best_err = np.clip(e0, -127.0, 127.0), None
    for e in (np.clip(e0, -127.0, 127.0), np.clip(e0 + 1.0, -127.0, 127.0)):
        d = np.exp2(e)[:, None]
        code = np.searchsorted(E2M1_MID, np.abs(w) / d)
        err = ((E2M1[code] * d - np.abs(w)) ** 2).sum(axis=1)
        if best_err is None:
            best_e, best_err = e, err
        else:
            better = err < best_err
            best_e, best_err = np.where(better, e, best_e), np.where(better, err, best_err)
    e8m0 = np.where(amax == 0, 0, (best_e + 127).astype(np.int64)).astype(np.uint8)
    d = np.exp2(best_e)[:, None]
    code = np.searchsorted(E2M1_MID, np.abs(w) / d).astype(np.uint8)
    code |= (w < 0).astype(np.uint8) << 3
    qs = code[:, :16] | (code[:, 16:] << 4)
    return np.concatenate([e8m0[:, None], qs], axis=1).astype(np.uint8).tobytes()


def quantize_q8_0(w):
    """w [.., K] float32 -> ggml block_q8_0 rows: half d, 32 int8."""
    w = w.reshape(-1, 32)
    amax = np.abs(w).max(axis=1)
    d = (amax / 127.0).astype("<f2")
    df = d.astype(np.float32)
    q = np.where(df[:, None] != 0, np.round(w / df[:, None]), 0.0)
    q = np.clip(q, -128, 127).astype(np.int8)
    blocks = np.empty((w.shape[0], 34), dtype=np.uint8)
    blocks[:, :2] = d.view(np.uint8).reshape(-1, 2)
    blocks[:, 2:] = q.view(np.uint8)
    return blocks.tobytes()


def convert(source, dest):
    raw, kvs, infos, data_start, alignment = parse(source)
    slots = install_gguf.loaded_tensors(install_gguf.Metadata(source))
    quantized = {n for n, types in slots.items() if "MXFP4" in types}
    embeddable = {
        n for n, types in slots.items() if "Q8_0" in types and "MXFP4" not in types
    }

    out_infos = []
    blob = bytearray()
    cursor = 0
    for info in infos:
        name, dims = info["name"], info["dims"]
        count = int(np.prod(dims))
        if name in embeddable:
            new_type, data = Q8_0, quantize_q8_0(tensor_data(raw, info, data_start))
        elif name in quantized and len(dims) >= 2 and dims[0] % 32 == 0:
            new_type, data = MXFP4, quantize_mxfp4(tensor_data(raw, info, data_start))
        elif info["type"] in (F16, BF16):
            new_type, data = F32, tensor_data(raw, info, data_start).astype("<f4").tobytes()
        elif info["type"] == F32:
            begin = data_start + info["offset"]
            new_type, data = F32, raw[begin : begin + count * 4].tobytes()
        else:
            raise ValueError(f"unsupported source type {info['type']} for {name}")
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
            struct.pack("<Q", len(key)) + key.encode()
            + struct.pack("<I", vtype) + value
        )
    info_bytes = [
        struct.pack("<Q", len(name)) + name.encode()
        + struct.pack("<I", len(dims))
        + struct.pack(f"<{len(dims)}Q", *dims)
        + struct.pack("<IQ", ttype, offset)
        for name, dims, ttype, offset in out_infos
    ]

    # Tensor offsets are relative to the data section start.
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
