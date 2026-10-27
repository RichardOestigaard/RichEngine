#!/usr/bin/env python3
"""Opencode traffic simulation: measures what an n=1-sequential agent loop
costs versus one n=N request packed into a single decode batch.

Phase A (the opencode shape): TURNS sequential chat requests, each resending
the full history — the agentic turn loop as real clients drive it.
Phase B: one request with n=TURNS — the same count of completions, N lanes
packed per decode step.

Prints JSON: per-phase wall time, decode batch-width histogram, tok/s, and
the packing ratio — how much of phase B's decode ran multi-lane.

Usage:
  .venv/bin/python dev/benchmarks/opencode_sim.py [BINARY] [N] [MODEL]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"


class Args:
    def __init__(self, binary: Path, model: str):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = 16384
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def chat(port, model, body_extra, timeout=600):
    body = smoke_real.chat_body(
        model, "", max_completion_tokens=256, temperature=0, **body_extra
    )
    return smoke_real.request(
        port, "POST", "/v1/chat/completions", body, timeout=timeout
    )


def batches(port):
    status, doc = smoke_real.request(port, "GET", "/status")
    assert status == 200
    m = doc["metrics"]
    return {
        "widths": dict(doc["scheduler"]["decode_batches_by_width"]),
        "decode_tokens": m["decode_output_tokens"],
        "decode_wall_ms": m["decode_wall_ms"],
    }


def batch_delta(before, after):
    widths = {
        k: int(after["widths"].get(k, 0)) - int(before["widths"].get(k, 0))
        for k in after["widths"]
    }
    return {
        "widths": widths,
        "decode_tokens": after["decode_tokens"] - before["decode_tokens"],
        "decode_wall_ms": after["decode_wall_ms"] - before["decode_wall_ms"],
        "decode_tok_s": round(
            1000
            * (after["decode_tokens"] - before["decode_tokens"])
            / max(after["decode_wall_ms"] - before["decode_wall_ms"], 1e-9),
            1,
        ),
    }


def main() -> int:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/richengine")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    model = sys.argv[3] if len(sys.argv) > 3 else MODEL
    server = smoke_real.RealServer(Args(binary, model))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        smoke_real.request(
            server.port, "POST", "/v1/chat/completions",
            smoke_real.chat_body(model, "Say ready.", max_completion_tokens=8),
            timeout=120,
        )
        prompt = (
            "Here is module http_client.py:\n\n```python\n"
            "def request(url, timeout=30):\n"
            "    response = urllib.request.urlopen(url, timeout=timeout)\n"
            "    return response.read()\n\n"
            "def get_json(url):\n"
            "    return json.loads(request(url))\n\n"
            "def post(url, data):\n"
            "    body = json.dumps(data).encode()\n"
            "    req = urllib.request.Request(url, data=body)\n"
            "    return request(req)\n"
            "```\n\n"
            "Rewrite the complete file adding retry logic: three attempts, "
            "exponential backoff starting at 100ms, retry only on transient "
            "HTTP status codes. Emit the whole file in one code block."
        )
        # Phase B prompt re-emits a larger file: the agentic edit-loop shape
        # the draft predicts best.
        pack_prompt = (
            "Here is module pipeline.py — emit it verbatim, unchanged, in one "
            "code block:\n\n```python\n"
            + "\n".join(
                f"def stage_{i}(items):\n"
                f"    return [x * {i} for x in items if x > {i}]"
                for i in range(12)
            )
            + "\n```"
        )

        # Phase A — opencode's shape: N sequential single-choice requests.
        before = batches(server.port)
        t0 = time.monotonic()
        serial_completion = 0
        for i in range(n):
            status, doc = chat(
                server.port, model,
                {"messages": [{"role": "user", "content":
                              prompt + f" Approach {i + 1}."}]},
            )
            assert status == 200, doc
            serial_completion += doc["usage"]["completion_tokens"]
        serial_s = time.monotonic() - t0
        serial_batches = batch_delta(before, batches(server.port))

        # Phase B — one n=N request: N lanes packed per decode step.
        before = batches(server.port)
        t0 = time.monotonic()
        status, doc = chat(
            server.port, model,
            {"messages": [{"role": "user", "content": pack_prompt}], "n": n},
        )
        packed_s = time.monotonic() - t0
        assert status == 200, doc
        packed_completion = doc["usage"]["completion_tokens"]
        packed_batches = batch_delta(before, batches(server.port))
        packed_total = sum(packed_batches["widths"].values())
        print(
            json.dumps(
                {
                    "model": model,
                    "n": n,
                    "serial": {
                        "wall_s": round(serial_s, 2),
                        "completion_tokens": serial_completion,
                        "tok_s": round(serial_completion / serial_s, 1),
                        "engine_decode_tok_s": serial_batches["decode_tok_s"],
                        "batches": serial_batches["widths"],
                    },
                    "packed_n": {
                        "wall_s": round(packed_s, 2),
                        "completion_tokens": packed_completion,
                        "tok_s": round(packed_completion / packed_s, 1),
                        "engine_decode_tok_s": packed_batches["decode_tok_s"],
                        "batches": packed_batches["widths"],
                        "multi_lane_share": round(
                            1 - packed_batches["widths"].get("b1", 0)
                            / max(packed_total, 1),
                            2,
                        ),
                    },
                    "speedup": round(serial_s / packed_s, 2),
                    "choices_distinct": len(
                        {c["message"]["content"] for c in doc["choices"]}
                    ),
                }
            )
        )
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
