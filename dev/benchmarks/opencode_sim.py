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
        model, "", max_completion_tokens=256, temperature=0.7, **body_extra
    )
    return smoke_real.request(
        port, "POST", "/v1/chat/completions", body, timeout=timeout
    )


def batches(port):
    status, doc = smoke_real.request(port, "GET", "/status")
    assert status == 200
    return dict(doc["scheduler"]["decode_batches_by_width"])


def batch_delta(before, after):
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


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
            "You are a coding agent. Propose an alternative implementation "
            "plan for adding retry logic to an HTTP client: distinct "
            "approaches, trade-offs, then pick one."
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
            {"messages": [{"role": "user", "content": prompt}], "n": n},
        )
        packed_s = time.monotonic() - t0
        assert status == 200, doc
        packed_completion = doc["usage"]["completion_tokens"]
        packed_batches = batch_delta(before, batches(server.port))
        packed_lanes = sum(
            int(k[1:]) * v
            for k, v in packed_batches.items()
            if k.startswith("b")
        )
        packed_total = sum(packed_batches.values())

        print(
            json.dumps(
                {
                    "model": model,
                    "n": n,
                    "serial": {
                        "wall_s": round(serial_s, 2),
                        "completion_tokens": serial_completion,
                        "tok_s": round(serial_completion / serial_s, 1),
                        "batches": serial_batches,
                    },
                    "packed_n": {
                        "wall_s": round(packed_s, 2),
                        "completion_tokens": packed_completion,
                        "tok_s": round(packed_completion / packed_s, 1),
                        "batches": packed_batches,
                        "multi_lane_share": round(
                            1 - packed_batches.get("b1", 0)
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
