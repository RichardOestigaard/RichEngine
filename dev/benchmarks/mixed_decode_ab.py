#!/usr/bin/env python3
"""Mixed constrained/free decode burst: aggregate tok/s across one binary.

Fires CONSTRAINED + FREE simultaneous chat requests -- half under a JSON
schema that cannot close early, half free text with ignore_eos -- and diffs
GET /status decode counters around the burst. Usage:

  python3 dev/benchmarks/mixed_decode_ab.py [BINARY] [REPS] [TOKENS] [MODEL]

BINARY defaults to build/richengine; pass another binary's path for A/B.
MODEL defaults to openbmb/MiniCPM5-2B-MLX.
"""

from __future__ import annotations

import concurrent.futures
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "openbmb/MiniCPM5-2B-MLX"
PACKAGE = ROOT / "install/models/openbmb/MiniCPM5-2B-MLX"

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "records",
        "schema": {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "minItems": 40,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "integer"},
                            "flag": {"type": "boolean"},
                        },
                        "required": ["name", "value", "flag"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["records"],
            "additionalProperties": False,
        },
    },
}


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


def free_body(model: str, tokens: int, seed: int) -> dict:
    return smoke_real.chat_body(
        model,
        f"Write a long continuous story about expedition {seed}. "
        "Do not stop early.",
        max_completion_tokens=tokens,
        ignore_eos=True,
    )


def constrained_body(model: str, tokens: int) -> dict:
    return smoke_real.chat_body(
        model,
        "List forty fictional city records.",
        max_completion_tokens=tokens,
        response_format=SCHEMA,
    )


def one(port: int, body: dict) -> dict:
    status, document = smoke_real.request(
        port, "POST", "/v1/chat/completions", body, timeout=600
    )
    assert status == 200, document
    return document


def status_metrics(port: int) -> dict:
    status, document = smoke_real.request(port, "GET", "/status")
    assert status == 200, document
    metrics = dict(document.get("metrics", {}))
    metrics["decode_batches"] = document.get("scheduler", {}).get(
        "decode_batches", 0
    )
    return metrics


def burst(
    port: int, model: str, tokens: int, rep: int
) -> tuple[float, float, float]:
    """Two constrained and two free requests at once.

    Returns (aggregate tok/s from the engine's decode counters, effective
    tok/s over the burst's wall span, decode batches delta)."""
    before = status_metrics(port)
    bodies = [
        constrained_body(model, tokens),
        constrained_body(model, tokens),
        free_body(model, tokens, 2 * rep),
        free_body(model, tokens, 2 * rep + 1),
    ]
    import time

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(one, port, body) for body in bodies]
        documents = [future.result() for future in futures]
    elapsed = time.monotonic() - started
    after = status_metrics(port)
    wall_ms = after["decode_wall_ms"] - before["decode_wall_ms"]
    tokens_out = (
        after["decode_output_tokens"] - before["decode_output_tokens"]
    )
    batches = after.get("decode_batches", 0) - before.get("decode_batches", 0)
    generated = sum(d["usage"]["completion_tokens"] for d in documents)
    return (
        1000.0 * tokens_out / max(wall_ms, 1e-9),
        generated / elapsed,
        float(batches),
    )


def main() -> int:
    binary = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build/richengine"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    model = sys.argv[4] if len(sys.argv) > 4 else MODEL
    server = smoke_real.RealServer(Args(binary, model))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        # Warmup burst compiles the constraint's mask path.
        burst(server.port, model, 64, 0)
        engine, effective, batches = [], [], []
        for rep in range(reps):
            e, f, b = burst(server.port, model, tokens, rep + 1)
            engine.append(e)
            effective.append(f)
            batches.append(b)
            print(
                f"rep {rep + 1}: engine {e:.1f} tok/s, "
                f"effective {f:.1f} tok/s, decode batches {b:.0f}",
                flush=True,
            )
        print(
            f"{binary}: engine median {statistics.median(engine):.1f} tok/s, "
            f"effective median {statistics.median(effective):.1f} tok/s, "
            f"median batches {statistics.median(batches):.0f}"
        )
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
