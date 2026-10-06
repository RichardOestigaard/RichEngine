#!/usr/bin/env python3
"""A/B an env flag on a Granite GGUF package: cold prefill TTFT and decode.

  .venv/bin/python dev/tests/granite_ab.py PACKAGE_DIR MODEL_ID FLAG [SAMPLES]

FLAG is given to the server as =1; the control legs run without it. ABBA
order cancels drift. Decode is measured from /status decode_wall_ms deltas.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_real  # noqa: E402
from submit_ahead_ab import prompt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class Args:
    max_context = 16384
    max_memory = None
    max_cache_disk = None
    max_image_pixels = None
    kv_format = "int8"

    def __init__(self, package: Path, model: str):
        self.package = package
        self.binary = ROOT / "build/richengine"
        self.model = model


def measure(port: int, model: str, seed: int, prompt_sentences: int,
            output_tokens: int) -> dict:
    _, before = smoke_real.request(port, "GET", "/status")
    status, body = smoke_real.request(
        port,
        "POST",
        "/v1/chat/completions",
        smoke_real.chat_body(
            model, prompt(seed, prompt_sentences),
            max_completion_tokens=output_tokens),
        timeout=300,
    )
    assert status == 200, body
    _, after = smoke_real.request(port, "GET", "/status")
    metrics = body["metrics"]
    delta = {
        "decode_wall_ms": after["metrics"]["decode_wall_ms"]
        - before["metrics"]["decode_wall_ms"],
        "decode_output_tokens": after["metrics"]["decode_output_tokens"]
        - before["metrics"]["decode_output_tokens"],
        "ttft_ms": metrics["request_latency"]["ttft_ms"],
        "prompt_tokens": body["usage"]["prompt_tokens"],
    }
    return delta


def leg(package: Path, model: str, env: dict, samples: int) -> list[dict]:
    server = smoke_real.RealServer(Args(package, model), environment=env)
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        return [measure(server.port, model, seed, 120, 64)
                for seed in range(samples)]
    finally:
        server.close()


def report(label: str, rows: list[dict]) -> None:
    decode = sorted(r["decode_wall_ms"] / max(r["decode_output_tokens"], 1)
                    for r in rows)
    ttft = sorted(r["ttft_ms"] for r in rows)
    print(f"{label}: decode {1000 / decode[len(decode)//2]:.1f} tok/s, "
          f"ttft {ttft[len(ttft)//2]:.0f} ms "
          f"(prompt {rows[0]['prompt_tokens']})", flush=True)


def main() -> int:
    package = Path(sys.argv[1])
    model = sys.argv[2]
    flag = sys.argv[3]
    samples = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    for label, env in [("off", {}), ("on", {flag: "1"}),
                       ("on", {flag: "1"}), ("off", {})]:
        rows = leg(package, model, env, samples)
        report(label, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
