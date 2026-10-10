#!/usr/bin/env python3
"""A/B RICHENGINE_SUBMIT_AHEAD: two concurrent cold prefills on Ornith.

The flag lets a second prefill's command overlap the running one's. With two
requests fired together, the queued request's TTFT is the metric.
"""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_real  # noqa: E402

MODEL = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit"
)

WORDS = (
    "anchor beacon cellar diagram ember fabric glacier harbor island "
    "jungle kernel ledger meadow needle orchard prairie quiver ridge "
    "saddle timber upland velvet willow xenon yield zephyr boulder"
).split()


def prompt(seed: int, sentences: int = 120) -> str:
    import random

    rng = random.Random(seed)
    parts = [
        f"Section {i}: the {rng.choice(WORDS)} report covers "
        f"{rng.randint(10, 99)} cases of {rng.choice(WORDS)} handling near "
        f"the {rng.choice(WORDS)} facility."
        for i in range(sentences)
    ]
    return "Summarize these notes in one line.\n\n" + "\n".join(parts)


class Args:
    package = PACKAGE
    binary = Path(__file__).resolve().parents[2] / "build/richengine"
    model = MODEL
    max_context = 16384
    max_memory = None
    max_cache_disk = None
    max_image_pixels = None
    kv_format = "int8"


def ttft(port: int, seed: int) -> float:
    status, body = smoke_real.request(
        port,
        "POST",
        "/v1/chat/completions",
        smoke_real.chat_body(MODEL, prompt(seed), max_completion_tokens=8),
        timeout=300,
    )
    assert status == 200, body
    return body["metrics"]["request_latency"]["ttft_ms"]


def pair(env: dict | None, rep: int) -> tuple[float, float]:
    server = smoke_real.RealServer(Args(), environment=env)
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            a = pool.submit(ttft, server.port, 1000 + rep)
            b = pool.submit(ttft, server.port, 2000 + rep)
            return a.result(), b.result()
    finally:
        server.close()


def main() -> int:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for label, env in [
        ("off", {}),
        ("on", {"RICHENGINE_SUBMIT_AHEAD": "1"}),
        ("on", {"RICHENGINE_SUBMIT_AHEAD": "1"}),
        ("off", {}),
    ]:
        times = [pair(env, r) for r in range(reps)]
        first = sorted(t[0] for t in times)
        second = sorted(max(t) for t in times)
        print(
            f"{label}: lead ttft median {first[len(first) // 2]:.0f} ms, "
            f"queued ttft median {second[len(second) // 2]:.0f} ms "
            f"({[f'{max(t):.0f}' for t in times]})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
