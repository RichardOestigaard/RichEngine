#!/usr/bin/env python3
"""Shared-prefix dedupe check for n>1 siblings: one request with n=4 over a
long prompt. If sibling lanes wait on the primary's prefill (WaitingPrefix),
prefill counters advance ~prompt_tokens once, not 4x.

Prints JSON: prefill token delta, per-sibling cache hits, wall time.
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
    def __init__(self, binary: Path):
        self.package = ROOT / "install/models" / MODEL
        self.binary = binary
        self.model = MODEL
        self.max_context = 16384
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def status_doc(port):
    s, d = smoke_real.request(port, "GET", "/status")
    return d if s == 200 else {}


def prefill_tokens(port):
    doc = status_doc(port)
    m = doc.get("metrics", {})
    return m.get("prefill_input_tokens", m.get("prefill_tokens", 0))


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    server = smoke_real.RealServer(Args(binary))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        smoke_real.request(
            server.port, "POST", "/v1/chat/completions",
            smoke_real.chat_body(MODEL, "Say ready.", max_completion_tokens=8),
            timeout=120,
        )

        # ~12k-token prompt: a big code file to rewrite.
        lines = "\n".join(
            f"def handler_{i}(request):\n"
            f"    value = request.get('field_{i}')\n"
            f"    if value is None:\n"
            f"        return error('missing field_{i}')\n"
            f"    return process_{i}(value)"
            for i in range(280)
        )
        prompt = f"Here is api.py — add docstrings to every function and emit the whole file in one code block:\n\n```python\n{lines}\n```"

        body = smoke_real.chat_body(
            MODEL, "", max_completion_tokens=64, temperature=0
        )
        body["messages"] = [{"role": "user", "content": prompt}]
        body["n"] = n

        before = prefill_tokens(server.port)
        t0 = time.monotonic()
        status_code, doc = smoke_real.request(
            server.port, "POST", "/v1/chat/completions", body, timeout=300
        )
        wall = time.monotonic() - t0
        after = prefill_tokens(server.port)
        assert status_code == 200, doc
        resp = doc
        usage = resp.get("usage", {})
        print(json.dumps({
            "n": n,
            "wall_s": round(wall, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "prefill_counter_delta": after - before,
            "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
            "choices": len(resp.get("choices", [])),
            "prompt_tokens_sum_hint": usage.get("prompt_tokens"),
        }))
    finally:
        server.close()


if __name__ == "__main__":
    main()
