#!/usr/bin/env python3
"""Exercise RICHENGINE_NGRAM_PREDRAFT against a live server.

A prompt that asks for a verbatim echo of a repeated block produces output
whose closing 3-grams already occurred in the prompt, so the n-gram table
should inject proposals on nearly every decode step. Losslessness is checked
by comparing the completion text against the same request without the flag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_real  # noqa: E402

MODEL = "incoai/Qwen3.8-27B-RichEngine"
PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "install/models/incoai/Qwen3.8-27B-RichEngine"
)
BINARY = Path(__file__).resolve().parents[2] / "build/richengine"

BLOCK = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    * 8
)
PROMPT = (
    "Repeat the following text verbatim, exactly once, and output nothing "
    "else:\n\n" + BLOCK
)


class Args:
    def __init__(self):
        self.package = PACKAGE
        self.binary = BINARY
        self.model = MODEL
        self.max_context = 8192
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"
        self.startup_timeout = 600


def complete(port: int) -> str:
    status, body = smoke_real.request(
        port,
        "POST",
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 96,
            "temperature": 0,
        },
    )
    assert status == 200, body
    message = body["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if content is None:
        content = json.dumps(body["choices"][0])
    return content


def run(env):
    args = Args()
    server = smoke_real.RealServer(args, environment=env)
    try:
        smoke_real.validate_status(
            server.wait_ready(600), args.kv_format
        )
        text = complete(server.port)
        log = server.tail()
        hits = sum(1 for line in log.splitlines() if "ngram-predraft" in line)
        return text, hits, log
    finally:
        server.close()


def main() -> int:
    base, base_hits, _ = run({})
    print(f"baseline chars={len(base)} ngram-hits={base_hits}", flush=True)
    flagged, hits, log = run(
        {"RICHENGINE_NGRAM_PREDRAFT": "1", "RICHENGINE_NGRAM_DEBUG": "1"}
    )
    print(f"flagged  chars={len(flagged)} ngram-hits={hits}", flush=True)
    if flagged != base:
        print("MISMATCH: outputs differ", file=sys.stderr)
        print("baseline:", base[:400], file=sys.stderr)
        print("flagged :", flagged[:400], file=sys.stderr)
        return 1
    if hits == 0:
        print("WARN: n-gram never injected on an echo prompt", file=sys.stderr)
        return 2
    print("PASS: identical output with n-gram injections", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
