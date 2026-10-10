#!/usr/bin/env python3
"""A/B the draft sliding window on Ornith-1.5-9B via two built binaries.

Each run starts a server, then measures greedy decodes whose context sits
past the small ring: a ~6K-token prompt and a 192-token completion. Reports
decode ms/token and draft acceptance from /status metric deltas.

  .venv/bin/python dev/tests/ornith_window_ab.py BINARY_A BINARY_B [SAMPLES]
"""

from __future__ import annotations

import random as _random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_real

MODEL = "ornith-ai/Ornith-1.5-9B-MLX-4bit"
PACKAGE = (
    Path(__file__).resolve().parents[2]
    / "install/models/ornith-ai/Ornith-1.5-9B-MLX-4bit"
)
ROOT = Path(__file__).resolve().parents[2]

# ~6K tokens of varied natural text so the decode position sits well past a
# 2048-slot ring. Sentences are distinct (repeated blocks would make every
# n-gram match ambiguous and confound draft quality).
_words = (
    "river market committee engine harvest signal border orchard pilot "
    "ledger fabric signal canvas harbor meadow beacon cellar diagram "
    "vertex pillar margin copper lantern prairie tunnel solder anchor"
).split()
_rng = _random.Random(20261006)
_sents = []
for _i in range(180):
    w = _rng.sample(_words, 6)
    _sents.append(
        f"In week {_i}, the {w[0]} team reviewed the {w[1]} ledger and "
        f"shipped {_rng.randint(3, 97)} units of {w[2]} to the {w[3]} "
        f"district before the {w[4]} audit flagged the {w[5]} account."
    )
PROMPT = (
    "Read the following notes, then continue them in the same style for "
    "the requested length.\n\n" + " ".join(_sents)
)


class Args:
    def __init__(self, binary: Path):
        self.package = PACKAGE
        self.binary = binary
        self.model = MODEL
        self.max_context = 16384
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"
        self.startup_timeout = 600


def sample(port: int) -> dict:
    _, before = smoke_real.request(port, "GET", "/status")
    status, body = smoke_real.request(
        port,
        "POST",
        "/v1/chat/completions",
        smoke_real.chat_body(MODEL, PROMPT, max_completion_tokens=192),
        timeout=300,
    )
    assert status == 200, body
    _, after = smoke_real.request(port, "GET", "/status")
    keys = [
        "decode_wall_ms",
        "decode_output_tokens",
        "drafted_tokens",
        "accepted_draft_tokens",
    ]
    delta = {k: after["metrics"][k] - before["metrics"][k] for k in keys}
    delta["prompt_tokens"] = body["usage"]["prompt_tokens"]
    delta["completion_tokens"] = body["usage"]["completion_tokens"]
    return delta


def run(binary: Path, samples: int) -> list[dict]:
    server = smoke_real.RealServer(Args(binary))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        return [sample(server.port) for _ in range(samples)]
    finally:
        server.close()


def report(label: str, rows: list[dict]) -> None:
    for i, r in enumerate(rows):
        ms_per_tok = r["decode_wall_ms"] / max(r["decode_output_tokens"], 1)
        acc = r["accepted_draft_tokens"] / max(r["drafted_tokens"], 1)
        print(
            f"{label} s{i}: prompt={r['prompt_tokens']} "
            f"out={r['completion_tokens']} {ms_per_tok:.2f} ms/tok "
            f"({1000 / ms_per_tok:.1f} tok/s) acceptance={acc:.3f}",
            flush=True,
        )
    med = sorted(r["decode_wall_ms"] / max(r["decode_output_tokens"], 1) for r in rows)[
        len(rows) // 2
    ]
    print(f"{label} median: {1000 / med:.1f} tok/s", flush=True)


def main() -> int:
    binaries = [Path(sys.argv[1]), Path(sys.argv[2])]
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    results = {}
    # ABBA order cancels drift between the two halves of the run.
    order = [0, 1, 1, 0]
    for leg, index in enumerate(order):
        rows = run(binaries[index].resolve(), samples)
        results.setdefault(index, []).extend(rows)
        report(f"bin{index} leg{leg}", rows)
    for index, rows in results.items():
        report(f"bin{index} ALL", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
