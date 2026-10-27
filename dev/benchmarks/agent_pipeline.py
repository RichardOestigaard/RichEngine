#!/usr/bin/env python3
"""Agentic pipeline simulation: a main agent spawns sub-agents to read
files, then synthesizes — the real multi-agent traffic shape.

Phases:
  A. sequential  — main turn, then 3 sub-agent reads run one at a time,
                   then the main synthesis turn (opencode's actual shape).
  B. concurrent  — same work, but the 3 sub-agent reads run in parallel
                   threads (packed decode lanes).

Measures aggregate decode tok/s from /status counter deltas
(decode_output_tokens / decode_wall_ms), plus wall tok/s and the
batch-width histogram.

Usage:
  .venv/bin/python dev/benchmarks/agent_pipeline.py [BINARY] [MODEL]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
        import os
        self.kv_format = os.environ.get("KV_FORMAT", "int8")


def chat(port, model, prompt, tokens=256):
    body = smoke_real.chat_body(
        model, "", max_completion_tokens=tokens, temperature=0
    )
    if "LFM" in model:
        body.pop("reasoning_effort", None)
    body["messages"] = [{"role": "user", "content": prompt}]
    s, doc = smoke_real.request(port, "POST", "/v1/chat/completions", body, timeout=300)
    assert s == 200, doc
    return doc


def counters(port):
    s, d = smoke_real.request(port, "GET", "/status")
    if s != 200:
        return None
    sched = d.get("scheduler", {})
    metrics = d.get("metrics", {})
    return {
        "widths": sched.get("decode_batches_by_width", {}),
        "decode_tokens": metrics.get("decode_output_tokens", 0),
        "decode_wall_ms": metrics.get("decode_wall_ms", 0),
        "prefill_tokens": metrics.get("prefill_input_tokens",
                                    metrics.get("prefill_tokens", 0)),
    }


def delta(before, after):
    widths = {}
    for key, count in after["widths"].items():
        widths[key] = count - before["widths"].get(key, 0)
    tokens = after["decode_tokens"] - before["decode_tokens"]
    wall_ms = after["decode_wall_ms"] - before["decode_wall_ms"]
    prefill = after["prefill_tokens"] - before["prefill_tokens"]
    multi = sum(v for k, v in widths.items() if k != "b1" and k != "1")
    return {
        "batches": widths,
        "decode_tokens": tokens,
        "decode_tok_s": round(1000 * tokens / wall_ms, 1) if wall_ms else 0,
        "prefill_tokens": prefill,
        "multi_lane_share": round(multi / sum(widths.values()), 2) if sum(widths.values()) else 0,
    }


def file_prompt(name, size_fns):
    """A 'read this file' sub-agent prompt: real-shaped code body."""
    lines = "\n".join(
        f"def {name}_{i}(ctx):\n"
        f"    row = ctx.query('SELECT * FROM {name} WHERE id = {{}}'.format({i}))\n"
        f"    if row.status == 'active':\n"
        f"        ctx.emit('event_{i}', row.payload)\n"
        f"    return row.as_dict()"
        for i in range(size_fns)
    )
    return (
        f"You read {name}.py. Summarize its structure: entry points, "
        f"side effects, and anything that looks wrong.\n\n```python\n{lines}\n```"
    )


def main():
    binary = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "build/richengine"
    model = sys.argv[2] if len(sys.argv) > 2 else MODEL
    n_sub = 4
    server = smoke_real.RealServer(Args(binary, model))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        chat(server.port, model, "Say ready.", tokens=8)

        main_prompt = (
            "You are the orchestrator. Three modules need review: "
            "parser.py, store.py, router.py, cache.py. Plan: spawn one reader per "
            "module, collect their summaries, then write a combined report. "
            "Emit only the plan."
        )
        subs = [file_prompt(name, 90) for name in ("parser", "store", "router", "cache")]
        synth_prompt = (
            "The readers reported: parser is clean but emits events on the "
            "hot path; store leaks connections on early return; router "
            "double-registers routes on reload. Write the combined report "
            "with severity ranking and concrete fixes."
        )

        out = {"model": model, "sub_agents": n_sub}

        # ---- Phase A: fully sequential (opencode's real shape) ----
        before = counters(server.port)
        t0 = time.monotonic()
        tokens = 0
        tokens += chat(server.port, model, main_prompt, 512)["usage"]["completion_tokens"]
        for prompt in subs:
            tokens += chat(server.port, model, prompt, 512)["usage"]["completion_tokens"]
        tokens += chat(server.port, model, synth_prompt, 512)["usage"]["completion_tokens"]
        wall = time.monotonic() - t0
        after = counters(server.port)
        out["sequential"] = {
            "wall_s": round(wall, 2),
            "completion_tokens": tokens,
            "wall_tok_s": round(tokens / wall, 1),
            **delta(before, after),
        }

        # ---- Phase B: sub-agents concurrent (packed decode lanes) ----
        before = counters(server.port)
        t0 = time.monotonic()
        tokens = chat(server.port, model, main_prompt, 512)["usage"]["completion_tokens"]
        with ThreadPoolExecutor(max_workers=n_sub) as pool:
            tokens += sum(
                doc["usage"]["completion_tokens"]
                for doc in pool.map(lambda p: chat(server.port, model, p, 512), subs)
            )
        tokens += chat(server.port, model, synth_prompt, 512)["usage"]["completion_tokens"]
        wall = time.monotonic() - t0
        after = counters(server.port)
        out["concurrent_subs"] = {
            "wall_s": round(wall, 2),
            "completion_tokens": tokens,
            "wall_tok_s": round(tokens / wall, 1),
            **delta(before, after),
        }

        out["speedup"] = round(
            out["sequential"]["wall_s"] / out["concurrent_subs"]["wall_s"], 2
        )
        print(json.dumps(out))
    finally:
        server.close()


if __name__ == "__main__":
    main()
