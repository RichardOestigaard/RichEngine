#!/usr/bin/env python3
"""Aggregate decode throughput under agentic traffic: AGENTS parallel
multi-turn sessions, each carrying its conversation history until the
prompt reaches CTX tokens (~12k default), measuring the server's decode
counters around the burst.

Sessions mix the shapes agent workloads produce: code-edit loops that
re-emit a file, tool-argument JSON, document Q&A and self-refinement — no
constraints, so every lane rides the fused decode path.

Usage:
  .venv/bin/python dev/benchmarks/agentic_decode.py [BINARY] [AGENTS] [CTX] [MODEL]

BINARY defaults to build/richengine, AGENTS 4, CTX 12000,
MODEL ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit. Prints one JSON line:
aggregate decode tok/s (engine counters), wall tok/s, acceptance, batch
histogram, per-agent rounds.
"""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

CODE = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def quick_check(items):
    total = 0
    for it in items:
        total += it.price * it.qty
    return total


def parse_row(line):
    parts = line.split(",")
    return {"name": parts[0], "value": int(parts[1])}
"""

DOC = """The kakapo (Strigops habroptilus) is a large, flightless, nocturnal parrot
endemic to New Zealand. It can weigh up to 4 kg and live over 90 years.
Kakapo breed only in years when rimu trees mast heavily, sometimes just
once every four years. As of 2024 fewer than 250 individuals remain, each
named and tracked by conservation rangers on predator-free islands. They
nest on the ground under dense vegetation, and the female alone incubates
the eggs and feeds the chick for up to six months after hatching."""


class Args:
    def __init__(self, binary: Path, model: str, ctx: int):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = max(16384, ctx + 4096)
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def body(model: str, messages: list, max_tokens: int) -> dict:
    return smoke_real.chat_body(
        model,
        "",
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=0,
        ignore_eos=True,
    )


def agent(port: int, model: str, agent_id: int, ctx_cap: int) -> dict:
    """One session's turns, alternating agentic shapes until the prompt
    budget runs out."""
    system = (
        "You are an autonomous coding agent. Keep every answer concrete: "
        "complete code blocks or JSON, no apologies, no hedging."
    )
    prompts = [
        # Code edit loop: re-emit the file with a change.
        f"Here is the module file:\n\n```python\n{CODE}```\n\n"
        "Rewrite the whole file so bubble_sort takes a reverse=False "
        "argument. Emit the complete file only.",
        "Now make quick_check tolerate items missing .qty (treat as 1). "
        "Emit the complete file again.",
        # Tool-argument JSON: emit the call payload.
        "Emit a JSON object for a tool call `run_tests` with arguments "
        "`paths`, `verbose`, `timeout_seconds` covering the module above. "
        "JSON only.",
        # Document Q&A.
        f"Given this field report:\n\n{DOC}\n\nList five intervention "
        "priorities for the species, each one sentence.",
        # Self-refinement.
        "Look at your last answer. Rewrite it as a JSON array of objects "
        "{priority, rationale, effort_weeks}. JSON only.",
        # Back to code: the loop keeps the file in context.
        "Add a main() to the module that reads stdin lines, calls parse_row "
        "on each, and prints a JSON summary. Emit the complete file.",
    ]
    messages = [{"role": "system", "content": system}]
    rounds = 0
    tokens_in = 0
    tokens_out = 0
    for turn in range(48):
        messages.append({"role": "user", "content": prompts[turn % len(prompts)]})
        status, document = smoke_real.request(
            port,
            "POST",
            "/v1/chat/completions",
            body(model, messages, 384),
            timeout=600,
        )
        assert status == 200, document
        choice = document["choices"][0]
        usage = document["usage"]
        tokens_in = usage["prompt_tokens"]
        tokens_out += usage["completion_tokens"]
        rounds += 1
        if tokens_in >= ctx_cap:
            break
        messages.append(
            {"role": "assistant", "content": choice["message"]["content"]}
        )
    return {"agent": agent_id, "rounds": rounds, "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out}


def status_metrics(port: int) -> dict:
    status, document = smoke_real.request(port, "GET", "/status")
    assert status == 200, document
    metrics = dict(document.get("metrics", {}))
    metrics["decode_batches"] = document.get("scheduler", {}).get(
        "decode_batches", {}
    )
    return metrics


def main() -> int:
    binary = Path(sys.argv[1] if len(sys.argv) > 1 else "build/richengine")
    agents = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    model = sys.argv[4] if len(sys.argv) > 4 else MODEL
    server = smoke_real.RealServer(Args(binary, model, ctx))
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        # One probe warms the weight mappings and the draft checkpoint.
        smoke_real.request(
            server.port, "POST", "/v1/chat/completions",
            smoke_real.chat_body(model, "Say ready.", max_completion_tokens=8),
            timeout=120,
        )
        before = status_metrics(server.port)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(agents) as pool:
            futures = [
                pool.submit(agent, server.port, model, i, ctx)
                for i in range(agents)
            ]
            results = [future.result() for future in futures]
        elapsed = time.monotonic() - started
        after = status_metrics(server.port)
        decode_ms = after["decode_wall_ms"] - before["decode_wall_ms"]
        decode_tokens = (
            after["decode_output_tokens"] - before["decode_output_tokens"]
        )
        drafted = after.get("drafted_tokens", 0) - before.get(
            "drafted_tokens", 0
        )
        accepted = after.get("accepted_draft_tokens", 0) - before.get(
            "accepted_draft_tokens", 0
        )
        if isinstance(after["decode_batches"], dict):
            batches = {
                k: int(after["decode_batches"].get(k, 0))
                - int(before["decode_batches"].get(k, 0))
                for k in after["decode_batches"]
            }
        else:
            batches = {
                "total": int(after["decode_batches"])
                - int(before["decode_batches"])
            }
        print(
            json.dumps(
                {
                    "model": model,
                    "agents": agents,
                    "ctx_cap": ctx,
                    "decode_wall_ms": decode_ms,
                    "decode_output_tokens": decode_tokens,
                    "aggregate_decode_tokens_per_second": round(
                        1000.0 * decode_tokens / max(decode_ms, 1e-9), 1
                    ),
                    "wall_elapsed_s": round(elapsed, 1),
                    "wall_tokens_per_second": round(
                        decode_tokens / max(elapsed, 1e-9), 1
                    ),
                    "drafted_tokens": drafted,
                    "accepted_draft_tokens": accepted,
                    "acceptance": round(
                        accepted / max(drafted, 1), 3
                    ),
                    "decode_batches": batches,
                    "agents_detail": results,
                }
            )
        )
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
