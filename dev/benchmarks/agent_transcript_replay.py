#!/usr/bin/env python3
"""Replay real public coding-agent transcripts against a serving instance.

Default source: thoughtworks/agentic-coding-trajectories, a 15k-session
normalized corpus built for LLM-serving benchmarks from OpenHands, SWE-smith,
and Klear trajectories. Rows are fetched through the Hugging Face dataset
viewer API and cached locally as JSONL; no parquet dependency is needed.

Shadow mode replays the recorded context exactly before each assistant turn,
while generated tokens still exercise live speculative decode. Live mode feeds
the model's own replies forward. Shadow mode is better for controlled A/Bs;
live mode is closer to production behavior.

Usage:
  .venv/bin/python dev/benchmarks/agent_transcript_replay.py \
      [BINARY] [MODEL] --sessions 4 --lanes 4 --max-turns 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import smoke_real  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"
DATASET = "thoughtworks/agentic-coding-trajectories"
CACHE = ROOT / "scratch/agent-transcripts/thoughtworks-agentic-coding.jsonl"


class Args:
    def __init__(self, binary: Path, model: str, max_context: int, max_memory: str | None):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = max_context
        self.max_memory = max_memory
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = "int8"


def fetch_rows(count: int, offset: int, cache: Path) -> list[dict]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if cache.exists():
        rows = [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    if len(rows) >= offset + count:
        return rows[offset:offset + count]
    need = offset + count
    written = len(rows)
    with cache.open("a", encoding="utf-8") as out:
        while written < need:
            length = min(20, need - written)
            query = urllib.parse.urlencode({
                "dataset": DATASET,
                "config": "default",
                "split": "train",
                "offset": written,
                "length": length,
            })
            url = "https://datasets-server.huggingface.co/rows?" + query
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            batch = [item["row"] for item in payload.get("rows", [])]
            if not batch:
                break
            for row in batch:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                rows.append(row)
                written += 1
    if len(rows) < offset + count:
        raise RuntimeError(f"dataset returned {len(rows)} rows, needed {offset + count}")
    return rows[offset:offset + count]


def normalize_messages(raw: list[dict], max_chars: int) -> list[dict]:
    messages = []
    for item in raw:
        role = item.get("role")
        content = item.get("content") or ""
        if len(content) > max_chars:
            content = content[:max_chars] + "\n[truncated for replay]"
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": "TOOL OBSERVATION:\n" + content})
    return messages


def assistant_indices(messages: list[dict]) -> list[int]:
    return [index for index, item in enumerate(messages) if item["role"] == "assistant"]


def chat(port: int, model: str, messages: list[dict], max_tokens: int, ignore_eos: bool = False) -> dict:
    body = smoke_real.chat_body(
        model, "", temperature=0, max_completion_tokens=max_tokens
    )
    if ignore_eos:
        body["ignore_eos"] = True
    body["messages"] = messages
    body.pop("tools", None)
    status, doc = smoke_real.request(
        port, "POST", "/v1/chat/completions", body, timeout=600
    )
    if status != 200:
        raise RuntimeError(f"chat failed {status}: {doc}")
    return doc


def metrics(doc: dict) -> dict:
    timings = doc.get("timings") or {}
    decode = (doc.get("metrics") or {}).get("decode") or {}
    return {
        "tok_s": timings.get("predicted_per_second") or 0,
        "accepted": decode.get("accepted_draft_tokens", 0),
        "offered": decode.get("drafted_tokens", 0),
        "completion_tokens": (doc.get("usage") or {}).get("completion_tokens", 0),
    }


def replay_session(port: int, model: str, row: dict, args, lane: int) -> dict:
    messages = normalize_messages(json.loads(row["messages_json"]), args.max_message_chars)
    result = {
        "lane": lane,
        "session_id": row.get("session_id"),
        "source_dataset": row.get("source_dataset"),
        "turns": [],
    }
    indices = assistant_indices(messages)
    if len(indices) < 2:
        result["error"] = "not enough assistant turns"
        return result
    # Start with every message before the second assistant turn. Each step
    # asks for the next assistant continuation, then either restores the
    # reference transcript (shadow) or keeps the live generation (live).
    cut = indices[1]
    history = messages[:cut]
    for turn, index in enumerate(indices[1:]):
        if turn >= args.max_turns:
            break
        if getattr(args, "barrier", None) is not None:
            args.barrier.wait()
        doc = chat(port, model, history, args.max_tokens, args.fixed_width)
        message = doc["choices"][0]["message"]
        generated = message.get("content") or ""
        result["turns"].append({"turn": turn, **metrics(doc)})
        reference = messages[index]["content"]
        history.append({"role": "assistant", "content": reference if args.mode == "shadow" else generated})
        # Add recorded user/tool observations up to the next assistant turn.
        next_index = indices[turn + 2] if turn + 2 < len(indices) else len(messages)
        for item in messages[index + 1:next_index]:
            history.append(item)
    return result


def status_snapshot(port: int) -> dict:
    status, doc = smoke_real.request(port, "GET", "/status", timeout=10)
    if status != 200:
        return {}
    return {"ngram": doc.get("ngram"), "corpus": doc.get("corpus"), "decode": doc.get("decode")}


def summarize(sessions: list[dict], wall_s: float) -> dict:
    turns = [turn for session in sessions for turn in session.get("turns", [])]
    speeds = [turn["tok_s"] for turn in turns if turn["tok_s"]]
    accepted = sum(turn["accepted"] for turn in turns)
    offered = sum(turn["offered"] for turn in turns)
    completion = sum(turn["completion_tokens"] for turn in turns)
    return {
        "turn_count": len(turns),
        "completion_tokens": completion,
        "wall_s": round(wall_s, 2),
        "wall_tok_s": round(completion / wall_s, 2) if wall_s else 0,
        "decode_tok_s_median": round(statistics.median(speeds), 2) if speeds else 0,
        "decode_tok_s_mean": round(statistics.mean(speeds), 2) if speeds else 0,
        "accepted_draft_tokens": accepted,
        "offered_draft_tokens": offered,
        "acceptance_rate": round(accepted / offered, 4) if offered else 0,
        "errors": [session.get("error") for session in sessions if session.get("error")],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("model", nargs="?", default=MODEL)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--max-memory", default="30G", help="server memory cap; use 0 for none")
    parser.add_argument("--max-message-chars", type=int, default=24000)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--fixed-width", action="store_true", help="pin sessions to lanes, synchronize turns, and ignore EOS")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="server environment override; repeatable for A/B experiments",
    )
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    rows = fetch_rows(args.sessions, args.offset, args.cache)
    if args.fixed_width:
        insufficient = []
        for index, row in enumerate(rows):
            messages = normalize_messages(json.loads(row["messages_json"]), args.max_message_chars)
            assistants = len(assistant_indices(messages))
            if assistants < args.max_turns + 1:
                insufficient.append((index, row.get("session_id"), assistants))
        if insufficient:
            raise RuntimeError(
                "--fixed-width requires every session to hold max-turns+1 assistant "
                f"messages; insufficient: {insufficient}"
            )
    if args.download_only:
        print(json.dumps({"downloaded": len(rows), "cache": str(args.cache)}))
        return 0
    binary = Path(args.binary).resolve()
    overrides = dict(item.split("=", 1) for item in args.env)
    server = smoke_real.RealServer(
        Args(binary, args.model, args.ctx, None if args.max_memory == "0" else args.max_memory), dict(os.environ, **overrides)
    )
    try:
        smoke_real.validate_status(server.wait_ready(600), "int8")
        started = time.monotonic()
        sessions = [None] * args.sessions
        lock = threading.Lock()
        next_index = 0
        lane_count = max(1, args.lanes)
        if args.fixed_width:
            if args.sessions != lane_count:
                raise RuntimeError("--fixed-width requires --sessions to equal --lanes")
            args.barrier = threading.Barrier(lane_count)

        def replay_one(index: int, lane: int):
            try:
                sessions[index] = replay_session(server.port, args.model, rows[index], args, lane)
            except Exception as error:
                sessions[index] = {
                    "lane": lane,
                    "session_id": rows[index].get("session_id"),
                    "source_dataset": rows[index].get("source_dataset"),
                    "turns": [],
                    "error": str(error),
                }

        def worker(lane: int):
            nonlocal next_index
            if args.fixed_width:
                replay_one(lane, lane)
                return
            while True:
                with lock:
                    if next_index >= args.sessions:
                        return
                    index = next_index
                    next_index += 1
                replay_one(index, lane)

        threads = [threading.Thread(target=worker, args=(lane,)) for lane in range(lane_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        wall_s = time.monotonic() - started
        server.log.flush()
        try:
            with open(server.log.name, "r", encoding="utf-8", errors="replace") as log_file:
                server_log_tail = log_file.read()[-12000:]
        except OSError:
            server_log_tail = ""
        payload = {
            "dataset": DATASET,
            "mode": args.mode,
            "sessions": sessions,
            "summary": summarize(sessions, wall_s),
            "status": status_snapshot(server.port),
            "server_log_tail": server_log_tail,
        }
        if any(session is None for session in sessions):
            raise RuntimeError("a replay worker did not finish its session")
        output = json.dumps(payload, indent=1)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(output)
        print(output)
        return 0
    finally:
        server.close()


if __name__ == "__main__":
    raise SystemExit(main())
