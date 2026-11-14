#!/usr/bin/env python3
"""A/B harness for n-gram corpus features (RICHENGINE_NGRAM_CORPUS*,
request corpus_opt_out) against a real model.

Each arm boots its own server with an env overlay, then runs:

  sequential — multi-turn agentic sessions (edit loops, tool-arg JSON,
               fix loops, verbatim quoting, chat control), per-request
               tok/s + accepted/offered draft tokens + output hash.
  wave       — four identical requests staggered ~0.5 s apart: the
               live-doc case, where a leading lane's in-flight stream can
               feed followers (identical lockstep lanes cannot — same
               position means zero followers).

Usage:
  .venv/bin/python dev/benchmarks/ngram_corpus_ab.py [BINARY]
      [--model MODEL] [--arm NAME=JSONENV ...] [--ctx TOKENS]
      [--out FILE]

BINARY defaults to build/richengine, MODEL to
ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit. Each --arm is a name plus a JSON
object merged into the server's environment, e.g.

  --arm corpus-off='{"RICHENGINE_NGRAM_CORPUS":"0"}'
  --arm nolive='{"RICHENGINE_NGRAM_CORPUS_LIVE":"0"}'

With no --arm the default pair is corpus on (ambient env) vs corpus off.
The first arm is the baseline the printed diff table compares against;
hash mismatches flag output changes (corpus work must be hash-identical
in single-lane traffic — near-tie argmax divergence under differing batch
widths is a known engine property, not a corpus bug).
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke_real  # noqa: E402
import agentic_traffic  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

# Long, strongly-structured output: leader stays in flight while followers
# overlap — the live-doc followers>0 case.
WAVE_PROMPT = (
    "Write a complete Python module for a JSON-backed note-taking CLI: an "
    "`add`, `list`, `search`, `tag`, and `delete` command, each as its own "
    "function, plus argparse wiring and a `main()`. Include docstrings. "
    "Output only code, no commentary."
)
WAVE_LANES = 4
WAVE_STAGGER_S = 0.5


def sessions():
    tr = agentic_traffic
    return [
        (
            "edit_loop",
            [
                f"Here is my code:\n{tr.CODE}\nRepeat it exactly but rename "
                "bubble_sort to sort_array everywhere. Output only code.",
                "Now emit it as a unified diff against the original. "
                "Output only the diff.",
                "Repeat the renamed file once more, verbatim.",
                "Also rename quick_check to fast_check and emit the whole "
                "file again, verbatim.",
            ],
        ),
        (
            "edit_again",
            [
                f"Here is my code:\n{tr.CODE}\nRepeat it exactly but rename "
                "bubble_sort to sort_array everywhere. Output only code.",
                "Emit that renamed file once more, verbatim.",
            ],
        ),
        (
            "toolargs",
            [
                'Call write_file with JSON: {"path": "src/sort.py", '
                '"content": "<the renamed file contents>"}. Output only '
                "JSON.",
                "Same call again, path src/sort_v2.py. Output only JSON.",
                "Same call again, path src/sort_v3.py. Output only JSON.",
            ],
        ),
        (
            "fix_loop",
            [
                "Write a Python function `parse_log_line(line)` returning "
                "(ts, level, worker, msg) from a log line. Output only "
                "code.",
                "It raised ValueError on empty lines — fix it and emit the "
                "full function again. Output only code.",
                "Still fails on lines with missing fields — handle that "
                "and emit the full function again. Output only code.",
            ],
        ),
        (
            "log_table",
            [
                "Render this log as a markdown table with columns "
                f"ts | level | worker | msg, one row per line:\n{tr.LOG}",
                "Same table again but only the WARN and ERROR rows.",
            ],
        ),
        (
            "rag",
            [
                f"Document:\n{tr.DOC}\n\nQuote the two sentences "
                "containing weight and lifespan, verbatim.",
                "Quote them again, same order.",
            ],
        ),
        (
            "chat",
            [
                "Write a short story about a lighthouse keeper who finds "
                "a strange fog bank."
            ],
        ),
    ]


class Args:
    def __init__(self, binary, model, ctx):
        self.package = ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = ctx
        self.max_memory = None
        self.max_cache_disk = None
        self.max_image_pixels = None
        self.kv_format = os.environ.get("KV_FORMAT", "int8")


def metrics_of(response):
    dec = (response.get("metrics") or {}).get("decode") or {}
    return {
        "tok_s": (response.get("timings") or {}).get("predicted_per_second"),
        "acc": dec.get("accepted_draft_tokens", 0),
        "off": dec.get("drafted_tokens", 0),
    }


def find_corpus(node):
    """The corpus counters object wherever /status nests it."""
    if isinstance(node, dict):
        if set(node) >= {"hits", "misses", "docs"}:
            return node
        for value in node.values():
            found = find_corpus(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_corpus(value)
            if found:
                return found
    return None


def status_corpus(port):
    status, document = smoke_real.request(port, "GET", "/status")
    if status != 200:
        return None
    return find_corpus(document)


def run_sequential(traffic, results, max_tokens):
    for name, turns in sessions():
        msgs = []
        for i, text in enumerate(turns):
            msgs.append({"role": "user", "content": text})
            reply = traffic.call(msgs, max_tokens) or ""
            msgs.append({"role": "assistant", "content": reply})
            entry = metrics_of(traffic.last)
            entry["hash"] = hashlib.sha256(reply.encode()).hexdigest()[:12]
            results[f"{name} t{i}"] = entry
        print(f"  {name}: done", flush=True)


def run_wave(base, model, results, max_tokens):
    replies = [None] * WAVE_LANES
    metrics = [None] * WAVE_LANES

    def lane(i):
        time.sleep(WAVE_STAGGER_S * i)
        try:
            traffic = agentic_traffic.Traffic(base, model)
            replies[i] = traffic.call(
                [{"role": "user", "content": WAVE_PROMPT}], max_tokens
            )
            metrics[i] = metrics_of(traffic.last)
        except Exception as error:  # noqa: BLE001 — report, keep running
            metrics[i] = {"error": str(error)[:200]}
            replies[i] = ""

    threads = [
        threading.Thread(target=lane, args=(i,)) for i in range(WAVE_LANES)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for i in range(WAVE_LANES):
        entry = metrics[i] or {}
        entry["hash"] = hashlib.sha256(
            (replies[i] or "").encode()
        ).hexdigest()[:12]
        results[f"wave lane{i}"] = entry
    print("  wave: done", flush=True)


def run_arm(binary, model, ctx, env_overlay, max_tokens):
    env = dict(os.environ)
    env.update(env_overlay)
    server = smoke_real.RealServer(Args(binary, model, ctx), env)
    results = {}
    try:
        smoke_real.validate_status(server.wait_ready(600), env.get("KV_FORMAT", "int8"))
        base = f"http://127.0.0.1:{server.port}"
        traffic = agentic_traffic.Traffic(base, model)
        run_sequential(traffic, results, max_tokens)
        run_wave(base, model, results, 640)
        results["_corpus"] = status_corpus(server.port)
    finally:
        server.close()
    return results


def diff_table(arms, results):
    baseline_name, baseline = arms[0], results[arms[0]]
    keys = [k for k in baseline if not k.startswith("_")]
    for arm in arms[1:]:
        other = results[arm]
        print(f"\n=== {baseline_name} -> {arm} ===")
        hash_diffs = 0
        for key in keys:
            base, new = baseline[key], other.get(key)
            if not new:
                print(f"{key:16s} missing")
                continue
            flag = ""
            if base.get("hash") != new.get("hash"):
                flag = " HASH-DIFF"
                hash_diffs += 1
            delta = ""
            if base.get("tok_s") and new.get("tok_s"):
                pct = (new["tok_s"] - base["tok_s"]) / base["tok_s"] * 100
                delta = f"{pct:+6.1f}%"
            print(
                f"{key:16s} {base.get('tok_s', 0):7.1f} -> "
                f"{new.get('tok_s', 0):7.1f} {delta}  "
                f"acc {base.get('acc', 0):4d}->{new.get('acc', 0):4d}  "
                f"off {base.get('off', 0):5d}->{new.get('off', 0):5d}{flag}"
            )
        print(f"hash diffs: {hash_diffs}")


def parse_arm(spec):
    name, sep, payload = spec.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(
            f"--arm wants NAME=JSONENV, got {spec!r}"
        )
    try:
        overlay = json.loads(payload or "{}")
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"--arm {name}: bad JSON env: {error}"
        ) from error
    if not isinstance(overlay, dict):
        raise argparse.ArgumentTypeError(f"--arm {name}: env must be a JSON object")
    return name, {str(k): str(v) for k, v in overlay.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--arm",
        type=parse_arm,
        action="append",
        default=None,
        metavar="NAME=JSONENV",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    arms = args.arm or [
        ("corpus-on", {}),
        ("corpus-off", {"RICHENGINE_NGRAM_CORPUS": "0"}),
    ]
    binary = Path(args.binary).resolve()
    results = {}
    for name, overlay in arms:
        print(f"arm {name}: {overlay or 'ambient env'}", flush=True)
        results[name] = run_arm(
            binary, args.model, args.ctx, overlay, args.max_tokens
        )

    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    if len(arms) > 1:
        diff_table([name for name, _ in arms], results)
    else:
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
