#!/usr/bin/env python3
"""ABBA A/B driver for real-agent transcript replay.

Compares neural-draft baseline against width-gated corpus-only GPU mixing on
fixed public coding-agent transcripts. Every repetition boots a fresh server,
uses identical transcript rows, and records the replay artifact for audit.

Usage:
  .venv/bin/python dev/benchmarks/agent_transcript_ab.py \
      build/richengine openbmb/MiniCPM5-2B-GGUF:Q4_K_M \
      --sessions 8 --lanes 4 --repeat 2
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ngram_corpus_ab  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPLAY = Path(__file__).resolve().parent / "agent_transcript_replay.py"
DEFAULT_MODEL = "ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit"

ARMS = {
    "neural": {
        "RICHENGINE_NGRAM_PREDRAFT": "0",
        "RICHENGINE_NGRAM_GPU_MIX": "0",
        "RICHENGINE_NGRAM_MIX": "0",
    },
    "corpus_mix": {
        "RICHENGINE_NGRAM_GPU_MIX": "1",
        "RICHENGINE_NGRAM_GPU_MIX_MIN_WIDTH": "2",
        "RICHENGINE_NGRAM_MIX_CORPUS_ONLY": "1",
    },
}


def ordering(repeat: int, order: str) -> list[tuple[int, str]]:
    names = list(ARMS)
    if order == "ab":
        return [(rep, name) for rep in range(repeat) for name in names]
    sequence = []
    for rep in range(repeat):
        pair = names if rep % 2 == 0 else list(reversed(names))
        sequence.extend((rep, name) for name in pair)
    return sequence


def run_one(binary: Path, model: str, args, arm: str, rep: int, out: Path) -> dict:
    env = ARMS[arm]
    command = [
        sys.executable,
        str(REPLAY),
        str(binary),
        model,
        "--sessions",
        str(args.sessions),
        "--lanes",
        str(args.lanes),
        "--offset",
        str(args.offset),
        "--max-turns",
        str(args.max_turns),
        "--max-tokens",
        str(args.max_tokens),
        "--ctx",
        str(args.ctx),
        "--max-memory",
        args.max_memory,
        "--max-message-chars",
        str(args.max_message_chars),
        "--mode",
        args.mode,
        "--out",
        str(out),
    ]
    if args.fixed_width:
        command.append("--fixed-width")
    for key, value in env.items():
        command.extend(("--env", f"{key}={value}"))
    ngram_corpus_ab.wait_for_memory()
    pressure_before = ngram_corpus_ab.memory_pressure_level()
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.monotonic() - started
    pressure_after = ngram_corpus_ab.memory_pressure_level()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"{arm} repetition {rep} failed after {elapsed:.1f}s\n"
            f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
        )
    payload = json.loads(out.read_text())
    payload["_host"] = {"memory_pressure_before": pressure_before, "memory_pressure_after": pressure_after, "elapsed_s": round(elapsed, 2)}
    errors = payload.get("summary", {}).get("errors") or []
    if errors:
        raise RuntimeError(f"{arm} repetition {rep} replay errors: {errors}")
    return payload


def scalar_metrics(payload: dict) -> dict:
    summary = payload.get("summary") or {}
    ngram = (payload.get("status") or {}).get("ngram") or {}
    return {
        "wall_tok_s": summary.get("wall_tok_s") or 0,
        "decode_tok_s_median": summary.get("decode_tok_s_median") or 0,
        "decode_tok_s_mean": summary.get("decode_tok_s_mean") or 0,
        "completion_tokens": summary.get("completion_tokens") or 0,
        "turn_count": summary.get("turn_count") or 0,
        "acceptance_rate": summary.get("acceptance_rate") or 0,
        "accepted_draft_tokens": summary.get("accepted_draft_tokens") or 0,
        "offered_draft_tokens": summary.get("offered_draft_tokens") or 0,
        "mixed_lanes": ngram.get("mixed_lanes") or 0,
        "mixed_corpus_lanes": ngram.get("mixed_corpus_lanes") or 0,
        "chain_tokens": ngram.get("chain_tokens") or 0,
        "verify_rows_total": ngram.get("verify_rows_total") or 0,
        "host_microseconds": ngram.get("host_microseconds") or 0,
        "agreement_queries": ngram.get("agreement_queries") or 0,
        "agreement_top": (ngram.get("agreement_labels") or {}).get("top") or 0,
        "agreement_candidate": (ngram.get("agreement_labels") or {}).get("candidate") or 0,
        "agreement_absent": (ngram.get("agreement_labels") or {}).get("absent") or 0,
        "agreement_unknown": (ngram.get("agreement_labels") or {}).get("unknown") or 0,
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for arm in ARMS:
        samples = [row["metrics"] for row in rows if row["arm"] == arm]
        if not samples:
            continue
        arm_result = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            arm_result[key + "_mean"] = statistics.mean(values)
            arm_result[key + "_median"] = statistics.median(values)
        result[arm] = arm_result
    return result


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compare(rows: list[dict]) -> dict:
    names = list(ARMS)
    control, treatment = names[0], names[1]
    neural = [row["metrics"]["wall_tok_s"] for row in rows if row["arm"] == control]
    mixed = [row["metrics"]["wall_tok_s"] for row in rows if row["arm"] == treatment]
    decode_neural = [row["metrics"]["decode_tok_s_median"] for row in rows if row["arm"] == control]
    decode_mixed = [row["metrics"]["decode_tok_s_median"] for row in rows if row["arm"] == treatment]
    mean_ratio = ratio(statistics.mean(mixed), statistics.mean(neural)) if neural and mixed else 0
    decode_ratio = ratio(statistics.mean(decode_mixed), statistics.mean(decode_neural)) if decode_neural and decode_mixed else 0
    paired = []
    reps = sorted({row["repetition"] for row in rows})
    for rep in reps:
        base = [row["metrics"]["wall_tok_s"] for row in rows if row["repetition"] == rep and row["arm"] == control]
        test = [row["metrics"]["wall_tok_s"] for row in rows if row["repetition"] == rep and row["arm"] == treatment]
        if base and test and base[0]:
            paired.append(test[0] / base[0])
    return {
        "wall_tok_s_ratio_mean": round(mean_ratio, 4),
        "wall_tok_s_delta_pct": round((mean_ratio - 1) * 100, 2),
        "decode_median_ratio_mean": round(decode_ratio, 4),
        "decode_median_delta_pct": round((decode_ratio - 1) * 100, 2),
        "paired_wall_ratio_geomean": round(math.exp(statistics.mean(math.log(x) for x in paired)), 4) if paired else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--max-memory", default="30G")
    parser.add_argument("--max-message-chars", type=int, default=6000)
    parser.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--experiment", choices=("neural-vs-mix", "agreement"), default="neural-vs-mix")
    parser.add_argument("--fixed-width", action="store_true", help="pin sessions to lanes, synchronize turns, and ignore EOS")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--order", choices=("abba", "ab"), default="abba")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--out", type=Path, default=ROOT / "scratch/agent-transcripts/transcript-ab.json")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if args.experiment == "agreement":
        base = {
            "RICHENGINE_NGRAM_GPU_MIX": "1",
            "RICHENGINE_NGRAM_GPU_MIX_MIN_WIDTH": "2",
            "RICHENGINE_NGRAM_MIX_CORPUS_ONLY": "1",
        }
        ARMS.clear()
        ARMS.update({
            "legacy_mix": dict(base),
            "agreement_mix": {**base, "RICHENGINE_NGRAM_DRAFT_AGREEMENT": "1"},
        })

    binary = Path(args.binary).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    try:
        for rep, arm in ordering(args.repeat, args.order):
            artifact = args.out.parent / f"{args.out.stem}-{arm}-rep{rep}.json"
            print(f"arm {arm} rep {rep + 1}/{args.repeat}", flush=True)
            try:
                payload = run_one(binary, args.model, args, arm, rep, artifact)
                rows.append({
                    "arm": arm,
                    "repetition": rep,
                    "artifact": str(artifact),
                    "metrics": scalar_metrics(payload),
                    "summary": payload.get("summary"),
                    "ngram": (payload.get("status") or {}).get("ngram"),
                    "host": payload.get("_host"),
                })
                print(f"  wall {rows[-1]['metrics']['wall_tok_s']} tok/s", flush=True)
            except Exception as error:
                failures.append({"arm": arm, "repetition": rep, "error": str(error)})
                print(f"  failed: {error}", flush=True)
                if not args.keep_going:
                    raise
    finally:
        payload = {
            "model": args.model,
            "sessions": args.sessions,
            "lanes": args.lanes,
            "mode": args.mode,
            "experiment": args.experiment,
            "repeat": args.repeat,
            "order": args.order,
            "runs": rows,
            "aggregate": aggregate(rows),
            "comparison": compare(rows),
            "failures": failures,
            "complete": not failures and len(rows) == len(ordering(args.repeat, args.order)),
        }
        args.out.write_text(json.dumps(payload, indent=1))
        print(json.dumps({"comparison": payload["comparison"], "failures": failures}, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
