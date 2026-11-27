#!/usr/bin/env python3
"""Paired three-arm A/B for closed-loop coding-agent tasks.

Arms: neural-only draft, legacy corpus mixing, and draft-agreement mixing.
Every arm boots a fresh memory-capped server and executes the same closed-loop
workspace tasks with identical ordering and concurrency. Repetitions rotate arm
order to balance thermal and host noise. Analysis uses per-task paired ratios,
success rates, decode-only time, GPU time, and bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ngram_corpus_ab  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SUITE = Path(__file__).resolve().parent / "agentic_task_suite.py"
DEFAULT_MODEL = "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:Q4_0"

ARMS = {
    "neural": {
        "RICHENGINE_NGRAM_PREDRAFT": "0",
        "RICHENGINE_NGRAM_GPU_MIX": "0",
        "RICHENGINE_NGRAM_MIX": "0",
        "RICHENGINE_NGRAM_DRAFT_AGREEMENT": "0",
    },
    "legacy_mix": {
        "RICHENGINE_NGRAM_GPU_MIX": "1",
        "RICHENGINE_NGRAM_GPU_MIX_MIN_WIDTH": "2",
        "RICHENGINE_NGRAM_MIX_CORPUS_ONLY": "1",
        "RICHENGINE_NGRAM_DRAFT_AGREEMENT": "0",
    },
    "agreement_mix": {
        "RICHENGINE_NGRAM_GPU_MIX": "1",
        "RICHENGINE_NGRAM_GPU_MIX_MIN_WIDTH": "2",
        "RICHENGINE_NGRAM_MIX_CORPUS_ONLY": "1",
        "RICHENGINE_NGRAM_DRAFT_AGREEMENT": "1",
    },
}


def run_one(binary: Path, model: str, args, arm: str, rep: int, out: Path) -> dict:
    command = [
        sys.executable,
        str(SUITE),
        str(binary),
        model,
        "--tasks", str(args.tasks),
        "--width", str(args.width),
        "--max-turns", str(args.max_turns),
        "--max-tokens", str(args.max_tokens),
        "--ctx", str(args.ctx),
        "--max-memory", args.max_memory,
        "--out", str(out),
    ]
    for key, value in ARMS[arm].items():
        command.extend(("--env", f"{key}={value}"))
    ngram_corpus_ab.wait_for_memory()
    pressure_before = ngram_corpus_ab.memory_pressure_level()
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.monotonic() - started
    pressure_after = ngram_corpus_ab.memory_pressure_level()
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"{arm} repetition {rep} failed\nstdout:\n{proc.stdout[-1600:]}\nstderr:\n{proc.stderr[-1600:]}"
        )
    payload = json.loads(out.read_text())
    payload["_host"] = {
        "memory_pressure_before": pressure_before,
        "memory_pressure_after": pressure_after,
        "elapsed_s": round(elapsed, 2),
    }
    return payload


def geomean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values)) if values else 0.0


def bootstrap_ci(values: list[float], iterations: int = 2000) -> list[float] | None:
    if len(values) < 4:
        return None
    rng = random.Random(7301)
    estimates = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(geomean(sample))
    estimates.sort()
    return [estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]]


def ratio_rows(runs: list[dict], control: str, treatment: str, key: str, successful_only: bool = True) -> list[float]:
    rows = []
    reps = sorted({row["repetition"] for row in runs})
    for rep in reps:
        selected = {row["arm"]: row["payload"] for row in runs if row["repetition"] == rep and row["arm"] in (control, treatment)}
        if len(selected) != 2:
            continue
        control_tasks = {task["task_id"]: task for task in selected[control]["tasks"]}
        treatment_tasks = {task["task_id"]: task for task in selected[treatment]["tasks"]}
        for task_id in sorted(control_tasks.keys() & treatment_tasks.keys()):
            base = control_tasks[task_id]
            test = treatment_tasks[task_id]
            if successful_only and not (base["success"] and test["success"]):
                continue
            denominator = base.get(key) or 0
            numerator = test.get(key) or 0
            if denominator > 0 and numerator > 0:
                rows.append(numerator / denominator)
    return rows


def suite_ratios(runs: list[dict], control: str, treatment: str, key: str) -> list[float]:
    rows = []
    for rep in sorted({row["repetition"] for row in runs}):
        selected = {row["arm"]: row["payload"]["summary"] for row in runs if row["repetition"] == rep and row["arm"] in (control, treatment)}
        if len(selected) != 2 and all(row["arm"] in selected for row in runs if row["repetition"] == rep):
            pass
        if control in selected and treatment in selected:
            base = selected[control].get(key) or 0
            test = selected[treatment].get(key) or 0
            if base and test:
                rows.append(test / base)
    return rows


def compare_pair(runs: list[dict], control: str, treatment: str) -> dict:
    result = {"control": control, "treatment": treatment}
    for key, label in (("wall_s", "task_wall"), ("decode_ms", "task_decode"), ("agent_ms", "task_agent")):
        rows = ratio_rows(runs, control, treatment, key)
        ci = bootstrap_ci(rows)
        result[label] = {
            "pairs": len(rows),
            "ratio_geomean": round(geomean(rows), 4),
            "ci95": [round(value, 4) for value in ci] if ci else None,
            "win_rate": round(sum(row < 1 for row in rows) / len(rows), 3) if rows else 0,
        }
    for key in (
        "wall_tok_s",
        "decode_gpu_ms_delta",
        "decode_gpu_ms_per_token",
        "decode_wall_ms_per_token",
        "verify_rows_per_token",
    ):
        rows = suite_ratios(runs, control, treatment, key)
        result["suite_" + key] = {
            "repetitions": len(rows),
            "ratio_geomean": round(geomean(rows), 4),
        }
    successes = {}
    for arm in (control, treatment):
        samples = [row["payload"]["summary"]["success_rate"] for row in runs if row["arm"] == arm]
        successes[arm] = round(statistics.mean(samples), 4) if samples else 0
    result["success_rate_mean"] = successes
    return result


def analyze(runs: list[dict]) -> dict:
    return {
        "agreement_vs_neural": compare_pair(runs, "neural", "agreement_mix"),
        "agreement_vs_legacy": compare_pair(runs, "legacy_mix", "agreement_mix"),
        "legacy_vs_neural": compare_pair(runs, "neural", "legacy_mix"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", nargs="?", default=str(ROOT / "build/richengine"))
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--max-memory", default="30G")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "scratch/agentic-tasks/nemotron-task-ab.json")
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    arms = list(ARMS)
    runs = []
    failures = []
    try:
        for rep in range(args.repeat):
            order = arms[rep % len(arms):] + arms[:rep % len(arms)]
            for arm in order:
                artifact = args.out.parent / f"{args.out.stem}-{arm}-rep{rep}.json"
                print(f"rep {rep + 1}/{args.repeat} arm {arm}", flush=True)
                try:
                    payload = None
                    for attempt in range(3):
                        payload = run_one(binary, args.model, args, arm, rep, artifact)
                        reason = payload["summary"].get("invalid_reason") or ""
                        if not reason:
                            break
                        print(f"  invalid: {reason}; retrying arm", flush=True)
                    else:
                        raise RuntimeError(f"arm remained invalid after retries: {reason}")
                    runs.append({
                        "arm": arm,
                        "repetition": rep,
                        "artifact": str(artifact),
                        "payload": payload,
                        "summary": payload["summary"],
                        "ngram": payload["status_after"]["ngram"],
                        "host": payload.get("_host"),
                    })
                    print("  " + json.dumps(payload["summary"], sort_keys=True), flush=True)
                except Exception as error:
                    failures.append({"arm": arm, "repetition": rep, "error": str(error)})
                    print(f"  failed: {error}", flush=True)
                    if not args.keep_going:
                        raise
    finally:
        # Artifact rows carry full task payloads for audit; analysis excludes failures.
        complete_runs = [row for row in runs if row["summary"].get("tasks")]
        payload = {
            "model": args.model,
            "tasks": args.tasks,
            "width": args.width,
            "repeat": args.repeat,
            "arms": ARMS,
            "runs": [
                {key: value for key, value in row.items() if key != "payload"}
                for row in runs
            ],
            "analysis": analyze(complete_runs),
            "failures": failures,
            "complete": len(failures) == 0,
        }
        args.out.write_text(json.dumps(payload, indent=1))
        print(json.dumps({"analysis": payload["analysis"], "failures": len(failures)}, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
