#!/usr/bin/env python3
"""Verify an installed model's output against recorded goldens.

Runs ``engine-tests/backend-benchmark --scenario decode,partial,short
--samples 1`` on a package and compares, per ``dev/tests/fixtures/
output_goldens.json`` entry:

- the decode scenario's ``output_token_hash`` at every batch width — the
  target's emitted tokens, deterministic under a given build and prompt;
- ``draft_acceptance_rate`` at every width, within 0.02 — a draft whose
  proposals corrupt (reassociated reductions, a stale embedding layout)
  collapses toward zero while the target's hash can still pass, so the
  hash alone does not cover the draft path;
- every partial and short request's literal ``output_tokens``.

``--record`` writes or replaces the package's golden instead of checking
it. Record only from a build whose output is trusted, and re-record when a
change legitimately shifts the argmax stream.

    python3 dev/tests/output_probe.py install/models/openbmb/MiniCPM5-2B-MLX
    python3 dev/tests/output_probe.py --record install/models/...

Without arguments every package with a golden is checked, in order.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "build/engine-tests/backend-benchmark"
METALLIB = ROOT / "build/richengine.metallib"
GOLDENS = ROOT / "dev/tests/fixtures/output_goldens.json"
SCENARIOS = "decode,partial,short"
ACCEPTANCE_TOLERANCE = 0.02


class ProbeFailure(RuntimeError):
    pass


def load_goldens() -> dict:
    return json.loads(GOLDENS.read_text()) if GOLDENS.exists() else {}


def run_benchmark(package: Path) -> dict:
    command = [
        str(BENCHMARK),
        str(METALLIB),
        str(package),
        "--scenario",
        SCENARIOS,
        "--samples",
        "1",
    ]
    print(f"backend-benchmark {package}", file=sys.stderr)
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode or not finished.stdout.strip():
        raise ProbeFailure(
            f"benchmark failed ({finished.returncode}):\n"
            + "\n".join(finished.stderr.splitlines()[-5:])
        )
    return json.loads(finished.stdout)


def observe(document: dict) -> dict:
    """The per-run output record a golden holds: decode hash and acceptance
    per width, and each measured request's output tokens."""
    decode = {}
    for sample in document["decode_throughput"]["samples"]:
        decode[str(sample["width"])] = {
            "hash": sample["output_token_hash"],
            "acceptance": round(sample["draft_acceptance_rate"], 4),
            "drafted_tokens": sample["drafted_tokens"],
        }
    requests = {}
    for measurement in document.get("measurements", []):
        if measurement["scenario"] == "short":
            name = f"short:{measurement['prompt_tokens'] - 1}"
        else:
            name = measurement["scenario"]
        requests[name] = measurement["output_tokens"]
    return {
        "device": document["identity"]["device"],
        "prompt_sha256": document["decode_throughput"].get("prompt_sha256"),
        "decode": decode,
        "requests": requests,
    }


def check(package: str, golden: dict, observed: dict) -> list[str]:
    failures = []
    if observed["prompt_sha256"] != golden["prompt_sha256"]:
        failures.append(
            "the benchmark prompt changed — re-record the goldens "
            f"({golden['prompt_sha256']} -> {observed['prompt_sha256']})"
        )
    if observed["device"] != golden["device"]:
        print(
            f"{package}: golden was recorded on {golden['device']!r}, "
            f"this is {observed['device']!r}",
            file=sys.stderr,
        )
    for width, want in golden["decode"].items():
        got = observed["decode"].get(width)
        if got is None:
            failures.append(f"decode B{width}: no sample")
            continue
        if got["hash"] != want["hash"]:
            failures.append(
                f"decode B{width}: output hash {got['hash']} != golden {want['hash']}"
            )
        if abs(got["acceptance"] - want["acceptance"]) > ACCEPTANCE_TOLERANCE:
            failures.append(
                f"decode B{width}: draft acceptance {got['acceptance']:.3f} "
                f"!= golden {want['acceptance']:.3f} (+/-{ACCEPTANCE_TOLERANCE})"
            )
        if not got["drafted_tokens"]:
            failures.append(f"decode B{width}: the draft proposed no tokens")
    for name, want in golden["requests"].items():
        got = observed["requests"].get(name)
        if got != want:
            first = next(
                (i for i, (a, b) in enumerate(zip(got or [], want)) if a != b),
                min(len(got or []), len(want)),
            )
            failures.append(
                f"{name}: output tokens diverge at index {first} "
                f"({len(got or [])} tokens vs {len(want)} golden)"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record", action="store_true", help="write goldens")
    parser.add_argument("packages", nargs="*", type=Path)
    args = parser.parse_args()
    goldens = load_goldens()
    packages = [str(p) for p in args.packages] or sorted(goldens)
    if not packages:
        print("no packages and no goldens", file=sys.stderr)
        return 2
    failed = False
    for name in packages:
        package = Path(name)
        if not package.is_absolute():
            package = ROOT / package
        try:
            observed = observe(run_benchmark(package))
        except ProbeFailure as error:
            print(f"{name}: {error}", file=sys.stderr)
            failed = True
            continue
        key = str(package.relative_to(ROOT)) if package.is_relative_to(ROOT) else name
        if args.record:
            goldens[key] = observed
            GOLDENS.write_text(json.dumps(goldens, indent=1) + "\n")
            print(f"{key}: recorded", file=sys.stderr)
            continue
        if key not in goldens:
            print(f"{key}: no golden — run with --record", file=sys.stderr)
            failed = True
            continue
        failures = check(key, goldens[key], observed)
        for failure in failures:
            print(f"{key}: {failure}", file=sys.stderr)
        if failures:
            failed = True
        else:
            print(f"{key}: output matches the golden", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
