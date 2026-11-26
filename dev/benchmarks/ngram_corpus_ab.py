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
import math
import statistics
import subprocess
import shlex
import os
import sys
import threading
import time
import urllib.request
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

# Heterogeneous concurrency: only lane zero emits the corpus-heavy module.
# Other lanes stress self-match, lookup-match, and mostly-open generation in
# the same batch, closer to independent agent workers than four copies of
# one prompt.
# Short-chain activation probe: the corpus is seeded with independent
# ten-word lines, then lanes re-emit those lines. Near every document end the
# retrieved continuation is shorter than full proposal width, exactly the
# case chain clipping must exploit.
SHORT_CHAIN_LINES = [
    'alpha beta gamma delta epsilon zeta eta theta iota kappa',
    'crimson scarlet vermilion ruby garnet coral amber saffron gold emerald',
    'orchard meadow forest river harbor valley prairie canyon ridge island',
    'copper silver nickel cobalt iron titanium chromium zinc tin platinum',
    'nebula comet asteroid meteor planet galaxy quasar pulsar satellite cosmos',
    'anchor beacon compass lantern harbor mast rudder keel sail horizon',
    'baker farmer mason painter plumber tailor weaver potter smith carpenter',
    'violin trumpet cello oboe clarinet harp flute drum viola trombone',
    'winter spring summer autumn monsoon glacier desert tundra wetland savanna',
    'purple orange yellow indigo violet magenta turquoise ivory charcoal azure',
    'pendulum piston turbine boiler condenser rotor valve gasket flywheel cylinder',
    'lexicon grammar syntax phoneme morpheme vowel consonant idiom proverb dialect',
    'falcon sparrow heron egret pelican swallow swift cardinal finch osprey',
    'granite basalt marble slate quartz obsidian limestone sandstone shale flint',
    'village market bridge tower chapel bakery library theater garden fountain',
    'antelope buffalo cheetah dolphin elephant falcon giraffe hyena iguana jaguar'
]
SHORT_CHAIN_WAVE_PROMPTS = [
    "Repeat these sixteen lines exactly, preserving every word and line break:\n"
    + "\n".join(SHORT_CHAIN_LINES)
    + "\nOutput only those lines."
    for _lane in range(4)
]

HETERO_WAVE_PROMPTS = [
    WAVE_PROMPT,
    "Repeat this exact JSON tool call twelve times, changing only the path "
    "index from 0 through 11: {\"tool\":\"write_file\",\"path\":"
    "\"src/out_0.py\",\"content\":\"alpha beta gamma\"}. Output only JSON lines.",
    "Write a Python function that validates nested configuration dictionaries, "
    "then write three distinct unit tests. Output only code.",
    "Write a markdown table of twelve deployment rows. Use repeated columns "
    "for service, region, status, latency, and owner. Output only the table.",
]


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
    def __init__(self, binary, model, ctx, max_memory="36G", package=None):
        self.package = Path(package) if package else ROOT / "install/models" / model
        self.binary = binary
        self.model = model
        self.max_context = ctx
        self.max_memory = max_memory
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


def find_named(node, name):
    """The first object member called `name`, wherever /status nests it."""
    if isinstance(node, dict):
        if name in node:
            return node[name]
        for value in node.values():
            found = find_named(value, name)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_named(value, name)
            if found is not None:
                return found
    return None


def status_snapshot(port):
    status, document = smoke_real.request(port, "GET", "/status")
    if status != 200:
        return None
    timing = find_named(document, "model_timing") or {}
    return {
        "corpus": find_corpus(document),
        "ngram": find_named(document, "ngram"),
        "decode": timing.get("decode") if isinstance(timing, dict) else None,
    }


def run_sequential(traffic, results, max_tokens, contaminated=None):
    for name, turns in sessions():
        if contaminated and contaminated.is_set():
            raise smoke_real.SmokeFailure("foreign native engine contaminated benchmark")
        msgs = []
        for i, text in enumerate(turns):
            msgs.append({"role": "user", "content": text})
            reply = traffic.call(msgs, max_tokens) or ""
            msgs.append({"role": "assistant", "content": reply})
            entry = metrics_of(traffic.last)
            entry["hash"] = hashlib.sha256(reply.encode()).hexdigest()[:12]
            results[f"{name} t{i}"] = entry
        print(f"  {name}: done", flush=True)


def seed_short_chain_corpus(base):
    for index, line in enumerate(SHORT_CHAIN_LINES):
        body = json.dumps({"text": "\n" + line}).encode()
        request = urllib.request.Request(
            base.rstrip("/") + "/v1/corpus",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise smoke_real.SmokeFailure("short-chain corpus seeding failed")


def run_wave(base, model, results, max_tokens, contaminated=None, heterogeneous=False, short_chain=False):
    if contaminated and contaminated.is_set():
        raise smoke_real.SmokeFailure("foreign native engine contaminated benchmark")
    replies = [None] * WAVE_LANES
    metrics = [None] * WAVE_LANES

    def lane(i):
        time.sleep((0.1 if short_chain else WAVE_STAGGER_S) * i)
        try:
            traffic = agentic_traffic.Traffic(base, model)
            replies[i] = traffic.call(
                [{"role": "user", "content": SHORT_CHAIN_WAVE_PROMPTS[i] if short_chain else HETERO_WAVE_PROMPTS[i] if heterogeneous else WAVE_PROMPT}],
                max_tokens,
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
    errors = [
        f"lane{i}: {metrics[i]['error']}"
        for i in range(WAVE_LANES)
        if metrics[i] and "error" in metrics[i]
    ]
    if errors:
        raise smoke_real.SmokeFailure(
            "wave request failed; benchmark invalid: " + "; ".join(errors)
        )
    for i in range(WAVE_LANES):
        entry = metrics[i] or {}
        entry["hash"] = hashlib.sha256(
            (replies[i] or "").encode()
        ).hexdigest()[:12]
        results[f"wave lane{i}"] = entry
    print("  wave: done", flush=True)


def memory_pressure_level():
    """macOS memorystatus pressure level; None where telemetry is unavailable."""
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def native_engine_pids():
    """Native serve-native PIDs, excluding shells merely mentioning them."""
    try:
        out = subprocess.run(
            ["/bin/ps", "-axo", "pid=,args="],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    pids = []
    for line in out.stdout.splitlines():
        try:
            pid_text, command = line.strip().split(None, 1)
            argv = shlex.split(command)
        except (ValueError, OSError):
            continue
        if len(argv) > 1 and argv[0].endswith("richengine") and argv[1] == "serve-native":
            pids.append(int(pid_text))
    return pids


def allow_foreign_engines():
    return os.environ.get("NGRAM_AB_ALLOW_FOREIGN") == "1"


def wait_for_memory(levels=(1, 2), timeout=600):
    """Wait out critical pressure and unrelated resident engines.

    A second model resident can consume both unified memory and the GPU. Its
    presence makes bootstrap flaky and invalidates latency measurements.
    """
    deadline = time.monotonic() + timeout
    while True:
        level = memory_pressure_level()
        busy = [] if allow_foreign_engines() else native_engine_pids()
        if (level is None or level in levels) and not busy:
            return level
        if time.monotonic() >= deadline:
            detail = []
            if level not in levels:
                detail.append(f"memory pressure level {level}")
            if busy:
                detail.append(f"native engines {busy}")
            raise smoke_real.SmokeFailure(
                "benchmark host stayed busy for " f"{timeout}s: {', '.join(detail)}"
            )
        time.sleep(2)


def run_arm(binary, model, ctx, env_overlay, max_tokens, max_memory, wave_only=False, package=None, hetero_wave=False, short_chain_wave=False):
    env = dict(os.environ)
    env.update(env_overlay)
    wait_for_memory()
    server = smoke_real.RealServer(Args(binary, model, ctx, max_memory, package), env)
    results = {}
    try:
        ready = server.wait_ready(600)
        smoke_real.validate_status(ready, env.get("KV_FORMAT", "int8"))
        if ready.get("memory_pressure") == "critical":
            raise smoke_real.SmokeFailure("server booted under critical memory pressure")
        results["_status_before"] = status_snapshot(server.port)
        base = f"http://127.0.0.1:{server.port}"
        traffic = agentic_traffic.Traffic(base, model)
        contaminated = threading.Event()
        stop_watch = threading.Event()
        own = set(server.native_pids())

        def watch_conflicts():
            while not stop_watch.is_set():
                if not allow_foreign_engines() and set(native_engine_pids()) - own:
                    contaminated.set()
                stop_watch.wait(1)

        watcher = threading.Thread(target=watch_conflicts, daemon=True)
        watcher.start()
        try:
            if not wave_only:
                run_sequential(traffic, results, max_tokens, contaminated)
            if short_chain_wave:
                seed_short_chain_corpus(base)
            run_wave(base, model, results, max(256, max_tokens), contaminated, hetero_wave, short_chain_wave)
        finally:
            stop_watch.set()
            watcher.join(3)
        if contaminated.is_set():
            raise smoke_real.SmokeFailure("foreign native engine contaminated benchmark")
        results["_corpus"] = status_corpus(server.port)
        results["_status_after"] = status_snapshot(server.port)
    except Exception as error:
        raise smoke_real.SmokeFailure(
            f"{error}\nserver tail:\n{server.tail()}"
        ) from error
    finally:
        server.close()
    return results


def validate_short_chain_results(arm_results):
    """Fail activation probes that produced no short mixed chains — or no
    clipped rows when the clipping arm explicitly requested them."""
    for name, record in arm_results.items():
        status = record.get("aggregate", {}).get("_status_after") or {}
        ngram = status.get("ngram") or {}
        exact = ngram.get("chain_lengths_exact") or []
        short = sum(exact[1:8]) if len(exact) >= 8 else 0
        mixed = ngram.get("mixed_lanes", 0)
        if not mixed:
            raise RuntimeError(f"arm {name}: short-chain probe mixed no lanes")
        if not short:
            raise RuntimeError(f"arm {name}: short-chain probe produced no length 1..7 chains")
        clip_enabled = (record.get("env") or {}).get("RICHENGINE_NGRAM_CHAIN_CLIP") == "1"
        if clip_enabled and not ngram.get("verify_rows_clipped", 0):
            raise RuntimeError(f"arm {name}: chain clipping enabled but clipped zero rows")


def aggregate_runs(runs):
    """Median aggregate over repeated server boots, with the modal hash."""
    keys = []
    for run in runs:
        for key in run:
            if not key.startswith("_") and key not in keys:
                keys.append(key)
    aggregate = {}
    for key in keys:
        entries = [run[key] for run in runs if key in run]
        item = {}
        for metric in ("tok_s", "acc", "off"):
            values = [entry.get(metric) for entry in entries]
            values = [value for value in values if isinstance(value, (int, float))]
            if values:
                item[metric] = statistics.median(values)
        hashes = [entry.get("hash") for entry in entries if entry.get("hash")]
        if hashes:
            item["hash"] = max(set(hashes), key=hashes.count)
            item["hash_variants"] = len(set(hashes))
        if any("error" in entry for entry in entries):
            item["error"] = next(
                entry["error"] for entry in entries if "error" in entry
            )
        aggregate[key] = item
    aggregate["_repetitions"] = len(runs)
    aggregate["_status_before"] = runs[0].get("_status_before") if runs else None
    aggregate["_status_after"] = runs[-1].get("_status_after") if runs else None
    aggregate["_corpus"] = runs[-1].get("_corpus") if runs else None
    return aggregate


def repetition_order(arms, repeat, order):
    if order == "serial" or len(arms) != 2:
        return [(arm_index, rep) for arm_index in range(len(arms)) for rep in range(repeat)]
    sequence = []
    for rep in range(repeat):
        arm_order = (0, 1) if rep % 2 == 0 else (1, 0)
        sequence.extend((arm_index, rep) for arm_index in arm_order)
    return sequence


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
                f"acc {base.get('acc', 0):7.1f}->{new.get('acc', 0):7.1f}  "
                f"off {base.get('off', 0):7.1f}->{new.get('off', 0):7.1f}{flag}"
            )
        print(f"hash diffs: {hash_diffs}")
        ratios = []
        for key in keys:
            base, new = baseline[key], other.get(key)
            if base.get("tok_s") and new and new.get("tok_s"):
                ratios.append(new["tok_s"] / base["tok_s"])
        if ratios:
            geomean = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
            print(
                f"tok/s ratio: geomean {geomean:.4f} "
                f"median {statistics.median(ratios):.4f} "
                f"mean {statistics.mean(ratios):.4f} samples {len(ratios)}"
            )


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
        "--max-memory",
        default="36G",
        help="per-server native memory cap passed to --max-memory (default 36G)",
    )
    parser.add_argument(
        "--arm",
        type=parse_arm,
        action="append",
        default=None,
        metavar="NAME=JSONENV",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--wave-only",
        action="store_true",
        help="skip single-lane sessions; useful for width-gated proposals",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse matching completed repetitions from --out",
    )
    parser.add_argument("--order", choices=("serial", "abba"), default="serial")
    parser.add_argument(
        "--hetero-wave",
        action="store_true",
        help="use different per-lane wave prompts instead of four copies",
    )
    parser.add_argument(
        "--short-chain-wave",
        action="store_true",
        help="use divergent JSON lanes designed for short corpus chains",
    )
    parser.add_argument(
        "--validate-short-chain",
        action="store_true",
        help="fail unless short chains activate, and clipping saves rows when enabled",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--package",
        type=Path,
        help="resolved model package when it differs from install/models/MODEL",
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    arms = args.arm or [
        ("corpus-on", {}),
        ("corpus-off", {"RICHENGINE_NGRAM_CORPUS": "0"}),
    ]
    binary = Path(args.binary).resolve()
    arm_results = {
        name: {"env": overlay, "runs": [None] * args.repeat}
        for name, overlay in arms
    }
    if args.resume and args.out and args.out.exists():
        try:
            prior = json.loads(args.out.read_text())
            if prior.get("model") == args.model:
                for name, overlay in arms:
                    old = prior.get("arms", {}).get(name)
                    if not old or old.get("env") != overlay:
                        continue
                    runs = old.get("runs", [])
                    for index in range(min(args.repeat, len(runs))):
                        arm_results[name]["runs"][index] = runs[index]
        except (OSError, ValueError):
            pass
    for arm_index, repetition in repetition_order(arms, args.repeat, args.order):
        name, overlay = arms[arm_index]
        if arm_results[name]["runs"][repetition] is not None:
            print(f"arm {name} rep {repetition + 1}/{args.repeat}: resumed", flush=True)
            continue
        print(
            f"arm {name} rep {repetition + 1}/{args.repeat}: "
            f"{overlay or 'ambient env'}",
            flush=True,
        )
        arm_results[name]["runs"][repetition] = run_arm(
            binary, args.model, args.ctx, overlay, args.max_tokens,
            args.max_memory, args.wave_only, args.package, args.hetero_wave, args.short_chain_wave,
        )
        if args.out:
            args.out.write_text(json.dumps({
                "schema_version": 1,
                "complete": False,
                "model": args.model,
                "repeat": args.repeat,
                "order": args.order,
                "max_memory": args.max_memory,
                "arms": arm_results,
            }, indent=1))

    results = {}
    for name, record in arm_results.items():
        if any(run is None for run in record["runs"]):
            raise RuntimeError(f"arm {name} did not complete every repetition")
        record["aggregate"] = aggregate_runs(record["runs"])
        results[name] = record["aggregate"]

    if args.out:
        payload = {
            "schema_version": 1,
            "complete": True,
            "model": args.model,
            "repeat": args.repeat,
            "order": args.order,
            "max_memory": args.max_memory,
            "arms": arm_results,
        }
        args.out.write_text(json.dumps(payload, indent=1))
    if args.validate_short_chain:
        validate_short_chain_results(arm_results)
    if len(arms) > 1:
        diff_table([name for name, _ in arms], results)
    else:
        print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
