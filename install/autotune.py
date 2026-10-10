"""Autotune: measure a model's optional engine paths on this chip and keep
the winners.

The engine's RICHENGINE_* switches are read once per process, so each
candidate runs as its own serve-native child under install/tuning's per-model
knob tables — gated by the chip, its unified-memory bandwidth and the
install's draft kind and target format, which is how the sweep skips paths
known slower than they can pay. Prompts are cut from real text by the
model's own tokenizer, so draft and n-gram knobs see a realistic acceptance
rate rather than the near-zero one arbitrary ids produced. Every candidate
is measured single-lane and at the full batch width — decode and prefill
both — each width REPETITIONS times, and a knob is kept only when some
metric improves past IMPROVEMENT while none of the others regress past it.
A recheck pass then gives every knob that won nothing one more try against
the final env, where a knob that only pays in combination can now win. The
result is a tuning.json beside the model's record, keyed by chip and
stamped with the engine binary it ran on; server/model_host.py applies it
as the engine's environment on every load while that binary is unchanged.
"""

import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

from . import layout, models
from .tuning import TuneContext, eligible, knobs_for

# The tuned result: beside the assembly's model.json, or beside the
# selection link for packages — a package's directory can sit inside the
# shared Hugging Face cache, where eviction would silently delete the
# record: {chip: {schema, engine, env, best, measured, verdicts, ...}}.
# Schema 2 added the engine fingerprint, the knob-table signature,
# per-width prefill metrics and resumable state; schema-1 entries measured
# a different workload and are ignored.
TUNING_RECORD = "tuning.json"
TUNING_SCHEMA = 2

# The synthetic workload: a fixed prompt long enough to exercise chunked
# prefill, greedy decode to a fixed output, warm-up excluded.
PROMPT_TOKENS = 512
OUTPUT_TOKENS = 128
# A full-width warm-up batch long enough to cover the EWMA warm-ups the
# n-gram knobs sweep (RICHENGINE_NGRAM_WARMUP reaches 16 rounds).
WARMUP_OUTPUT_TOKENS = 64
# The engine's lane count (ExecutionLimits::maximumBatchWidth) — the batched
# measurement submits a full batch.
BATCH_WIDTH = 4
# A candidate must move a metric by more than this share to count as an
# improvement — or as a regression; smaller deltas are run-to-run noise.
IMPROVEMENT = 0.015
# Each width is measured this many times inside one engine and the medians
# are compared; a single pass read scheduler noise as signal.
REPETITIONS = 3
# A measurement taken while the host is thermally throttled or critically
# memory-pressured is flagged and never kept. The sweep waits this long for
# pressure to clear before giving up and measuring anyway.
PRESSURE_WAIT_SECONDS = 90.0
PRESSURE_POLL_SECONDS = 5.0

# Bad-path pruning. An ordered knob's first candidate that loses by more
# than this decode share is deep — the remaining values deviate further
# from the default and are skipped unmeasured. The same depth decides which
# first-pass losers the interaction recheck still owes a second try.
PRUNE_DEPTH = 0.08
RECHECK_DEPTH = 0.08

# Cross-model priors, shared under the models root:
# {chip: {family: {"KNOB=value": {"wins": n, "losses": n}}}} — a knob's
# history pools only within its model family, since one family's loser
# says little about another's. Records written by older sweeps pool the
# stats flat under the chip; reads still fold them in so nothing learned
# is lost. A Quick sweep skips candidates whose Beta(1+wins, 1+losses)
# win rate cannot reach the bound even at its optimistic edge after
# enough observations; Complete measures them anyway.
PRIORS_NAME = ".tuning-priors.json"
PRIORS_MIN_OBSERVATIONS = 5
PRIORS_SKIP_BOUND = 0.15

# The kernel pass: one run of the engine's kernel tuner (paths.TUNE_BINARY),
# whose paired in-process measurement fits each affine Linear plan to this
# machine — seconds per workload where an env knob costs a whole engine
# launch. Its winners reach every later measurement through this env var,
# which ops/DeviceTuning.cpp parses and lets outrank the shipped tables.
# Seconds bound each workload's measurement; Quick spends less per key.
KERNEL_PLANS_ENV = "RICHENGINE_LINEAR_PLANS"
KERNEL_PASS_SECONDS = {"quick": 5, "complete": 10}

# machdep.cpu.brand_string's chip → the Metal GPU family generation (Apple7
# is M1 … Apple10+ is M4 and newer) and its unified-memory bandwidth in GB/s.
_GPU_FAMILY = {"M1": 7, "M2": 8, "M3": 9, "M4": 10, "M5": 11}
_BANDWIDTH_GBPS = {
    "M1": 68,
    "M1 Pro": 200,
    "M1 Max": 400,
    "M1 Ultra": 800,
    "M2": 100,
    "M2 Pro": 200,
    "M2 Max": 400,
    "M2 Ultra": 800,
    "M3": 100,
    "M3 Pro": 150,
    "M3 Max": 400,
    "M3 Ultra": 800,
    "M4": 120,
    "M4 Pro": 273,
    "M4 Max": 546,
    "M5": 153,
    "M5 Pro": 307,
    "M5 Max": 614,
}

# The sweep's prompt text: plausible user traffic, tiled to whatever length
# the prompt pool needs. Real text is what makes the draft knobs' measured
# acceptance rate mean anything.
_CORPUS = (
    "The deployment checklist says the gateway cert rotates on the first "
    "of the month, but the staging cluster still presents the old chain on "
    "half its pods. Write me the kubectl commands to find every ingress "
    "whose TLS secret predates last Tuesday, then draft the rollback note "
    "for the change record — keep it under a page, and flag anything that "
    "touches the payments namespace for review before it runs. Once that "
    "is done, summarize the three slowest queries from last night's "
    "report: the join on orders and shipments is suspected, but the "
    "planner output suggests the index on customer_id is being skipped "
    "because of the implicit cast. Propose the migration that fixes the "
    "column type, note whether it can run online on our Postgres version, "
    "and estimate the lock window. If the estimate is over five minutes, "
    "outline the pt-online-schema-change fallback instead.\n\n"
    "For the mobile release, the crash report shows a spike in "
    "WatchdogTermination on iPhone 12 devices since the 4.2 rollout, all "
    "inside the sync engine's compaction path. List the plausible causes "
    "in order of likelihood, write the radar-style repro steps for the top "
    "one, and sketch the fix — assume the compactor is holding a file lock "
    "across a suspension boundary. Also review the attached diff that "
    "moves prefetching off the main queue; call out any data races it "
    "introduces and whether the added barrier is actually sufficient on "
    "arm64.\n\n"
    "Finally, the board packet needs a paragraph on infrastructure spend. "
    "The storage line item grew 22% quarter over quarter, driven almost "
    "entirely by retained inference traces from the evaluation harness. "
    "Explain the lifecycle policy we'd adopt to cap it, the trade-off "
    "against debuggability, and one sentence on why the growth is expected "
    "to flatten once sampling rolls out to the smaller tenants."
)


def detect_chip():
    """(chip name, GPU family, memory bandwidth GB/s) of this machine; the
    family and bandwidth are 0 when the chip is not in the table. A 0
    bandwidth then reads as unproven — bandwidth-gated knobs are measured
    rather than excluded — while a 0 family still keeps family-gated knobs
    out, since their path may not exist on an unrecognized GPU."""
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        brand = ""
    chip = brand.removeprefix("Apple").strip() or platform_chip_name()
    family = 0
    bandwidth = 0
    for name in sorted(_BANDWIDTH_GBPS, key=len, reverse=True):
        if chip == name or chip.startswith(name + " "):
            bandwidth = _BANDWIDTH_GBPS[name]
            break
    match = re.match(r"M(\d+)", chip)
    if match:
        family = _GPU_FAMILY.get("M" + match.group(1), 0)
    return chip, family, bandwidth


def platform_chip_name():
    """The bare SoC name when sysctl had no brand string."""
    machine = subprocess.run(
        ["sysctl", "-n", "hw.model"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    ).stdout.strip()
    return machine


def _draft_kind(family):
    """The draft kind a ModelFamily loads, as the Knob draft gate spells it."""
    if family is None or family.draft is None:
        return "none"
    signature = dict(family.draft.signature)
    architectures = signature.get("architectures") or ()
    if "DFlash2DraftModel" in architectures:
        return "dflash2"
    if "DFlashDraftModel" in architectures:
        return "dflash"
    if "markov_head_type" in signature:
        return "dspark"
    return "none"


def context_for(assembly_dir: Path) -> TuneContext:
    """The sweep context of a held assembly or package directory: its family
    record plus this chip's family and bandwidth."""
    from . import families

    kind = models.installation_kind(assembly_dir)
    if kind == models.ASSEMBLY:
        record = models.read_json(assembly_dir / layout.ASSEMBLY_RECORD)
        family_name = record["family"]
        target_format = record["target_format"]
        diffusion = False
    elif kind == models.PACKAGE:
        manifest = models.read_json(assembly_dir / layout.PACKAGE_MANIFEST)
        # Locally packed manifests type their family; Hub snapshots record
        # only the model name, which for these packages is the family.
        family_name = manifest.get("family") or manifest["model"]
        target_format = manifest["format"]["name"]
        diffusion = isinstance(manifest.get("diffusion"), dict)
    else:
        raise models.ModelError(
            f"{assembly_dir} is not an installed model assembly or package"
        )
    family = next((f for f in families.FAMILIES if f.name == family_name), None)
    chip, gpu_family, bandwidth = detect_chip()
    return TuneContext(
        family=family_name,
        chip=chip,
        gpu_family=gpu_family,
        bandwidth_gbps=bandwidth,
        model_type=(
            "" if family is None else dict(family.signature).get("model_type", "")
        ),
        target_format=target_format,
        draft_kind=_draft_kind(family),
        diffusion=diffusion,
    )


def engine_fingerprint(binary: Path) -> dict:
    """Size and mtime of the engine binary and its metallib — the record's
    'measured on' stamp. A rebuilt engine changes them, which is exactly
    when a prior sweep's winners stop being trustworthy."""
    try:
        stat = binary.stat()
    except OSError:
        return {}
    fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    try:
        metallib = (binary.parent / "richengine.metallib").stat()
        fingerprint["metallib_size"] = metallib.st_size
        fingerprint["metallib_mtime_ns"] = metallib.st_mtime_ns
    except OSError:
        pass
    return fingerprint


def _knobs_signature(knobs) -> str:
    """The sweep's identity for resume: which knobs and values it asked."""
    ordered = [[knob.env, *knob.values] for knob in knobs]
    return hashlib.sha1(json.dumps(ordered).encode()).hexdigest()[:16]


def _prompts(assembly_dir: Path, windows: int) -> tuple[list[tuple[int, ...]], bool]:
    """`windows` distinct prompt windows of PROMPT_TOKENS each, cut from the
    corpus by the model's own tokenizer so decode runs on text the draft
    model can actually predict. Distinct windows keep any timed batch off
    the prefix cache a warm-up or earlier repetition primed. Arbitrary ids
    remain the fallback when the install has no readable tokenizer — then
    the second element reports False, since near-zero draft acceptance on
    synthetic ids makes every draft knob's verdict a different workload's."""
    try:
        from tokenizers import Tokenizer

        path = assembly_dir / layout.TOKENIZER / "tokenizer.json"
        ids = list(
            Tokenizer.from_file(str(path)).encode(_CORPUS, add_special_tokens=False).ids
        )
    except Exception:
        ids = []
    needed = windows * PROMPT_TOKENS
    if not ids:
        return [
            tuple((lane * 977 + i) % 65536 + 1 for i in range(PROMPT_TOKENS))
            for lane in range(windows)
        ], False
    ids = (ids * math.ceil(needed / len(ids)))[:needed]
    return [
        tuple(ids[lane * PROMPT_TOKENS : (lane + 1) * PROMPT_TOKENS])
        for lane in range(windows)
    ], True


def _cpu_speed_limit():
    """pmset's CPU speed limit as a percent; under 100 the host is
    thermally throttling. None when the probe fails or reports nothing —
    'pmset -g therm' only prints the limit while it is in force."""
    try:
        output = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", output)
    return int(match.group(1)) if match else None


def _memory_pressure_level():
    """kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical.
    None when the probe fails."""
    try:
        return int(
            subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def system_pressure():
    """A short reason string while the host's timing would be contaminated
    — thermal throttle or critical memory pressure — else None."""
    limit = _cpu_speed_limit()
    if limit is not None and limit < 100:
        return f"thermal throttle (CPU speed limit {limit}%)"
    level = _memory_pressure_level()
    if level is not None and level >= 4:
        return "critical memory pressure"
    return None


def _wait_for_pressure(log):
    """The current pressure reason after waiting for it to clear; None when
    the host is (or became) clean."""
    reason = system_pressure()
    if reason is None:
        return None
    log(f"host shows {reason}; waiting for it to clear")
    deadline = time.monotonic() + PRESSURE_WAIT_SECONDS
    while reason is not None and time.monotonic() < deadline:
        time.sleep(PRESSURE_POLL_SECONDS)
        reason = system_pressure()
    return reason


def _engine_command(binary: Path, assembly_dir: Path) -> list[str]:
    """serve-native argv for a tuning child: the package's own auto limits."""
    return [str(binary), "serve-native", str(assembly_dir), "auto", "auto"]


def _request(prompt: tuple[int, ...], output_tokens: int):
    """One synthetic generation frame: greedy sampling, EOS ignored so
    every run decodes exactly output_tokens."""
    from server import protocol as wire

    return wire.RequestFrame(
        request_id=0,
        priority=wire.RequestPriority.NORMAL,
        absolute_deadline_unix_micros=0,
        remaining_deadline_micros=0,
        logical_max_output_tokens=output_tokens,
        prompt_tokens=prompt,
        sampling=wire.SamplingParameters(),
        seed=0,
        constraint=wire.ConstraintMode.NONE,
        image_spans=(),
        image_pixels=b"",
        return_progress=False,
        score_tokens=(),
        generation_prompt_tokens=0,
        flags=wire.RequestFlag.IGNORE_END_OF_SEQUENCE,
    )


def _run_batch(runtime, prompts, output_tokens: int) -> dict:
    """Submit one generation per prompt; report aggregated decode and
    per-lane prefill throughput from the DoneEvents."""
    from server import runtime as engine_runtime

    calls = [
        runtime.submit(
            engine_runtime.GenerationRequest(
                _request(prompt, output_tokens), math.inf, None, None
            )
        )
        for prompt in prompts
    ]
    prompt_tokens = 0
    completion_tokens = 0
    decode_micros = 0
    prefill_rates = []
    for call in calls:
        event = call.result(timeout=600)
        prompt_tokens += event.prompt_tokens
        completion_tokens += event.completion_tokens
        decode_micros = max(decode_micros, event.decode_micros)
        if event.prefill_micros:
            prefill_rates.append(event.prompt_tokens / (event.prefill_micros / 1e6))
    seconds = decode_micros / 1e6 if decode_micros else 0.0
    return {
        "width": len(prompts),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_micros": decode_micros,
        "tokens_per_second": round(completion_tokens / seconds, 2) if seconds else 0.0,
        # Mean per-lane prefill rate; at width 1 the same single rate.
        "prefill_tokens_per_second": (
            round(statistics.fmean(prefill_rates), 2) if prefill_rates else 0.0
        ),
    }


def _median_samples(samples: list[dict]) -> dict:
    """One width's repetitions merged to medians, repetition count kept."""
    merged = {
        "width": samples[0]["width"],
        "repetitions": len(samples),
    }
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "decode_micros",
        "tokens_per_second",
        "prefill_tokens_per_second",
    ):
        merged[key] = statistics.median(sample[key] for sample in samples)
    return merged


def measure(
    binary: Path,
    assembly_dir: Path,
    env: dict,
    *,
    prompts: list[tuple[int, ...]],
    widths=(1, BATCH_WIDTH),
    repetitions: int = REPETITIONS,
    incumbent: dict | None = None,
) -> tuple[dict, dict]:
    """One engine under `env`: a warmup pass, then REPETITIONS batches at
    each width, merged to medians: {width: metrics}. `prompts` is a pool of
    distinct windows; every batch consumes `width` fresh ones so no timed
    prefill rides the cache a warm-up or earlier repetition primed. Raises
    on a failed start. With `incumbent` ({width: metrics}) the exact
    early-kill applies: once a majority of a width's reps each regress past
    IMPROVEMENT on one metric, the median must regress too — the candidate
    can never be kept, so the rest of its batches are skipped. Returns
    (medians, samples), samples as {width: {metric: [per-rep values]}} —
    the baseline's samples seed the keep rule's noise floor."""
    from server import runtime as engine_runtime

    runtime = engine_runtime.MultiplexedRuntime(
        _engine_command(binary, assembly_dir),
        # Popen replaces the child's whole environment; overlay the swept
        # knobs on this process's.
        env={**os.environ, **env},
        startup_timeout=600.0,
        pending_limit=max(widths),
        eager_start=True,
    )
    try:
        if not runtime.wait_ready():
            raise RuntimeError("the tuning engine did not become ready")
        cursor = 0

        def batch(width: int, output_tokens: int) -> dict:
            nonlocal cursor
            window = prompts[cursor : cursor + width]
            assert len(window) == width, "prompt pool exhausted"
            cursor += width
            return _run_batch(runtime, window, output_tokens)

        batch(max(widths), WARMUP_OUTPUT_TOKENS)
        merged = {}
        samples = {}
        doomed = False
        for width in widths:
            reps = []
            while len(reps) < repetitions and not doomed:
                reps.append(batch(width, OUTPUT_TOKENS))
                base = (incumbent or {}).get(str(width)) or {}
                # A majority of reps each regressing past IMPROVEMENT on the
                # same metric means the median regresses — guaranteed loss.
                if incumbent is not None and any(
                    before > 0
                    and sum(
                        (rep.get(metric) or 0.0) / before - 1 < -IMPROVEMENT
                        for rep in reps
                    )
                    * 2
                    > repetitions
                    for metric in _METRICS
                    for before in [base.get(metric) or 0.0]
                ):
                    doomed = True
            merged[str(width)] = _median_samples(reps)
            samples[str(width)] = {
                metric: [rep.get(metric, 0.0) for rep in reps] for metric in _METRICS
            }
            if doomed:
                break
        return merged, samples
    finally:
        runtime.close()


_METRICS = ("tokens_per_second", "prefill_tokens_per_second")


def _keeps(candidate: dict, incumbent: dict, sigma: dict | None = None) -> bool:
    """True when the candidate wins somewhere without losing anywhere: a
    metric must improve past IMPROVEMENT at some width and none may regress
    past it at any width — the 'wins at both' the sweep always claimed,
    extended to prefill so a decode-only metric can no longer hide a prefill
    regression (or keep a pure-noise 'win'). When `sigma` carries the
    baseline's per-rep relative spread per (width, metric), the bar rises
    to max(IMPROVEMENT, 2σ): a delta inside the measured noise floor counts
    as neither win nor loss."""
    improved = False
    for width, cand in candidate.items():
        base = incumbent.get(width)
        if not base:
            continue
        for metric in _METRICS:
            after = cand.get(metric) or 0.0
            before = base.get(metric) or 0.0
            if after <= 0 or before <= 0:
                continue
            floor = IMPROVEMENT
            if sigma is not None:
                floor = max(floor, 2 * sigma.get((width, metric), 0.0))
            delta = after / before - 1
            if delta < -floor:
                return False
            improved = improved or delta > floor
    return improved


def _baseline_sigma(samples: dict) -> dict:
    """Relative spread per (width, metric) of the baseline's per-rep
    values — the measurement-noise floor the keep rule reads."""
    out = {}
    for width, per_metric in samples.items():
        if not isinstance(per_metric, dict):
            continue
        for metric, values in per_metric.items():
            values = [v for v in values if type(v) in (int, float) and v > 0]
            if len(values) < 2:
                continue
            mean = statistics.fmean(values)
            if mean > 0:
                out[(width, metric)] = statistics.stdev(values) / mean
    return out


def _deep_loss(candidate: dict, incumbent: dict, depth: float) -> bool:
    """The candidate's decode delta regresses past `depth` at every width
    it measured — beyond recovery for this candidate, and for an ordered
    knob's remaining values, which deviate further from the default."""
    deltas = []
    for width, cand in candidate.items():
        base = incumbent.get(width)
        if not base:
            continue
        before = base.get("tokens_per_second") or 0.0
        after = cand.get("tokens_per_second") or 0.0
        if before > 0 and after > 0:
            deltas.append(after / before - 1)
    return bool(deltas) and max(deltas) < -depth


def _read_priors(path: Path) -> dict:
    """The priors document, {} when absent or unreadable."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _family_stats(doc: dict, chip: str, family: str) -> dict:
    """The (chip, family) candidate stats, merged with the chip's legacy
    flat pool — records written before priors keyed by family."""
    chip_doc = doc.get(chip)
    if not isinstance(chip_doc, dict):
        return {}
    stats = {key: value for key, value in chip_doc.items() if "=" in key}
    bucket = chip_doc.get(family)
    if isinstance(bucket, dict):
        stats.update(bucket)
    return stats


def _prior_stats(doc: dict, chip: str, family: str, *, engine: str, knobs: str) -> dict:
    """The family stats a sweep may trust: the file's fingerprint must
    match this engine and knob table — a rebuilt binary or a changed
    table expires every prior at once. Files without a fingerprint
    (written before priors were stamped) still read, matching how
    tuning.json's own resume check treats legacy records here: stats
    older than stamping carry no way to know their engine."""
    meta = doc.get("_meta")
    if isinstance(meta, dict) and (
        meta.get("engine") != engine or meta.get("knobs") != knobs
    ):
        return {}
    return _family_stats(doc, chip, family)


def _prior_skips(stats: dict, knob, value: str) -> bool:
    """Whether Quick skips this candidate: with enough observations, the
    Beta(1+wins, 1+losses) posterior's optimistic edge — mean + 2σ — still
    under the win-rate bound means it has never paid on this chip."""
    entry = stats.get(f"{knob.env}={value}")
    if not isinstance(entry, dict):
        return False
    wins = entry.get("wins", 0) or 0
    losses = entry.get("losses", 0) or 0
    n = wins + losses
    if n < PRIORS_MIN_OBSERVATIONS:
        return False
    mean = (wins + 1) / (n + 2)
    spread = math.sqrt(mean * (1 - mean) / (n + 3))
    return mean + 2 * spread < PRIORS_SKIP_BOUND


def _update_priors(
    path: Path,
    chip: str,
    family: str,
    results: dict,
    best_env: dict,
    *,
    engine: str,
    knobs: str,
):
    """Fold a finished sweep's verdicts into the (chip, family) priors: a
    candidate that made the final env counts as a win, every other clean
    measurement a loss. Pressured entries teach nothing and stay out;
    elimination measurements are the absence of a knob, not a candidate,
    and stay out too. The file stamps the engine and knob-table
    fingerprints its wins were measured under — a rebuilt engine or a
    changed table expiries every prior, like tuning.json's own check."""
    doc = _read_priors(path)
    doc["_meta"] = {"engine": engine, "knobs": knobs}
    stats = doc.setdefault(chip, {}).setdefault(family, {})
    for key, entry in results.items():
        if (
            key == "baseline"
            or key.startswith("eliminate:")
            or not isinstance(entry, dict)
            or entry.get("pressured")
        ):
            continue
        name, _, value = key.removeprefix("recheck:").partition("=")
        if not name or not value:
            continue
        slot = stats.setdefault(f"{name}={value}", {"wins": 0, "losses": 0})
        slot["wins" if best_env.get(name) == value else "losses"] += 1
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)


def tuning_path(assembly_dir: Path) -> Path:
    """The model's tuning record. Assemblies keep it beside model.json.
    Packages — whose directory can live inside the shared Hugging Face
    cache — keep it beside the selection link under models_root instead,
    so cache eviction cannot delete it."""
    if models.installation_kind(assembly_dir) == models.PACKAGE:
        return assembly_dir.parent / (assembly_dir.name + ".tuning.json")
    return assembly_dir / TUNING_RECORD


def _read_record(assembly_dir: Path) -> dict:
    """The whole tuning.json document, {} when absent or unreadable. A
    package's record may still sit at the legacy path inside its
    directory — written before records moved beside the link."""
    for path in dict.fromkeys(
        (tuning_path(assembly_dir), assembly_dir / TUNING_RECORD)
    ):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            return document
    return {}


def _write_record(assembly_dir: Path, record: dict):
    """Atomically replace tuning.json — an interrupted write never leaves
    half a record."""
    path = tuning_path(assembly_dir)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, path)


def _entry(assembly_dir: Path, chip: str | None = None) -> dict | None:
    """This chip's record entry, whatever its schema, or None."""
    if chip is None:
        chip, _, _ = detect_chip()
    entry = _read_record(assembly_dir).get(chip)
    return entry if isinstance(entry, dict) else None


def load_tuning(
    assembly_dir: Path, chip: str | None = None, binary: Path | None = None
) -> dict:
    """The chip's tuned env for this model, {} when untuned, another
    chip's, an old schema's, or — when `binary` is given — recorded against
    a different engine build's."""
    entry = _entry(assembly_dir, chip)
    if entry is None or entry.get("schema") != TUNING_SCHEMA:
        return {}
    recorded = entry.get("engine")
    if binary is not None and recorded and recorded != engine_fingerprint(binary):
        return {}
    env = entry.get("env")
    return dict(env) if isinstance(env, dict) else {}


def tuning_note(
    assembly_dir: Path, chip: str | None = None, binary: Path | None = None
) -> str | None:
    """Why this chip's record is not applied, for the load log; None when
    there is no record or it applies cleanly."""
    if chip is None:
        chip, _, _ = detect_chip()
    entry = _entry(assembly_dir, chip)
    if entry is None:
        return None
    if entry.get("schema") != TUNING_SCHEMA:
        return "recorded by an older sweep; re-run 'richengine tune'"
    recorded = entry.get("engine")
    if binary is not None and recorded and recorded != engine_fingerprint(binary):
        return "measured on another engine build; re-run 'richengine tune'"
    return None


def tuning_state(
    assembly_dir: Path, chip: str | None = None, binary: Path | None = None
) -> str:
    """This chip's tune state for the model: "untuned" (no record for the
    chip at all), "stale" (a record load_tuning rejects — an old schema's
    or another engine build's), "partial" (a valid record an interrupted
    or running sweep has not finished; its winners already apply) or
    "tuned" (a finished sweep's, whether it kept knobs or not)."""
    if chip is None:
        chip, _, _ = detect_chip()
    entry = _entry(assembly_dir, chip)
    if entry is None:
        return "untuned"
    if tuning_note(assembly_dir, chip=chip, binary=binary) is not None:
        return "stale"
    return "tuned" if entry.get("complete") else "partial"


def tuned_environment(
    assembly_dir: Path, chip: str | None = None, binary: Path | None = None
) -> dict | None:
    """The process environment for an engine serving this assembly: the
    tuned knobs merge over the caller's, an explicitly set RICHENGINE_* or
    a user's own value always winning over the tuned one. None — the child
    inherits — when the model is untuned."""
    tuned = load_tuning(assembly_dir, chip=chip, binary=binary)
    if not tuned:
        return None
    env = dict(os.environ)
    for name, value in tuned.items():
        env.setdefault(name, str(value))
    return env


def _kernel_pass(
    binary: Path, assembly_dir: Path, *, seconds: float, log=print
) -> dict | None:
    """The engine's kernel tuner over this model's projection plans: one
    process, every candidate measured paired against its policy default
    through the production encoders. Returns the record fragment —
    {"measured", "changed", "spec", "keys"} where spec is the
    RICHENGINE_LINEAR_PLANS value engine plan selection reads — an
    {"error"} marker when the pass ran and failed, or None when there is
    no tuner to run, which a resume retries rather than remembers."""
    from . import paths

    tool = paths.TUNE_BINARY
    if not tool.is_file():
        log(f"kernels · no kernel tuner at {tool} — pass skipped")
        return None
    report_path = tuning_path(assembly_dir).with_suffix(".kernel-report.json")
    command = [
        str(tool),
        str(binary.parent / "richengine.metallib"),
        str(assembly_dir),
        "--seconds",
        str(seconds),
        "--json",
        str(report_path),
    ]
    log(f"kernels · fitting the model's Linear plans ({seconds:g} s per key)")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # The launcher's own process group — the API's killpg takes the
            # tuner down with a cancelled sweep.
        )
    except OSError as error:
        log(f"kernels · could not start the kernel tuner: {error}")
        return {"error": str(error)}
    for line in process.stdout or ():
        log(line.rstrip("\n"))
    returncode = process.wait()
    if returncode != 0:
        return {"error": f"the kernel tuner exited {returncode}"}
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"error": "the kernel tuner wrote no report"}
    keys = {
        key["workload"]: key["winner"]
        for key in report.get("keys", [])
        if isinstance(key, dict) and key.get("changed") and key.get("complete")
    }
    measured = sum(isinstance(key, dict) for key in report.get("keys", []))
    return {
        "measured": measured,
        "changed": len(keys),
        "spec": ";".join(f"{workload}={winner}" for workload, winner in keys.items()),
        "keys": keys,
    }


def tune(
    binary: Path,
    assembly_dir: Path,
    *,
    include_quality: bool = False,
    interactions: bool = True,
    kernel_pass: bool = True,
    priors_path: Path | None = None,
    mode: str | None = None,
    contaminated: str | None = None,
    log=print,
) -> dict:
    """Sweep this model's eligible knobs on this chip: baseline, then each
    knob's candidates against the running-best env, keeping improvements,
    then one recheck pass for knobs that won nothing, then a backward-
    elimination pass dropping winners the final env no longer needs.
    Writes tuning.json after every measurement — an interrupted sweep
    resumes where it stopped — and returns the tuned env. With
    `priors_path`, a finished sweep folds its verdicts into the chip's
    shared priors; a flagless (Quick) sweep also consults them and skips
    candidates that have never won on this chip. `contaminated` names a
    condition that invalidates every measurement for keeping purposes — a
    loaded engine sharing the GPU; entries still record what ran, flagged,
    but nothing can win and nothing reaches the priors."""
    context = context_for(assembly_dir)
    tuning = knobs_for(context.family)
    if tuning is None:
        raise models.ModelError("no tunable-knob table for this model's family")
    all_eligible = eligible(tuning, context)
    quality_skipped = sum(1 for knob in all_eligible if knob.quality_sensitive)
    # A variable the user exported reaches every measurement, and at serve
    # time model_host's setdefault keeps it over any tuned winner — a kept
    # value could never apply, so the knob is not swept at all.
    user_set = {knob.env for knob in all_eligible if knob.env in os.environ}
    knobs = tuple(
        knob
        for knob in all_eligible
        if (include_quality or not knob.quality_sensitive) and knob.env not in user_set
    )
    log(
        f"chip {context.chip} · family {context.gpu_family} · "
        f"{context.bandwidth_gbps} GB/s · {len(knobs)} of "
        f"{len(tuning.knobs)} knobs eligible"
    )
    if quality_skipped:
        log(
            f"{quality_skipped} quality-trading knobs stay out; "
            "--include-quality-knobs sweeps them too"
        )
    if user_set:
        log(
            f"{', '.join(sorted(user_set))} already set in the "
            "environment — excluded; the value always wins at serve"
        )
    inherited = sorted(
        name
        for name in os.environ
        if name.startswith("RICHENGINE_") and name not in user_set
    )
    if inherited:
        log(
            f"inherited {', '.join(inherited)} reach every candidate and "
            "shift the baseline; they are not measured"
        )
    if contaminated:
        log(
            f"measuring under {contaminated} — results are recorded but "
            "cannot win and never reach the priors"
        )

    widths = (1, BATCH_WIDTH)
    prompts, real_prompts = _prompts(
        assembly_dir, max(widths) + REPETITIONS * sum(widths)
    )
    if not real_prompts and any(knob.draft_kinds for knob in knobs):
        # Synthetic ids decode to text no draft predicts: acceptance sits
        # near zero and every draft knob's verdict measures a workload
        # real traffic never sees. Drop them like the user-set ones.
        knobs = tuple(knob for knob in knobs if not knob.draft_kinds)
        log(
            "synthetic prompts — no readable tokenizer.json; draft knobs "
            "stay out, they would measure near-zero acceptance"
        )
    fingerprint = engine_fingerprint(binary)
    signature = _knobs_signature(knobs)

    record = _read_record(assembly_dir)
    prior = record.get(context.chip)
    results: dict = {}
    verdicts: dict = {}
    best_env: dict[str, str] = {}
    best_metrics: dict | None = None
    # The kernel pass's record fragment: the measured Linear plan winners,
    # or an {"error"} marker that keeps a resume from re-running a pass
    # that already failed. A resumable entry carries its own.
    plans: dict | None = None
    if (
        isinstance(prior, dict)
        and prior.get("schema") == TUNING_SCHEMA
        and prior.get("engine") == fingerprint
        and prior.get("knobs") == signature
        and not prior.get("complete")
        and isinstance(prior.get("measured"), dict)
        and isinstance(prior["measured"].get("baseline"), dict)
        and not prior["measured"]["baseline"].get("pressured")
        and isinstance(prior.get("env"), dict)
        and prior.get("best")
    ):
        # Pressured measurements never won a knob; drop them so a resume
        # re-measures them clean rather than skipping them forever. A
        # pressured baseline leaves "baseline" out, restarting the sweep.
        results = {
            key: entry
            for key, entry in prior["measured"].items()
            if not (isinstance(entry, dict) and entry.get("pressured"))
        }
        if isinstance(prior.get("verdicts"), dict):
            verdicts = dict(prior["verdicts"])
        best_env = dict(prior["env"])
        best_metrics = prior["best"]
        if isinstance(prior.get("plans"), dict):
            plans = prior["plans"]
        log(f"resuming · {len(results) - 1} candidates already measured")

    sigma = _baseline_sigma((results.get("baseline") or {}).get("samples") or {})

    def write():
        record[context.chip] = {
            "schema": TUNING_SCHEMA,
            "engine": fingerprint,
            "knobs": signature,
            "mode": mode
            or ("complete" if include_quality and interactions else "quick"),
            "gpu_family": context.gpu_family,
            "bandwidth_gbps": context.bandwidth_gbps,
            "env": best_env,
            "best": best_metrics,
            "measured": results,
            "verdicts": verdicts,
            "plans": plans,
            "complete": False,
            "updated_unix": int(time.time()),
        }
        _write_record(assembly_dir, record)

    # The kernel pass fits the model's affine Linear plans to this machine
    # once, before the env sweep: its winners ride every later measurement
    # through RICHENGINE_LINEAR_PLANS in best_env — both modes run it, Quick
    # with the smaller per-key budget. A user-set value always wins at
    # serve, so the pass is skipped then; a contaminated sweep cannot keep
    # anything and skips the pass too. Its record is written before any
    # measurement so an interrupted sweep resumes past it.
    if plans is None and kernel_pass and not contaminated:
        if KERNEL_PLANS_ENV in os.environ:
            log(
                f"kernels · {KERNEL_PLANS_ENV} already set in the "
                "environment — pass skipped; the value always wins at serve"
            )
            plans = {}
        else:
            plans = (
                _kernel_pass(
                    binary,
                    assembly_dir,
                    seconds=KERNEL_PASS_SECONDS[
                        mode
                        or (
                            "complete" if include_quality and interactions else "quick"
                        )
                    ],
                    log=log,
                )
                or {}
            )
        write()

    def sweep(candidates, prefix=""):
        """Measure each (knob, value) against the running best; keep the
        winners. `prefix` namespaces recheck keys from first-pass ones.
        Candidates are contiguous per knob, so an ordered knob whose first
        value deep-loses prunes the rest of its values unmeasured — the
        verdict remembers them so no later pass re-asks the question."""
        nonlocal best_env, best_metrics
        baseline_metrics = (results.get("baseline") or {}).get("metrics") or {}
        dead: set[str] = set()
        for knob, value in candidates:
            candidate_key = f"{knob.env}={value}"
            if knob.env in dead:
                verdicts.setdefault(candidate_key, "pruned")
                continue
            if best_env.get(knob.env) == value:
                continue
            key = prefix + candidate_key
            if key in results:
                continue
            env = dict(best_env)
            env[knob.env] = value
            pressured = contaminated or _wait_for_pressure(log)
            log(f"trying {knob.env}={value}")
            try:
                measured, _ = measure(
                    binary,
                    assembly_dir,
                    env,
                    prompts=prompts,
                    incumbent=best_metrics,
                )
            except (OSError, RuntimeError) as error:
                log(f"{knob.env}={value} · failed ({error})")
                verdicts[candidate_key] = "failed"
                continue
            log(f"{knob.env}={value} · {measured}")
            results[key] = {"env": env, "metrics": measured}
            if pressured:
                results[key]["pressured"] = pressured
                verdicts[candidate_key] = "pressured"
            elif _keeps(measured, best_metrics, sigma):
                best_env[knob.env] = value
                best_metrics = measured
                verdicts[candidate_key] = "kept"
                log("  kept")
            elif knob.ordered and _deep_loss(measured, baseline_metrics, PRUNE_DEPTH):
                dead.add(knob.env)
                verdicts[candidate_key] = "rejected"
                log(
                    f"  rest of {knob.env} skipped — ordered knob's "
                    "first value lost beyond prune depth"
                )
            else:
                verdicts[candidate_key] = "rejected"
            write()

    if best_metrics is None:
        pressured = contaminated or _wait_for_pressure(log)
        best_metrics, baseline_samples = measure(
            binary, assembly_dir, {}, prompts=prompts
        )
        sigma = _baseline_sigma(baseline_samples)
        log(f"baseline · {best_metrics}")
        results["baseline"] = {
            "env": {},
            "metrics": best_metrics,
            "samples": baseline_samples,
        }
        if pressured:
            results["baseline"]["pressured"] = pressured
        write()

    # The kernel pass's winners as candidate zero: measured once
    # whole-model against the baseline, kept only when the keep rule says
    # the fitted plans pay end to end — the same A/B gate the shipped
    # DeviceTuning tables took before a policy change. On a resume the
    # entry already sits in results and in best_env when it kept.
    if plans and plans.get("spec") and "kernels" not in results:
        pressured = contaminated or _wait_for_pressure(log)
        log(
            f"kernels · {plans['changed']} of {plans['measured']} plans "
            "changed — measuring the fitted env"
        )
        try:
            kernel_metrics, _ = measure(
                binary,
                assembly_dir,
                {KERNEL_PLANS_ENV: plans["spec"]},
                prompts=prompts,
                incumbent=best_metrics,
            )
        except (OSError, RuntimeError) as error:
            log(f"kernels · fitted env failed ({error})")
        else:
            log(f"kernels · {kernel_metrics}")
            results["kernels"] = {
                "env": {KERNEL_PLANS_ENV: plans["spec"]},
                "metrics": kernel_metrics,
            }
            if pressured:
                results["kernels"]["pressured"] = pressured
                verdicts["kernels"] = "pressured"
            elif _keeps(kernel_metrics, best_metrics, sigma):
                best_env[KERNEL_PLANS_ENV] = plans["spec"]
                best_metrics = kernel_metrics
                verdicts["kernels"] = "kept"
                log("  kept")
            else:
                verdicts["kernels"] = "rejected"
            write()

    first_pass_candidates = [(knob, value) for knob in knobs for value in knob.values]
    if priors_path is not None and not include_quality and not interactions:
        # A Quick sweep trusts the chip's priors; any flag widens the scope
        # past Quick and every candidate is measured regardless of history.
        stats = _prior_stats(
            _read_priors(priors_path),
            context.chip,
            context.family,
            engine=fingerprint,
            knobs=signature,
        )
        measured_once = [
            (knob, value)
            for knob, value in first_pass_candidates
            if not _prior_skips(stats, knob, value)
        ]
        for knob, value in first_pass_candidates:
            if (knob, value) not in measured_once:
                verdicts[f"{knob.env}={value}"] = "prior-skipped"
        pruned = len(first_pass_candidates) - len(measured_once)
        if pruned:
            log(
                f"priors · {pruned} candidates never won for this family "
                "on this chip — skipped in Quick; --complete measures "
                "everything"
            )
        first_pass_candidates = measured_once
    # The count feeds the job's progress; on a resume, only what the sweep
    # still has to measure counts — best_env hits and measured keys skip.
    remaining = [
        (knob, value)
        for knob, value in first_pass_candidates
        if best_env.get(knob.env) != value and f"{knob.env}={value}" not in results
    ]
    log(f"sweep · {len(remaining)} candidates")
    sweep(first_pass_candidates)
    if interactions:
        # Knobs that won nothing get one more try against the final env: a
        # knob that only pays alongside a winner — or that lost against a
        # weaker incumbent — can win now.
        recheck = [
            (knob, value)
            for knob in knobs
            if knob.env not in best_env
            for value in knob.values
        ]
        # A candidate that already lost beyond the depth at every width
        # owes nothing to a recheck — no plausible interaction rescues it;
        # an ordered-pruned value was never measured at all and owes the
        # same nothing. Unmeasured (failed) and pressured entries still
        # get the pass.
        base = (results.get("baseline") or {}).get("metrics") or {}
        owed, deep = [], []
        for knob, value in recheck:
            candidate_key = f"{knob.env}={value}"
            if verdicts.get(candidate_key) == "pruned":
                deep.append((knob, value))
                continue
            entry = results.get(candidate_key)
            metrics = (entry or {}).get("metrics") if isinstance(entry, dict) else None
            if not isinstance(metrics, dict) or not _deep_loss(
                metrics, base, RECHECK_DEPTH
            ):
                owed.append((knob, value))
            else:
                deep.append((knob, value))
        if recheck:
            log(
                f"recheck · {len(owed)} candidates against the final env"
                + (f" · {len(deep)} deep losses already out" if deep else "")
            )
        sweep(owed, prefix="recheck:")

    # Backward elimination: a knob kept early may no longer pay beside the
    # winners that followed it. Re-measure the final env with each winner
    # removed; drop the knob when its removal costs nothing within the
    # noise floor — or gains. Bounded: one engine per retained winner.
    winners = sorted(best_env)
    if winners:
        log(f"prune · {len(winners)} winners against the final env")
    for name in winners:
        value = best_env[name]
        trial = {key: val for key, val in best_env.items() if key != name}
        pressured = contaminated or _wait_for_pressure(log)
        log(f"trying -{name}={value}")
        try:
            measured, _ = measure(
                binary,
                assembly_dir,
                trial,
                prompts=prompts,
            )
        except (OSError, RuntimeError) as error:
            log(f"-{name}={value} · failed ({error}) — kept")
            continue
        log(f"-{name}={value} · {measured}")
        results[f"eliminate:{name}={value}"] = {
            "env": trial,
            "metrics": measured,
        }
        # The env without the knob must lose to it for the knob to stay:
        # no meaningful win means the knob contributes nothing.
        if pressured or _keeps(best_metrics, measured, sigma):
            continue
        del best_env[name]
        best_metrics = measured
        verdicts[f"{name}={value}"] = "eliminated"
        log("  removed — the final env pays without it")
        write()

    record[context.chip]["complete"] = True
    _write_record(assembly_dir, record)
    if priors_path is not None:
        try:
            _update_priors(
                priors_path,
                context.chip,
                context.family,
                results,
                best_env,
                engine=fingerprint,
                knobs=signature,
            )
        except OSError:
            pass
    return best_env
