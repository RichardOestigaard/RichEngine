"""Autotune sweeps as a background job for the HTTP API.

The job runs `launcher.py tune` in a subprocess, exactly like Installer runs
models.py: the API holds a job record, the client polls it and can cancel it.
The sweep itself lives in install/autotune.py — this file only spawns it and
reports what it reported.
"""

import argparse
import collections
import fcntl
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .errors import APIError

TAIL_LINES = 200  # bounded in memory; status() serves a shorter window
STATUS_LINES = 60

# A measured candidate's output line: "RICHENGINE_X=v · {...}" — first-pass
# and recheck measurements log the same shape. "sweep · N candidates" and
# "recheck · N candidates" announce each phase's workload.
_CANDIDATE_LINE = re.compile(r"^\S*RICHENGINE_[A-Z_0-9]*=")
_TOTAL_LINE = re.compile(r"(?:sweep|recheck) · (\d+)")


def _command(
    models_root,
    model,
    *,
    revision=None,
    draft_model=None,
    language_only=False,
    mode="quick",
    selection=None,
    include_quality=False,
    interactions=None,
):
    """The launcher's tune CLI, spelled like the installer's models.py call:
    the install interpreter (or ours) running the repo's launcher. `mode`
    picks the sweep's defaults — "complete" maps to --complete — while
    include_quality and interactions stay per-flag overrides of it."""
    from install import paths

    python = paths.PYTHON if paths.PYTHON.is_file() else Path(sys.executable)
    command = [
        str(python),
        str(paths.ROOT / "install" / "launcher.py"),
        "tune",
        "--models",
        str(models_root),
    ]
    if selection is not None:
        # The link's name under the models root; the launcher resolves it
        # directly, skipping the model-and-options derivation.
        command += ["--selection", selection]
    else:
        command += ["--model", model]
        if revision:
            command += ["--revision", revision]
        if draft_model:
            command += ["--draft-model", draft_model]
        if language_only:
            command.append("--language-only")
    if mode == "complete":
        command.append("--complete")
    if include_quality:
        command.append("--include-quality-knobs")
    if interactions is True:
        command.append("--interaction-pass")
    elif interactions is False:
        command.append("--no-interaction-pass")
    return command, paths.ROOT


def _kept(job):
    """The tuned env a finished sweep recorded, as "NAME=value" strings —
    RICHENGINE_ stripped like model_host's applied-knobs log. Best effort:
    an unreadable record reports []."""
    try:
        from install import autotune, paths

        tuned = autotune.load_tuning(
            job["link"],
            binary=paths.BINARY if paths.BINARY.is_file() else None,
        )
    except Exception:
        return []
    return sorted(
        f"{name.removeprefix('RICHENGINE_')}={value}"
        for name, value in tuned.items()
    )


def _width_tps(metrics):
    """Decode throughput at the batched width, the single-lane width when
    that is missing; 0 when the metrics are absent or invalid."""
    if not isinstance(metrics, dict):
        return 0.0
    for width in ("4", "1"):
        section = metrics.get(width)
        tps = (
            section.get("tokens_per_second")
            if isinstance(section, dict)
            else None
        )
        if type(tps) in (int, float) and tps > 0:
            return float(tps)
    return 0.0


def _results(job):
    """Every measured candidate's swing against the baseline at the batched
    width, as "KNOB=value +4.1%" lines — biggest movers first, pressured
    measurements skipped. Best effort like _kept."""
    try:
        from install import autotune

        entry = autotune._entry(job["link"])
    except Exception:
        return []
    measured = entry.get("measured") if isinstance(entry, dict) else None
    if not isinstance(measured, dict):
        return []
    baseline = measured.get("baseline")
    before = _width_tps(
        baseline.get("metrics") if isinstance(baseline, dict) else None
    )
    if not before:
        return []
    rows = []
    for key, candidate in measured.items():
        if (
            key == "baseline"
            or not isinstance(candidate, dict)
            or candidate.get("pressured")
        ):
            continue
        after = _width_tps(candidate.get("metrics"))
        if not after:
            continue
        delta = after / before - 1
        name = key.removeprefix("recheck:").removeprefix("RICHENGINE_")
        rows.append((abs(delta), f"{name} {delta:+.1%}"))
    rows.sort(key=lambda row: -row[0])
    return [text for _, text in rows[:12]]


def _headline(job):
    """The payoff number for the done row: the final env's decode swing vs
    the baseline — "decode +9.8% @b4". None when the record lacks either
    side or the baseline was pressured."""
    try:
        from install import autotune

        entry = autotune._entry(job["link"])
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None
    measured = entry.get("measured")
    baseline = measured.get("baseline") if isinstance(measured, dict) else None
    if not isinstance(baseline, dict) or baseline.get("pressured"):
        return None
    base_metrics = baseline.get("metrics")
    best_metrics = entry.get("best")
    if not isinstance(base_metrics, dict) or not isinstance(best_metrics, dict):
        return None
    for width in ("4", "1"):
        before = base_metrics.get(width, {}).get("tokens_per_second")
        after = best_metrics.get(width, {}).get("tokens_per_second")
        if (
            type(before) in (int, float)
            and type(after) in (int, float)
            and before > 0
            and after > 0
        ):
            return f"decode {after / before - 1:+.1%} @b{width}"
    return None


def _failure(job):
    """The sweep's own error line for a failed run — the launcher's
    'error: …' when it printed one, else the last nonempty output line."""
    lines = [line for line in job["lines"] if line.strip()]
    for line in reversed(lines):
        if "error:" in line:
            return line.split("error:", 1)[-1].strip()
    return lines[-1].strip() if lines else None


def _other_serve_running():
    """A held serve-<port>.lock means a serve's engine may share the GPU —
    the launcher warns on the same locks; the API refuses instead."""
    from install import paths

    for lock in paths.RUNTIME.glob("serve-*.lock"):
        try:
            handle = lock.open("a+")
        except OSError:
            continue
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()
    return False


class Tuner:
    """Runs one tune at a time; the job outlives the request that started it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job = None

    def start(
        self,
        models_root,
        model,
        *,
        revision=None,
        draft_model=None,
        language_only=False,
        mode="quick",
        selection=None,
        include_quality=False,
        interaction_pass=None,
        allow_loaded=False,
        serving=False,
    ):
        """Validate and spawn the sweep; APIError on a bad request, a 409
        when a serve runs without allow_loaded (its engine shares the GPU
        and contaminates every measurement) or a tune already runs."""
        from install import models as model_artifacts

        if mode not in ("quick", "complete"):
            raise APIError(400, '"mode" must be "quick" or "complete"')
        try:
            model_artifacts.parse_model_id(model)
        except argparse.ArgumentTypeError as error:
            raise APIError(400, str(error)) from None
        if selection is not None:
            # The link's name under the models root, verbatim. Membership
            # in selection_links is exact path equality — "../x" never is
            # one — so the name cannot escape the root.
            link = Path(models_root) / selection
            if link not in model_artifacts.selection_links(Path(models_root)):
                raise APIError(
                    400, f"unknown selection: {selection}", "invalid_request"
                )
        else:
            link = model_artifacts.Selection.of(
                models_root,
                model,
                revision=revision,
                language_only=language_only,
                draft_model=draft_model,
            ).link
        if model_artifacts.installation_kind(link) is None:
            raise APIError(404, f"{model} is not installed", "model_not_found")
        if not allow_loaded and (serving or _other_serve_running()):
            raise APIError(
                409,
                "a model is loaded; its engine shares the GPU and skews "
                'every measurement — unload it first or set "allow_loaded"',
                "engine_busy",
            )
        with self._lock:
            if self._job is not None and not self._job["done"]:
                raise APIError(
                    409,
                    f'a tune of {self._job["model"]} is already running',
                    "engine_busy",
                )
            command, cwd = _command(
                models_root,
                model,
                revision=revision,
                draft_model=draft_model,
                language_only=language_only,
                mode=mode,
                selection=selection,
                include_quality=include_quality,
                interactions=interaction_pass,
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    # Its own session so cancel signals the launcher's whole
                    # tree — the serve-native it is measuring included —
                    # never the server.
                    start_new_session=True,
                )
            except OSError as error:
                raise APIError(
                    500, f"could not start the tune: {error}"
                ) from None
            self._job = {
                "model": model,
                "mode": mode,
                "link": link,
                "started": time.time(),
                "done": False,
                "returncode": None,
                "cancelled": False,
                "kept": None,
                "results": None,
                "headline": None,
                "error": None,
                "candidates_done": 0,
                "candidates_total": 0,
                "lines": collections.deque(maxlen=TAIL_LINES),
                "process": process,
            }
            threading.Thread(
                target=self._collect,
                args=(self._job,),
                name="model tune",
                daemon=True,
            ).start()
        return self.status()

    def _collect(self, job):
        """Drain the launcher's output until it exits — one line per
        candidate it measured — counting them and each phase's announced
        workload for the progress fields."""
        process = job["process"]
        try:
            for line in process.stdout or ():
                line = line.rstrip("\n")
                with self._lock:
                    job["lines"].append(line)
                    if _CANDIDATE_LINE.match(line):
                        job["candidates_done"] += 1
                    total = _TOTAL_LINE.search(line)
                    if total:
                        job["candidates_total"] += int(total.group(1))
        finally:
            returncode = process.wait()
            with self._lock:
                job["returncode"] = returncode
                job["done"] = True
                # Done however it ended — a cancelled or failed sweep may
                # still have written winners to the partial record.
                job["kept"] = _kept(job)
                job["results"] = _results(job)
                job["headline"] = (
                    _headline(job) if job["returncode"] == 0 else None
                )
                job["error"] = (
                    _failure(job)
                    if job["returncode"] != 0 and not job["cancelled"]
                    else None
                )

    def status(self):
        """The job for GET /v1/models/tune: a snapshot safe to serialize."""
        with self._lock:
            job = self._job
            if job is None:
                return {"running": False, "job": None}
            return {
                "running": not job["done"],
                "model": job["model"],
                "mode": job["mode"],
                "started": job["started"],
                "done": job["done"],
                "ok": job["done"]
                and job["returncode"] == 0
                and not job["cancelled"],
                "cancelled": job["cancelled"],
                "returncode": job["returncode"],
                "kept": job["kept"],
                "results": job["results"],
                "headline": job.get("headline"),
                "error": job.get("error"),
                "candidates_done": job["candidates_done"],
                "candidates_total": job["candidates_total"],
                "tail": list(job["lines"])[-STATUS_LINES:],
            }

    def cancel(self):
        """SIGTERM the launcher's process group — the sweep dies with the
        serve-native it is measuring, and tuning.json's partial record lets
        the next tune resume. False when nothing is running."""
        with self._lock:
            job = self._job
            if job is None or job["done"]:
                return False
            job["cancelled"] = True
            try:
                os.killpg(os.getpgid(job["process"].pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                job["process"].terminate()
            return True
