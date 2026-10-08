"""Model installation as a background job for the HTTP API."""

import argparse
import collections
import subprocess
import sys
import threading
import time
from pathlib import Path

from .errors import APIError

# Enough of `install/models.py prepare` output to show what a long download is
# doing without buffering the whole log in memory.
TAIL_LINES = 200
STATUS_LINES = 60


def _command(
    models_root, model, *, revision=None, draft_model=None, language_only=False
):
    """The installer's own CLI, as the launcher spells it: the install
    package's interpreter so huggingface_hub resolves, and the repository
    root as the working directory so `import install.models` inside
    models.py's __main__ guard finds the package."""
    from install import paths

    python = paths.PYTHON if paths.PYTHON.is_file() else Path(sys.executable)
    command = [
        str(python),
        str(paths.ROOT / "install" / "models.py"),
        "--models",
        str(models_root),
        "--model",
        model,
    ]
    if revision:
        command += ["--revision", revision]
    if draft_model:
        command += ["--draft-model", draft_model]
    if language_only:
        command.append("--language-only")
    command.append("prepare")
    return command, paths.ROOT


class Installer:
    """Runs `install/models.py prepare` for one model at a time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._job = None

    def start(
        self, models_root, model, *, revision=None, draft_model=None,
        language_only=False,
    ):
        """Validate the selection and spawn the installer; the job outlives
        the request and is polled over GET /v1/models/install."""
        from install import models as model_artifacts

        try:
            model_artifacts.parse_model_id(model)
        except argparse.ArgumentTypeError as error:
            raise APIError(400, str(error)) from None
        if draft_model is not None:
            try:
                model_artifacts.parse_draft_model(draft_model)
            except argparse.ArgumentTypeError as error:
                raise APIError(400, f'"draft_model": {error}') from None
        with self._lock:
            if self._job is not None and not self._job["done"]:
                raise APIError(
                    409,
                    f'an install of {self._job["model"]} is already running',
                    "engine_busy",
                )
            command, cwd = _command(
                models_root,
                model,
                revision=revision,
                draft_model=draft_model,
                language_only=language_only,
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    # Its own session so cancel signals the installer tree,
                    # never the server, and a server SIGINT does not.
                    start_new_session=True,
                )
            except OSError as error:
                raise APIError(
                    500, f"could not start the installer: {error}"
                ) from None
            self._job = {
                "model": model,
                "started": time.time(),
                "done": False,
                "returncode": None,
                "cancelled": False,
                "lines": collections.deque(maxlen=TAIL_LINES),
                "process": process,
            }
            threading.Thread(
                target=self._collect,
                args=(self._job,),
                name="model install",
                daemon=True,
            ).start()
        return self.status()

    def _collect(self, job):
        process = job["process"]
        try:
            for line in process.stdout or ():
                with self._lock:
                    job["lines"].append(line.rstrip("\n"))
        finally:
            returncode = process.wait()
            with self._lock:
                job["returncode"] = returncode
                job["done"] = True

    def status(self):
        """The current or last job, without the Popen handle."""
        with self._lock:
            job = self._job
            if job is None:
                return {"running": False, "job": None}
            return {
                "running": not job["done"],
                "model": job["model"],
                "started": job["started"],
                "done": job["done"],
                "ok": job["done"]
                and job["returncode"] == 0
                and not job["cancelled"],
                "cancelled": job["cancelled"],
                "returncode": job["returncode"],
                "tail": list(job["lines"])[-STATUS_LINES:],
            }

    def cancel(self):
        """SIGTERM the running installer; False when nothing is running. The
        Hub cache keeps partial downloads, so a retry resumes them."""
        with self._lock:
            job = self._job
            if job is None or job["done"]:
                return False
            job["cancelled"] = True
            job["process"].terminate()
            return True
