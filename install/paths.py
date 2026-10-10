"""Immutable program files and writable per-user data, for source or release."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()
DATA = Path.home() / "Library/Application Support/RichEngine" if PACKAGED else ROOT
MODELS = DATA / "models" if PACKAGED else ROOT / "install/models"
RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/richengine" if PACKAGED else "build/richengine")
# The autotune's kernel pass: the dev/tuning measurement harness shipped
# beside the engine (Makefile's TUNE_TOOL).
TUNE_BINARY = ROOT / ("engine/richengine-tune" if PACKAGED else "build/richengine-tune")
