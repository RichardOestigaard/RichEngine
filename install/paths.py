"""Immutable program files and writable per-user data, for source or release."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = (ROOT / "release.json").is_file()


def _data_dir() -> Path:
    """RichEngine's per-user data directory on this platform."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/RichEngine"
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg) if xdg else Path.home() / ".local/share") / "RichEngine"


DATA = _data_dir()


def default_models_root() -> Path:
    """The models root without user configuration.

    Source checkouts with install/models already present keep using it: a real
    directory is an existing development install, and a symlink names the store
    its owner chose. Only a checkout without either uses the per-user data
    directory. Packaged installs always use the per-user directory.
    """
    legacy = ROOT / "install/models"
    if not PACKAGED:
        if legacy.is_symlink():
            return legacy.resolve()
        if legacy.is_dir():
            return legacy
    return DATA / "models"


# The default root, kept as a module value for existing callers and tests.
# models_root() is the resolution every new call site uses.
MODELS = default_models_root()


def _configured_models_dir() -> Path | None:
    """The optional models_dir of DATA/config.json; a malformed file configures nothing."""
    try:
        value = json.loads((DATA / "config.json").read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    configured = value.get("models_dir")
    if not isinstance(configured, str) or not configured.strip():
        return None
    path = Path(configured).expanduser()
    return path if path.is_absolute() else DATA / path


def packed_cache_root() -> Path:
    """The default root of the native engine's packed weight-image cache."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/RichEngine/packed"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "RichEngine/packed"


def models_root(override=None) -> tuple[Path, str]:
    """The models root and what chose it: flag, env, config or default.

    An explicit --models/--models-dir value wins, then RICHENGINE_MODELS_DIR,
    then the optional models_dir key of DATA/config.json. The default keeps an
    existing source checkout's install/models store and otherwise uses
    DATA/models.
    """
    if override is not None and str(override).strip():
        return Path(override).expanduser().absolute(), "flag"
    configured = os.environ.get("RICHENGINE_MODELS_DIR")
    if configured and configured.strip():
        return Path(configured).expanduser().absolute(), "env"
    configured = _configured_models_dir()
    if configured is not None:
        return configured.absolute(), "config"
    return Path(MODELS), "default"


RUNTIME = DATA / "runtime" if PACKAGED else ROOT / "build/runtime"
PYTHON = ROOT / ("python/bin/python3" if PACKAGED else ".venv/bin/python")
BINARY = ROOT / ("engine/richengine" if PACKAGED else "build/richengine")
# The autotune's kernel pass: the dev/tuning measurement harness shipped
# beside the engine (Makefile's TUNE_TOOL).
TUNE_BINARY = ROOT / ("engine/richengine-tune" if PACKAGED else "build/richengine-tune")
# The plan-parity checker's token dump driver (dev/benchmarks/decode_profile.mm);
# packaged installs ship no benchmarks, so checks must degrade when absent.
PROFILE_BINARY = ROOT / ("engine/decode-profile" if PACKAGED else "build/engine-tests/decode-profile")
