"""Local model directories: what other tools already downloaded.

A candidate is a directory holding a config.json beside safetensors
weights (an MLX or oQ checkpoint), or root *.gguf files. scan() finds
them under the roots model_dirs() returns — LM Studio's
~/.lmstudio/models by default, plus the directories oMLX's settings
name and ~/.omlx/models — and probe() reports each one's family, format
and quantization as the installer decides them, before anything is
selected. find() maps a name back to a directory.

Nothing here downloads; probing a safetensors directory costs its
config.json, a GGUF directory its header.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path

from . import families, gguf, models, pack

LMSTUDIO_MODELS = Path.home() / ".lmstudio" / "models"
OMLX_DIR = Path.home() / ".omlx"

# The affine quantization the runtime loads (MLX uniform Q4/64 and the
# oMLX oQ matrix), shared with upstream._mlx_target.
OQ_BITS = (4, 5, 6, 8)
OQ_GROUPS = (64, 128)


def model_dirs() -> list[Path]:
    """The roots scan() walks: LM Studio's model folder, the directories
    oMLX's settings.json names (model_dirs and the singular model_dir it
    also writes) and ~/.omlx/models, each existing exactly once."""
    roots = [LMSTUDIO_MODELS]
    try:
        settings = json.loads((OMLX_DIR / "settings.json").read_text(encoding="utf-8"))
        model = settings.get("model") if isinstance(settings, dict) else None
        if isinstance(model, dict):
            for key in ("model_dirs", "model_dir"):
                value = model.get(key)
                for entry in value if isinstance(value, list) else [value]:
                    if isinstance(entry, str) and entry:
                        roots.append(Path(entry).expanduser())
    except (OSError, json.JSONDecodeError):
        pass
    roots.append(OMLX_DIR / "models")
    unique = []
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved.is_dir() and resolved not in unique:
            unique.append(resolved)
    return unique


def _weight_bytes(path: Path) -> int:
    total = 0
    for name in path.iterdir():
        if name.is_file() and name.name.endswith((".safetensors", ".gguf")):
            total += name.stat().st_size
    return total


def _is_candidate(path: Path) -> bool:
    """Whether path looks like a model: its own files, at its root."""
    if not path.is_dir():
        return False
    root_files = {name.name for name in path.iterdir() if name.is_file()}
    if any(name.endswith(".gguf") for name in root_files):
        return True
    return "config.json" in root_files and any(
        name.endswith(".safetensors") for name in root_files
    )


def _scan_root(root: Path):
    """The candidate directories under one root: a model directory at any
    level up to <root>/<author>/<model>, its subtree not searched again
    (LM Studio's own .cache folders never look like models anyway)."""
    queue = collections.deque(
        entry for entry in sorted(root.iterdir()) if entry.is_dir()
    )
    while queue:
        path = queue.popleft()
        if path.name.startswith("."):
            continue
        try:
            if _is_candidate(path):
                yield path
                continue
            if len(path.relative_to(root).parts) >= 2:
                continue
            queue.extend(entry for entry in sorted(path.iterdir()) if entry.is_dir())
        except OSError:
            yield path


def _quantization(config):
    """The checkpoint's quantization kind and a readable summary, as the
    runtime's probe decides it: 'affine' for the uniform 4-bit/group-64 the
    legacy path plans, 'oQ' for the oMLX mixed-precision matrix, else None."""
    quant = config.get("quantization")
    if not isinstance(quant, dict) or not affine_supported(quant, {}):
        return None, None
    entries = [entry for entry in quant.values() if isinstance(entry, dict)]
    # An entry may spell "bits": null explicitly; drop the Nones before the
    # numeric set is sorted.
    bits = {
        bit
        for bit in {quant.get("bits")}
        | {entry.get("bits", quant.get("bits")) for entry in entries}
        if bit is not None
    }
    groups = {
        group
        for group in {quant.get("group_size")}
        | {entry.get("group_size", quant.get("group_size")) for entry in entries}
        if group is not None
    }
    bits_text = (
        "/".join(sorted(str(bit) for bit in bits))
        + "-bit"
        + (" mixed" if entries else "")
    )
    group_text = (
        "group-64"
        if groups == {64}
        else "group-128"
        if groups == {128}
        else "group-64/128"
    )
    kind = "affine" if bits == {4} and groups == {64} and not entries else "oQ"
    return kind, f"{bits_text} {group_text} affine"


def affine_supported(entry, defaults) -> bool:
    """Whether a quantization dict entry is in the oQ matrix: affine mode at
    4-8 bits over groups of 64 or 128, fields defaulting to the global's."""
    return (
        entry.get("mode", defaults.get("mode", "affine")) == "affine"
        and entry.get("bits", defaults.get("bits")) in OQ_BITS
        and entry.get("group_size", defaults.get("group_size")) in OQ_GROUPS
    )


def quantization_support(config):
    """None when config's MLX quantization is one the runtime loads, else a
    reason it is not: anything but the oQ affine matrix (GPTQ/AWQ ship no
    "quantization" dict, so an absent one is a non-MLX source)."""
    quant = config.get("quantization")
    if not isinstance(quant, dict):
        return "not an MLX quantized checkpoint"
    if not affine_supported(quant, {}):
        return (
            f"quantization mode {quant.get('mode')!r}, "
            f"{quant.get('bits')} bits/group-{quant.get('group_size')}"
        )
    for name, entry in quant.items():
        if isinstance(entry, dict) and not affine_supported(entry, quant):
            return f"{name} at {entry.get('bits')} bits/group-{entry.get('group_size')}"
    return None


@dataclass
class Candidate:
    """A local model directory and what probing it found."""

    path: Path
    name: str  # <author>/<name> or <name>, as its parent dirs read
    format: str  # "mlx", "gguf" or "packed"
    family: str | None
    quantization: str | None
    bytes: int
    reason: str | None  # why it cannot be served, when it cannot


def probe(path: Path, root: Path) -> Candidate:
    """The candidate at path: its display name under root, its format,
    family and a reason it cannot serve when there is one."""
    try:
        name = path.relative_to(root).as_posix()
    except ValueError:
        name = path.name
    root_files = {entry.name for entry in path.iterdir() if entry.is_file()}
    ggufs = sorted(name for name in root_files if name.endswith(".gguf"))
    weight_bytes = _weight_bytes(path)
    reason = None
    family = None
    quantization = None
    if ggufs:
        # The target GGUF the installer selects, as it selects it: the only
        # non-projector file, else reported as ambiguous.
        try:
            from . import upstream

            chosen, _ = upstream.select_gguf(root_files, None)
            with (path / chosen).open("rb") as stream:
                header = gguf.Metadata(stream, tensors=False)
            config = gguf.model_config(header)
            family = families.family_for(
                config, name=f"{name} {header.values.get('general.name', '')}"
            ).name
            quantization = (
                f"{chosen} ({len(ggufs)} file{'s' if len(ggufs) > 1 else ''})"
            )
            return Candidate(
                path, name, "gguf", family, quantization, weight_bytes, None
            )
        except models.ModelError as error:
            reason = str(error)
            return Candidate(
                path, name, "gguf", family, quantization, weight_bytes, reason
            )
    if "config.json" not in root_files:
        return Candidate(path, name, "gguf", None, None, weight_bytes, "no config.json")
    try:
        config = models.read_json(path / "config.json")
    except models.ModelError as error:
        return Candidate(path, name, "mlx", None, None, weight_bytes, str(error))
    if pack.packable(config):
        format_name, _ = pack.target_format(config)
        return Candidate(
            path, name, "packed", format_name, "packed at install", weight_bytes, None
        )
    reason = quantization_support(config)
    try:
        family = families.family_for(config, name=name).name
    except models.ModelError as error:
        reason = reason or str(error)
    kind, quantization = _quantization(config)
    return Candidate(
        path,
        name,
        "mlx",
        family,
        quantization,
        weight_bytes,
        None if reason is None else reason,
    )


def scan(roots=None):
    """Every candidate under the roots, probed. A directory that is
    unreadable or half-downloaded still lists, with its reason."""
    found = []
    for root in roots if roots is not None else model_dirs():
        for path in _scan_root(root):
            try:
                found.append(probe(path, root))
            # A corrupt config or GGUF raises past probe()'s own guards; one
            # bad directory must not abort the scan's other candidates.
            except Exception as error:
                name = (
                    path.relative_to(root).as_posix()
                    if path.is_relative_to(root)
                    else path.name
                )
                found.append(Candidate(path, name, "mlx", None, None, 0, str(error)))
    return sorted(found, key=lambda candidate: candidate.name.lower())


def find(name, roots=None) -> Path:
    """The candidate directory name selects: an exact <author>/<name> or
    the unique <name> match under any root. An ambiguous or missing name
    is a ModelError the caller reports."""
    matches = [
        candidate
        for candidate in scan(roots)
        if candidate.name == name or Path(candidate.name).name == name
    ]
    exact = [candidate for candidate in matches if candidate.name == name]
    if exact:
        matches = exact
    if len(matches) == 1:
        return matches[0].path
    if matches:
        raise models.ModelError(
            f"{name} names several local models: "
            + ", ".join(candidate.name for candidate in matches)
        )
    raise models.ModelError(f"no local model named {name}")
