#!/usr/bin/env python3

"""Install and verify the model a RichEngine selection names.

Terms the installer modules share:
- selection: what `--model OWNER/REPO[:VARIANT]` and its source options name
  (Selection); its selection link, under the models root, points to what
  serves it.
- assembly: the directory an upstream model is served from, of links to
  source snapshot files and its record, model.json (assembly.py), built and
  published by upstream.py. A legacy RichEngine package is served from its own
  Hub snapshot instead (legacy.py).
- source snapshot: a repository at one commit in the Hub cache, or a local
  draft directory (hub.py).

This module holds what the installers share, and the command line.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

if __name__ == "__main__" and not __package__:
    # Run as a script by make, the launcher and `python install/models.py`:
    # import siblings as the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

from . import layout, paths

MODELS = paths.MODELS
# The bound on one JSON metadata file.
MAX_JSON_BYTES = 4 * 1024 * 1024
REPO_ID = re.compile(
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]*[A-Za-z0-9_])?/"
    r"[A-Za-z0-9_](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9_])?"
)
VARIANT_SEPARATOR = ":"
VARIANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# What installation_kind finds at a selection link.
ASSEMBLY, PACKAGE = "assembly", "package"
InstallationKind = Literal["assembly", "package"]
# Staging an interrupted installation leaves, which garbage collection
# removes: a selection link being replaced (beside the link), and an assembly
# or derived metadata entry being written (beside the published entries).
LINK_STAGING = layout.LINK_STAGING
ENTRY_STAGING = layout.ENTRY_STAGING


class ModelError(RuntimeError):
    pass


def _styled(text, *codes, stream=sys.stdout):
    """Wrap codes around text on a terminal with NO_COLOR unset — the
    launcher's gate (install/launcher.py _ansi): piped output and log files
    stay byte-plain."""
    if stream.isatty() and "NO_COLOR" not in os.environ:
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"
    return text


def accent(text):
    """The launcher help's cyan: model and file names the user may reuse."""
    return _styled(str(text), "36")


def dim(text):
    """The launcher help's 'Usage' gray: detail the user can skip."""
    return _styled(str(text), "2")


def warn(message):
    """Report something the user may need to act on, on stderr."""
    print(
        f"{_styled('Warning:', '33', stream=sys.stderr)} {message}",
        file=sys.stderr,
        flush=True,
    )


def is_hex_digest(value, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", value) is not None
    )


def is_safe_path(name) -> bool:
    """Whether name is a plain relative POSIX path: no absolute, empty, "."
    or ".." part, no backslash or control character, and none of the
    characters Hub download patterns read as globs (* ? [ ])."""
    if not isinstance(name, str):
        return False
    path = PurePosixPath(name)
    return (
        bool(path.parts)  # "" and "." have none
        and not path.is_absolute()
        and path.as_posix() == name
        and ".." not in path.parts
        and not any(character in name for character in "\\*?[]")
        and all(ord(character) >= 32 for character in name)
    )


def validate_repo_id(value: str) -> str:
    # Keep argument validation available before the Hub dependency is installed.
    if (
        not isinstance(value, str)
        or not REPO_ID.fullmatch(value)
        or "--" in value
        or ".." in value
        or value.endswith(".git")
    ):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    return value


def split_model_id(value: str) -> tuple[str, str | None]:
    """owner/repo[:variant] -> (repository ID, variant or None)."""
    if not isinstance(value, str):
        raise ModelError("model must be a full Hugging Face repository ID (owner/repo)")
    repo_id, separator, variant = value.partition(VARIANT_SEPARATOR)
    validate_repo_id(repo_id)
    if not separator:
        return repo_id, None
    if not VARIANT.fullmatch(variant) or ".." in variant:
        raise ModelError(
            "model variant must be a short name such as UD-Q4_K_M "
            f"(owner/repo{VARIANT_SEPARATOR}VARIANT)"
        )
    return repo_id, variant


def _local_directory(value: str) -> Path | None:
    """The local model directory value names, when it names one: an
    existing path that looks like a path (absolute, ~, or . — a bare
    owner/repo stays a repository ID even if it shadows a directory),
    local:<author>/NAME or local:NAME, or a bare NAME a local cache
    holds (install/local.py)."""
    from . import local

    if not value:
        return None
    if value.startswith("local:"):
        found = local.find(value.removeprefix("local:"))
        return found.resolve() if found is not None else None
    expanded = Path(value).expanduser()
    if Path(value).is_absolute() or value.startswith(("~", ".")):
        if expanded.is_dir():
            return expanded.resolve()
        raise ModelError(f"model directory not found: {expanded}")
    if "/" not in value:
        # Not a repository ID; the local caches may hold it by name.
        try:
            found = local.find(value)
        except ModelError as error:
            raise ModelError(
                f"{error}; a repository needs its owner (owner/repo)"
            ) from error
        return found.resolve() if found is not None else None
    return None


def parse_model_id(value: str) -> str:
    try:
        split_model_id(value)
    except ModelError as error:
        # A local directory or scanned name resolves at selection time.
        if (
            (Path(value).is_absolute() or value.startswith(("~", ".")))
            and Path(value).expanduser().is_dir()
        ) or value.startswith("local:") or "/" not in value:
            return value
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def parse_draft_model(value: str) -> str:
    # A local draft directory is recorded as an absolute path, so the
    # installation it selects does not depend on the working directory.
    if value and (local := Path(value).expanduser()).is_dir():
        # --draft-model is an explicit local-artifact flag: any existing
        # directory wins, including a relative path under the cwd. (The
        # target install path stays strict — a bare owner/repo is a Hub
        # ID even when it shadows a directory.)
        return str(local.resolve())
    try:
        return validate_repo_id(value)
    except ModelError:
        raise argparse.ArgumentTypeError(
            "must be a local DFlash2 draft directory or a Hugging Face "
            "repository ID (owner/repo)"
        ) from None


def hash_file(path: Path, digest) -> str:
    """The hex digest of path's content fed to digest, a hashlib object."""
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    return hash_file(path, hashlib.sha256())


def read_json(path: Path):
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise ModelError(f"JSON metadata is too large: {path}")
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelError(f"expected a JSON object in {path}")
    return value


def json_bytes(value) -> bytes:
    """Canonical JSON: the same value always encodes to the same bytes."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode()


def selection_link(
    models: Path,
    model_id: str,
    *,
    revision=None,
    language_only=False,
    draft_model=None,
    thin=False,
) -> Path:
    """The selection link of model_id with these source options:
    OWNER/REPO[:VARIANT] under models for the model alone, else
    .selections/<hash of the selection> — a local model directory's too,
    which has no repository ID to name it by. Every name resolves through
    _local_directory so local paths and bare names take the same link
    Selection.of computes."""
    local_path = _local_directory(model_id)
    key_id = str(local_path) if local_path is not None else model_id
    if (
        revision
        or language_only
        or draft_model
        or thin
        or local_path is not None
        or Path(model_id).is_absolute()
        or model_id.startswith("local:")
    ):
        options = [key_id, revision, language_only, draft_model]
        if thin:
            options.append(True)
        selection = json.dumps(options, separators=(",", ":"))
        return (
            models / layout.SELECTIONS / hashlib.sha256(selection.encode()).hexdigest()
        )
    repo_id, variant = split_model_id(model_id)
    if variant is None:
        return models / repo_id
    return models / f"{repo_id}{VARIANT_SEPARATOR}{variant}"


def selection_links(models: Path):
    """Every selection link under models, in the places selection_link names:
    OWNER/REPO and .selections/<hash>. A stray symlink elsewhere — under
    .resolved, .metadata or .packed — is not one."""
    return [
        path
        for path in models.glob("*/*")
        if path.is_symlink()
        and (
            path.parent.name == layout.SELECTIONS
            or not path.parent.name.startswith(".")
        )
    ]


@dataclass(frozen=True)
class Selection:
    """What --model and its source options select, and its selection link."""

    model: str
    repo_id: str
    variant: str | None
    revision: str | None
    language_only: bool
    draft_model: str | None
    models_root: Path
    thin: bool
    link: Path

    @classmethod
    def of(
        cls,
        models_root,
        model,
        *,
        revision=None,
        language_only=False,
        draft_model=None,
        thin=False,
    ):
        path = _local_directory(model)
        if path is not None:
            # A local model directory: its path is the source Repository
            # resolve reads (hub.Repository.resolve), its name the display.
            repo_id, variant, display, link_key = str(path), None, path.name, str(path)
        else:
            repo_id, variant = split_model_id(model)
            display, link_key = model, model
        models_root = Path(models_root).resolve()
        options = {
            "revision": revision,
            "language_only": language_only,
            "draft_model": draft_model,
        }
        if thin:
            options["thin"] = True
        link = selection_link(models_root, link_key, **options)
        return cls(
            display,
            repo_id,
            variant,
            revision,
            language_only,
            draft_model,
            models_root,
            thin,
            link,
        )


def installation_kind(link: Path) -> InstallationKind | None:
    """What serves a selection link: an assembly (its model.json), a legacy
    RichEngine package (its manifest.json), or nothing (None)."""
    if (link / layout.ASSEMBLY_RECORD).exists():
        return ASSEMBLY
    if (link / layout.PACKAGE_MANIFEST).exists():
        return PACKAGE
    return None


def _legacy_repo_id(link: Path) -> str:
    """The Hub repository ID of a legacy package's installed snapshot."""
    from . import legacy

    snapshot = legacy.installed_snapshot(link)
    folder = snapshot.parent.parent.name
    if not folder.startswith("models--"):
        raise ModelError(
            f"installed package is not in a Hugging Face model cache: {snapshot}"
        )
    return folder.removeprefix("models--").replace("--", "/")


def verify_installation(link: Path, *, repo_id=None, full=False):
    """Verify the assembly or package one installed selection link serves."""
    from . import assembly, legacy, pack

    kind = installation_kind(link)
    if kind == ASSEMBLY:
        return assembly.verify(link, full=full)
    if kind == PACKAGE:
        if pack.is_built(link):
            return pack.verify(link, full=full)
        return legacy.verify(
            link, repo_id or _legacy_repo_id(link), full=full
        )
    raise ModelError(f"no model is installed at {link}")


def _installation_pin_pairs(link: Path):
    """The (snapshot, repository ID) pairs an installation may have pinned."""
    from . import assembly, hub, legacy, pack

    kind = installation_kind(link)
    if kind == ASSEMBLY:
        return assembly.recorded_pins(link)
    if kind != PACKAGE:
        return []
    if not pack.is_built(link):
        snapshot = legacy.installed_snapshot(link)
        return [(snapshot, _legacy_repo_id(link))]
    pairs = []
    for source in pack.manifest_of(link).get("sources", {}).values():
        if not isinstance(source, dict):
            continue
        repo, revision = source.get("repo"), source.get("revision")
        if isinstance(repo, str) and is_hex_digest(revision, 40):
            pairs.append((hub.snapshot(repo, revision), repo))
    return pairs


def remove_installation(models_root: Path, link: Path):
    """Remove one selection's installation and collect entries only it used.

    The link itself is the removal boundary: shared assemblies, blobs and Hub
    snapshots stay until their last selection or pin is gone. A server holding
    the assembly keeps it valid for that server's lifetime.
    """
    from . import assembly, hub

    models_root = Path(models_root)
    link = Path(link)
    if link not in selection_links(models_root):
        raise ModelError(f"{link} is not a selection under {models_root}")
    if installation_kind(link) is None:
        raise ModelError(f"no model is installed at {link}")
    with installation_lock(models_root):
        pins = [
            hub.pinned(snapshot, link) for snapshot, _ in _installation_pin_pairs(link)
        ]
        link.unlink()
        parent = link.parent
        if parent != models_root and parent.name != layout.SELECTIONS:
            try:
                parent.rmdir()
            except OSError:
                pass
        hub.retire_other_pins([], pins)
        assembly.collect_garbage(models_root)


@contextmanager
def installation_lock(models: Path):
    """Exclusive access to everything written under models."""
    lock_path = models / layout.INSTALL_LOCK
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                dim("Another RichEngine model installation is running; waiting..."),
                flush=True,
            )
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def link_selection(link: Path, target: Path):
    """Point the selection link at target, atomically. Only a link is ever
    replaced: a directory at its path is someone's own files."""
    if link.exists() and not link.is_symlink():
        raise ModelError(f"refusing to replace non-symlink model path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f"{LINK_STAGING}{link.name}-", dir=link.parent)
    )
    temporary = stage / "model"
    try:
        os.symlink(target, temporary, target_is_directory=True)
        os.replace(temporary, link)
    finally:
        temporary.unlink(missing_ok=True)
        stage.rmdir()


def _rewrite_internal_links(source: Path, destination: Path):
    """Move copied internal links and assembly file records into destination.

    Assembly records name absolute file paths. Rewriting one changes the
    record digest, so its .resolved directory is renamed to the new digest and
    selection links are pointed there in the same pass.
    """
    source_real = Path(os.path.realpath(source))
    destination_real = Path(os.path.realpath(destination))
    resolved = destination_real / ".resolved"
    renamed = {}

    def inside_source(path):
        try:
            return Path(os.path.realpath(path)).relative_to(source_real)
        except ValueError:
            return None

    if resolved.is_dir():
        for assembly_dir in sorted(resolved.iterdir()):
            record_path = assembly_dir / "model.json"
            if not record_path.is_file():
                continue
            try:
                record = json.loads(record_path.read_text())
            except ValueError:
                continue
            changed = False
            files = record.get("files")
            if isinstance(files, dict):
                for entry in files.values():
                    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                        continue
                    inside = inside_source(entry["path"])
                    if inside is None:
                        continue
                    entry["path"] = str(destination_real / inside)
                    changed = True
            if not changed:
                continue
            encoded = json_bytes(record)
            record_path.write_bytes(encoded)
            digest = hashlib.sha256(encoded).hexdigest()
            if assembly_dir.name != digest:
                renamed[assembly_dir.name] = digest
                os.rename(assembly_dir, resolved / digest)

    for link in destination.rglob("*"):
        if not link.is_symlink():
            continue
        raw = os.readlink(link)
        if not os.path.isabs(raw):
            continue
        inside = inside_source(raw)
        if inside is None:
            continue
        parts = inside.parts
        if len(parts) >= 2 and parts[0] == ".resolved":
            assembly_name = renamed.get(parts[1], parts[1])
            inside = Path(".resolved", assembly_name, *parts[2:])
        target = destination_real / inside
        link.unlink()
        link.symlink_to(target, target_is_directory=target.is_dir())


def _write_config(config):
    """Atomically write DATA/config.json, preserving unrelated keys."""
    paths.DATA.mkdir(parents=True, exist_ok=True)
    path = paths.DATA / "config.json"
    existing = {}
    try:
        loaded = json.loads(path.read_text())
        existing = loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError):
        pass
    existing.update(config)
    temporary = tempfile.NamedTemporaryFile(
        dir=paths.DATA, prefix=".config-", delete=False, mode="w"
    )
    try:
        with temporary:
            json.dump(existing, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, path)
    except BaseException:
        Path(temporary.name).unlink(missing_ok=True)
        raise


def _tree_file_totals(root: Path):
    """Regular-file count and bytes copytree's copy function will process."""
    files = 0
    bytes_ = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            try:
                if path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            files += 1
            bytes_ += size
    return files, bytes_


def _tree_bytes(root: Path):
    """The apparent bytes of a tree, following no symlink."""
    total = 0
    for directory, dirs, files in os.walk(root):
        for name in (*dirs, *files):
            try:
                total += (Path(directory) / name).lstat().st_size
            except OSError:
                continue
    return total


def _format_bytes(bytes_: int) -> str:
    value = float(bytes_)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{bytes_}B"


def verify_installations(models_root: Path) -> int:
    """Quick-verify every installed selection under models_root; return the count."""
    verified = 0
    for link in selection_links(Path(models_root)):
        if installation_kind(link) is not None:
            verify_installation(link, full=False)
            verified += 1
    return verified


def migrate_models_root(destination: Path):
    """Copy the store chosen without --models to destination and configure it.

    Regular files are hardlinked where the filesystem permits, so same-volume
    migration does not duplicate weights. Symlinks are preserved, except links
    into the copied store itself, which are rewritten to the new copy. The new
    copy is verified before config.json starts pointing at it.
    """
    source, source_kind = paths.models_root()
    destination = destination.expanduser().absolute()
    if not source.is_dir():
        raise ModelError(f"the current models root does not exist: {source}")
    source_real = Path(os.path.realpath(source))
    destination_real = Path(os.path.realpath(destination))
    if source_real == destination_real:
        raise ModelError("the migration destination is already the models root")
    if destination_real.is_relative_to(source_real):
        raise ModelError("the migration destination cannot be inside the models root")
    if destination.exists() and any(destination.iterdir()):
        raise ModelError(f"the migration destination is not empty: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_files, total_bytes = _tree_file_totals(source)
    progress = {"files": 0, "bytes": 0, "next": 0.1}
    print(
        f"Copying {source} -> {destination}: {total_files} files, "
        f"{_format_bytes(total_bytes)} (hardlinked where possible)...",
        flush=True,
    )

    def copy_with_progress(file_source, file_destination):
        file_source = Path(file_source)
        file_destination = Path(file_destination)
        try:
            size = file_source.stat().st_size
        except OSError:
            size = 0
        try:
            os.link(file_source, file_destination)
        except FileExistsError:
            pass
        except OSError:
            # Cross-volume copies can sit on one multi-GB weight for minutes:
            # progress within the file matters more than a file count does.
            copied_file = 0
            next_file_report = 256 * 1024 * 1024
            temporary = tempfile.NamedTemporaryFile(
                dir=file_destination.parent, prefix=".migrating-", delete=False
            )
            temporary_path = Path(temporary.name)
            try:
                with temporary as output, file_source.open("rb") as input:
                    while chunk := input.read(8 * 1024 * 1024):
                        output.write(chunk)
                        copied_file += len(chunk)
                        if copied_file >= next_file_report:
                            print(
                                f"migrate: {file_source.name} "
                                f"{_format_bytes(copied_file)}/{_format_bytes(size)}",
                                flush=True,
                            )
                            next_file_report += 256 * 1024 * 1024
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_path, file_destination)
                shutil.copystat(file_source, file_destination)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        progress["files"] += 1
        progress["bytes"] += size
        complete = progress["files"] == total_files
        fraction = progress["bytes"] / total_bytes if total_bytes else 1.0
        if complete or fraction >= progress["next"]:
            print(
                f"migrate: {progress['files']}/{total_files} files, "
                f"{_format_bytes(progress['bytes'])}/{_format_bytes(total_bytes)}",
                flush=True,
            )
            while progress["next"] <= fraction:
                progress["next"] += 0.1
        return file_destination

    with installation_lock(source):
        shutil.copytree(
            source,
            destination,
            symlinks=True,
            copy_function=copy_with_progress,
            ignore=shutil.ignore_patterns(".install.lock"),
            dirs_exist_ok=True,
        )
        _rewrite_internal_links(source, destination)
    try:
        verified = verify_installations(destination)
    except (ModelError, OSError) as error:
        raise ModelError(
            f"the migrated copy failed verification; config.json was not changed: {error}"
        ) from error
    _write_config({"models_dir": str(destination)})
    print(f"Migrated models root {source} -> {destination}")
    print(f"Verified {verified} migrated installation{'s' if verified != 1 else ''}.")
    print(f"Recorded models_dir in {paths.DATA / 'config.json'}")
    print(
        f"The previous root was left in place at {source} "
        f"({_format_bytes(_tree_bytes(source))}, resolved as {source_kind})."
    )
    print(
        "Same-volume migration uses hardlinks: old and new files share inodes, "
        "so removing the old root frees only data no new-root link still uses. "
        "After a successful serve from the new root, remove the old root manually "
        "(for example with rm -rf)."
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Install RichEngine runtime weights")
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help="models directory (default: RICHENGINE_MODELS_DIR, config.json or the install root's)",
    )
    parser.add_argument(
        "--model",
        type=parse_model_id,
        help="Hugging Face repository ID (owner/repo[:variant]) or a local model directory",
    )
    parser.add_argument("--revision", help="optional upstream branch, tag or commit")
    parser.add_argument(
        "--draft-model",
        type=parse_draft_model,
        help="override the automatically selected DFlash2 repository or local directory",
    )
    parser.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading",
    )
    parser.add_argument(
        "--thin",
        action="store_true",
        help=(
            "experimental, MLX affine targets only: keep serving weights in the "
            "packed cache and drop target payload links on the start after that "
            "cache validates; the cache is RICHENGINE_PACKED_CACHE_ROOT or "
            "~/Library/Caches/RichEngine/packed, and losing it downloads and "
            "rebuilds the payload"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("verify").add_argument("--full", action="store_true")
    commands.add_parser("link", help="print the selection link")
    commands.add_parser("root", help="print the resolved models root and its source")
    commands.add_parser(
        "migrate",
        help="copy the current models root to --models and record it in config.json",
    )
    commands.add_parser(
        "local", help="list the models the local caches hold (LM Studio, oMLX)"
    )
    args = parser.parse_args(argv)
    if args.command not in ("local", "root", "migrate") and not args.model:
        parser.error("--model is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    models_root, models_source = paths.models_root(args.models)
    if args.command == "root":
        print(models_root)
        print(f"source: {models_source}")
        return 0
    if args.command == "migrate":
        try:
            if args.models is None:
                raise ModelError("migrate requires --models DESTINATION")
            migrate_models_root(args.models)
        except (ModelError, OSError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        return 0
    if args.command == "local":
        from . import local

        candidates = local.scan()
        if not candidates:
            roots = ", ".join(str(root) for root in local.model_dirs()) or "none"
            print(f"No models found under the local model directories ({roots}).")
            return 0
        width = max(len(candidate.name) for candidate in candidates)
        for candidate in candidates:
            size = f"{candidate.bytes / 1e9:.1f} GB" if candidate.bytes else "-"
            verdict = candidate.reason or candidate.family or ""
            detail = (
                f"{candidate.quantization}; " if candidate.quantization else ""
            ) + verdict
            print(f"{candidate.name:<{width}}  {candidate.format:<6} {detail} ({size})")
        return 0
    # The installer imports this module, so it imports it once it exists.
    from . import upstream

    try:
        selection = Selection.of(
            models_root,
            args.model,
            revision=args.revision,
            language_only=args.language_only,
            draft_model=args.draft_model,
            thin=args.thin,
        )
        if args.command == "link":
            print(selection.link)
            return 0
        # The launcher starts this with the stop signals blocked, so that one
        # sent while it starts is not lost; it arrives here.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
        if args.command == "prepare":
            upstream.prepare(selection)
        else:
            if installation_kind(selection.link) is None:
                raise ModelError(f"{args.model} is not installed in {models_root}")
            verify_installation(
                selection.link, repo_id=selection.repo_id, full=args.full
            )
            print(
                f"RichEngine model {args.model} preflight passed "
                f"({'full' if args.full else 'quick'})."
            )
    except (ModelError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    # The other installer modules import this file by its module name. Run
    # that module, not this __main__ copy, so the ModelError they raise is the
    # one main() reports.
    import importlib

    installer = importlib.import_module("install.models")
    raise SystemExit(installer.main())
