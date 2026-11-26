"""Assemblies: the directories upstream models are served from.

An assembly is models/.resolved/<SHA-256 of its record>: links to its source
snapshots' files (hub.py) and to metadata derived from a GGUF
(models/.metadata/<key>), and model.json, the record naming its sources and
the identity of every linked file. It never changes once published.
Selection links point to assemblies; one that no selection links and no
server holds is collected.

The paths an assembly links, and what reads each:
  config.json            the target configuration: the MLX config.json, or
                         the one derived from the GGUF (ModelDescriptor.mm
                         inspectSourceModel)
  target/config.json     the same MLX file again, where
                         SafetensorsCheckpoint.mm reads a checkpoint's
                         configuration
  target/<shard>         the MLX safetensors shards (SafetensorsCheckpoint.mm)
  target/<name>.gguf     the GGUF target (GgufTarget.cpp findTargetGguf)
  tokenizer/config.json  the same configuration again, from which the
                         server's AutoTokenizer.from_pretrained chooses the
                         tokenizer class
  tokenizer/<file>       the MLX tokenizer files, or tokenizer.json,
                         tokenizer_config.json and chat_template.jinja derived
                         from the GGUF (server.py --tokenizer)
  draft/config.json, draft/<name>.safetensors
                         the DFlash2 checkpoint (ModelDescriptor.mm,
                         DraftCheckpoint.cpp)
  vision/config.json, vision/<shard>
                         the MLX shards holding vision_tower.*
                         (VisionLoader.cpp, through SafetensorsCheckpoint.mm)
  vision/mmproj.gguf     the GGUF vision projector (VisionLoader.cpp)
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from . import gguf, hub, layout, models, paths

GGUF_VISION = layout.GGUF_VISION
TargetFormat = Literal["mlx-affine", "gguf"]
VisionFormat = Literal["none", "safetensors", "gguf"]
TARGET_FORMATS: tuple[TargetFormat, ...] = ("mlx-affine", "gguf")
VISION_FORMATS: tuple[VisionFormat, ...] = ("none", "safetensors", "gguf")
RECORD_KEYS = {
    "version",
    "model",
    "family",
    "signature",
    "serving",
    "target_format",
    "vision_format",
    "sources",
    "files",
    "thin",
}
FILE_RECORD_KEYS = {"path", "bytes", "mtime_ns", "ctime_ns", "digest"}


def build(models_root: Path, record, files) -> Path:
    """The assembly of record, whose files map each assembly path to its
    source file: an existing one if it verifies, else built again."""
    encoded = models.json_bytes(record)

    def write(stage):
        for name, path in files.items():
            link = stage / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(path.absolute())
        _write_durably(stage / layout.ASSEMBLY_RECORD, encoded)

    destination = models_root / layout.RESOLVED / hashlib.sha256(encoded).hexdigest()
    _reuse_or_write(destination, write, verify)
    return destination


def _reuse_or_write(destination, write, check):
    """Reuse the entry at destination if check passes; else write it with
    write(stage) into a staging folder beside it, published by one rename and
    checked again. Call it under the installation lock, which serializes the
    writers of every entry."""
    if destination.exists():
        try:
            check(destination)
            return
        except (models.ModelError, OSError) as error:
            print(
                f"Rebuilding the damaged {models.dim(destination)}: {error}",
                flush=True,
            )
            shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=layout.ENTRY_STAGING, dir=destination.parent))
    try:
        write(stage)
        os.rename(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    check(destination)


def _write_durably(path, data):
    """Write a file of an entry, on disk before the entry is published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_files(models_root: Path, files):
    """Copy-or-hardlink every source file outside models_root into .blobs.

    The returned mapping keeps the same assembly paths. Files already owned by
    the models root (derived GGUF metadata, packed artifacts) are unchanged;
    Hub snapshots and local draft files become model-store blobs named by
    their content identity. Hardlinks make same-volume adoption zero-copy;
    copy is the cross-volume and unsupported-filesystem fallback. Call it
    under the installation lock.
    """
    root = models_root / layout.BLOBS
    root.mkdir(parents=True, exist_ok=True)
    owned_root = models_root.resolve()
    result = {}
    for name, source in files.items():
        source = Path(source).absolute()
        try:
            if source.resolve().is_relative_to(owned_root):
                result[name] = source
                continue
        except OSError:
            pass
        digest = _digest(source)
        destination = root / digest
        if not destination.exists():
            resolved = source.resolve()
            try:
                os.link(resolved, destination)
            except FileExistsError:
                pass
            except OSError:
                temporary = tempfile.NamedTemporaryFile(
                    dir=root, prefix=".copying-", delete=False
                )
                temporary_path = Path(temporary.name)
                try:
                    with temporary as output, resolved.open("rb") as input:
                        shutil.copyfileobj(input, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary_path, destination)
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise
        result[name] = destination
    return result


THIN_MANIFEST = "packed-manifest.json"
THIN_SOURCES = "packed-sources"


def thin_payload_name(name) -> bool:
    """Whether an assembly path is a thin install's droppable target payload."""
    return (
        isinstance(name, str)
        and name.startswith(layout.TARGET + "/")
        and name.endswith(".safetensors")
    )


def _thin_checkpoint_documents(files):
    """The thin target manifest and packed-sources stamp lines for files.

    files maps assembly paths to owned source files after materialization. The
    JSON document preserves every target shard's tensor index; the text document
    is the exact shard stamp the native packed-cache key uses. Both are small
    metadata files even for a many-shard checkpoint.
    """
    shards = []
    stamp = []
    for name in sorted(n for n in files if thin_payload_name(n)):
        path = Path(files[name])
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 2 <= size <= 1 << 20:
                raise models.ModelError(f"invalid safetensors header in {name}")
            try:
                header = json.loads(stream.read(size))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise models.ModelError(f"invalid safetensors header in {name}") from error
        tensors = {}
        for tensor, record in sorted(header.items()):
            if tensor == "__metadata__":
                continue
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("dtype"), str)
                or not isinstance(record.get("shape"), list)
                or not isinstance(record.get("data_offsets"), list)
                or len(record["data_offsets"]) != 2
            ):
                raise models.ModelError(f"invalid safetensors tensor record in {name}")
            tensors[tensor] = {
                "dtype": record["dtype"],
                "shape": record["shape"],
                "data_offsets": record["data_offsets"],
            }
        digest = path.name if models.is_hex_digest(path.name, 64) else models.sha256(path)
        shards.append(
            {
                "name": Path(name).name,
                "digest": digest,
                "bytes": path.stat().st_size,
                "data_offset": 8 + size,
                "tensors": tensors,
            }
        )
        stamp.append(f"{Path(name).name} digest {digest}\n")
    manifest = {"version": 1, "shards": shards}
    return models.json_bytes(manifest), "".join(stamp).encode()


def _write_blob(models_root: Path, data: bytes) -> Path:
    """Write a small content-addressed model-store blob and return its path."""
    root = models_root / layout.BLOBS
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    destination = root / digest
    if not destination.exists():
        temporary = tempfile.NamedTemporaryFile(
            dir=root, prefix=".copying-", delete=False
        )
        temporary_path = Path(temporary.name)
        try:
            with temporary as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    return destination


def add_thin_documents(models_root: Path, files):
    """Add the thin target's metadata documents to an assembly file mapping."""
    manifest, sources = _thin_checkpoint_documents(files)
    files = dict(files)
    files[layout.TARGET + "/" + THIN_MANIFEST] = _write_blob(models_root, manifest)
    files[layout.TARGET + "/" + THIN_SOURCES] = _write_blob(models_root, sources)
    return files


def _packed_cache_root():
    configured = os.environ.get("RICHENGINE_PACKED_CACHE_ROOT")
    if configured is not None:
        return Path(configured).expanduser() if configured.strip() else None
    return paths.packed_cache_root()


def packed_cache_entry_valid(entry: Path) -> bool:
    """Whether a shared packed-cache entry is complete as its stamp records it."""
    try:
        lines = (entry / ".packed").read_text().splitlines()
    except OSError:
        return False
    images = []
    for line in lines:
        parts = line.split()
        if parts and parts[0].endswith(".bin") and len(parts) >= 5:
            try:
                images.append((parts[0], int(parts[4])))
            except ValueError:
                return False
    if not images:
        return False
    for name, size in images:
        try:
            if (entry / name).stat().st_size != size:
                return False
        except OSError:
            return False
    return True


def _engine_fingerprint_text() -> str | None:
    """The native binary fingerprint .validated markers carry."""
    try:
        from . import autotune

        fingerprint = autotune.engine_fingerprint(paths.BINARY)
    except OSError:
        return None
    metal_size = fingerprint.get("metallib_size")
    metal_mtime = fingerprint.get("metallib_mtime_ns")
    metal = "" if metal_size is None else f"{metal_size}:{metal_mtime}"
    return f"{fingerprint['size']}:{fingerprint['mtime_ns']}:{metal}"


def thin_cache_valid(assembly: Path) -> bool:
    """Whether this assembly has a complete packed cache validated by the runtime.

    The native loader writes .validated/<assembly digest> only after reading or
    completing the entry keyed by the same plan it is running. This is the thin
    installation's canonical-storage check; an absent or incomplete entry makes
    verification fail so prepare reinstalls the source payload.
    """
    root = _packed_cache_root()
    if root is None or not root.is_dir():
        return False
    try:
        owner = assembly.resolve().name
    except OSError:
        return False
    fingerprint = _engine_fingerprint_text()
    if fingerprint is None:
        return False
    for entry in root.iterdir():
        marker = entry / ".validated" / owner
        if not entry.is_dir() or not marker.is_file():
            continue
        try:
            lines = marker.read_text().splitlines()
        except OSError:
            continue
        if len(lines) >= 2 and lines[1] == fingerprint and packed_cache_entry_valid(entry):
            return True
    return False


def drop_thin_sources(assembly: Path, models_root: Path) -> int:
    """Drop a thin assembly's target payload links once its cache is validated.

    Owned blobs are unlinked only when no other assembly record references the
    same content. The record intentionally keeps its original file entries:
    they are the repair manifest and the refcount metadata, while missing
    payload links are what verify() tolerates for a thin install.
    """
    try:
        record = models.read_json(assembly / layout.ASSEMBLY_RECORD)
    except models.ModelError:
        return 0
    if record.get("thin") is not True or not thin_cache_valid(assembly):
        return 0
    blobs = (models_root / layout.BLOBS).resolve()
    other_references = set()
    resolved = models_root / layout.RESOLVED
    current = assembly.resolve()
    for other in resolved.iterdir() if resolved.is_dir() else ():
        if other.resolve() == current:
            continue
        try:
            files = models.read_json(other / layout.ASSEMBLY_RECORD).get("files", {})
        except models.ModelError:
            continue
        for entry in files.values() if isinstance(files, dict) else ():
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                other_references.add(entry["path"])
    dropped = 0
    for name, entry in record.get("files", {}).items():
        if not thin_payload_name(name):
            continue
        link = assembly / name
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
                dropped += 1
        except OSError:
            pass
        try:
            blob = Path(entry["path"])
            if blob.parent.resolve() == blobs and str(blob) not in other_references:
                blob.unlink(missing_ok=True)
        except OSError:
            pass
    return dropped


def retire_thin_source_pin(link: Path, record):
    """Retire the target snapshot pin of a validated thin assembly."""
    if record.get("thin") is not True or not thin_cache_valid(link):
        return
    source = record.get("sources", {}).get(layout.TARGET, {})
    revision = source.get("revision")
    repo = source.get("repo")
    if isinstance(repo, str) and models.is_hex_digest(revision, 40):
        hub.retire_other_pins([], [hub.pinned(hub.snapshot(repo, revision), link)])


def file_record(path: Path):
    """What model.json records of a linked file: its path, size and times,
    which verify compares on every start, and its content digest, which
    verify(full=True) recomputes (_digest_of)."""
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "digest": _digest(path),
    }


def _digest(path):
    """A Hub cache file's name in blobs/: the digest the Hub publishes for it,
    the Git blob SHA-1 of a small file (40 hex digits) or the SHA-256 of an
    LFS file (64). Any other file's SHA-256."""
    blob = path.resolve()
    if blob.parent.name == "blobs" and (
        models.is_hex_digest(blob.name, 40) or models.is_hex_digest(blob.name, 64)
    ):
        return blob.name
    return models.sha256(path)


def _digest_of(path, digest):
    """path's content digest in the scheme of digest (_digest)."""
    if len(digest) == 64:
        return models.sha256(path)
    size = path.stat().st_size
    return models.hash_file(path, hashlib.sha1(f"blob {size}\0".encode()))


def verify(assembly: Path, *, full=False):
    """The record of the assembly at assembly (an entry of .resolved, or a
    selection link to one) if it verifies: model.json has the shape build
    records, and each assembly path still links the file it records, with
    the recorded size and times (with full, the recorded content). Needs no
    Hub access."""
    record = models.read_json(assembly / layout.ASSEMBLY_RECORD)
    if not _well_formed(record):
        raise models.ModelError("invalid resolved model record")
    if _packed_draft(record["files"]):
        raise models.ModelError("its draft is not a DFlash2 checkpoint")
    thin_valid = record.get("thin") is True and thin_cache_valid(assembly)
    for name, entry in record["files"].items():
        path = assembly / name
        try:
            stat = path.stat()
        except OSError:
            if thin_valid and thin_payload_name(name):
                continue
            raise models.ModelError("resolved model file missing: " + name) from None
        # A blob is content-addressed and may be a hardlink of its original
        # cache object. Linking or unlinking either name updates the shared
        # inode's ctime, so ctime is recorded only for format compatibility,
        # never verified. Hub snapshot files keep the mtime check; owned blobs
        # are immutable content addressed by their name.
        owned_blob = Path(entry["path"]).parent.name == layout.BLOBS
        if (
            not path.is_symlink()
            or str(path.readlink()) != entry["path"]
            or stat.st_size != entry["bytes"]
            or (not owned_blob and stat.st_mtime_ns != entry["mtime_ns"])
        ):
            raise models.ModelError("resolved model file changed: " + name)
        if full and _digest_of(path, entry["digest"]) != entry["digest"]:
            raise models.ModelError("source content hash mismatch: " + name)
    return record


def _well_formed(record):
    gguf_target = record.get("target_format") == "gguf"
    sources = record.get("sources")
    files = record.get("files")
    return (
        # signature and serving joined the record after version 1 shipped:
        # a record without them is older, not invalid.
        set(record) <= RECORD_KEYS | ({"metadata"} if gguf_target else set())
        and RECORD_KEYS - {"signature", "serving", "thin"} <= set(record)
        and type(record["version"]) is int
        and record["version"] == 1
        and isinstance(record["model"], str)
        and isinstance(record["family"], str)
        and record["target_format"] in TARGET_FORMATS
        and record["vision_format"] in VISION_FORMATS
        and (not gguf_target or models.is_hex_digest(record["metadata"], 64))
        and isinstance(sources, dict)
        # A draft-less family (Granite) records its target source alone.
        and set(sources) in ({layout.TARGET}, {layout.TARGET, layout.DRAFT})
        and all(
            isinstance(source, dict)
            and set(source) == {"repo", "revision"}
            and isinstance(source["repo"], str)
            and (
                source["revision"] is None
                or models.is_hex_digest(source["revision"], 40)
            )
            for source in sources.values()
        )
        and isinstance(files, dict)
        and ("thin" not in record or type(record["thin"]) is bool)
        and (record.get("thin") is not True or record["target_format"] == "mlx-affine")
        and all(
            models.is_safe_path(name)
            and isinstance(entry, dict)
            and set(entry) == FILE_RECORD_KEYS
            and isinstance(entry["path"], str)
            and all(
                type(entry[key]) is int for key in ("bytes", "mtime_ns", "ctime_ns")
            )
            and (
                models.is_hex_digest(entry["digest"], 40)
                or models.is_hex_digest(entry["digest"], 64)
            )
            for name, entry in files.items()
        )
    )


def _packed_draft(files):
    """Whether the assembly links a packed draft, draft/model.bin and
    draft/layer-N.bin, as assemblies did before drafts were prepared from
    their DFlash2 checkpoints; the runtime loads only a checkpoint now."""
    return any(
        name.startswith(layout.DRAFT + "/") and name.endswith(".bin") for name in files
    )


def pins(record, assembly: Path | None = None):
    """The Hub snapshot of each source an assembly links files from, with its
    repository ID: the snapshots its installation pins. Each is found from a
    linked file's path, not the cache location: HF_HUB_CACHE may name another
    folder now than when the assembly was built."""
    thin_valid = (
        assembly is not None
        and record.get("thin") is True
        and thin_cache_valid(assembly)
    )
    sources = {
        (hub.folder_name(source["repo"]), source["revision"]): source["repo"]
        for kind, source in record["sources"].items()
        if source["revision"] is not None
        and not (thin_valid and kind == layout.TARGET)
    }
    snapshots = {}
    for entry in record["files"].values():
        path = hub.snapshot_of(entry["path"])
        key = path and (path.parent.parent.name, path.name)
        if key in sources:
            snapshots[path] = sources[key]
    # Owned installations link .blobs rather than snapshot files. Pin their
    # source snapshots in the cache that served them too: legacy assemblies
    # are covered by the linked-path scan above, while new ones keep the
    # offline repair reference without depending on the cache for serving.
    for (folder_name, revision), repo in sources.items():
        path = hub.snapshot(repo, revision)
        if path.is_dir():
            snapshots.setdefault(path, repo)
    return sorted(snapshots.items())


def recorded_pins(link: Path):
    """pins of the assembly a selection link names, as its record states
    them whether or not the assembly still verifies; none without a record
    of the shape build writes."""
    try:
        record = models.read_json(link / layout.ASSEMBLY_RECORD)
    except models.ModelError:
        return []
    return pins(record) if _well_formed(record) else []


def _metadata_inputs(files):
    """The assembly paths of the GGUF files the derived metadata comes from,
    in order: the target, then the vision projector."""
    targets = [
        n
        for n in sorted(files)
        if n.startswith(layout.TARGET + "/") and n.endswith(".gguf")
    ]
    return targets + ([GGUF_VISION] if GGUF_VISION in files else [])


def _metadata_key(sources):
    """The .metadata entry derived from these source file records. The same
    content at another path or time is the same source; the adapter
    (gguf.py) and the tokenizers library that writes tokenizer.json are part
    of the derivation."""
    from importlib.metadata import version

    identity = {
        "sources": [{"bytes": s["bytes"], "digest": s["digest"]} for s in sources],
        "adapter": models.sha256(Path(gguf.__file__)),
        "tokenizers": version("tokenizers"),
    }
    return hashlib.sha256(models.json_bytes(identity)).hexdigest()


def metadata_key(file_records):
    """The key of the metadata this release derives from an assembly's GGUF
    files, given each assembly path's file record."""
    return _metadata_key(
        [file_records[name] for name in _metadata_inputs(file_records)]
    )


def derived_metadata(models_root: Path, files):
    """The key and the files, by assembly path, of the metadata derived from
    the GGUF files among files (assembly path -> source file): derived once,
    and again when its entry is damaged. Call it under the installation lock."""
    inputs = [files[name] for name in _metadata_inputs(files)]
    sources = [file_record(path) for path in inputs]
    key = _metadata_key(sources)

    def write(stage):
        contents = gguf.derived_files(*inputs)
        if [file_record(path) for path in inputs] != sources:
            raise models.ModelError("GGUF source changed while reading metadata")
        for name, data in contents.items():
            _write_durably(stage / name, data)
        hashes = {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        }
        _write_durably(stage / layout.METADATA_RECORD, models.json_bytes(hashes))

    entry = models_root / layout.METADATA / key
    _reuse_or_write(entry, write, _check_metadata)
    derived = {name: entry / name for name in gguf.DERIVED_FILES}
    derived[layout.TOKENIZER + "/config.json"] = entry / "config.json"
    return key, derived


def _check_metadata(entry):
    """Raise unless a metadata entry holds its files as its files.json lists them."""
    hashes = models.read_json(entry / layout.METADATA_RECORD)
    if set(hashes) != set(gguf.DERIVED_FILES):
        raise models.ModelError("invalid prepared GGUF metadata record")
    for name in gguf.DERIVED_FILES:
        if models.sha256(entry / name) != hashes[name]:
            raise models.ModelError("prepared GGUF metadata changed: " + name)


def hold(link: Path, models_root: Path):
    """The directory a selection link serves from, and what holds it: for an
    assembly, its model.json opened under a shared lock. While that file is
    open, in this process or one that inherited it, collect_garbage keeps the
    assembly. The link is read under the installation lock, where collection
    runs, so there is no moment the assembly is neither linked nor held. A
    legacy package is served from the link itself: (link, None)."""
    if models.installation_kind(link) != models.ASSEMBLY:
        return link, None
    with models.installation_lock(models_root):
        assembly = link.resolve(strict=True)
        record = (assembly / layout.ASSEMBLY_RECORD).open("rb")
        fcntl.flock(record, fcntl.LOCK_SH)
    return assembly, record


def is_held(assembly: Path) -> bool:
    """Whether a server holds the assembly (hold)."""
    try:
        with (assembly / layout.ASSEMBLY_RECORD).open("rb") as record:
            fcntl.flock(record, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    return False


def collect_garbage(models_root: Path):
    """Remove what no installation uses: assemblies that no selection links
    and no server holds, metadata entries the remaining assemblies do not
    link, and staging an interrupted installation left. Call it under the
    installation lock, where all of these are written."""
    resolved = models_root / layout.RESOLVED
    derived = models_root / layout.METADATA
    linked = {link.resolve() for link in models.selection_links(models_root)}
    # A package's tuning record lives beside its link (autotune.tuning_path);
    # a sidecar whose link is gone has nothing left to tune. The second glob
    # reaches the .selections links — "*" skips dot-directories.
    for sidecar in (
        *models_root.glob("*/*.tuning.json"),
        *(models_root / layout.SELECTIONS).glob("*.tuning.json"),
    ):
        if not (
            sidecar.parent / sidecar.name.removesuffix(".tuning.json")
        ).is_symlink():
            sidecar.unlink(missing_ok=True)
    # Locally packed packages (install/pack.py) are collected like
    # assemblies: a package no selection links is deleted, its staging too.
    packed = models_root / layout.PACKED
    for entry in sorted(packed.iterdir()) if packed.is_dir() else ():
        if entry.name.startswith(layout.PACK_STAGING) or entry not in linked:
            shutil.rmtree(entry, ignore_errors=True)
    used = set()
    blobs = (models_root / layout.BLOBS).resolve()
    used_blobs = set()
    for assembly in sorted(resolved.iterdir()) if resolved.is_dir() else ():
        if assembly.resolve() not in linked and not is_held(assembly):
            shutil.rmtree(assembly)
            continue
        # A damaged record protects no metadata: its assembly is rebuilt.
        try:
            record = models.read_json(assembly / layout.ASSEMBLY_RECORD)
            files = record.get("files")
        except models.ModelError:
            continue
        thin_valid = record.get("thin") is True and thin_cache_valid(assembly)
        for name, entry in files.items() if isinstance(files, dict) else ():
            path = Path(entry.get("path", "")) if isinstance(entry, dict) else None
            if thin_valid and thin_payload_name(name):
                continue
            if path and path.is_relative_to(derived):
                used.add(path.relative_to(derived).parts[0])
            if path and path.parent.resolve() == blobs:
                used_blobs.add(path.name)
    for entry in sorted(derived.iterdir()) if derived.is_dir() else ():
        if entry.name not in used:
            shutil.rmtree(entry)
    for blob in sorted(blobs.iterdir()) if blobs.is_dir() else ():
        if blob.name.startswith(".copying-") or blob.name not in used_blobs:
            blob.unlink(missing_ok=True)
    for stage in models_root.glob(f"*/{layout.LINK_STAGING}*"):
        shutil.rmtree(stage)
