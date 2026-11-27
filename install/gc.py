"""Garbage-collect the Hugging Face cache against installed assemblies.

Resolved models symlink into the Hub cache for tokenizer, config and
weight files; everything else the cache holds is dead weight — repos
browsed but never installed, unselected revisions, orphaned Xet store
payloads. The plan walks the models root for those links, then asks
huggingface_hub's cache manager to drop what nothing references. The
cache is shared with other tools, so deletion only ever removes entries
no assembly links to — and only on `disk gc --yes`.
"""

import os
import shutil
from pathlib import Path


def hf_cache_dir():
    """The Hub cache root huggingface_hub scans (HF_HOME/hub)."""
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path(
            os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")
        ) / "hub"


def _preserved_snapshot_paths(models_root, cache_dir):
    """Hub-cache snapshot entries the assemblies under models_root link to.

    A link's immediate target is the snapshot path that must keep
    existing; resolving it reaches the blob or shared-store payload the
    download machinery refcounted.
    """
    preserved = set()
    if not models_root.is_dir():
        return preserved
    hub = os.path.normpath(str(cache_dir))
    for path in models_root.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            raw = os.readlink(path)
        except OSError:
            continue
        target = raw if os.path.isabs(raw) else os.path.join(path.parent, raw)
        target = os.path.normpath(target)
        # Links may spell the cache through a symlinked parent (macOS
        # /tmp); resolve the directory part — not the file, which is
        # itself a link into blobs/ or the shared store.
        target = os.path.normpath(
            os.path.join(
                os.path.realpath(os.path.dirname(target)),
                os.path.basename(target),
            )
        )
        if target == hub or target.startswith(hub + os.sep):
            preserved.add(target)
    return preserved


def _live_shared_paths(preserved, cache_dir):
    """Shared-store payload paths assemblies still resolve to."""
    try:
        from huggingface_hub.utils import _shared_blobs

        store_dir = os.path.normpath(str(_shared_blobs.shared_blobs_dir(cache_dir)))
    except Exception:
        store_dir = os.path.normpath(str(cache_dir / "blobs"))
    live = set()
    for target in preserved:
        try:
            final = Path(target).resolve()
        except OSError:
            continue
        s = os.path.normpath(str(final))
        if s.startswith(store_dir + os.sep):
            live.add(s)
    return live


def _live_assemblies(models_root):
    """Selection-linked or server-held assemblies under models_root."""
    from . import assembly, layout, models

    root = Path(models_root)
    resolved = root / layout.RESOLVED
    linked = {link.resolve() for link in models.selection_links(root)}
    if not resolved.is_dir():
        return []
    return [
        entry
        for entry in sorted(resolved.iterdir())
        if entry.resolve() in linked or assembly.is_held(entry)
    ]


def _live_pinned_snapshots(models_root):
    """(repo folder, commit) pairs live installations may pin for repair.

    Pin-owner hashes are irreversible, and roots or selection hashes can
    change while the pinned snapshot remains exactly the repair source a live
    installation records. A pin therefore protects a revision when a live
    installation needs that revision, whichever selection originally wrote
    the ref. Pins of revisions no live installation needs — including an
    abandoned thin target payload — protect nothing and do not block gc.
    """
    from . import assembly, models

    needed = set()
    for link in models.selection_links(Path(models_root)):
        try:
            if models.installation_kind(link) is None:
                continue
            if models.installation_kind(link) == models.ASSEMBLY:
                record = models.read_json(link / "model.json")
                pairs = assembly.pins(record, link)
            else:
                pairs = models._installation_pin_pairs(link)
        except Exception:
            # One malformed install must not make every repair pin look dead.
            continue
        for snapshot, _repo in pairs:
            if models.is_hex_digest(snapshot.name, 40):
                needed.add((snapshot.parent.parent.name, snapshot.name))
    return needed


def owned_blob_plan(models_root):
    """Unreferenced files of the model store's content-addressed .blobs root.

    A blob is live when a selection-linked or server-held assembly records it.
    assembly.collect_garbage uses the same ownership rule after installations;
    exposing it here lets `disk gc` account for orphans interrupted installs
    left behind.
    """
    from . import assembly, layout, models

    root = Path(models_root)
    blobs = (root / layout.BLOBS).resolve()
    if not blobs.is_dir():
        return {}
    referenced = set()
    for entry in _live_assemblies(root):
        try:
            assembly_record = models.read_json(entry / layout.ASSEMBLY_RECORD)
            files = assembly_record.get("files")
        except models.ModelError:
            continue
        thin_valid = assembly_record.get("thin") is True and assembly.thin_cache_valid(entry)
        for name, record in files.items() if isinstance(files, dict) else ():
            if assembly.thin_payload_name(name) and thin_valid:
                continue
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                continue
            path = Path(record["path"])
            try:
                if path.parent.resolve() == blobs:
                    referenced.add(path.name)
            except OSError:
                pass
    orphans = {}
    for blob in blobs.iterdir():
        if not blob.is_file() or blob.name in referenced:
            continue
        try:
            orphans[blob] = blob.stat().st_size
        except OSError:
            pass
    return orphans


def packed_cache_dir():
    """The shared packed weight-image cache root, or None when disabled."""
    configured = os.environ.get("RICHENGINE_PACKED_CACHE_ROOT")
    if configured is not None:
        return Path(configured).expanduser() if configured.strip() else None
    from . import paths

    return paths.packed_cache_root()


def _entry_bytes(entry):
    total = 0
    for path in entry.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def packed_cache_plan(models_root, cache_root=None):
    """Shared packed-cache entries no live assembly's owner marker references.

    The native loader registers each use as .owners/<assembly digest> inside
    the cache entry. Entries from retired assemblies, interrupted direct
    loads and earlier keying schemes have no live owner and are collectible;
    the cache is derived data, so deletion never invalidates an install.
    """
    root = Path(cache_root) if cache_root is not None else packed_cache_dir()
    if root is None or not root.is_dir():
        return {}
    live = {entry.name for entry in _live_assemblies(Path(models_root))}
    orphans = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        owners = entry / ".owners"
        try:
            referenced = any(
                marker.name in live
                for marker in owners.iterdir()
                if marker.is_file() and not marker.name.endswith(".tmp")
            ) if owners.is_dir() else False
        except OSError:
            referenced = False
        if not referenced:
            orphans[entry] = _entry_bytes(entry)
    return orphans


def plan(models_root=None, cache_dir=None):
    """What `disk gc` would free. Reads only; nothing is deleted.

    A revision is live while any file an assembly links to sits in it;
    repos whose revisions are all dead go whole. Shared-store payloads
    are collectible when no manifest reference survives — minus the ones
    assemblies resolve to directly.
    """
    from huggingface_hub import scan_cache_dir

    if models_root is None:
        from . import paths

        models_root = paths.models_root()[0]
    if cache_dir is None:
        cache_dir = hf_cache_dir()
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return {
            "cache_dir": cache_dir,
            "info": None,
            "preserved": 0,
            "dead_revisions": [],
            "dead_repos": [],
            "live_revisions": 0,
            "shared_blobs": {},
            "incomplete": [],
            "repos": 0,
            "owned_blobs": owned_blob_plan(models_root),
            "packed_cache": packed_cache_plan(models_root),
        }
    # scan_cache_dir reports resolved paths; match them so preserved links
    # compare equal even when HF_HOME sits under a symlink.
    cache_dir = cache_dir.resolve()

    preserved = _preserved_snapshot_paths(Path(models_root), cache_dir)
    live_shared = _live_shared_paths(preserved, cache_dir)
    info = scan_cache_dir(cache_dir)

    # Assemblies pin the snapshots they link under refs/richengine/ so a
    # missing cache file can be repaired without a download; a pinned
    # commit stays live even when nothing symlinks to it today.
    pinned = set()
    try:
        from . import models as install_models

        live_pins = _live_pinned_snapshots(models_root)
        for repo_info in info.repos:
            refs = repo_info.repo_path / "refs" / "richengine"
            if not refs.is_dir():
                continue
            for ref in refs.rglob("*"):
                if (
                    ref.is_file()
                    and (repo_info.repo_path.name, ref.name) in live_pins
                    and install_models.is_hex_digest(ref.name, 40)
                ):
                    pinned.add(ref.name)
    except Exception:
        pass

    dead_revisions = []
    dead_repos = []
    live_revisions = 0
    for repo in info.repos:
        live = False
        for revision in repo.revisions:
            if revision.commit_hash in pinned or any(
                os.path.normpath(str(f.file_path)) in preserved
                for f in revision.files
            ):
                live = True
                live_revisions += 1
            else:
                dead_revisions.append(revision.commit_hash)
        if not live:
            dead_repos.append(repo.repo_id)

    shared_blobs = {}
    try:
        from huggingface_hub.utils import _shared_blobs

        for store_path, size in _shared_blobs.unreferenced_shared_blobs(
            cache_dir
        ).items():
            if os.path.normpath(str(store_path)) not in live_shared:
                shared_blobs[str(store_path)] = size
    except Exception:
        pass

    incomplete = [f.file_path for f in info.incomplete_files]
    return {
        "cache_dir": cache_dir,
        "info": info,
        "preserved": len(preserved),
        "dead_revisions": dead_revisions,
        "dead_repos": dead_repos,
        "live_revisions": live_revisions,
        "shared_blobs": shared_blobs,
        "incomplete": incomplete,
        "repos": len(info.repos),
        "owned_blobs": owned_blob_plan(models_root),
        "packed_cache": packed_cache_plan(models_root),
    }


def execute_detailed(result):
    """Delete everything plan() marked dead; returns bytes freed by category."""
    info = result["info"]
    freed = {
        "owned_blobs": 0,
        "packed_cache": 0,
        "revisions": 0,
        "incomplete": 0,
        "shared_blobs": 0,
    }
    for path, size in result.get("owned_blobs", {}).items():
        try:
            Path(path).unlink()
            freed["owned_blobs"] += size
        except OSError:
            pass
    for path, size in result.get("packed_cache", {}).items():
        try:
            shutil.rmtree(path)
            freed["packed_cache"] += size
        except OSError:
            pass
    if result["dead_revisions"]:
        strategy = info.delete_revisions(*sorted(set(result["dead_revisions"])))
        freed["revisions"] += strategy.expected_freed_size
        strategy.execute()
    for path in result["incomplete"]:
        try:
            size = Path(path).stat().st_size
            Path(path).unlink()
            freed["incomplete"] += size
        except OSError:
            pass
    try:
        from huggingface_hub.utils import _shared_blobs

        for store_path in result["shared_blobs"]:
            freed["shared_blobs"] += _shared_blobs.sweep_shared_blob(
                Path(store_path), cache_dir=result["cache_dir"]
            )
    except Exception:
        pass
    return freed


def execute(result):
    """Delete everything plan() marked dead; returns bytes freed."""
    return sum(execute_detailed(result).values())


def reclaimable_hint(models_root=None):
    """A one-line 'disk gc would free X' note after installs, or None."""
    try:
        result = plan(models_root)
        if result is None:
            return None
        estimate = sum(result["shared_blobs"].values())
        # Revisions/repos sizes need the delete strategy's accounting; the
        # shared-blob and incomplete bytes are the cheap lower bound, and a
        # nonzero dead-revision count means there is more.
        if not result["dead_repos"] and not result["dead_revisions"]:
            if estimate < (1 << 30):
                return None
        dead = len(result["dead_revisions"])
        repos = len(result["dead_repos"])
        detail = []
        if repos:
            detail.append(f"{repos} repos")
        if dead:
            detail.append(f"{dead} revisions")
        if result["shared_blobs"]:
            detail.append(f"{len(result['shared_blobs'])} shared blobs")
        return (
            "richengine disk gc could free ~"
            f"{_human_size(estimate)} of Hub cache "
            f"({' + '.join(detail) or 'unreferenced files'})"
        )
    except Exception:
        return None


def _human_size(num):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024 or unit == "TB":
            return f"{num:.1f}{unit}" if unit != "B" else f"{num}B"
        num /= 1024
