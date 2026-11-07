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

        models_root = paths.MODELS
    if cache_dir is None:
        cache_dir = hf_cache_dir()
    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        return None
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

        for repo_info in info.repos:
            refs = repo_info.repo_path / "refs" / "richengine"
            if not refs.is_dir():
                continue
            for ref in refs.rglob("*"):
                if ref.is_file() and install_models.is_hex_digest(ref.name, 40):
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
    }


def execute(result):
    """Delete everything plan() marked dead; returns bytes freed."""
    info = result["info"]
    freed = 0
    if result["dead_revisions"]:
        strategy = info.delete_revisions(*sorted(set(result["dead_revisions"])))
        freed += strategy.expected_freed_size
        strategy.execute()
    for path in result["incomplete"]:
        try:
            size = Path(path).stat().st_size
            Path(path).unlink()
            freed += size
        except OSError:
            pass
    try:
        from huggingface_hub.utils import _shared_blobs

        for store_path in result["shared_blobs"]:
            freed += _shared_blobs.sweep_shared_blob(
                Path(store_path), cache_dir=result["cache_dir"]
            )
    except Exception:
        pass
    return freed


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
