"""Model disk stores and wipe/dedupe actions for the HTTP API."""

import argparse
import contextlib
import io
import json
import os
import shutil
from pathlib import Path

from .errors import APIError


def _api_error(error):
    """A launcher refusal as an HTTP error: "is serving" conflicts with the
    running engine; anything else is the request's fault."""
    status = 409 if "is serving" in str(error) else 400
    message = str(error)
    if error.hint:
        message += f" · hint: {error.hint}"
    return APIError(status, message)


def _last_document(text):
    """The last JSON document a disk command printed: dedupe --apply --json
    prints the preview and then the outcome."""
    decoder = json.JSONDecoder()
    documents = []
    index = 0
    while True:
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            break
        document, index = decoder.raw_decode(text, index)
        documents.append(document)
    return documents[-1]


def _run(command, arguments):
    """One launcher disk function's JSON stdout. The launcher is imported on
    first use so serve startup never touches the install package."""
    from install import launcher

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            command(arguments)
    except launcher.LauncherError as error:
        raise _api_error(error) from None
    return _last_document(captured.getvalue())


def _dir_bytes(path):
    """The bytes a directory tree holds, following nothing: a symlink stays
    the link. Same rule as the launcher's own accounting."""
    total = 0
    for entry in Path(path).rglob("*"):
        try:
            if entry.is_file() and not entry.is_symlink():
                total += entry.stat().st_size
        except OSError:
            pass
    return total


def stores(port):
    """Every store's rows as `disk --calc --json` lists them, largest first."""
    from install import launcher
    from install import paths as install_paths

    rows = launcher._store_rows(calc=True)
    rows.sort(key=lambda row: -(row["bytes"] or 0))
    owner = launcher._serve_owner(port)
    serving = owner is not None and launcher._pid_running(owner["pid"])
    for row in rows:
        # A splash store cannot be wiped while this engine serves from it.
        row["locked"] = bool(serving and row["name"] in launcher.SPLASH_STORES)
    models_root = install_paths.models_root()[0]
    # The packed weight-image cache sits outside the models root and is not
    # one of the wiped stores; name it so the UI can account for its bytes.
    packed_root = install_paths.packed_cache_root()
    packed = {
        "path": str(packed_root),
        "exists": packed_root.is_dir(),
        "bytes": _dir_bytes(packed_root) if packed_root.is_dir() else 0,
    }
    return {
        "stores": rows,
        "free_bytes": shutil.disk_usage(Path.home()).free,
        "models_root": str(models_root),
        "packed_cache": packed,
    }


def wipe(store, model, port):
    from install import launcher

    # The CLI's argparse choices guard this; here an unknown name would die as
    # a bare StopIteration inside _disk_wipe.
    store = launcher._canonical_store(store)
    if store not in launcher.MODEL_STORES:
        raise APIError(400, f"unknown store {store!r}")
    return _run(
        launcher._disk_wipe,
        argparse.Namespace(store=store, model=model, yes=True, json=True, port=port),
    )


def dedupe(variants, apply):
    from install import launcher

    return _run(
        launcher.dedupe,
        argparse.Namespace(variants=variants, json=True, apply=apply, yes=True),
    )


def _json_plan(result):
    """A gc plan as JSON-ready values: paths as strings, the scan_cache_dir
    object dropped. Byte maps stay {path: bytes} — the shape the web UI's
    disk page sums."""
    return {
        "cache_dir": str(result["cache_dir"]),
        "repos": result["repos"],
        "live_revisions": result["live_revisions"],
        "preserved": result["preserved"],
        "dead_repos": result["dead_repos"],
        "dead_revisions": result["dead_revisions"],
        "incomplete": [str(path) for path in result["incomplete"]],
        "shared_blobs": dict(result["shared_blobs"]),
        "owned_blobs": {
            str(path): size for path, size in result["owned_blobs"].items()
        },
        "packed_cache": {
            str(path): size for path, size in result["packed_cache"].items()
        },
    }


def gc_plan(models_root):
    """What `disk gc` would free, read-only. Safe beside serving: the plan
    preserves everything a selection links or a running server holds —
    held assemblies count as live in every category."""
    from install import gc as hub_gc

    return _json_plan(hub_gc.plan(Path(models_root)))


def gc_apply(models_root):
    """Collect everything a fresh gc plan marks dead; bytes freed by
    category and in total. The installation lock serializes against an
    install finalizing its assembly — blobs it just wrote must not read as
    orphans mid-write."""
    from install import gc as hub_gc
    from install import models as install_models

    root = Path(models_root)
    if root.is_dir():
        with install_models.installation_lock(root):
            result = hub_gc.plan(root)
            freed = hub_gc.execute_detailed(result)
    else:
        freed = hub_gc.execute_detailed(hub_gc.plan(root))
    return {"freed": sum(freed.values()), "detail": freed}


def uninstall(models_root, selection, serving):
    """Remove one installed selection: its link, then whatever no remaining
    selection links — the assembly, derived metadata, orphaned blobs and the
    tuning sidecar — through assembly.collect_garbage under the installation
    lock. `serving` is the set of model ids the server currently serves; a
    serving or otherwise held selection refuses with a conflict."""
    from install import assembly, launcher, layout
    from install import models as install_models

    models_root = Path(models_root)
    link = None
    for candidate in install_models.selection_links(models_root):
        if str(candidate.relative_to(models_root)) == selection:
            link = candidate
            break
    if link is None:
        raise APIError(404, f"no installed selection {selection!r}", "not_found")
    # Which model id the selection serves as decides the conflict: the
    # record's target repo for an optioned install, the link name otherwise —
    # exactly how /v1/models/available names its rows.
    row = next(
        (
            entry
            for entry in launcher._installed_details(models_root)
            if entry["selection"] == selection
        ),
        None,
    )
    if row is not None and row["model"] in serving:
        raise APIError(
            409,
            f"{row['model']} is serving; unload it before uninstalling",
            "engine_busy",
        )
    try:
        target = link.resolve(strict=True)
    except OSError:
        target = None
    # A held assembly stays live even after its link goes — collected only
    # once the holder exits — so uninstalling one would report bytes freed
    # that are not. Refuse instead; defense in depth behind the serving
    # check, which cannot see other serving processes.
    # resolve() canonicalizes the whole chain (macOS /tmp is itself a
    # symlink), so the .resolved spelling must canonicalize the same way
    # for the parent comparison to match.
    resolved = Path(os.path.realpath(models_root / layout.RESOLVED))
    if (
        target is not None
        and target.parent == resolved
        and assembly.is_held(target)
    ):
        raise APIError(
            409,
            f"{selection} is in use by a running server; unload it before uninstalling",
            "engine_busy",
        )
    # The bytes the removal reclaims, for the report: the link's target when
    # no other selection shares it, zero when it is shared or broken. Shared
    # Hub-cache payload the assembly symlinks to survives either way.
    freed = 0
    if target is not None and target.is_dir():
        shared = False
        for other in install_models.selection_links(models_root):
            if other == link:
                continue
            try:
                if other.resolve() == target:
                    shared = True
                    break
            except OSError:
                continue
        if not shared and str(target).startswith(
            os.path.realpath(models_root) + os.sep
        ):
            freed = _dir_bytes(target)
    with install_models.installation_lock(models_root):
        link.unlink(missing_ok=True)
        assembly.collect_garbage(models_root)
    return {"uninstalled": selection, "bytes": freed}
