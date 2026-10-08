"""Model disk stores and wipe/dedupe actions for the HTTP API."""

import argparse
import contextlib
import io
import json
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


def stores(port):
    """Every store's rows as `disk --calc --json` lists them, largest first."""
    from install import launcher

    rows = launcher._store_rows(calc=True)
    rows.sort(key=lambda row: -(row["bytes"] or 0))
    owner = launcher._serve_owner(port)
    serving = owner is not None and launcher._pid_running(owner["pid"])
    for row in rows:
        # A splash store cannot be wiped while this engine serves from it.
        row["locked"] = bool(serving and row["name"] in launcher.SPLASH_STORES)
    return {"stores": rows, "free_bytes": shutil.disk_usage(Path.home()).free}


def wipe(store, model, port):
    from install import launcher

    # The CLI's argparse choices guard this; here an unknown name would die as
    # a bare StopIteration inside _disk_wipe.
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
