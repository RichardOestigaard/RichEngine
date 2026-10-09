#!/usr/bin/env python3
"""Serve in the foreground, or connect an installed agent to the local server."""

import argparse
import difflib
import errno
import fcntl
import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

if __name__ == "__main__" and not __package__:
    # Run as a script by the PATH wrappers and ./richengine: import siblings as
    # the install package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "install"

# The options serve shares with the server, in modules of the standard
# library alone: the launcher runs before .venv exists.
from server import serve_options

from . import assembly, catalog, clients, paths
from . import models as model_artifacts

ROOT = paths.ROOT
RUNTIME_DIR = paths.RUNTIME
PORT = 8000
# Either stops `richengine serve` wherever it is. The programs with handlers of
# their own, the installer it runs and the server it executes, start with
# them blocked, not ignored, until those handlers are in place, so one sent
# meanwhile waits for its handler instead of being lost or ending the
# program in a traceback. make and the device check run with them unblocked.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class LauncherError(RuntimeError):
    """A failure reported as 'error: <message>', with an optional 'hint:'
    line suggesting the next step."""

    def __init__(self, message, hint=None):
        super().__init__(message)
        self.hint = hint


class StopSignal(KeyboardInterrupt):
    """One of STOP_SIGNALS, raised where it arrives, as Ctrl+C is."""

    def __init__(self, number):
        super().__init__(number)
        self.number = number


def _interrupt(number, _frame):
    raise StopSignal(number)


def _run_held(command, **options):
    """Run a program that unblocks the stop signals itself, holding them from
    its spawn. One the launcher takes meanwhile ends the program too."""
    signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
    try:
        with subprocess.Popen(command, **options) as program:
            try:
                signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
                return program.wait()
            except BaseException:
                program.kill()
                raise
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)


def _ansi(stream, *codes):
    """Wrap codes around text written to stream only when it is a terminal and
    NO_COLOR is unset: piped output and log files stay byte-plain."""
    return stream.isatty() and "NO_COLOR" not in os.environ


def _styled(text, *codes, stream=sys.stdout):
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if _ansi(stream) else text


# Python 3.14's argparse colors help with its own theme; _HelpFormatter colors
# it instead, in the palette the top-level help uses, on every version.
try:
    argparse.ArgumentParser(color=False)
    _PARSER_OPTIONS = {"color": False}
except TypeError:
    _PARSER_OPTIONS = {}


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """The same styling the top-level help uses: bold section headers, cyan
    flags, a dim 'Usage' label — only ever on a terminal."""

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None and _ansi(sys.stdout):
            prefix = _styled("Usage", "2") + ": "
        return super()._format_usage(usage, actions, groups, prefix)

    def start_section(self, heading):
        super().start_section(_styled(heading, "1") if heading else heading)

    def _format_action_invocation(self, action):
        # Python 3.14's formatter colors the invocation with its own theme
        # even when the parser's color is off; strip it and re-color in the
        # shared palette.
        text = re.sub(
            r"\x1b\[[0-9;]*m", "", super()._format_action_invocation(action)
        )
        if not _ansi(sys.stdout):
            return text
        return "".join(
            _styled(part, "36") if part.startswith("-") else part
            for part in re.split(r"(, | )", text)
        )


def _base_url(port):
    return f"http://127.0.0.1:{port}"


def _request_json(path, timeout=2, *, port=PORT):
    request = urllib.request.Request(_base_url(port) + path)
    if key := os.environ.get("RICHENGINE_API_KEY"):
        request.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 401:
            raise LauncherError(
                "RichEngine authentication failed; set RICHENGINE_API_KEY to the server's key"
            ) from None
        return None
    except (
        OSError,
        UnicodeDecodeError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return None


def _ensure_installed(selection):
    if not paths.PACKAGED:
        # Serialize builds across ports; make keeps the lock if the launcher exits.
        with (RUNTIME_DIR / "build.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for command in (
                ["make", "platform-check", "install-environment"],
                ["make", "-j4", "all"],
            ):
                if subprocess.run(
                    command, cwd=ROOT, pass_fds=(lock.fileno(),)
                ).returncode:
                    raise LauncherError("source build failed; see the output above")
    # The engine refuses an unsupported Mac only once the model is prepared;
    # its own check refuses it before tens of GB are downloaded.
    check = subprocess.run(
        [str(paths.BINARY), "device-check"], capture_output=True, text=True
    )
    if check.returncode:
        # The binary's own refusal is its last line; one that dies before
        # main() (dyld on an older macOS) leaves a report worth showing whole.
        report = check.stderr.strip()
        raise LauncherError(
            report.splitlines()[-1].removeprefix("error: ")
            if check.returncode > 0 and report
            else f"the engine's device check failed: {report or f'status {check.returncode}'}"
        )
    command = [
        str(paths.PYTHON),
        str(ROOT / "install/models.py"),
        "--models",
        str(selection.models_root),
        "--model",
        selection.model,
        "prepare",
    ]
    for flag, value in (
        ("--revision", selection.revision),
        ("--draft-model", selection.draft_model),
    ):
        if value is not None:
            command[-1:-1] = [flag, value]
    if selection.language_only:
        command.insert(-1, "--language-only")
    if _run_held(command, cwd=ROOT):
        raise LauncherError(
            "model download or verification failed",
            hint=_model_suggestion(selection.model),
        )


def _installed_details(models_root=None):
    """Each servable selection under the models root, as the record its
    assembly carries: model ID, family, formats and linked bytes. A
    .selections/<hash> link is named by the target repository its record
    states; a legacy package without a record reports its link name alone."""
    from install import autotune
    from install.tuning import knobs_for

    if models_root is None:
        models_root = paths.MODELS
    installed = []
    chip, _, _ = autotune.detect_chip()
    binary = paths.BINARY if paths.BINARY.is_file() else None

    def tune_state(link):
        # tuning.json sits in the link's target; the link reads through it.
        # A partial record's winners already apply, so the knob count is
        # real even while the sweep still runs.
        return autotune.tuning_state(link, chip=chip, binary=binary), len(
            autotune.load_tuning(link, chip=chip, binary=binary)
        )

    for link in model_artifacts.selection_links(models_root):
        if model_artifacts.installation_kind(link) is None:
            continue
        name = str(link.relative_to(models_root))
        entry = {"model": name, "selection": name, "family": None,
                 "target_format": None, "vision_format": None, "bytes": None}
        try:
            record = model_artifacts.read_json(link / "model.json")
        except model_artifacts.ModelError:
            entry["tuning"], entry["tuned_knobs"] = tune_state(link)
            entry["tunable"] = knobs_for(entry["family"]) is not None
            installed.append(entry)
            continue
        try:
            if name.startswith(".selections/"):
                entry["model"] = record["sources"]["target"]["repo"]
            entry["family"] = record.get("family")
            entry["target_format"] = record.get("target_format")
            entry["vision_format"] = record.get("vision_format")
            files = record.get("files")
            if isinstance(files, dict):
                entry["bytes"] = sum(
                    f["bytes"]
                    for f in files.values()
                    if isinstance(f, dict) and type(f.get("bytes")) is int
                )
        except (KeyError, TypeError):
            pass
        entry["tuning"], entry["tuned_knobs"] = tune_state(link)
        entry["tunable"] = knobs_for(entry["family"]) is not None
        installed.append(entry)
    installed.sort(key=lambda entry: entry["model"])
    return installed


def _installed_models():
    """The model IDs whose selection link under the models root resolves to a
    servable installation."""
    return [entry["model"] for entry in _installed_details()]


def _size(bytes_):
    if bytes_ < 1024:
        return f"{bytes_}B"
    value = bytes_
    for unit in ("K", "M", "G"):
        value /= 1024
        if value < 1024 or unit == "G":
            return f"{value:.1f}G" if unit == "G" else f"{value:.0f}{unit}"


def _suggested_models():
    """The upstream models of install/completions/suggested-models.txt, the
    same file shell completion reads."""
    try:
        text = (ROOT / "install/completions/suggested-models.txt").read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeDecodeError):
        return []
    return sorted({line.strip() for line in text.splitlines() if line.strip()})


def _model_suggestion(model_id):
    """A 'did you mean' hint for a model ID that failed, from every model ID
    this installation knows: the catalog, the suggested list and the
    installed selections."""
    candidates = sorted(
        set(catalog.official_ids())
        | set(_suggested_models())
        | set(_installed_models())
    )
    close = difflib.get_close_matches(model_id, candidates, n=3, cutoff=0.5)
    if not close and "/" in model_id:
        # An owner typo hides a right repository name; match the name alone.
        name = model_id.split("/", 1)[1].split(":")[0]
        by_name = {candidate.split("/", 1)[-1].split(":")[0]: candidate for candidate in candidates}
        names = difflib.get_close_matches(name, list(by_name), n=3, cutoff=0.7)
        close = [by_name[match] for match in names]
    return f"did you mean {', '.join(close)}?" if close else None


def _serve_lock_owner(lock):
    try:
        lock.seek(0)
        owner = json.load(lock)
    except (OSError, UnicodeError, ValueError):
        return ""
    if not isinstance(owner, dict):
        return ""
    pid, model, port = owner.get("pid"), owner.get("model"), owner.get("port")
    if (
        type(pid) is not int
        or pid <= 0
        # A server can run with no model loaded; the field is then null.
        or not (model is None or (isinstance(model, str) and model.isprintable()))
        or type(port) is not int
        or not 1 <= port <= 65535
    ):
        return ""
    return f" (PID {pid}, model {model or 'none'}, port {port})"


def _check_port(host, port):
    """Raise OSError if another process owns host:port. The probe binds with
    SO_REUSEADDR, as the HTTP listener does, so closed connections in
    TIME_WAIT do not block a restart; a live listener at the address still
    refuses the bind. The option also lets the bind succeed beside another
    process's listener at a wider or narrower address of the port (0.0.0.0
    or a dual-stack :: beside 127.0.0.1, or the reverse), and the two would
    then split the address's connections, so a listener that accepts one
    there owns the port too."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
        address = probe.getsockname()[0]
    # A wildcard address is tried on loopback, where the launcher's clients
    # connect.
    if address == "0.0.0.0":
        address = "127.0.0.1"
    with socket.socket() as client:
        client.settimeout(1)
        if client.connect_ex((address, port)) == 0:
            raise OSError(errno.EADDRINUSE, os.strerror(errno.EADDRINUSE))


def tune(args):
    """Run the autotune sweep against an installed model and record the
    winning engine knobs in its tuning.json (install/autotune.py)."""
    from install import assembly, autotune

    models_root = Path(args.models).resolve() if args.models else paths.MODELS
    if args.selection is not None:
        # The link's name under the models root, taken verbatim — installs
        # made with options live under .selections/<hash>, unreachable by
        # model ID. Membership in selection_links blocks path escapes.
        link = models_root / args.selection
        if link not in model_artifacts.selection_links(models_root):
            raise LauncherError(
                f"{args.selection} is not a selection under {models_root}"
            )
    elif args.model is not None:
        link = model_artifacts.Selection.of(
            models_root,
            args.model,
            revision=args.revision,
            language_only=args.language_only,
            draft_model=args.draft_model,
        ).link
    else:
        raise LauncherError(
            "tune needs a model: --model OWNER/REPO or --selection NAME"
        )
    if model_artifacts.installation_kind(link) is None:
        raise LauncherError(
            f"{args.selection or args.model} is not installed; install it with "
            f"'richengine serve --model {args.model or '<MODEL>'}' first"
        )
    if not paths.BINARY.is_file():
        raise LauncherError(f"no engine binary at {paths.BINARY}; run make all")
    # A running serve shares the GPU with the sweep's engines; every number
    # it measures is then contaminated. Warn rather than fail — servers on
    # different ports coexist deliberately.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for lock in RUNTIME_DIR.glob("serve-*.lock"):
        with lock.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(handle, fcntl.LOCK_UN)
            except BlockingIOError:
                print(
                    _styled("warn:", "33;1", stream=sys.stderr)
                    + " a serve is running; its engine shares this GPU "
                    "and skews every measurement",
                    file=sys.stderr,
                )
                break
    # One sweep at a time across CLI and UI — the server spawns this same
    # command, so the lock excludes both directions. Two sweeps would
    # contend for the GPU and corrupt each other's measurements.
    with (RUNTIME_DIR / "tune.lock").open("a+") as tune_lock:
        try:
            fcntl.flock(tune_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError("a tune is already running") from None
        directory, record = assembly.hold(link, models_root)
        try:
            env = autotune.tune(
                paths.BINARY,
                directory,
                include_quality=args.complete or args.include_quality_knobs,
                interactions=(args.complete or args.interaction_pass)
                and not args.no_interaction_pass,
                priors_path=models_root / autotune.PRIORS_NAME,
            )
        except model_artifacts.ModelError as error:
            raise LauncherError(str(error)) from error
        finally:
            if record is not None:
                record.close()
        if env:
            print(f"\n{_styled('Tuned', '32;1')} · {len(env)} knobs kept:")
            for name, value in sorted(env.items()):
                print(f"  {name}={value}")
        else:
            print(f"\n{_styled('Tuned', '32;1')} · the defaults already won every knob")
        print(f"  written to {autotune.tuning_path(directory)}")
    return 0


def serve(args):
    if args.model is None and (
        args.dry_run
        or args.revision is not None
        or args.language_only
        or args.draft_model is not None
    ):
        raise LauncherError("a serve without a model takes no model options")
    if args.dry_run:
        # The plan alone: resolve the selection and report what is installed,
        # without locks, downloads, builds or an engine check.
        selection = model_artifacts.Selection.of(
            paths.MODELS,
            args.model,
            revision=args.revision,
            language_only=args.language_only,
            draft_model=args.draft_model,
        )
        kind = model_artifacts.installation_kind(selection.link)
        fields = [
            ("model", selection.model),
            ("revision", selection.revision or _styled("repository default", "2")),
            ("draft", selection.draft_model or _styled("auto (family's draft)", "2")),
            (
                "vision",
                _styled("skipped", "2") if selection.language_only else "prepared",
            ),
            ("variant", selection.variant or _styled("none (MLX or single GGUF)", "2")),
            (
                "installed",
                f"{_styled(kind, '32')} at {_styled(str(selection.link), '2')}"
                if kind
                else _styled("no — serve would download it", "33"),
            ),
        ]
        for label, value in fields:
            print(f"  {_styled(f'{label:<11}', '2')}{value}")
        return 0
    # Started in the background from a non-interactive shell, the launcher
    # inherits SIGINT as ignored; take both stop signals from the start.
    for number in STOP_SIGNALS:
        signal.signal(number, _interrupt)
    if args.offline:
        # The installer, the catalog refresh and the server all read it.
        os.environ["HF_HUB_OFFLINE"] = "1"
    # Keep both locks across exec until the foreground server exits.
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with (
        (RUNTIME_DIR / "serve.lock").open("a+") as installation,
        (RUNTIME_DIR / f"serve-{args.port}.lock").open("a+") as lock,
    ):
        # Servers share the installation; upgrades require exclusive access.
        try:
            fcntl.flock(installation, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                "RichEngine is being upgraded; wait for the upgrade to finish"
            ) from None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LauncherError(
                f"RichEngine is already serving{_serve_lock_owner(lock)}; "
                "stop it with Ctrl+C first",
                hint="inspect it with 'richengine status'",
            ) from None
        lock.seek(0)
        lock.truncate()
        json.dump(
            {"pid": os.getpid(), "model": args.model, "port": args.port}, lock
        )
        lock.flush()
        # Fail before downloads/builds if another service owns the selected port.
        # The HTTP server also binds before loading weights, closing the race.
        try:
            _check_port(args.host, args.port)
        except OSError as error:
            raise LauncherError(
                f"cannot bind {args.host}:{args.port}: {error}",
                hint="choose a different port with --port",
            ) from None
        record = None
        if args.model is not None:
            selection = model_artifacts.Selection.of(
                paths.MODELS,
                args.model,
                revision=args.revision,
                language_only=args.language_only,
                draft_model=args.draft_model,
            )
            _ensure_installed(selection)
            # A concurrent install may advance the selection link. Keep this
            # process's tokenizer, draft and target on one immutable
            # assembly, held until the server exits.
            root, record = assembly.hold(selection.link, selection.models_root)
            if record is not None:
                os.set_inheritable(record.fileno(), True)
        # The server package of this installation, from any working
        # directory: -P keeps the directory, which may hold a package of the
        # same name, off sys.path, and PYTHONPATH names the root.
        command = [
            str(paths.PYTHON),
            "-u",
            "-P",
            "-m",
            "server.server",
            *(
                [
                    str(root),
                    "--tokenizer",
                    str(root / "tokenizer"),
                    "--model",
                    args.model,
                ]
                if args.model is not None
                else []
            ),
            "--models-dir",
            str(paths.MODELS),
            "--binary",
            str(paths.BINARY),
            "--port",
            str(args.port),
            *serve_options.serve_argv(args),
        ]
        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            TRANSFORMERS_VERBOSITY="error",
            PYTHONPATH=str(ROOT),
            **serve_options.serve_environment(args),
        )
        # Detached, because execve replaces this process a line later and a
        # thread would not survive it. Failure is silent by design.
        catalog.spawn_refresh()
        os.set_inheritable(installation.fileno(), True)
        os.set_inheritable(lock.fileno(), True)
        # The exec resets the handlers; the server unblocks the signals once
        # its own are in place, past its imports.
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        os.execve(command[0], command, environment)


def coding_client(args):
    path = clients.find_executable(args.command)
    listing = _request_json("/v1/models", port=args.port)
    if listing is None:
        raise LauncherError(
            f"No ready RichEngine server at {_base_url(args.port)}",
            hint="run 'richengine serve --model <HF_REPO_ID>' "
            "in another terminal first",
        )
    models = listing.get("data", []) if isinstance(listing, dict) else []
    if (
        not isinstance(models, list)
        or not models
        or not isinstance(models[0], dict)
        or models[0].get("owned_by") != "richengine"
        or type(models[0].get("context_length")) is not int
        or models[0]["context_length"] <= 0
    ):
        raise LauncherError("Could not identify the local RichEngine server")
    # The first entry is the name responses report.
    model, context = models[0].get("id"), models[0]["context_length"]
    # Only opencode needs its major version: the launch defaults changed
    # between its first and second major releases. A failed probe adds nothing.
    client_version = (
        clients.probe_major_version(path) if args.command == "opencode" else None
    )
    command, environment = clients.command(
        args.command,
        path,
        _base_url(args.port),
        model,
        context,
        input_modalities=models[0].get("input_modalities"),
        client_args=args.client_args,
        client_version=client_version,
    )
    print(f"Starting {args.command}: {model} · {context:,} context tokens", flush=True)
    if args.command == "claude":
        print(
            "Claude hosted WebSearch is unavailable. "
            "WebFetch, local tools and MCP are unchanged.",
            flush=True,
        )
    elif args.command == "codex":
        print(
            "Codex hosted WebSearch is disabled: RichEngine does not provide "
            "OpenAI's search service. Local tools and MCP are unchanged.",
            flush=True,
        )
    os.execvpe(path, command, environment)


def list_models(args):
    installed = _installed_details()
    known = sorted(set(catalog.official_ids()) | set(_suggested_models()))
    installed_ids = {entry["model"] for entry in installed}
    if args.json:
        print(
            json.dumps(
                {
                    "installed": installed if args.verbose else installed_ids,
                    "available": known,
                },
                indent=2,
            )
        )
        return 0
    if not installed and not known:
        print("No models installed and the catalog has not been fetched yet.")
        print("Serve one with: richengine serve --model <MODEL>")
        return 0
    if args.verbose:
        rows = [
            (
                entry["model"],
                "installed",
                entry["family"] or "",
                entry["target_format"] or "",
                entry["vision_format"] or "",
                _size(entry["bytes"]) if type(entry["bytes"]) is int else "",
            )
            for entry in installed
        ]
        rows += [
            (model, "", "", "", "", "")
            for model in known
            if model not in installed_ids
        ]
        header = ("MODEL", "STATUS", "FAMILY", "FORMAT", "VISION", "SIZE")
    else:
        rows = [(entry["model"], "installed") for entry in installed]
        rows += [
            (model, "") for model in known if model not in installed_ids
        ]
        header = ("MODEL", "STATUS")
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    print(_styled("  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip(), "1"))
    for row in sorted(rows):
        cells = [
            _styled(cell, "32") if i == 1 and cell == "installed" else cell
            for i, cell in enumerate(row)
        ]
        # Pad every column but the last; ANSI in 'installed' doesn't count.
        parts = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            pad = len(row[i])
            parts.append(cell + " " * (w - pad) if i < len(cells) - 1 else cell)
        print("  ".join(parts).rstrip())
    print("\nServe one with: richengine serve --model <MODEL>")
    return 0


def list_flags(args):
    """Every option of every command, read back out of the parser so the
    listing cannot drift from what parses."""
    parser = _build_parser()
    subcommands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    plain = argparse.HelpFormatter("richengine")
    styled = _HelpFormatter("richengine")
    sections = [
        (
            f"richengine {name}",
            [
                (plain._format_action_invocation(a), styled._format_action_invocation(a), a.help)
                for a in subcommands.choices[name]._actions
                if a.option_strings
            ],
        )
        for name in ("serve", "models", "status", "flags", "doctor", "disk")
    ]
    sections.append(
        (
            "richengine <agent>",
            [
                (
                    "-- <args>",
                    _styled("-- <args>", "36"),
                    "arguments after -- are passed to the agent, including --help",
                )
            ],
        )
    )
    sections.append(
        (
            "richengine",
            [
                (
                    name,
                    _styled(name, "36"),
                    help_text,
                )
                for name, help_text in (
                    ("--version", "print the version"),
                    ("-h, --help", "this help"),
                )
            ],
        )
    )
    width = max(len(invocation) for _, rows in sections for invocation, _, _ in rows)
    for title, rows in sections:
        print(_styled(title, "1"))
        for invocation, colored, help_text in rows:
            padding = " " * (width - len(invocation) + 2)
            print(f"  {colored}{padding}{help_text or ''}".rstrip())
        print()
    return 0


def _serve_owner(port):
    """The metadata serve-{port}.lock records, as _serve_lock_owner validates
    it, or None."""
    try:
        owner = json.loads((RUNTIME_DIR / f"serve-{port}.lock").read_text())
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(owner, dict):
        return None
    pid, model = owner.get("pid"), owner.get("model")
    if (
        type(pid) is not int
        or pid <= 0
        or not (model is None or (isinstance(model, str) and model.isprintable()))
        or owner.get("port") != port
    ):
        return None
    return {"pid": pid, "model": model, "port": port}


def _pid_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def status(args):
    listing = _request_json("/v1/models", port=args.port)
    entries = listing.get("data", []) if isinstance(listing, dict) else []
    served = next(
        (
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("owned_by") == "richengine"
        ),
        None,
    )
    owner = _serve_owner(args.port)
    if served is not None:
        fields = {
            "state": "serving",
            "model": served.get("id"),
            "address": _base_url(args.port),
            "context_length": served.get("context_length"),
            "input_modalities": served.get("input_modalities"),
            "pid": owner["pid"] if owner else None,
        }
        # The richer /status payload: queue depth, restarts and cache rates.
        detail = _request_json("/status", port=args.port)
        if isinstance(detail, dict):
            transport = detail.get("transport")
            if isinstance(transport, dict):
                pending = transport.get("pending")
                limit = transport.get("pending_limit")
                if type(pending) is int:
                    fields["requests"] = (
                        f"{pending} in flight"
                        + (f" of {limit} admitted" if type(limit) is int else "")
                    )
                if type(transport.get("restarts")) is int and transport["restarts"]:
                    fields["restarts"] = transport["restarts"]
            cache = detail.get("grammar_cache")
            if isinstance(cache, dict) and type(cache.get("hits")) is int:
                total = cache["hits"] + cache.get("misses", 0)
                if total:
                    fields["grammar_cache"] = (
                        f"{cache['hits'] * 100 // total}% hits of {total}"
                    )
        if args.json:
            print(json.dumps(fields, indent=2))
            return 0
        print(f"RichEngine · {_styled('serving', '32;1')}")
        for key, label in (
            ("model", "model"),
            ("address", "address"),
            ("context_length", "context"),
            ("input_modalities", "modalities"),
            ("requests", "requests"),
            ("grammar_cache", "grammars"),
            ("restarts", "restarts"),
            ("pid", "pid"),
        ):
            value = fields.get(key)
            if value is None:
                continue
            if key == "context_length" and type(value) is int:
                value = f"{value:,} tokens"
            elif isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            print(f"  {_styled(f'{label:<11}', '2')}{value}")
        return 0
    starting = owner is not None and _pid_running(owner["pid"])
    if args.json:
        print(
            json.dumps(
                {
                    "state": "starting" if starting else "absent",
                    "model": owner["model"] if starting else None,
                    "address": _base_url(args.port),
                    "pid": owner["pid"] if starting else None,
                },
                indent=2,
            )
        )
        return 1
    if starting:
        print(f"RichEngine · {_styled('starting', '33;1')}")
        print(f"  {_styled('model      ', '2')}{owner['model']}")
        print(f"  {_styled('address    ', '2')}{_base_url(args.port)}")
        print(f"  {_styled('pid        ', '2')}{owner['pid']}")
        print("Not ready yet; wait for 'Ready' in the serve terminal.")
        return 1
    print(f"RichEngine · {_styled('absent', '31;1')}")
    print(f"  No server at {_base_url(args.port)}.")
    print("  Start one with: richengine serve --model <MODEL>")
    return 1


def doctor(args):
    """One health line per check: ok green, warn yellow, fail red, info dim.
    A hard failure — a missing engine or a refused device — ends in 1."""
    ok = _styled("ok  ", "32")
    warn = _styled("warn", "33")
    fail = _styled("fail", "31")
    info = _styled("info", "2")
    report = lambda mark, name, detail: print(f"  {mark} {name:<10}{detail}")
    failed = False

    # The engine binary and its own device check, which refuses an
    # unsupported Mac before a model downloads.
    if not paths.BINARY.is_file():
        report(fail, "engine", f"no binary at {paths.BINARY}; run make all")
        failed = True
    else:
        check = subprocess.run(
            [str(paths.BINARY), "device-check"], capture_output=True, text=True
        )
        if check.returncode:
            report(fail, "engine", check.stderr.strip().splitlines()[-1]
                   if check.stderr.strip() else f"device check exited {check.returncode}")
            failed = True
        else:
            report(ok, "engine", "device check passed")

    # The venv the server and installer run under.
    if paths.PYTHON.is_file():
        report(ok, "python", str(paths.PYTHON))
    else:
        report(fail, "python", f"missing {paths.PYTHON}; run make install-environment")
        failed = True

    # Unified memory against the 36 GB the 4-bit examples need.
    memory = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
    )
    try:
        gb = int(memory.stdout.strip()) / 1024**3
        mark = ok if gb >= 36 else warn
        report(mark, "memory", f"{gb:.0f} GB unified" + ("" if gb >= 36 else "; the 4-bit examples need 36 GB"))
    except ValueError:
        report(info, "memory", "could not read hw.memsize")

    # Room on the volume holding the models directory.
    usage = subprocess.run(
        ["df", "-g", str(paths.MODELS if paths.MODELS.exists() else paths.DATA)],
        capture_output=True,
        text=True,
    )
    try:
        free_gb = int(usage.stdout.splitlines()[-1].split()[3])
        mark = ok if free_gb >= 36 else warn
        report(mark, "disk", f"{free_gb} GB free" + ("" if free_gb >= 36 else "; the 4-bit examples need ~36 GB"))
    except (IndexError, ValueError):
        report(info, "disk", "could not read free space")

    # The model catalog completion and `models` read.
    stale = catalog.is_stale()
    report(
        info if stale else ok,
        "catalog",
        "cache is stale or missing; refreshes on next serve" if stale else "cache is fresh",
    )

    # Whatever the selected port serves.
    served = _request_json("/v1/models", timeout=1)
    entries = served.get("data", []) if isinstance(served, dict) else []
    if entries and isinstance(entries[0], dict) and entries[0].get("owned_by") == "richengine":
        report(ok, "server", f"{entries[0].get('id')} at {_base_url(PORT)}")
    else:
        report(info, "server", f"none at {_base_url(PORT)}")
    return 1 if failed else 0


# The splash store keys are user-facing — MODEL_STORES keys are what
# `disk wipe` accepts — and predate the RichEngine rename; the values must
# not change.
STORE_SPLASH = "splash"
STORE_SPLASH_CACHE = "splash-cache"
# Both splash keys together: the "is serving" lock and disk.py's version of
# it guard them as one.
SPLASH_STORES = (STORE_SPLASH, STORE_SPLASH_CACHE)

# The directories other local model tools keep their weights in; each name is
# what `disk` lists and `disk wipe` accepts. Every store lists candidates —
# tools have moved these between versions.
# Each store: display name, candidate directories, and how deep its model
# entries nest (2 for author/repo layouts like LM Studio's and our own).
MODEL_STORES = {
    STORE_SPLASH: (
        "Splash/RichEngine",
        lambda: [paths.MODELS],
        2,
    ),
    STORE_SPLASH_CACHE: (
        "RichEngine caches",
        lambda: [
            # Compiled weights and the KV prefix cache; both rebuild on demand.
            Path.home() / "Library/Caches/RichEngine",
        ],
        1,
    ),
    "omlx": (
        "oMLX",
        lambda: [
            # oMLX self-installs under ~/.omlx (binary, cluster state, weight
            # caches), keeps config/logs under Application Support, and an
            # ANE cache under Library/Caches.
            Path.home() / ".omlx/cache",
            Path.home() / "Library/Application Support/omlx",
            Path.home() / "Library/Caches/omlx",
        ],
        1,
    ),
    "ollama": (
        "Ollama",
        lambda: [
            Path.home() / ".ollama/models",
        ],
        1,
    ),
    "lm-studio": (
        "LM Studio",
        lambda: [
            Path.home() / ".lmstudio/models",
            Path.home() / ".cache/lm-studio/models",
        ],
        2,
    ),
    "huggingface": (
        "Hugging Face cache",
        lambda: [
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "hub",
            # Xet's deduplicating blob store sits beside hub/.
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            / "xet",
        ],
        1,
    ),
    "rapidmlx": (
        "RapidMLX",
        lambda: [
            Path.home() / ".rapid-mlx",
            Path.home() / ".rapidmlx/models",
            Path.home() / ".cache/rapidmlx",
        ],
        1,
    ),
    "basert": (
        "BaseRT",
        lambda: [
            # The weights cache; ~/.basert holds the binaries, not models.
            Path.home() / "Library/Caches/baseRT/models",
            Path.home() / ".basert/models",
            Path.home() / "Library/Application Support/BaseRT/models",
        ],
        1,
    ),
    "coreai": (
        "CoreAI",
        lambda: [
            Path.home() / "Library/Caches/coreai-cache",
            Path.home() / "Library/Application Support/CoreAIKit",
        ],
        1,
    ),
    "unsloth": (
        "Unsloth",
        lambda: [
            Path.home() / ".unsloth",
            Path.home() / ".local/share/unsloth",
        ],
        1,
    ),
    "mlx-serve": (
        "mlx-serve",
        lambda: [Path.home() / ".mlx-serve"],
        1,
    ),
}


def _dir_bytes(path):
    """The bytes path's tree holds, following nothing: a symlink stays the
    size of the link so shared blobs are not counted twice."""
    total = 0
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            try:
                total += (Path(root) / name).lstat().st_size
            except OSError:
                continue
    return total


def _store_rows(calc=True):
    """Every store with its existing directories and the model entries they
    hold. Sizes are counted only with calc — walking a 200 GB cache is slow."""
    rows = []
    for key, (title, candidates, depth) in MODEL_STORES.items():
        existing = [path for path in candidates() if path.exists()]
        bytes_ = 0
        entries = []
        for path in existing:
            if calc:
                bytes_ += _dir_bytes(path)
            try:
                children = [
                    child
                    for child in path.iterdir()
                    if not child.name.startswith(".")
                ]
            except OSError:
                continue
            # Depth-2 stores hold author/<model>: descend into directories
            # that themselves hold directories; anything else lists as is.
            if depth == 2:
                expanded = []
                for child in children:
                    if child.is_dir() and not child.is_symlink():
                        try:
                            grandchildren = [
                                grandchild
                                for grandchild in child.iterdir()
                                if grandchild.is_dir()
                                and not grandchild.name.startswith(".")
                            ]
                        except OSError:
                            grandchildren = []
                        if grandchildren:
                            expanded += [
                                child / grandchild.name
                                for grandchild in grandchildren
                            ]
                            continue
                    expanded.append(child)
                children = expanded
            for child in children:
                if key == "huggingface":
                    # The Hub cache names directories models--OWNER--REPO and
                    # shares blobs/, snapshots internals are not models.
                    if not child.name.startswith("models--"):
                        continue
                    name = child.name[8:].replace("--", "/")
                elif depth == 2 and child.parent != path:
                    name = f"{child.parent.name}/{child.name}"
                else:
                    name = child.name
                entries.append(
                    {
                        "name": name,
                        "bytes": _dir_bytes(child) if calc else None,
                    }
                )
        # Largest first where sizes were counted, else by name.
        entries.sort(
            key=lambda e: (-(e["bytes"] or 0), e["name"]) if calc else e["name"]
        )
        rows.append(
            {
                "name": key,
                "title": title,
                "paths": [str(path) for path in existing],
                "bytes": bytes_ if calc else None,
                "models": entries,
                "exists": bool(existing),
            }
        )
    return rows


def disk(args):
    if args.disk_command == "wipe":
        return _disk_wipe(args)
    if args.disk_command == "dedupe":
        return dedupe(args)
    rows = _store_rows(calc=args.calc)
    if args.calc:
        rows.sort(key=lambda row: -(row["bytes"] or 0))
    if args.json:
        print(json.dumps({"stores": rows}, indent=2))
        return 0
    width = max(len(row["title"]) for row in rows)
    for row in rows:
        if not row["exists"]:
            print(
                f"{row['title']:<{width}}  {_styled('—', '2')}"
            )
            continue
        where = ", ".join(row["paths"])
        size = (
            _styled(_size(row["bytes"]), "36")
            if args.calc
            else _styled("(skipped)", "2")
        )
        print(
            f"{row['title']:<{width}}  {size}  {_styled(where, '2')}"
        )
        for entry in row["models"][:10]:
            if entry["bytes"] is None:
                print(f"{'':<{width}}    {_styled(entry['name'], '2')}")
            else:
                print(
                    f"{'':<{width}}    {entry['name']}  "
                    f"{_styled(_size(entry['bytes']), '2')}"
                )
        if len(row["models"]) > 10:
            more = f"… {len(row['models']) - 10} more"
            print(f"{'':<{width}}    {_styled(more, '2')}")
    print(
        f"\nWipe a store with: richengine disk wipe "
        f"<{'|'.join(MODEL_STORES)}> --yes"
    )
    return 0


# Stores a model may be served or linked from: dedupe never removes from
# them, so splash assemblies (symlink trees into the Hub cache) stay valid.
DEDUPE_KEEPS = (STORE_SPLASH, "huggingface")
# Stores whose entries are not model trees at all.
DEDUPE_SKIPS = (STORE_SPLASH_CACHE,)
# Wipe order when a model is absent from the keeps: earlier wins.
DEDUPE_PRIORITY = (
    "lm-studio",
    "omlx",
    "basert",
    "coreai",
    "ollama",
    "unsloth",
    "mlx-serve",
    "rapidmlx",
)


# Suffixes that mark an encoding or quantization of the same base model —
# never an architecture part (DFlash drafts, MTP heads keep their names).
_VARIANT_SUFFIX = re.compile(
    r"[-_](?:gguf|mlx|exl\d?|awq|gptq|bnb|safetensors?|fp16|bf16|fp8(?:e4m3)?|"
    r"mxfp4|nvfp4|\d+bit|[oiq]q\d\S*|qat\S*|ud-\S*|iq\d\S*)$",
    re.IGNORECASE,
)


def _dedupe_key(store, name, variants=False):
    """The normalized owner/repo an entry stands for. Splash links may carry
    a :VARIANT; bare LM Studio dirs have no owner and match weakly. With
    variants, format suffixes strip to the base model name."""
    if store == STORE_SPLASH:
        name = name.split(":", 1)[0]
    owner, _, repo = name.rpartition("/")
    if variants:
        while repo and (stripped := _VARIANT_SUFFIX.sub("", repo)) != repo:
            repo = stripped
    return f"{owner}/{repo}".lower() if owner else repo.lower()


def _dedupe_groups(variants=False):
    """Model occurrences shared across stores: [{model, keep, remove, weak}].
    keep holds entries in splash/huggingface (or the priority winner);
    remove holds the rest. weak marks a basename-only match."""
    rows = {
        row["name"]: row
        for row in _store_rows(calc=True)
        if row["exists"] and row["name"] not in DEDUPE_SKIPS
    }
    occurrences = {}
    weak_pending = []
    for store in MODEL_STORES:
        row = rows.get(store)
        if row is None:
            continue
        for entry in row["models"]:
            key = _dedupe_key(store, entry["name"], variants)
            occurrence = {
                "store": store,
                "title": row["title"],
                "name": entry["name"],
                "bytes": entry["bytes"] or 0,
            }
            if "/" in key:
                occurrences.setdefault(key, []).append(occurrence)
            else:
                weak_pending.append(occurrence)
    # A bare name joins the owned group whose repo tail it spells; generic
    # directory names (a "Models" folder) prove nothing and stay out.
    for occurrence in weak_pending:
        key = _dedupe_key("", occurrence["name"], variants)
        if key in ("models", "model", "cache", "blobs", "snapshots"):
            continue
        match = next(
            (k for k in occurrences if k.rsplit("/", 1)[-1] == key), None
        )
        if match is None:
            occurrences.setdefault(key, []).append(occurrence)
        else:
            occurrence["weak"] = True
            occurrences[match].append(occurrence)
    if variants:
        # Group keys are base names now; a group whose entries spell the same
        # full name is an exact duplicate, anything else is a format variant.
        variant_groups = {
            key
            for key, found in occurrences.items()
            if len({f["name"].lower().split(":")[0] for f in found}) > 1
        }
    else:
        variant_groups = set()
    # Coverage means real weights: splash links are tiny by measure (symlink
    # trees) but ride on the Hub cache, so a Hub directory of ≥100 MiB covers
    # the model. Keep-store entries are never removed; stubs only list.
    REAL_BYTES = 100 * 1024 * 1024
    groups = []
    for model, found in occurrences.items():
        if len(found) < 2:
            continue
        kept_stores = [f for f in found if f["store"] in DEDUPE_KEEPS]
        covered = any(
            f["store"] == "huggingface" and f["bytes"] >= REAL_BYTES
            for f in found
        )
        others = [f for f in found if f["store"] not in DEDUPE_KEEPS]
        if covered:
            keep, remove = kept_stores, others
        elif others:
            # Nothing canonical holds real weights: keep the priority winner.
            ordered = sorted(
                others,
                key=lambda f: (
                    DEDUPE_PRIORITY.index(f["store"])
                    if f["store"] in DEDUPE_PRIORITY
                    else len(DEDUPE_PRIORITY),
                    -(f["bytes"]),
                ),
            )
            keep, remove = kept_stores + ordered[:1], ordered[1:]
        else:
            continue
        if not remove:
            continue
        groups.append(
            {
                "model": model,
                "keep": keep,
                "remove": remove,
                "reclaimable": sum(f["bytes"] for f in remove),
                "variant": model in variant_groups,
            }
        )
    return groups


def _dedupe_remove_path(store, occurrence):
    row = next(r for r in _store_rows(calc=False) if r["name"] == store)
    return _model_paths(store, row, occurrence["name"])


def dedupe(args):
    groups = _dedupe_groups(variants=args.variants)
    total = sum(group["reclaimable"] for group in groups)
    if args.json:
        print(json.dumps({"groups": groups, "reclaimable": total}, indent=2))
        if not args.apply:
            return 0
    elif not groups:
        print("No duplicate models across stores.")
        return 0
    if not args.apply:
        if not args.json:
            print(
                f"{_styled('Duplicates', '1')} · {len(groups)} models · "
                f"{_styled(_size(total), '36')} reclaimable"
            )
            for group in groups:
                variant = _styled(" · formats", "35") if group["variant"] else ""
                print(f"\n  {_styled(group['model'], '1')}{variant}")
                for entry in group["keep"]:
                    weak = _styled(" (unverified)", "33") if entry.get("weak") else ""
                    print(
                        f"    {_styled('keep  ', '32')} {entry['title']}  "
                        f"{entry['name']}  {_styled(_size(entry['bytes']), '2')}{weak}"
                    )
                for entry in group["remove"]:
                    weak = _styled(" (unverified — kept)", "33") if entry.get("weak") else ""
                    print(
                        f"    {_styled('remove' if not entry.get('weak') else 'skip  ', '31' if not entry.get('weak') else '33')} "
                        f"{entry['title']}  "
                        f"{entry['name']}  {_styled(_size(entry['bytes']), '2')}{weak}"
                    )
            print("\nApply with: richengine disk dedupe --apply --yes")
        return 0
    if not args.yes:
        raise LauncherError(
            f"refusing to remove {_size(total)} across {len(groups)} models without confirmation",
            hint="run 'richengine disk dedupe --apply --yes'",
        )
    removed, freed = [], 0
    for group in groups:
        for entry in group["remove"]:
            # A basename-only match is listed but never removed.
            if entry.get("weak"):
                continue
            for path in _dedupe_remove_path(entry["store"], entry):
                freed += _dir_bytes(path)
                try:
                    if path.is_symlink() or path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path)
                    removed.append(str(path))
                except OSError as error:
                    raise LauncherError(f"could not wipe {path}: {error}") from None
    if args.json:
        print(json.dumps({"removed": removed, "freed": freed}, indent=2))
    else:
        print(f"Removed {len(removed)} duplicates: {_size(freed)} freed.")
    return 0


def _model_paths(store, row, model):
    """The filesystem paths `model` names inside `store`, each a direct
    member of one of the store's own directories — never a path that escapes
    it."""
    paths_ = []
    for root in row["paths"]:
        root = Path(root)
        if store == "huggingface":
            # owner/repo as the Hub cache spells it: models--OWNER--REPO.
            if "/" not in model:
                continue
            candidate = root / ("models--" + model.replace("/", "--"))
        else:
            # Model names must stay inside the store: one path component, or
            # owner/repo where the store nests two deep.
            depth = MODEL_STORES[store][2]
            components = model.split("/")
            if (
                model.startswith("/")
                or ".." in components
                or any(not c or c.startswith(".") for c in components)
                or len(components) > (2 if depth == 2 else 1)
            ):
                continue
            candidate = root / model
        try:
            if candidate.name.startswith(".") or not candidate.exists():
                continue
        except OSError:
            continue
        paths_.append(candidate)
    return paths_


def _splash_assembly(link):
    """The .resolved assembly a splash selection link serves, where no other
    selection still points: wiped with the link. Shared targets stay."""
    try:
        target = link.resolve(strict=True)
    except OSError:
        return None
    if target.parent != paths.MODELS / ".resolved":
        return None
    for other in model_artifacts.selection_links(paths.MODELS):
        if other != link:
            try:
                if other.resolve() == target:
                    return None
            except OSError:
                continue
    return target


def _disk_wipe(args):
    name = args.store
    row = next(row for row in _store_rows() if row["name"] == name)
    if not row["exists"]:
        if args.json:
            print(json.dumps({"wiped": name, "paths": [], "bytes": 0}))
        else:
            print(f"{row['title']}: nothing to wipe.")
        return 0
    owner = _serve_owner(args.port)
    if name in SPLASH_STORES and owner is not None and _pid_running(owner["pid"]):
        raise LauncherError(
            "RichEngine is serving from the splash stores",
            hint="stop the server (Ctrl+C) before wiping it",
        )
    if args.model is not None:
        return _disk_wipe_model(args, row)
    if not args.yes:
        listing = ", ".join(row["paths"])
        raise LauncherError(
            f"refusing to wipe {row['title']} ({listing}, {_size(row['bytes'])}) without confirmation",
            hint=f"run 'richengine disk wipe {name} --yes'",
        )
    wiped = []
    for path in row["paths"]:
        # A store may be one symlinked directory: replace it with the real
        # tree so the link's target is not wiped instead.
        real = Path(path).resolve()
        try:
            shutil.rmtree(real)
            wiped.append(str(real))
        except OSError as error:
            raise LauncherError(f"could not wipe {real}: {error}") from None
    if args.json:
        print(json.dumps({"wiped": name, "paths": wiped, "bytes": row["bytes"]}))
    else:
        print(
            f"Wiped {row['title']}: {_size(row['bytes'])} across "
            f"{len(wiped)} director{'y' if len(wiped) == 1 else 'ies'}."
        )
    return 0


def _disk_wipe_model(args, row):
    targets = _model_paths(args.store, row, args.model)
    if not targets:
        raise LauncherError(
            f"no model {args.model!r} in {row['title']}",
            hint="models there: "
            + ", ".join(entry["name"] for entry in row["models"][:8]),
        )
    if not args.yes:
        raise LauncherError(
            f"refusing to wipe {args.model} ({', '.join(str(t) for t in targets)}) without confirmation",
            hint=f"run 'richengine disk wipe {args.store} {args.model} --yes'",
        )
    wiped, bytes_ = [], 0
    for path in targets:
        bytes_ += _dir_bytes(path)
        # A splash link's assembly must resolve before the link goes away.
        assembly = _splash_assembly(path) if args.store == STORE_SPLASH else None
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            wiped.append(str(path))
        except OSError as error:
            raise LauncherError(f"could not wipe {path}: {error}") from None
        if assembly is not None:
            try:
                bytes_ += _dir_bytes(assembly)
                shutil.rmtree(assembly)
                wiped.append(str(assembly))
            except OSError as error:
                raise LauncherError(f"could not wipe {assembly}: {error}") from None
    if args.json:
        print(
            json.dumps(
                {"wiped": args.store, "model": args.model, "paths": wiped, "bytes": bytes_},
                indent=2,
            )
        )
    else:
        print(f"Wiped {args.model} from {row['title']}: {_size(bytes_)}.")
    return 0


def _parse_port(value):
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "port must be an integer from 1 to 65535"
        ) from None
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _version():
    if not paths.PACKAGED:
        return "RichEngine (source checkout)"
    return "RichEngine " + str(
        json.loads((paths.ROOT / "release.json").read_text())["version"]
    )


def _print_help():
    """The top-level help: grouped commands, with color only on a terminal."""
    bold = lambda text: _styled(text, "1")
    command = lambda name: _styled(f"{name:<11}", "36")
    print(
        f"{bold('RichEngine')} — serve a local model, connect an installed agent\n"
        f"\n{_styled('Usage:', '2')} richengine {_styled('<command>', '36')} [options]\n"
        f"\n{bold('Server')}\n"
        f"  {command('serve')}run the local server; Ctrl+C stops it\n"
        f"  {command('models')}list installed and known models\n"
        f"  {command('status')}report the running server\n"
        f"  {command('flags')}list every option of every command\n"
        f"  {command('doctor')}check the engine, environment and server\n"
        f"  {command('disk')}show or wipe every app's model storage\n"
        f"\n{bold('Agents')}\n"
        + "".join(
            f"  {command(name)}connect {name} to the running server\n"
            for name in clients.INSTALL_URLS
        )
        + f"\n{bold('Global')}\n"
        f"  {command('--version')}print the version\n"
        f"  {command('-h, --help')}this help\n"
        f"\n{bold('Quick start')}\n"
        "  richengine serve --model mlx-community/Qwen3.8-27B-4bit\n"
        "  richengine opencode    # in another terminal, after Ready\n"
        f"\n{bold('Environment')}\n"
        "  RICHENGINE_PORT     default port for serve, status and clients (8000)\n"
        "  RICHENGINE_API_KEY  bearer token sent to the server, when set"
    )


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="richengine",
        description=__doc__,
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
        epilog=(
            "Quick start:\n"
            "  richengine serve --model mlx-community/Qwen3.8-27B-4bit\n"
            "  richengine opencode  # in another terminal, after Ready\n\n"
            "Use richengine serve --help for server settings. Client arguments,\n"
            "including --help, are passed through to the installed agent.\n\n"
            "Environment:\n"
            "  RICHENGINE_PORT      the default port for serve, status and clients (8000)\n"
            "  RICHENGINE_API_KEY   the bearer token clients and status send, when set"
        ),
    )
    parser.add_argument("--version", action="version", version=_version())
    commands = parser.add_subparsers(
        dest="command", required=True, metavar="command"
    )
    server = commands.add_parser(
        "serve",
        help="run the local server; Ctrl+C stops it",
        description="Load an upstream model, automatically select its DFlash2 draft, and serve in the foreground.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
        epilog=(
            "Examples:\n"
            "  richengine serve --model mlx-community/Qwen3.8-27B-4bit\n"
            "  richengine serve --model unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M --max-context 128K\n\n"
            "After Ready, open http://127.0.0.1:8000 or connect an installed agent.\n"
            "The startup summary and /status report the effective context limit.\n"
            "A client may impose a smaller limit. Keep this terminal open; Ctrl+C stops serving."
        ),
    )
    server.add_argument(
        "--port",
        type=_parse_port,
        default=os.environ.get("RICHENGINE_PORT", str(PORT)),
        help="HTTP port (default: RICHENGINE_PORT or 8000)",
    )
    server.add_argument(
        "--model",
        type=model_artifacts.parse_model_id,
        metavar="OWNER/REPO[:VARIANT]",
        help="upstream Hugging Face model, with a GGUF variant after ':' (e.g. :UD-Q4_K_M); "
        "omit to start unloaded and load one later over the API",
    )
    server.add_argument(
        "--revision",
        help="optional model branch, tag or commit (default: repository default)",
    )
    server.add_argument(
        "--draft-model",
        type=model_artifacts.parse_draft_model,
        help="override the automatically selected DFlash2 repository or local directory",
    )
    server.add_argument(
        "--language-only",
        action="store_true",
        help="skip vision preparation and loading",
    )
    server.add_argument(
        "--offline",
        action="store_true",
        help="start the installed model without contacting the Hugging Face Hub (as HF_HUB_OFFLINE=1)",
    )
    server.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved selection and exit without downloading or serving",
    )
    serve_options.add_serve_arguments(server)
    models_command = commands.add_parser(
        "models",
        help="list installed and known models",
        description="List the models installed locally and every model ID the catalog and the suggested list name.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    models_command.add_argument(
        "--json", action="store_true", help="print the lists as JSON"
    )
    models_command.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="add family, format, vision and installed size columns",
    )
    tune_command = commands.add_parser(
        "tune",
        help="measure and keep this chip's fastest engine knobs for a model",
        description=(
            "Sweep the model's tunable engine paths — the subset its family "
            "offers that this chip and memory bandwidth can run — measuring "
            "decode throughput single-lane and at the full batch width, and "
            "write the winners to the model's tuning.json, applied on every "
            "serve.\n\n"
            "Quick is the default: quality-trading knobs stay out and the "
            "interaction recheck pass is skipped. --complete runs both. "
            "--include-quality-knobs and --interaction-pass enable each "
            "half on its own."
        ),
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    tune_command.add_argument(
        "--model",
        type=model_artifacts.parse_model_id,
        metavar="OWNER/REPO[:VARIANT]",
        help="the installed model to tune",
    )
    tune_command.add_argument(
        "--selection",
        metavar="NAME",
        help="an installed selection link named relative to the models "
        "directory (e.g. .selections/<hash>), for installs whose options "
        "the model ID alone does not reproduce; skips --model resolution",
    )
    tune_command.add_argument(
        "--models",
        metavar="DIR",
        help="models directory holding the install (default: the install root's)",
    )
    tune_command.add_argument(
        "--revision",
        help="optional model branch, tag or commit (default: repository default)",
    )
    tune_command.add_argument(
        "--draft-model",
        type=model_artifacts.parse_draft_model,
        help="the draft override the installation was made with, if any",
    )
    tune_command.add_argument(
        "--language-only",
        action="store_true",
        help="the language-only installation, if that is what was installed",
    )
    tune_command.add_argument(
        "--complete",
        action="store_true",
        help="the complete sweep: quality-trading knobs plus the "
        "interaction recheck pass (Quick, the default, runs neither)",
    )
    tune_command.add_argument(
        "--include-quality-knobs",
        action="store_true",
        help="also sweep knobs that trade output quality for speed "
        "(DiffusionGemma's step cap and exit thresholds); off by default "
        "and implied by --complete",
    )
    tune_command.add_argument(
        "--interaction-pass",
        action="store_true",
        help="run the recheck that re-measures losing knobs against the "
        "final env; implied by --complete",
    )
    tune_command.add_argument(
        "--no-interaction-pass",
        action="store_true",
        help="skip the recheck pass even under --complete",
    )
    status_command = commands.add_parser(
        "status",
        help="report the running server",
        description="Report whether a RichEngine server is ready, starting, or absent on the selected port. Exits 0 while serving, 1 otherwise.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    status_command.add_argument(
        "--port",
        type=_parse_port,
        default=os.environ.get("RICHENGINE_PORT", str(PORT)),
        help="HTTP port (default: RICHENGINE_PORT or 8000)",
    )
    status_command.add_argument(
        "--json", action="store_true", help="print the status as JSON"
    )
    commands.add_parser(
        "flags",
        help="list every option of every command",
        description="Every option of every command, generated from the parser itself.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    commands.add_parser(
        "doctor",
        help="check the engine, environment and server",
        description="Check the engine binary and device, the Python environment, memory, disk, the model catalog and the running server. Exits 1 on a hard failure.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    disk_command = commands.add_parser(
        "disk",
        help="show or wipe every app's model storage",
        description="Report the model stores of Splash/RichEngine, oMLX, LM Studio, the Hugging Face cache, RapidMLX and BaseRT; 'wipe' removes one.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    disk_command.add_argument(
        "--json", action="store_true", help="print the stores as JSON"
    )
    disk_command.add_argument(
        "--calc",
        action="store_true",
        help="count every store's bytes; the listing skips the walk without it",
    )
    disk_command.add_argument(
        "--port",
        type=_parse_port,
        default=os.environ.get("RICHENGINE_PORT", str(PORT)),
        help="HTTP port the splash-serving check probes (default: RICHENGINE_PORT or 8000)",
    )
    disk_commands = disk_command.add_subparsers(dest="disk_command")
    dedupe_command = disk_commands.add_parser(
        "dedupe",
        help="find models duplicated across stores",
        description="Find the same owner/repo in several stores, keep the Splash/Hugging Face copy, and list or remove the rest.",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    dedupe_command.add_argument(
        "--apply", action="store_true", help="remove the duplicates (needs --yes)"
    )
    dedupe_command.add_argument(
        "--json", action="store_true", help="print the groups as JSON"
    )
    dedupe_command.add_argument(
        "--variants",
        action="store_true",
        help="also match format variants (X-GGUF, X-MLX, X-4bit, X-oQ4e…) of one base model",
    )
    dedupe_command.add_argument(
        "--yes",
        action="store_true",
        help="confirm removal; required with --apply, there is no undo",
    )
    wipe = disk_commands.add_parser(
        "wipe",
        help="delete one store's model files",
        formatter_class=_HelpFormatter,
        **_PARSER_OPTIONS,
    )
    wipe.add_argument(
        "store",
        choices=list(MODEL_STORES),
        metavar="STORE",
        help="one of: " + ", ".join(MODEL_STORES),
    )
    wipe.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="one entry of the store (e.g. owner/repo); omit to wipe the whole store",
    )
    wipe.add_argument(
        "--yes",
        action="store_true",
        help="confirm the wipe; required, there is no undo",
    )
    for name in clients.INSTALL_URLS:
        commands.add_parser(
            name, help=f"connect {name} to the running server", **_PARSER_OPTIONS
        )
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["-h"] or argv == ["--help"]:
        # A bare `richengine` prints help rather than the required-command
        # error argparse raises.
        _print_help()
        raise SystemExit(0)
    client_args = []
    if argv and argv[0] in clients.INSTALL_URLS:
        argv, client_args = argv[:1], argv[1:]
        if client_args[:1] == ["--"]:
            client_args = client_args[1:]
    elif "--" in argv:
        boundary = argv.index("--")
        argv, client_args = argv[:boundary], argv[boundary + 1 :]
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve_options.check_serve_arguments(parser, args)
    if args.command in clients.INSTALL_URLS:
        try:
            args.port = _parse_port(os.environ.get("RICHENGINE_PORT", str(PORT)))
        except argparse.ArgumentTypeError as error:
            parser.error(f"RICHENGINE_PORT: {error}")
    if client_args and args.command not in clients.INSTALL_URLS:
        parser.error("arguments after -- are only supported for coding clients")
    args.client_args = client_args
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.command == "serve":
            return serve(args)
        if args.command == "models":
            return list_models(args)
        if args.command == "tune":
            return tune(args)
        if args.command == "status":
            return status(args)
        if args.command == "flags":
            return list_flags(args)
        if args.command == "doctor":
            return doctor(args)
        if args.command == "disk":
            return disk(args)
        return coding_client(args)
    except (LauncherError, clients.ClientError, OSError) as error:
        print(_styled("error:", "31;1", stream=sys.stderr), error, file=sys.stderr)
        if getattr(error, "hint", None):
            print(_styled("hint:", "36", stream=sys.stderr), error.hint, file=sys.stderr)
        return 1
    except StopSignal as stop:
        # The status a shell gives a program the signal ends: 130 for
        # SIGINT, 143 for SIGTERM.
        return 128 + stop.number
    except KeyboardInterrupt:
        return 128 + signal.SIGINT


if __name__ == "__main__":
    raise SystemExit(main())
