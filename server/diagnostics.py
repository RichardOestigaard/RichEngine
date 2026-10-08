"""Concise console diagnostics without request-body logging."""

import os
import re
import sys
import time
import traceback
from pathlib import Path

# The server package, whose innermost frame locates an unexpected error.
_PACKAGE = Path(__file__).resolve().parent


def log_unexpected(error):
    """Print the type of an unexpected error and the innermost line of the
    server it passed through, never its message, which may carry request
    data."""
    try:
        location = ""
        for frame, line in traceback.walk_tb(error.__traceback__):
            path = Path(frame.f_code.co_filename).resolve()
            if path.parent == _PACKAGE:
                location = f" · {_PACKAGE.name}/{path.name}:{line}"
        print_status(
            f"Error · internal_server_error · {type(error).__name__}{location}",
            error=True,
        )
    except Exception:
        pass


def _ansi(stream):
    """The launcher's gate: colors only on a terminal with NO_COLOR unset, so
    piped output and log files stay byte-plain."""
    return stream.isatty() and "NO_COLOR" not in os.environ


def _styled(text, *codes, stream=sys.stdout):
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if _ansi(stream) else text


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def accent(text):
    """The help palette's cyan: models, addresses and agent commands."""
    return _styled(text, "36", stream=sys.stdout)


def dim(text):
    """The help's 'Usage' gray: detail the operator can skip."""
    return _styled(text, "2", stream=sys.stdout)


# The status word's palette, install/launcher.py's: Ready in the serving
# green-bold, Done and a recovery in the ok green, warnings and the stopping
# state in yellow, errors in red, Loading in the help's cyan, and the
# follow-up hint and the chat-template label in its 'Usage' gray. The rest
# stays plain, as the launcher's info rows do.
_STATUS_COLORS = {
    "Ready": "32;1",
    "Done": "32",
    "Engine restarted": "32",
    "Cancelled": "33",
    "Warning": "33",
    "Refused": "33",
    "Stopping": "33",
    "Loading": "36",
    "Next": "2",
    "Chat template": "2",
    "Error": "31",
    "Engine failed": "31",
    "Engine restart failed": "31",
    "Engine stopped": "31",
    "Template error": "31",
}


def print_status(message, *, error=False):
    # One write per line, newline included, so that lines written at once by
    # request threads, or by the native runtime on the shared stderr, stay
    # whole in a terminal or in one log file.
    stream = sys.stderr if error else sys.stdout
    stamp = time.strftime("%H:%M:%S")
    if _ansi(stream):
        word, separator, rest = message.partition(" · ")
        if word in _STATUS_COLORS:
            message = (
                _styled(word, _STATUS_COLORS[word], stream=stream)
                + separator
                + rest
            )
        stamp = _styled(stamp, "2", stream=stream)
    else:
        # accent()/dim() a call site embedded still leave the log byte-plain.
        message = _ANSI.sub("", message)
    stream.write(f"{stamp} {message}\n")
    stream.flush()


def print_request(record):
    outcome = record["outcome"]
    if outcome == "error":
        print_status(f"Error · {record.get('error_code', 'runtime_error')}", error=True)
        return
    metrics = record.get("metrics", {})
    latency = metrics.get("request_latency", {})
    parts = [
        "Cancelled" if outcome == "cancelled" else "Done",
        f"{dim('input')} {record['prompt_tokens']:,}",
        f"{dim('cached')} {metrics.get('cache', {}).get('matched_tokens', 0):,}",
        f"{dim('output')} {record.get('completion_tokens', 0):,}",
    ]
    tools = record.get("tools")
    if isinstance(tools, dict) and tools.get("count"):
        signature = dim(tools["signature"]) if tools.get("signature") else ""
        parts.append(f"{dim('tools')} {tools['count']}·{signature}")
    ttft = latency.get("ttft_ms")
    speed = latency.get("stream_tokens_per_second")
    if ttft is not None:
        parts.append(f"{dim('TTFT')} {ttft / 1000:.1f}s")
    if speed is not None:
        parts.append(accent(f"{speed:.1f} tok/s"))
    print_status(" · ".join(parts))
