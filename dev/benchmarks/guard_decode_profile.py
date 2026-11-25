#!/usr/bin/env python3
"""Guarded launcher for the decode-profile benchmark.

Fails closed: it refuses to launch, and kills the child it launched, whenever
the host cannot prove it has the memory headroom the run is allowed. It is a
seatbelt for a benchmark, not a guarantee against kernel panics — it never
touches processes it did not launch.

  guard_decode_profile.py --memory-cap-mib MIB -- COMMAND [ARGS...]
"""

import argparse
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

MIB = 1024 * 1024
HOST_RESERVE_MIB = 6 * 1024
HOST_RESERVE = HOST_RESERVE_MIB * MIB
POLL_SECONDS = 0.5
TIMEOUT_SECONDS = 120.0
KILL_GRACE_SECONDS = 2.0
REFUSING = {"backend-benchmark", "decode-profile"}


def fail(message):
    print(f"guard-decode-profile: {message}", file=sys.stderr)
    return 1


def tool(name):
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"{name} is unavailable")
    return path


def sysctl_int(sysctl, key):
    out = subprocess.run(
        [sysctl, "-n", key], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"sysctl {key} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def pressure_normal(sysctl):
    return sysctl_int(sysctl, "kern.memorystatus_vm_pressure_level") == "1"


def host_available_bytes(sysctl, vm_stat):
    page_size = int(sysctl_int(sysctl, "hw.pagesize"))
    if page_size <= 0:
        raise RuntimeError("vm_stat page size must be positive")
    out = subprocess.run(
        [vm_stat], capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError(f"vm_stat failed: {out.stderr.strip()}")
    pages = {}
    for match in re.finditer(r"^(.+?):\s+(\d+)\.", out.stdout, re.MULTILINE):
        pages[match.group(1).strip()] = int(match.group(2))
    try:
        # vm_stat's "Pages free" already excludes speculative pages.
        # Count file-backed cache and purgeable pages as reclaimable.
        page_count = (
            pages["Pages free"]
            + pages["File-backed pages"]
            + pages["Pages purgeable"]
        )
    except KeyError as error:
        raise RuntimeError(f"vm_stat is missing {error}") from error
    return page_size * page_count


def conflicting_process(pgrep):
    for name in REFUSING:
        result = subprocess.run(
            [pgrep, "-x", name], capture_output=True, timeout=10)
        if result.returncode == 0:
            return name
        if result.returncode != 1:
            raise RuntimeError(f"pgrep -x {name} failed")
    return None


def terminate_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + KILL_GRACE_SECONDS
    while time.monotonic() < deadline:
        if child.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def main(argv):
    parser = argparse.ArgumentParser(
        description="launch decode-profile only with proven memory headroom")
    parser.add_argument("--memory-cap-mib", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if not args.memory_cap_mib or args.memory_cap_mib <= 0:
        return fail("--memory-cap-mib must be a positive MiB count")
    cap_bytes = args.memory_cap_mib * MIB

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        return fail("no command after --")
    if "--memory-cap-mib" in command:
        return fail(
            "the guarded command already carries --memory-cap-mib; "
            "the launcher sets the authoritative cap")
    command = command + ["--memory-cap-mib", str(args.memory_cap_mib)]

    # One guarded run per user at a time; a held lock means another run is
    # already spending this host's headroom.
    lock_path = Path(tempfile.gettempdir()) / (
        f"richengine-decode-profile-{os.getuid()}.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return fail(f"another guarded decode-profile holds {lock_path}")

    try:
        sysctl = tool("sysctl")
        vm_stat = tool("vm_stat")
        pgrep = tool("pgrep")
        available = host_available_bytes(sysctl, vm_stat)
        normal = pressure_normal(sysctl)
        conflict = conflicting_process(pgrep)
    except (OSError, RuntimeError, subprocess.SubprocessError,
            ValueError) as error:
        return fail(f"telemetry unavailable, refusing: {error}")

    if not normal:
        return fail("system memory pressure is not normal, refusing")
    if conflict:
        return fail(f"{conflict} is running, refusing")
    required = cap_bytes + HOST_RESERVE
    if available < required:
        return fail(
            f"estimated host memory {available / MIB:.0f} MiB is below "
            f"{required / MIB:.0f} MiB (cap + {HOST_RESERVE_MIB} MiB reserve), refusing")

    try:
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as error:
        return fail(f"could not launch benchmark: {error}")
    started = time.monotonic()
    status = None
    try:
        while child.poll() is None:
            if time.monotonic() - started > TIMEOUT_SECONDS:
                status = f"exceeded {TIMEOUT_SECONDS:.0f}s, terminating"
                break
            if not pressure_normal(sysctl):
                status = "memory pressure rose, terminating"
                break
            if host_available_bytes(sysctl, vm_stat) < HOST_RESERVE:
                status = f"host headroom fell under {HOST_RESERVE_MIB} MiB, terminating"
                break
            time.sleep(POLL_SECONDS)
    except (OSError, RuntimeError, subprocess.SubprocessError,
            ValueError) as error:
        status = f"telemetry lost ({error}), terminating"
    except KeyboardInterrupt:
        status = "interrupted, terminating"
    finally:
        if child.poll() is None:
            terminate_group(child)

    if status is not None:
        return fail(status)
    return child.wait()


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    sys.exit(main(sys.argv[1:]))
