"""The Mac's thermal state, reported in /status and logged on each change.

NSProcessInfo.thermalState is read over the Objective-C runtime so the
server needs no platform Python extension. A daemon thread polls it;
slowing down under heat is otherwise invisible until requests stall.
"""

import ctypes
import threading
import time

from .diagnostics import accent, print_status

POLL_SECONDS = 2.0
# NSProcessInfoThermalState's four levels.
STATES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}

_monitor = None
_monitor_lock = threading.Lock()


def _read_state():
    """The raw NSProcessInfo.thermalState, or None when it cannot be read."""
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_long
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        process_info = objc.objc_msgSend(
            objc.objc_getClass(b"NSProcessInfo"),
            objc.sel_registerName(b"processInfo"),
        )
        if not process_info:
            return None
        return objc.objc_msgSend(
            process_info, objc.sel_registerName(b"thermalState")
        )
    except (AttributeError, OSError):
        return None


class _Monitor:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = _read_state()
        self.thread = threading.Thread(
            target=self._run, name="thermal state", daemon=True
        )
        self.thread.start()

    def _run(self):
        while True:
            state = _read_state()
            with self.lock:
                changed = state != self.state
                self.state = state
            if changed:
                name = STATES.get(state, "unknown" if state is None else f"unknown ({state})")
                print_status(f"Thermal · state {accent(name)}")
            time.sleep(POLL_SECONDS)

    def current(self):
        with self.lock:
            return self.state


def start():
    """Start reporting; idempotent."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = _Monitor()
        return _monitor


def state():
    """The thermal state name for /status, or None when unavailable."""
    monitor = _monitor
    if monitor is None:
        return None
    current = monitor.current()
    if current is None:
        return None
    return STATES.get(current, f"unknown ({current})")
