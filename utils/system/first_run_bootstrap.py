"""
First-run bootstrap for installs where install.py never ran.

ComfyUI Desktop and similar managed setups clone custom nodes without executing
install.py, leaving engine dependencies missing on every startup. When the
background dependency check reports missing engines and no installer state file
exists (.cache/install_state.json is only written after a successful run), this
module runs install.py once in a detached subprocess with a hard timeout, logs
output to .cache/bootstrap.log, and records the in-flight process in
.cache/bootstrap_active.json so concurrent or crashed sessions stay safe.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from utils.system.dependency_checker import DependencyChecker

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".cache"
_INSTALL_STATE_PATH = _CACHE_DIR / "install_state.json"
_ACTIVE_MARKER_PATH = _CACHE_DIR / "bootstrap_active.json"
_LOG_PATH = _CACHE_DIR / "bootstrap.log"
_RESULT_PREFIX = "BOOTSTRAP_RESULT: "

_DEFAULT_TIMEOUT_SECONDS = 1800


def _log(message: str) -> None:
    print(f"🔧 [First-Run Bootstrap] {message}")


def _timeout_seconds() -> int:
    raw = os.environ.get("TTS_AUDIO_SUITE_BOOTSTRAP_TIMEOUT", "")
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return _DEFAULT_TIMEOUT_SECONDS


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) terminates processes on Windows - use the API instead.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.WaitForSingleObject.restype = ctypes.c_uint32
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

        SYNCHRONIZE = 0x00100000
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            return False
        try:
            # A process handle is signalled (WAIT_OBJECT_0 == 0) once the process
            # exits; anything else means it is still running.
            already_exited = kernel32.WaitForSingleObject(handle, 0) == 0
        finally:
            kernel32.CloseHandle(handle)
        return not already_exited
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_active_marker():
    """Return the marker dict for a live repair, or None (stale markers are removed)."""
    try:
        marker = json.loads(_ACTIVE_MARKER_PATH.read_text(encoding="utf-8"))
        pid = int(marker.get("pid", 0))
    except (OSError, ValueError):
        return None
    if not _pid_alive(pid):
        # Server was killed mid-repair; the child died with it.
        _ACTIVE_MARKER_PATH.unlink(missing_ok=True)
        return None
    return marker


def echo_last_outcome() -> None:
    """Print how the previous automatic repair ended when the env is still unrepaired."""
    if _INSTALL_STATE_PATH.exists() or not _LOG_PATH.exists():
        return
    try:
        lines = _LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in reversed(lines):
        if line.startswith(_RESULT_PREFIX):
            _log(f"Previous automatic repair ended with: {line[len(_RESULT_PREFIX):].strip()}")
            break


def _kill_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True)
    else:
        import signal

        os.killpg(os.getpgid(pid), signal.SIGTERM)


def _watchdog(process: subprocess.Popen) -> None:
    timeout = _timeout_seconds()
    try:
        code = process.wait(timeout=timeout)
        outcome = "success" if code == 0 else f"failed (exit code {code})"
    except subprocess.TimeoutExpired:
        _kill_tree(process.pid)
        outcome = f"timed out after {timeout}s and was killed"

    _ACTIVE_MARKER_PATH.unlink(missing_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_RESULT_PREFIX}{outcome}\n")

    if outcome == "success":
        _log("Dependency repair finished successfully. Restart ComfyUI so every engine picks up the new packages.")
    else:
        _log(f"Automatic dependency repair did not complete ({outcome}). "
             "See .cache/bootstrap.log, or run install.py manually.")


def start_background_repair(engine_issues: dict) -> bool:
    """Run install.py once in a detached subprocess. Returns True when spawned."""
    if not engine_issues or _INSTALL_STATE_PATH.exists():
        return False

    missing_total = sum(len(missing) for missing in engine_issues.values())
    if os.environ.get("TTS_AUDIO_SUITE_AUTO_INSTALL", "1") == "0":
        _log(f"{missing_total} engine packages are missing and automatic repair is disabled "
             "(TTS_AUDIO_SUITE_AUTO_INSTALL=0). Manual command: install.py")
        return False
    if _read_active_marker():
        _log("A repair process is already running - progress: .cache/bootstrap.log")
        return False

    _CACHE_DIR.mkdir(exist_ok=True)
    log_handle = _LOG_PATH.open("a", encoding="utf-8")
    log_handle.write(f"\n=== bootstrap started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [sys.executable, str(_REPO_ROOT / "install.py")],
        cwd=_REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=creationflags,
        start_new_session=(sys.platform != "win32"),
    )
    _ACTIVE_MARKER_PATH.write_text(json.dumps({"pid": process.pid, "started": time.time()}), encoding="utf-8")
    log_handle.close()

    _log(f"install.py never completed in this environment ({missing_total} missing engine packages). "
         f"Repairing automatically in the background (pid {process.pid}, timeout {_timeout_seconds()}s). "
         "Progress: .cache/bootstrap.log")
    threading.Thread(target=_watchdog, args=(process,), daemon=True).start()
    return True


def dependency_check_callback(warnings: list) -> None:
    """AsyncDependencyChecker callback that keeps the default output and bootstraps."""
    print("📋 System Dependencies (background check):")
    for warning in warnings:
        print(f"   {warning}")

    engine_issues = {}
    for engine in DependencyChecker.ENGINE_DEPENDENCIES:
        missing = DependencyChecker.check_engine_dependencies(engine)
        if missing:
            engine_issues[engine] = missing

    if not engine_issues or _INSTALL_STATE_PATH.exists():
        # Repaired env still missing something, or core-level problem: keep manual guidance only.
        return

    echo_last_outcome()
    try:
        start_background_repair(engine_issues)
    except Exception as error:
        _log(f"Failed to start automatic repair: {error}")
