"""One-shot adapter for an official IndexTTS checkout and its own interpreter."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import psutil
import soundfile


_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


InterruptCheck = Callable[[], bool]


def _comfyui_interrupt_requested() -> bool:
    try:
        from comfy.model_management import processing_interrupted
    except ImportError:
        return False
    return bool(processing_interrupted())


def _raise_processing_interrupted(engine_label: str, diagnostic: str) -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except ImportError:
        pass
    else:
        throw_exception_if_processing_interrupted()
    raise InterruptedError(f"{engine_label} external subprocess interrupted: {diagnostic}")


def _clear_processing_interrupt() -> None:
    try:
        from comfy.model_management import interrupt_current_processing
    except ImportError:
        return
    interrupt_current_processing(False)


class _WindowsIOCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _WindowsJobBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _WindowsJobExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _WindowsJobBasicLimitInformation),
        ("IoInfo", _WindowsIOCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _WindowsKillOnCloseJob:
    """Kernel containment assigned before a suspended runner may create children."""

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._kernel32 = kernel32
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _WindowsJobExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            _WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            self._handle,
            _WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            kernel32.CloseHandle(self._handle)
            self._handle = None
            raise error

    def assign(self, process_handle: int) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        if not self._kernel32.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ExternalIndexTTSSubprocessProxy:
    """Expose the official ``IndexTTS2.infer`` shape across a process boundary."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        model_dir: str | Path,
        device: str,
        use_fp16: bool,
        use_cuda_kernel: bool | None = None,
        use_deepspeed: bool = False,
        use_torch_compile: bool = False,
        use_accel: bool = False,
        timeout_seconds: float = 900.0,
        termination_grace_seconds: float = 5.0,
        temp_root: str | Path | None = None,
        interrupt_check: InterruptCheck | None = None,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.device = device
        self.use_fp16 = bool(use_fp16)
        self.use_cuda_kernel = use_cuda_kernel
        self.use_deepspeed = bool(use_deepspeed)
        self.use_torch_compile = bool(use_torch_compile)
        self.use_accel = bool(use_accel)
        self.timeout_seconds = float(timeout_seconds)
        self.termination_grace_seconds = float(termination_grace_seconds)
        self.temp_root = Path(temp_root).resolve() if temp_root is not None else None
        self.interrupt_check = interrupt_check or _comfyui_interrupt_requested
        self.python_executable = self._resolve_python_executable()
        self.runner_path = Path(__file__).with_name("external_subprocess_runner.py").resolve()
        self._validate_runtime()

    def _resolve_python_executable(self) -> Path:
        if os.name == "nt":
            return self.source_root / ".venv" / "Scripts" / "python.exe"
        return self.source_root / ".venv" / "bin" / "python"

    def _validate_runtime(self) -> None:
        if not self.source_root.is_dir():
            raise RuntimeError(f"IndexTTS source_root is not a directory: {self.source_root}")
        if not (self.source_root / "indextts" / "infer_v2.py").is_file():
            raise RuntimeError(f"IndexTTS public inference API is missing: {self.source_root}")
        if not self.model_dir.is_dir() or not (self.model_dir / "config.yaml").is_file():
            raise RuntimeError(f"IndexTTS model directory is incomplete: {self.model_dir}")
        if not self.python_executable.is_file():
            raise RuntimeError(
                "IndexTTS compatible interpreter is missing: "
                f"{self.python_executable}. Create the checkout-local .venv from the official lockfile."
            )
        if not self.runner_path.is_file():
            raise RuntimeError(f"IndexTTS subprocess runner is missing: {self.runner_path}")
        if self.timeout_seconds <= 0:
            raise ValueError("IndexTTS subprocess timeout must be positive")
        if self.termination_grace_seconds <= 0:
            raise ValueError("IndexTTS subprocess termination grace must be positive")
        if self.temp_root is not None and not self.temp_root.is_dir():
            raise RuntimeError(f"IndexTTS temporary root is not a directory: {self.temp_root}")

    def infer(self, spk_audio_prompt, text, output_path, **inference_kwargs):
        voice_path = Path(spk_audio_prompt).resolve()
        if not voice_path.is_file():
            raise RuntimeError(f"IndexTTS speaker reference audio is missing: {voice_path}")

        temporary_directory = tempfile.TemporaryDirectory(
            prefix="tts-audio-suite-indextts-",
            dir=str(self.temp_root) if self.temp_root is not None else None,
        )
        primary_error: BaseException | None = None
        try:
            temporary_path = Path(temporary_directory.name)
            for directory_name in ("pycache", "numba-cache", "matplotlib"):
                (temporary_path / directory_name).mkdir(parents=True, exist_ok=True)
            child_output = temporary_path / "output.wav"
            manifest_path = temporary_path / "request.json"
            manifest = {
                "source_root": str(self.source_root),
                "model_dir": str(self.model_dir),
                "output_path": str(child_output),
                "constructor": {
                    "device": self.device,
                    "use_fp16": self.use_fp16,
                    "use_cuda_kernel": self.use_cuda_kernel,
                    "use_deepspeed": self.use_deepspeed,
                    "use_torch_compile": self.use_torch_compile,
                    "use_accel": self.use_accel,
                },
                "inference": {
                    "spk_audio_prompt": str(voice_path),
                    "text": str(text),
                    **inference_kwargs,
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            command = [str(self.python_executable), str(self.runner_path), str(manifest_path)]
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": str(temporary_path / "pycache"),
                    "NUMBA_CACHE_DIR": str(temporary_path / "numba-cache"),
                    "MPLCONFIGDIR": str(temporary_path / "matplotlib"),
                }
            )
            popen_kwargs: dict[str, Any] = {
                "cwd": str(self.source_root),
                "env": environment,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            process = self._start_process(command, popen_kwargs)
            stdout, stderr = self._communicate_with_control(process, "IndexTTS")

            try:
                self._close_windows_job(process)
            except Exception as exc:
                raise RuntimeError(f"Windows Job Object cleanup failed: {exc}") from exc

            if stdout:
                print(f"[IndexTTS external stdout]\n{stdout.rstrip()}")
            if stderr:
                print(f"[IndexTTS external stderr]\n{stderr.rstrip()}", file=sys.stderr)
            if process.returncode != 0:
                diagnostic = (stderr or stdout or "no child diagnostics").strip()
                raise RuntimeError(
                    f"External IndexTTS subprocess exited {process.returncode}: {diagnostic}"
                )
            if not child_output.is_file():
                raise RuntimeError("External IndexTTS subprocess completed without an output WAV")

            samples, sample_rate = soundfile.read(
                child_output,
                dtype="float32",
                always_2d=True,
            )
            if sample_rate <= 0 or samples.shape[0] == 0:
                raise RuntimeError("External IndexTTS subprocess produced an empty WAV")
            if output_path is not None:
                requested_output = Path(output_path)
                requested_output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(child_output, requested_output)
            return int(sample_rate), samples
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                temporary_directory.cleanup()
            except Exception as cleanup_error:
                diagnostic = f"temporary directory cleanup failed: {cleanup_error}"
                if primary_error is None:
                    raise RuntimeError(diagnostic) from cleanup_error
                primary_error.args = (f"{primary_error}; {diagnostic}",)

    @staticmethod
    def _start_process(command, popen_kwargs):
        """Start a Windows runner suspended, contain it, then allow it to execute."""
        options = dict(popen_kwargs)
        if os.name != "nt":
            return subprocess.Popen(command, **options)
        options["creationflags"] = (
            int(options.get("creationflags", 0)) | _WINDOWS_CREATE_SUSPENDED
        )
        process = subprocess.Popen(command, **options)
        # Lightweight Popen doubles used by unit tests have no native process
        # handle and were never actually suspended.
        if not hasattr(process, "_handle"):
            return process
        job: _WindowsKillOnCloseJob | None = None
        try:
            job = _WindowsKillOnCloseJob()
            job.assign(int(process._handle))
            psutil.Process(process.pid).resume()
            setattr(process, "_tts_windows_job", job)
            return process
        except Exception as exc:
            if job is not None:
                try:
                    job.close()
                except Exception:
                    pass
            try:
                process.kill()
            except Exception:
                pass
            raise RuntimeError(
                f"failed to assign suspended IndexTTS runner to Windows Job Object: {exc}"
            ) from exc

    @staticmethod
    def _close_windows_job(process) -> bool:
        job = getattr(process, "_tts_windows_job", None)
        if job is None:
            return False
        try:
            job.close()
        finally:
            setattr(process, "_tts_windows_job", None)
        return True

    def _cleanup_timed_out_process(self, process) -> tuple[str, str, str, bool]:
        """Best-effort bounded cleanup that never replaces the primary timeout."""
        notes: list[str] = []
        tree_exit_verified = False
        deadline = time.monotonic() + self.termination_grace_seconds
        try:
            tree_diagnostic = self._terminate_process_tree(
                process,
                min(
                    self.termination_grace_seconds,
                    max(0.0, deadline - time.monotonic()),
                ),
            )
            tree_exit_verified = True
            if tree_diagnostic:
                notes.append(str(tree_diagnostic))
        except Exception as exc:
            notes.append(f"tree termination failed: {exc}")

        try:
            still_running = process.poll() is None
        except Exception as exc:
            notes.append(f"process status check failed: {exc}")
            still_running = True
        if still_running:
            try:
                process.kill()
            except Exception as exc:
                notes.append(f"direct kill failed: {exc}")

        try:
            remaining = min(
                self.termination_grace_seconds,
                max(0.0, deadline - time.monotonic()),
            )
            stdout, stderr = process.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            notes.append(
                f"cleanup communicate exceeded remaining {remaining:g}s "
                f"of {self.termination_grace_seconds:g}s grace"
            )
            stdout, stderr = "", ""
        except Exception as exc:
            notes.append(f"cleanup communicate failed: {exc}")
            stdout, stderr = "", ""
        try:
            if process.poll() is None:
                notes.append("process exit could not be verified")
                tree_exit_verified = False
        except Exception as exc:
            notes.append(f"final process status check failed: {exc}")
            tree_exit_verified = False
        return stdout, stderr, "; ".join(notes), tree_exit_verified

    def _communicate_with_control(self, process, engine_label: str) -> tuple[str, str]:
        deadline = time.monotonic() + self.timeout_seconds
        timeout_error: subprocess.TimeoutExpired | None = None
        while True:
            if self.interrupt_check():
                stdout, stderr, cleanup, tree_exit_verified = (
                    self._cleanup_timed_out_process(process)
                )
                diagnostic = (stderr or stdout or "interrupted").strip()
                if cleanup:
                    diagnostic = f"{diagnostic}; cleanup: {cleanup}"
                if not tree_exit_verified:
                    _clear_processing_interrupt()
                    raise RuntimeError(
                        f"{engine_label} interruption cleanup failed: {diagnostic}"
                    )
                _raise_processing_interrupted(engine_label, diagnostic)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                stdout, stderr, cleanup, _tree_exit_verified = (
                    self._cleanup_timed_out_process(process)
                )
                diagnostic = (stderr or stdout or str(timeout_error or "deadline exceeded")).strip()
                if cleanup:
                    diagnostic = f"{diagnostic}; cleanup: {cleanup}"
                raise TimeoutError(
                    f"External {engine_label} subprocess exceeded {self.timeout_seconds:g}s: {diagnostic}"
                ) from timeout_error
            try:
                return process.communicate(timeout=min(0.25, remaining))
            except subprocess.TimeoutExpired as exc:
                timeout_error = exc

    @staticmethod
    def _remaining_before(deadline: float, action: str, notes: list[str]) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            notes.append(f"process-tree deadline exhausted before {action}")
            return 0.0
        return remaining

    @staticmethod
    def _windows_process_identity(candidate, deadline: float, notes: list[str]):
        if not ExternalIndexTTSSubprocessProxy._remaining_before(
            deadline,
            f"identity check for PID {candidate.pid}",
            notes,
        ):
            return None
        try:
            identity = (int(candidate.pid), float(candidate.create_time()))
        except psutil.NoSuchProcess:
            return None
        except Exception as exc:
            notes.append(f"identity check for PID {candidate.pid} failed: {exc}")
            return None
        if time.monotonic() >= deadline:
            notes.append(
                f"process-tree deadline exhausted during identity check for PID {candidate.pid}"
            )
            return None
        return identity

    @staticmethod
    def _windows_identity_is_alive(candidate, identity, deadline: float, notes: list[str]):
        if not ExternalIndexTTSSubprocessProxy._remaining_before(
            deadline,
            f"liveness check for PID {candidate.pid}",
            notes,
        ):
            return None
        try:
            current_identity = (int(candidate.pid), float(candidate.create_time()))
            if current_identity != identity:
                notes.append(f"PID {candidate.pid} was reused; skipped stale process handle")
                return False
            if not ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"is_running check for PID {candidate.pid}",
                notes,
            ):
                return None
            return bool(candidate.is_running())
        except psutil.NoSuchProcess:
            return False
        except Exception as exc:
            notes.append(f"liveness check for PID {candidate.pid} failed: {exc}")
            return True

    @staticmethod
    def _bounded_windows_fallback(process, deadline: float, notes: list[str]) -> bool:
        """Suspend parents before direct snapshots so the discovered tree is stable."""
        if not ExternalIndexTTSSubprocessProxy._remaining_before(
            deadline,
            f"opening root PID {process.pid}",
            notes,
        ):
            raise TimeoutError("; ".join(notes))
        try:
            pending = deque([psutil.Process(process.pid)])
        except psutil.NoSuchProcess:
            return False
        except Exception as exc:
            notes.append(f"opening root PID {process.pid} failed: {exc}")
            raise RuntimeError("; ".join(notes)) from exc

        captured: dict[tuple[int, float], Any] = {}
        while pending:
            candidate = pending.popleft()
            identity = ExternalIndexTTSSubprocessProxy._windows_process_identity(
                candidate,
                deadline,
                notes,
            )
            if identity is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError("; ".join(notes))
                continue
            if identity in captured:
                continue
            captured[identity] = candidate
            alive = ExternalIndexTTSSubprocessProxy._windows_identity_is_alive(
                candidate,
                identity,
                deadline,
                notes,
            )
            if alive is None:
                raise TimeoutError("; ".join(notes))
            if not alive:
                continue

            if not ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"suspending PID {candidate.pid}",
                notes,
            ):
                raise TimeoutError("; ".join(notes))
            try:
                candidate.suspend()
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                notes.append(
                    f"suspend PID {candidate.pid} failed; spawn race cannot be excluded: {exc}"
                )

            if not ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"snapshotting direct children of PID {candidate.pid}",
                notes,
            ):
                raise TimeoutError("; ".join(notes))
            try:
                pending.extend(candidate.children(recursive=False))
            except psutil.NoSuchProcess:
                pass
            except Exception as exc:
                notes.append(f"direct-child snapshot for PID {candidate.pid} failed: {exc}")

        # Every captured parent is now suspended, so its child list cannot grow.
        # Terminate only after the complete stable tree is known; terminating a
        # launcher shim early can otherwise take its still-unvisited child with it.
        for _identity, candidate in reversed(captured.items()):
            if not ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"terminating PID {candidate.pid}",
                notes,
            ):
                raise TimeoutError("; ".join(notes))
            try:
                candidate.terminate()
            except psutil.NoSuchProcess:
                pass
            except Exception as exc:
                notes.append(f"terminate PID {candidate.pid} failed: {exc}")

        survivors: list[int] = []
        for identity, candidate in captured.items():
            alive = ExternalIndexTTSSubprocessProxy._windows_identity_is_alive(
                candidate,
                identity,
                deadline,
                notes,
            )
            if alive is None:
                raise TimeoutError("; ".join(notes))
            if not alive:
                continue
            remaining = ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"waiting for PID {candidate.pid} after terminate",
                notes,
            )
            if remaining <= 0:
                raise TimeoutError("; ".join(notes))
            try:
                candidate.wait(timeout=min(0.05, remaining))
            except (psutil.TimeoutExpired, AttributeError):
                pass
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                notes.append(f"wait for PID {candidate.pid} failed: {exc}")
            alive = ExternalIndexTTSSubprocessProxy._windows_identity_is_alive(
                candidate,
                identity,
                deadline,
                notes,
            )
            if alive is None:
                raise TimeoutError("; ".join(notes))
            if not alive:
                continue
            if not ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"killing PID {candidate.pid}",
                notes,
            ):
                raise TimeoutError("; ".join(notes))
            try:
                candidate.kill()
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                notes.append(f"kill PID {candidate.pid} failed: {exc}")
            remaining = ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                f"waiting for PID {candidate.pid} after kill",
                notes,
            )
            if remaining <= 0:
                raise TimeoutError("; ".join(notes))
            try:
                candidate.wait(timeout=min(0.05, remaining))
            except (psutil.TimeoutExpired, AttributeError):
                pass
            except psutil.NoSuchProcess:
                continue
            except Exception as exc:
                notes.append(f"post-kill wait for PID {candidate.pid} failed: {exc}")
            alive = ExternalIndexTTSSubprocessProxy._windows_identity_is_alive(
                candidate,
                identity,
                deadline,
                notes,
            )
            if alive is None:
                raise TimeoutError("; ".join(notes))
            if alive:
                survivors.append(candidate.pid)
        if survivors:
            raise RuntimeError(
                "; ".join(
                    [*notes, f"process-tree PIDs still alive: {', '.join(map(str, survivors))}"]
                )
            )
        return True

    @staticmethod
    def _terminate_process_tree(process, grace_seconds: float) -> str:
        if process.poll() is not None:
            raise RuntimeError("process tree exit could not be verified after root exit")
        if grace_seconds <= 0:
            raise TimeoutError("process-tree termination deadline was already exhausted")
        deadline = time.monotonic() + grace_seconds
        notes: list[str] = []
        tree_exit_verified = False
        if os.name == "nt":
            if getattr(process, "_tts_windows_job", None) is not None:
                if not ExternalIndexTTSSubprocessProxy._remaining_before(
                    deadline,
                    "closing Windows Job Object",
                    notes,
                ):
                    raise TimeoutError("; ".join(notes))
                try:
                    ExternalIndexTTSSubprocessProxy._close_windows_job(process)
                    notes.append("Windows Job Object closed with kill-on-close")
                    tree_exit_verified = True
                except Exception as exc:
                    notes.append(f"Windows Job Object close failed: {exc}")

            if process.poll() is None and getattr(process, "_tts_windows_job", None) is None:
                remaining = ExternalIndexTTSSubprocessProxy._remaining_before(
                    deadline,
                    "taskkill",
                    notes,
                )
                if remaining <= 0:
                    raise TimeoutError("; ".join(notes))
                taskkill_timeout = min(remaining, grace_seconds / 2.0)
                taskkill_failed = False
                try:
                    result = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=taskkill_timeout,
                    )
                    if result.returncode != 0:
                        taskkill_failed = True
                        diagnostic = (
                            result.stderr or result.stdout or "taskkill failed"
                        ).strip()
                        notes.append(f"taskkill failed ({result.returncode}): {diagnostic}")
                    else:
                        tree_exit_verified = True
                except subprocess.TimeoutExpired:
                    taskkill_failed = True
                    notes.append(f"taskkill exceeded {taskkill_timeout:g}s")
                except Exception as exc:
                    taskkill_failed = True
                    notes.append(f"taskkill invocation failed: {exc}")
                if taskkill_failed or process.poll() is None:
                    tree_exit_verified = (
                        ExternalIndexTTSSubprocessProxy._bounded_windows_fallback(
                            process,
                            deadline,
                            notes,
                        )
                        or tree_exit_verified
                    )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                tree_exit_verified = True
            else:
                tree_exit_verified = True
        if process.poll() is None:
            remaining = ExternalIndexTTSSubprocessProxy._remaining_before(
                deadline,
                "waiting for process-tree exit",
                notes,
            )
            if remaining <= 0:
                raise TimeoutError("; ".join(notes))
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                diagnostic = "; ".join(notes) if notes else "no child diagnostics"
                raise TimeoutError(
                    f"process tree did not exit within {grace_seconds:g}s: {diagnostic}"
                ) from exc
            except Exception as exc:
                diagnostic = "; ".join(notes) if notes else "no child diagnostics"
                raise RuntimeError(f"process tree wait failed: {exc}; {diagnostic}") from exc
        if not tree_exit_verified:
            raise RuntimeError(
                "; ".join([*notes, "process tree exit could not be verified"])
            )
        return "; ".join(notes)

    def to(self, device):
        self.device = str(device)
        return self

    def cleanup(self) -> None:
        """No persistent child exists between inference calls."""

    close = cleanup
