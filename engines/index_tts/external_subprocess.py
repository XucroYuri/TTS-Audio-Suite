"""One-shot adapter for an official IndexTTS checkout and its own interpreter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from typing import Any

import soundfile


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

        with tempfile.TemporaryDirectory(
            prefix="tts-audio-suite-indextts-",
            dir=str(self.temp_root) if self.temp_root is not None else None,
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
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

            process = subprocess.Popen(command, **popen_kwargs)
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                stdout, stderr, cleanup_diagnostic = self._cleanup_timed_out_process(process)
                diagnostic = (stderr or stdout or str(exc)).strip()
                if cleanup_diagnostic:
                    diagnostic = f"{diagnostic}; cleanup: {cleanup_diagnostic}"
                raise TimeoutError(
                    f"External IndexTTS subprocess exceeded {self.timeout_seconds:g}s: {diagnostic}"
                ) from exc

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

    def _cleanup_timed_out_process(self, process) -> tuple[str, str, str]:
        """Best-effort bounded cleanup that never replaces the primary timeout."""
        notes: list[str] = []
        try:
            self._terminate_process_tree(process, self.termination_grace_seconds)
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
            stdout, stderr = process.communicate(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            notes.append(
                f"cleanup communicate exceeded {self.termination_grace_seconds:g}s"
            )
            stdout, stderr = "", ""
        except Exception as exc:
            notes.append(f"cleanup communicate failed: {exc}")
            stdout, stderr = "", ""
        try:
            if process.poll() is None:
                notes.append("process exit could not be verified")
        except Exception as exc:
            notes.append(f"final process status check failed: {exc}")
        return stdout, stderr, "; ".join(notes)

    @staticmethod
    def _terminate_process_tree(process, grace_seconds: float) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=grace_seconds,
            )
            if result.returncode != 0 and process.poll() is None:
                diagnostic = (result.stderr or result.stdout or "taskkill failed").strip()
                raise RuntimeError(diagnostic)
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"process tree did not exit within {grace_seconds:g}s"
            ) from exc

    def to(self, device):
        self.device = str(device)
        return self

    def cleanup(self) -> None:
        """No persistent child exists between inference calls."""

    close = cleanup
