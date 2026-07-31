"""One-shot official-checkout GPT-SoVITS subprocess adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import soundfile

from engines.index_tts.external_subprocess import ExternalIndexTTSSubprocessProxy


class ExternalGPTSovitsSubprocessProxy(ExternalIndexTTSSubprocessProxy):
    """Expose official ``TTS.run`` while keeping it outside ComfyUI."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        gpt_weight: str | Path,
        sovits_weight: str | Path,
        bert_path: str | Path,
        cnhubert_path: str | Path,
        device: str,
        use_fp16: bool,
        version: str,
        python_executable: str | Path | None = None,
        timeout_seconds: float = 900.0,
        termination_grace_seconds: float = 5.0,
        temp_root: str | Path | None = None,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.gpt_weight = Path(gpt_weight).resolve()
        self.sovits_weight = Path(sovits_weight).resolve()
        self.bert_path = Path(bert_path).resolve()
        self.cnhubert_path = Path(cnhubert_path).resolve()
        self.device = str(device)
        self.use_fp16 = bool(use_fp16)
        self.version = str(version)
        self.timeout_seconds = float(timeout_seconds)
        self.termination_grace_seconds = float(termination_grace_seconds)
        self.temp_root = Path(temp_root).resolve() if temp_root is not None else None
        self.python_executable = (
            Path(python_executable).resolve()
            if python_executable is not None
            else self._resolve_python_executable()
        )
        self.runner_path = Path(__file__).with_name("external_subprocess_runner.py").resolve()
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        if not self.source_root.is_dir():
            raise RuntimeError(f"GPT-SoVITS source_root is not a directory: {self.source_root}")
        official_api = self.source_root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py"
        if not official_api.is_file():
            raise RuntimeError(f"GPT-SoVITS public inference API is missing: {official_api}")
        for label, path, expected_kind in (
            ("GPT weight", self.gpt_weight, "file"),
            ("SoVITS weight", self.sovits_weight, "file"),
            ("BERT directory", self.bert_path, "directory"),
            ("CNHuBERT directory", self.cnhubert_path, "directory"),
        ):
            exists = path.is_file() if expected_kind == "file" else path.is_dir()
            if not exists:
                raise RuntimeError(f"GPT-SoVITS {label} is missing: {path}")
        if not self.python_executable.is_file():
            raise RuntimeError(
                "GPT-SoVITS compatible interpreter is missing: "
                f"{self.python_executable}. Reuse the checkout-local environment prepared from official requirements."
            )
        if not self.runner_path.is_file():
            raise RuntimeError(f"GPT-SoVITS subprocess runner is missing: {self.runner_path}")
        if self.timeout_seconds <= 0:
            raise ValueError("GPT-SoVITS subprocess timeout must be positive")
        if self.termination_grace_seconds <= 0:
            raise ValueError("GPT-SoVITS subprocess termination grace must be positive")
        if self.temp_root is not None and not self.temp_root.is_dir():
            raise RuntimeError(f"GPT-SoVITS temporary root is not a directory: {self.temp_root}")

    def run(self, inputs: dict[str, Any]):
        inference = dict(inputs)
        voice_path = Path(str(inference.get("ref_audio_path") or "")).resolve()
        if not voice_path.is_file():
            raise RuntimeError(f"GPT-SoVITS reference audio is missing: {voice_path}")
        inference["ref_audio_path"] = str(voice_path)

        temporary_directory = tempfile.TemporaryDirectory(
            prefix="tts-audio-suite-gptsovits-",
            dir=str(self.temp_root) if self.temp_root is not None else None,
        )
        primary_error: BaseException | None = None
        try:
            temporary_path = Path(temporary_directory.name)
            child_output = temporary_path / "output.wav"
            manifest_path = temporary_path / "request.json"
            manifest = {
                "source_root": str(self.source_root),
                "output_path": str(child_output),
                "runtime_config_path": str(temporary_path / "runtime-config.yaml"),
                "config": {
                    "gpt_weight": str(self.gpt_weight),
                    "sovits_weight": str(self.sovits_weight),
                    "bert_path": str(self.bert_path),
                    "cnhubert_path": str(self.cnhubert_path),
                    "device": self.device,
                    "use_fp16": self.use_fp16,
                    "version": self.version,
                },
                "inference": inference,
            }
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, allow_nan=False),
                encoding="utf-8",
            )
            command = [str(self.python_executable), str(self.runner_path), str(manifest_path)]
            environment = os.environ.copy()
            environment.update(
                {
                    "TTS_AUDIO_SUITE_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "MODELSCOPE_OFFLINE": "1",
                    "PYTHONNOUSERSITE": "1",
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
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                stdout, stderr, cleanup_diagnostic = self._cleanup_timed_out_process(process)
                diagnostic = (stderr or stdout or str(exc)).strip()
                if cleanup_diagnostic:
                    diagnostic = f"{diagnostic}; cleanup: {cleanup_diagnostic}"
                raise TimeoutError(
                    f"External GPT-SoVITS subprocess exceeded {self.timeout_seconds:g}s: {diagnostic}"
                ) from exc

            try:
                self._close_windows_job(process)
            except Exception as exc:
                raise RuntimeError(f"Windows Job Object cleanup failed: {exc}") from exc
            if stdout:
                print(f"[GPT-SoVITS external stdout]\n{stdout.rstrip()}")
            if stderr:
                print(f"[GPT-SoVITS external stderr]\n{stderr.rstrip()}", file=sys.stderr)
            if process.returncode != 0:
                diagnostic = (stderr or stdout or "no child diagnostics").strip()
                raise RuntimeError(
                    f"External GPT-SoVITS subprocess exited {process.returncode}: {diagnostic}"
                )
            if not child_output.is_file():
                raise RuntimeError("External GPT-SoVITS subprocess completed without an output WAV")

            samples, sample_rate = soundfile.read(child_output, dtype="float32", always_2d=False)
            samples = np.asarray(samples)
            if samples.ndim == 2 and samples.shape[1] == 1:
                samples = samples[:, 0]
            if sample_rate <= 0 or samples.ndim != 1 or samples.size == 0:
                raise RuntimeError(
                    "External GPT-SoVITS subprocess produced invalid audio: "
                    f"sample_rate={sample_rate}, shape={samples.shape}"
                )
            if not np.isfinite(samples).all():
                raise RuntimeError("External GPT-SoVITS subprocess produced non-finite audio")
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

    def cleanup(self) -> None:
        """No persistent child exists between inference calls."""

    close = cleanup
