"""One-shot official-checkout CosyVoice subprocess adapter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import soundfile
import torch

from engines.index_tts.external_subprocess import (
    ExternalIndexTTSSubprocessProxy,
    InterruptCheck,
    _private_child_path,
    _comfyui_interrupt_requested,
    _cleanup_temporary_directory,
)


class ExternalCosyVoiceSubprocessProxy(ExternalIndexTTSSubprocessProxy):
    """Expose official CosyVoice inference while keeping it outside ComfyUI."""

    sample_rate = 24000

    def __init__(
        self,
        *,
        source_root: str | Path,
        model_dir: str | Path,
        device: str,
        use_fp16: bool,
        load_trt: bool = False,
        load_vllm: bool = False,
        timeout_seconds: float = 900.0,
        termination_grace_seconds: float = 5.0,
        temp_root: str | Path | None = None,
        interrupt_check: InterruptCheck | None = None,
    ) -> None:
        self.source_root = Path(source_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.device = str(device)
        self.use_fp16 = bool(use_fp16)
        self.load_trt = bool(load_trt)
        self.load_vllm = bool(load_vllm)
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
            raise RuntimeError(f"CosyVoice source_root is not a directory: {self.source_root}")
        if not (self.source_root / "cosyvoice" / "cli" / "cosyvoice.py").is_file():
            raise RuntimeError(f"CosyVoice public inference API is missing: {self.source_root}")
        if not (self.source_root / "third_party" / "Matcha-TTS").is_dir():
            raise RuntimeError(f"CosyVoice Matcha-TTS checkout is missing: {self.source_root}")
        if not self.model_dir.is_dir():
            raise RuntimeError(f"CosyVoice model directory is missing: {self.model_dir}")

        config_name = next(
            (
                name
                for name in ("cosyvoice.yaml", "cosyvoice2.yaml", "cosyvoice3.yaml")
                if (self.model_dir / name).is_file()
            ),
            None,
        )
        if config_name is None:
            raise RuntimeError(f"CosyVoice model configuration is missing: {self.model_dir}")
        required = ["llm.pt", "flow.pt", "hift.pt", "campplus.onnx"]
        required.append(
            "speech_tokenizer_v1.onnx"
            if config_name == "cosyvoice.yaml"
            else "speech_tokenizer_v2.onnx"
            if config_name == "cosyvoice2.yaml"
            else "speech_tokenizer_v3.onnx"
        )
        missing = [name for name in required if not (self.model_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"CosyVoice model directory is incomplete: missing {missing}")
        if not self.python_executable.is_file():
            raise RuntimeError(
                "CosyVoice compatible interpreter is missing: "
                f"{self.python_executable}. Reuse a checkout-local environment prepared from the official requirements."
            )
        if not self.runner_path.is_file():
            raise RuntimeError(f"CosyVoice subprocess runner is missing: {self.runner_path}")
        if self.timeout_seconds <= 0:
            raise ValueError("CosyVoice subprocess timeout must be positive")
        if self.termination_grace_seconds <= 0:
            raise ValueError("CosyVoice subprocess termination grace must be positive")
        if self.temp_root is not None and not self.temp_root.is_dir():
            raise RuntimeError(f"CosyVoice temporary root is not a directory: {self.temp_root}")

    def inference_cross_lingual(
        self,
        tts_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        del zero_shot_spk_id
        yield self._run(
            mode="cross_lingual",
            text=tts_text,
            prompt_wav=prompt_wav,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
        )

    def inference_zero_shot(
        self,
        tts_text,
        prompt_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        del zero_shot_spk_id
        yield self._run(
            mode="zero_shot",
            text=tts_text,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
        )

    def inference_instruct2(
        self,
        tts_text,
        instruct_text,
        prompt_wav,
        zero_shot_spk_id="",
        stream=False,
        speed=1.0,
        text_frontend=True,
    ):
        del zero_shot_spk_id
        yield self._run(
            mode="instruct",
            text=tts_text,
            prompt_wav=prompt_wav,
            instruct_text=instruct_text,
            speed=speed,
            stream=stream,
            text_frontend=text_frontend,
        )

    def _run(
        self,
        *,
        mode: str,
        text: str,
        prompt_wav: str,
        prompt_text: str = "",
        instruct_text: str = "",
        speed: float = 1.0,
        stream: bool = False,
        text_frontend: bool = True,
    ) -> dict[str, torch.Tensor]:
        if stream:
            raise RuntimeError("Streaming is not supported by the one-shot CosyVoice runtime")
        voice_path = Path(prompt_wav).resolve()
        if not voice_path.is_file():
            raise RuntimeError(f"CosyVoice speaker reference audio is missing: {voice_path}")

        temporary_directory = tempfile.TemporaryDirectory(
            prefix="tts-audio-suite-cosyvoice-",
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
                    "use_fp16": self.use_fp16,
                    "load_trt": self.load_trt,
                    "load_vllm": self.load_vllm,
                },
                "mode": str(mode),
                "text": str(text),
                "prompt_wav": str(voice_path),
                "prompt_text": str(prompt_text or ""),
                "instruct_text": str(instruct_text or ""),
                "speed": float(speed),
                "stream": False,
                "text_frontend": bool(text_frontend),
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
                    "PYTHONPYCACHEPREFIX": _private_child_path(temporary_path / "pycache"),
                    "NUMBA_CACHE_DIR": _private_child_path(temporary_path / "numba-cache"),
                    "MPLCONFIGDIR": _private_child_path(temporary_path / "matplotlib"),
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
            stdout, stderr = self._communicate_with_control(process, "CosyVoice")

            try:
                self._close_windows_job(process)
            except Exception as exc:
                raise RuntimeError(f"Windows Job Object cleanup failed: {exc}") from exc

            if stdout:
                print(f"[CosyVoice external stdout]\n{stdout.rstrip()}")
            if stderr:
                print(f"[CosyVoice external stderr]\n{stderr.rstrip()}", file=sys.stderr)
            if process.returncode != 0:
                diagnostic = (stderr or stdout or "no child diagnostics").strip()
                raise RuntimeError(
                    f"External CosyVoice subprocess exited {process.returncode}: {diagnostic}"
                )
            if not child_output.is_file():
                raise RuntimeError("External CosyVoice subprocess completed without an output WAV")

            samples, sample_rate = soundfile.read(child_output, dtype="float32", always_2d=True)
            if sample_rate != self.sample_rate or samples.shape[0] == 0:
                raise RuntimeError(
                    f"External CosyVoice subprocess produced invalid audio: sample_rate={sample_rate}, frames={samples.shape[0]}"
                )
            return {"tts_speech": torch.from_numpy(samples.T.copy())}
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                _cleanup_temporary_directory(temporary_directory, temporary_path)
            except Exception as cleanup_error:
                diagnostic = f"temporary directory cleanup failed: {cleanup_error}"
                if primary_error is None:
                    raise RuntimeError(diagnostic) from cleanup_error
                primary_error.args = (f"{primary_error}; {diagnostic}",)

    def cleanup(self) -> None:
        """No persistent child exists between inference calls."""

    close = cleanup
