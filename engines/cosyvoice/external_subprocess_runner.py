"""Child entrypoint for plugin-owned official CosyVoice one-shot inference."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import sys
import traceback

import soundfile
import torch
import torchaudio


_TARGET_SAMPLE_RATE = 24000
_COSYVOICE3_PREFIX = "You are a helpful assistant.<|endofprompt|>"


@contextmanager
def _offline_runtime():
    """Force local cache resolution and fail closed on any network attempt."""
    import modelscope

    original_snapshot_download = modelscope.snapshot_download
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def local_snapshot_download(*args, **kwargs):
        kwargs["local_files_only"] = True
        return original_snapshot_download(*args, **kwargs)

    def blocked_connect(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled for the CosyVoice one-shot runtime")

    modelscope.snapshot_download = local_snapshot_download
    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_connect
    try:
        yield
    finally:
        modelscope.snapshot_download = original_snapshot_download
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def _model_generation(model_dir: Path) -> str:
    for config_name, generation in (
        ("cosyvoice.yaml", "v1"),
        ("cosyvoice2.yaml", "v2"),
        ("cosyvoice3.yaml", "v3"),
    ):
        if (model_dir / config_name).is_file():
            return generation
    raise RuntimeError(f"No supported CosyVoice configuration found in {model_dir}")


def _without_cosyvoice3_prefix(value: str) -> str:
    return value[len(_COSYVOICE3_PREFIX):] if value.startswith(_COSYVOICE3_PREFIX) else value


def _collect_audio(outputs) -> torch.Tensor:
    chunks = []
    for output in outputs:
        chunk = output.get("tts_speech")
        if not isinstance(chunk, torch.Tensor) or chunk.numel() == 0:
            raise RuntimeError("CosyVoice returned an empty or invalid audio chunk")
        chunks.append(chunk.detach().float().cpu())
    if not chunks:
        raise RuntimeError("CosyVoice returned no audio chunks")
    audio = torch.cat(chunks, dim=-1)
    if not torch.isfinite(audio).all():
        raise RuntimeError("CosyVoice returned non-finite audio samples")
    return audio


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: external_subprocess_runner.py <request.json>", file=sys.stderr)
        return 2

    try:
        payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        source_root = Path(payload["source_root"]).resolve()
        model_dir = Path(payload["model_dir"]).resolve()
        output_path = Path(payload["output_path"]).resolve()
        sys.path.insert(0, str(source_root))
        matcha_path = source_root / "third_party" / "Matcha-TTS"
        sys.path.insert(0, str(matcha_path))
        generation = _model_generation(model_dir)

        with _offline_runtime():
            from cosyvoice.cli.cosyvoice import AutoModel

            official_module = sys.modules["cosyvoice.cli.cosyvoice"]
            imported_source = Path(official_module.__file__).resolve()
            if not imported_source.is_relative_to(source_root):
                raise RuntimeError(f"CosyVoice imported outside registered source_root: {imported_source}")

            requested_constructor = dict(payload["constructor"])
            constructor = {
                "model_dir": str(model_dir),
                "load_trt": bool(requested_constructor.get("load_trt", False)),
                "fp16": bool(requested_constructor.get("use_fp16", True)),
            }
            if generation != "v1":
                constructor["load_vllm"] = bool(requested_constructor.get("load_vllm", False))
            model = AutoModel(**constructor)

            text = str(payload["text"])
            prompt_text = str(payload.get("prompt_text") or "")
            if generation == "v1":
                text = _without_cosyvoice3_prefix(text)
                prompt_text = _without_cosyvoice3_prefix(prompt_text)
            common = {
                "tts_text": text,
                "prompt_wav": str(Path(payload["prompt_wav"]).resolve()),
                "stream": False,
                "speed": float(payload.get("speed", 1.0)),
                "text_frontend": bool(payload.get("text_frontend", True)),
            }
            mode = str(payload["mode"])
            if mode == "cross_lingual":
                outputs = model.inference_cross_lingual(**common)
            elif mode == "zero_shot":
                outputs = model.inference_zero_shot(prompt_text=prompt_text, **common)
            elif mode == "instruct" and generation != "v1":
                outputs = model.inference_instruct2(
                    instruct_text=str(payload.get("instruct_text") or ""),
                    **common,
                )
            elif mode == "instruct":
                raise RuntimeError("CosyVoice v1 does not support prompt-audio instruct2 mode")
            else:
                raise RuntimeError(f"Unsupported CosyVoice inference mode: {mode}")

            audio = _collect_audio(outputs)
            source_sample_rate = int(model.sample_rate)
            if source_sample_rate != _TARGET_SAMPLE_RATE:
                audio = torchaudio.functional.resample(
                    audio,
                    orig_freq=source_sample_rate,
                    new_freq=_TARGET_SAMPLE_RATE,
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(
                output_path,
                audio.squeeze(0).numpy(),
                _TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )
        if not output_path.is_file():
            raise RuntimeError("CosyVoice inference returned without writing the output WAV")
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
