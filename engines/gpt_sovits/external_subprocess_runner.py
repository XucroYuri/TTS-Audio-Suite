"""Child entrypoint for plugin-owned official GPT-SoVITS one-shot inference."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
import soundfile


@contextmanager
def _offline_runtime():
    """Fail closed if the official runtime attempts any network access."""
    original_connect = socket.socket.connect
    original_create_connection = socket.create_connection

    def blocked_connect(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("network access is disabled for the GPT-SoVITS one-shot runtime")

    socket.socket.connect = blocked_connect
    socket.create_connection = blocked_connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.create_connection = original_create_connection


def _collect_audio(result) -> tuple[int, np.ndarray]:
    fragments = [result] if isinstance(result, tuple) else list(result)
    if not fragments:
        raise RuntimeError("Official GPT-SoVITS runtime returned no audio")
    sample_rate = int(fragments[0][0])
    chunks = []
    for fragment_rate, audio in fragments:
        if int(fragment_rate) != sample_rate:
            raise RuntimeError("Official GPT-SoVITS runtime returned inconsistent sample rates")
        chunk = np.asarray(audio).reshape(-1)
        if chunk.size == 0 or not np.isfinite(chunk).all():
            raise RuntimeError("Official GPT-SoVITS runtime returned invalid audio")
        chunks.append(chunk)
    return sample_rate, np.concatenate(chunks)


def _configure_official_speaker_encoder(path: str) -> None:
    """Override GPT-SoVITS' cwd-relative speaker encoder path when registered."""
    if not path:
        return
    import sv

    speaker_encoder = Path(path).resolve()
    if not speaker_encoder.is_file():
        raise RuntimeError(f"GPT-SoVITS speaker encoder checkpoint is missing: {speaker_encoder}")
    sv.sv_path = str(speaker_encoder)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: external_subprocess_runner.py <request.json>", file=sys.stderr)
        return 2

    try:
        payload = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        source_root = Path(payload["source_root"]).resolve()
        package_root = source_root / "GPT_SoVITS"
        output_path = Path(payload["output_path"]).resolve()
        runtime_config_path = Path(payload["runtime_config_path"]).resolve()
        if output_path.is_relative_to(source_root) or runtime_config_path.is_relative_to(source_root):
            raise RuntimeError("GPT-SoVITS output and runtime config must stay outside registered source_root")

        sys.path[:0] = [
            str(source_root),
            str(package_root),
            str(package_root / "eres2net"),
        ]
        config = dict(payload["config"])
        custom_config = {
            "device": str(config["device"]),
            "is_half": bool(config["use_fp16"]),
            "version": str(config["version"]),
            "t2s_weights_path": str(Path(config["gpt_weight"]).resolve()),
            "vits_weights_path": str(Path(config["sovits_weight"]).resolve()),
            "bert_base_path": str(Path(config["bert_path"]).resolve()),
            "cnhuhbert_base_path": str(Path(config["cnhubert_path"]).resolve()),
        }
        for key in ("t2s_weights_path", "vits_weights_path"):
            if not Path(custom_config[key]).is_file():
                raise RuntimeError(f"GPT-SoVITS registered weight is missing: {custom_config[key]}")
        for key in ("bert_base_path", "cnhuhbert_base_path"):
            if not Path(custom_config[key]).is_dir():
                raise RuntimeError(f"GPT-SoVITS registered model directory is missing: {custom_config[key]}")

        with _offline_runtime():
            from TTS_infer_pack.TTS import TTS, TTS_Config

            _configure_official_speaker_encoder(str(config.get("sv_path", "")))

            official_module = sys.modules["TTS_infer_pack.TTS"]
            imported_source = Path(official_module.__file__).resolve()
            if not imported_source.is_relative_to(source_root):
                raise RuntimeError(
                    f"GPT-SoVITS imported outside registered source_root: {imported_source}"
                )
            runtime_config = TTS_Config({"custom": custom_config})
            runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_config.configs_path = str(runtime_config_path)
            runtime = TTS(runtime_config)
            sample_rate, audio = _collect_audio(runtime.run(dict(payload["inference"])))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(output_path, audio, sample_rate, subtype="PCM_16")
            print(
                "GPT_SOVITS_OFFLINE_ENFORCED "
                f"source={imported_source} config={runtime_config_path}"
            )
        if not output_path.is_file():
            raise RuntimeError("GPT-SoVITS inference returned without writing the output WAV")
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
