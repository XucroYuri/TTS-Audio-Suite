"""Behavioral coverage for GPT-SoVITS child-process cache isolation."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import wave

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_external_gpt_subprocess_module():
    # Loading the adapter should not require the heavy IndexTTS engine package;
    # expose only its filesystem package path so the shared subprocess helper
    # can be imported in this focused unit test.
    previous_index_package = sys.modules.get("engines.index_tts")
    if previous_index_package is None:
        index_package = types.ModuleType("engines.index_tts")
        index_package.__path__ = [str(REPO_ROOT / "engines" / "index_tts")]
        sys.modules["engines.index_tts"] = index_package
    path = REPO_ROOT / "engines" / "gpt_sovits" / "external_subprocess.py"
    spec = importlib.util.spec_from_file_location("external_gpt_private_cache_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_index_package is None:
            sys.modules.pop("engines.index_tts", None)

    return module


def _prepare_runtime(tmp_path: Path):
    source_root = tmp_path / "gpt-source"
    package_root = source_root / "GPT_SoVITS"
    official_api = package_root / "TTS_infer_pack" / "TTS.py"
    official_api.parent.mkdir(parents=True)
    official_api.write_text("class TTS: pass\nclass TTS_Config: pass\n", encoding="utf-8")
    (package_root / "eres2net").mkdir()

    interpreter = source_root / ".venv" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()
    pretrained = package_root / "pretrained_models"
    pretrained.mkdir()
    gpt_weight = pretrained / "s1.ckpt"
    sovits_weight = pretrained / "s2.pth"
    gpt_weight.touch()
    sovits_weight.touch()
    bert_path = pretrained / "bert"
    cnhubert_path = pretrained / "cnhubert"
    bert_path.mkdir()
    cnhubert_path.mkdir()

    voice_path = source_root / "voice.wav"
    with wave.open(str(voice_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x01\x00" * 160)

    temp_root = tmp_path / "private-temp"
    temp_root.mkdir()
    return (
        source_root,
        interpreter,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    )


def _path_without_windows_extended_prefix(value: str) -> Path:
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
        if value.startswith("UNC\\"):
            value = "\\\\" + value[4:]
    return Path(value)


@pytest.mark.unit
def test_gpt_child_cache_directories_exist_before_start_and_are_private(monkeypatch, tmp_path):
    module = _load_external_gpt_subprocess_module()
    (
        source_root,
        interpreter,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    ) = _prepare_runtime(tmp_path)
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 6242
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            environment = kwargs["env"]
            for variable in ("PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
                cache_path = _path_without_windows_extended_prefix(environment[variable])
                # This assertion runs from the Popen replacement, proving the
                # child-private directories exist before process creation.
                assert cache_path.is_dir(), f"{variable} missing before child start"
                assert cache_path.is_relative_to(temp_root.resolve())
                if os.name == "nt":
                    assert environment[variable].startswith("\\\\?\\")
            assert not (temp_root.parent / "numba-cache").exists()

        def communicate(self, timeout):
            manifest = json.loads(Path(observed["command"][2]).read_text(encoding="utf-8"))
            with wave.open(manifest["output_path"], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(32000)
                handle.writeframes(b"\x10\x00" * 320)
            return "official stdout", ""

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    proxy = module.ExternalGPTSovitsSubprocessProxy(
        source_root=source_root,
        gpt_weight=gpt_weight,
        sovits_weight=sovits_weight,
        bert_path=bert_path,
        cnhubert_path=cnhubert_path,
        python_executable=interpreter,
        device="cuda",
        use_fp16=True,
        version="v2",
        temp_root=temp_root,
        interrupt_check=lambda: False,
    )

    sample_rate, samples = proxy.run(
        {
            "text": "private cache behavior",
            "text_lang": "en",
            "ref_audio_path": str(voice_path),
            "prompt_text": "reference",
            "prompt_lang": "en",
        }
    )

    assert sample_rate == 32000
    assert samples.shape == (320,)
    assert not list(temp_root.iterdir())
