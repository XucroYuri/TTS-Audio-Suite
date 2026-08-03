import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import types
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_NODE_IDS = {
    "ASRPunctuationTruecaseNode",
    "CharacterVoicesNode",
    "ChatterBoxAudioAnalyzer",
    "ChatterBoxAudioAnalyzerOptions",
    "ChatterBoxEngineNode",
    "ChatterBoxF5TTSEditOptions",
    "ChatterBoxF5TTSEditVoice",
    "ChatterBoxOfficial23LangEngineNode",
    "ChatterBoxVoiceCapture",
    "CosyVoiceEngineNode",
    "DotsTTSEngineNode",
    "EchoTTSEngineNode",
    "F5TTSEngineNode",
    "FishAudioS2EngineNode",
    "GPTSovitsEngineNode",
    "GraniteASREngineNode",
    "HiggsAudioEngineNode",
    "HiggsAudioV3EngineNode",
    "IndexTTSEmotionOptionsNode",
    "IndexTTSEngineNode",
    "LoadRVCModelNode",
    "MergeAudioNode",
    "MossClipStagingNode",
    "MossDatasetPrepNode",
    "MossDatasetRowsNode",
    "MossSoundEffectV2EngineNode",
    "MossTrainingConfigNode",
    "MossTTSEngineNode",
    "MouthMovementAnalyzer",
    "OmniVoiceEngineNode",
    "OmniVoiceInstructionBuilderNode",
    "PhonemeTextNormalizer",
    "Qwen3TTSEngineNode",
    "QwenEmotionNode",
    "RefreshVoiceCacheNode",
    "RVCDatasetPrepNode",
    "RVCEngineNode",
    "RVCPitchOptionsNode",
    "RVCTrainingConfigNode",
    "SaveCharacterVoiceNode",
    "SRTAdvancedOptionsNode",
    "StepAudioEditXAudioEditorNode",
    "StepAudioEditXEngineNode",
    "StringMultilineTagEditor",
    "TextToSRTBuilderNode",
    "UnifiedASRTranscribeNode",
    "UnifiedModelTrainingNode",
    "UnifiedSoundEffectsNode",
    "UnifiedTTSSRTNode",
    "UnifiedTTSTextNode",
    "UnifiedVoiceChangerNode",
    "UnifiedVoiceDesignerNode",
    "VibeVoiceEngineNode",
    "VisemeDetectionOptionsNode",
    "VocalRemovalNode",
    "VoiceFixerNode",
    "TTSExternalGPTSovitsEngine",
    "TTSExternalIndexTTSEngine",
    "TTSExternalCosyVoiceEngine",
    "TTSExternalAudioAsset",
}

TARGETS_PROFILE_NODE_IDS = {
    "TTSExternalGPTSovitsEngine",
    "TTSExternalIndexTTSEngine",
    "TTSExternalCosyVoiceEngine",
    "TTSExternalAudioAsset",
    "UnifiedTTSTextNode",
}

OPTIONAL_ENGINE_NODE_IDS = {
    "ChatterBoxEngineNode",
    "F5TTSEngineNode",
    "GPTSovitsEngineNode",
    "IndexTTSEngineNode",
    "CosyVoiceEngineNode",
}


@pytest.mark.unit
def test_plugin_loader_probe_keeps_the_parent_process_unchanged():
    module_names_before = set(sys.modules)
    sys_path_before = list(sys.path)
    numpy_module_before = sys.modules.get("numpy")
    missing = object()
    numpy_bool_before = getattr(numpy_module_before, "bool", missing)

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        cwd=REPO_ROOT,
        env={**os.environ, "COMFYUI_TESTING": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    missing_node_ids = sorted(EXPECTED_NODE_IDS - set(payload["node_ids"]))
    assert not missing_node_ids, f"nodes.py did not register expected node IDs: {missing_node_ids}"
    assert set(sys.modules) == module_names_before
    assert sys.path == sys_path_before
    assert sys.modules.get("numpy") is numpy_module_before
    assert getattr(numpy_module_before, "bool", missing) is numpy_bool_before
    if importlib.util.find_spec("scipy") is not None:
        from scipy.io import wavfile

        assert callable(wavfile.write)


@pytest.mark.unit
def test_plugin_loader_probe_preserves_the_child_numpy_bool_alias():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        cwd=REPO_ROOT,
        env={**os.environ, "COMFYUI_TESTING": "1"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["numpy_bool_unchanged"] is True


@pytest.mark.unit
def test_plugin_loader_probe_cleans_up_its_temporary_directories(tmp_path: Path):
    temporary_root = tmp_path / "probe-temp"
    temporary_root.mkdir()

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "COMFYUI_TESTING": "1",
            "TEMP": str(temporary_root),
            "TMP": str(temporary_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert list(temporary_root.iterdir()) == []


@pytest.mark.unit
def test_targets_profile_registers_bridge_and_dependency_clean_workflow_nodes():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "COMFYUI_TESTING": "1",
            "TTS_AUDIO_SUITE_INSTALL_PROFILE": "tts_more_targets",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert set(payload["node_ids"]) == TARGETS_PROFILE_NODE_IDS
    assert not OPTIONAL_ENGINE_NODE_IDS.intersection(payload["node_ids"])


def _install_comfyui_test_stubs(probe_root: Path):
    for module_name in (
        "comfy",
        "comfy.model_management",
        "comfy.utils",
        "nodes",
        "server",
        "execution",
        "comfy_extras",
    ):
        sys.modules[module_name] = MagicMock()

    folder_paths = types.ModuleType("folder_paths")
    for directory_name in ("models", "input", "output", "temp"):
        (probe_root / directory_name).mkdir()
    folder_paths.models_dir = str(probe_root / "models")
    folder_paths.input_directory = str(probe_root / "input")
    folder_paths.output_directory = str(probe_root / "output")
    folder_paths.temp_directory = str(probe_root / "temp")
    folder_paths.get_folder_paths = lambda *args, **kwargs: []
    folder_paths.add_model_folder_path = lambda *args, **kwargs: None
    folder_paths.get_input_directory = lambda: folder_paths.input_directory
    folder_paths.get_output_directory = lambda: folder_paths.output_directory
    folder_paths.get_temp_directory = lambda: folder_paths.temp_directory
    folder_paths.get_annotated_filepath = lambda path: path
    folder_paths.exists_annotated_filepath = lambda path: os.path.exists(path)
    sys.modules["folder_paths"] = folder_paths


def _install_optional_dependency_stubs():
    """Keep the plugin loader test independent of optional engine packages."""
    audio_separator = types.ModuleType("audio_separator")
    audio_separator_separator = types.ModuleType("audio_separator.separator")
    audio_separator_separator.Separator = type("Separator", (), {})
    audio_separator.separator = audio_separator_separator
    sys.modules["audio_separator"] = audio_separator
    sys.modules["audio_separator.separator"] = audio_separator_separator

    rvc_audio = types.ModuleType("rvc_audio")
    rvc_audio.audio_to_bytes = lambda *args, **kwargs: b""
    rvc_audio.save_input_audio = lambda *args, **kwargs: None
    rvc_audio.load_input_audio = lambda *args, **kwargs: None
    rvc_audio.get_audio = lambda *args, **kwargs: (None, 0)
    sys.modules["rvc_audio"] = rvc_audio

    rvc_utils = types.ModuleType("rvc_utils")
    rvc_utils.get_filenames = lambda *args, **kwargs: []
    rvc_utils.get_hash = lambda *args, **kwargs: ""
    rvc_utils.get_optimal_torch_device = lambda *args, **kwargs: "cpu"
    sys.modules["rvc_utils"] = rvc_utils

    rvc_downloader = types.ModuleType("rvc_downloader")
    for name in (
        "KARAFAN_MODELS",
        "MDX_MODELS",
        "VR_MODELS",
        "ZFTURBO_MODELS",
        "MELBAND_MODELS",
    ):
        setattr(rvc_downloader, name, [])
    rvc_downloader.RVC_DOWNLOAD_LINK = ""
    rvc_downloader.ZFTURBO_DOWNLOAD_LINK = ""
    rvc_downloader.MELBAND_DOWNLOAD_LINK = ""
    rvc_downloader.download_file = lambda *args, **kwargs: None
    sys.modules["rvc_downloader"] = rvc_downloader

    rvc_lib = types.ModuleType("lib")
    rvc_karafan = types.ModuleType("lib.karafan")
    rvc_lib.karafan = rvc_karafan
    sys.modules["lib"] = rvc_lib
    sys.modules["lib.karafan"] = rvc_karafan


def _probe_node_ids():
    with tempfile.TemporaryDirectory(prefix="tts_suite_probe_") as temporary_root:
        _install_comfyui_test_stubs(Path(temporary_root))
        _install_optional_dependency_stubs()
        spec = importlib.util.spec_from_file_location(
            "tts_audio_suite_all_node_registration_probe", REPO_ROOT / "nodes.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return sorted(module.NODE_CLASS_MAPPINGS)


def _run_probe():
    logs = io.StringIO()
    import numpy as np

    numpy_bool_before = getattr(np, "bool", None)
    try:
        with redirect_stdout(logs), redirect_stderr(logs):
            node_ids = _probe_node_ids()
    except Exception:
        print(f"{traceback.format_exc()}Captured loader logs:\n{logs.getvalue()}", file=sys.stderr)
        return 1
    print(json.dumps({
        "node_ids": node_ids,
        "numpy_bool_unchanged": getattr(np, "bool", None) is numpy_bool_before,
    }))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit("expected --probe")
    raise SystemExit(_run_probe())
