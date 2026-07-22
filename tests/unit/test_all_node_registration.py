import importlib.util
from pathlib import Path
import sys
import types

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


def _install_optional_dependency_stubs(monkeypatch):
    """Keep the plugin loader test independent of optional engine packages."""
    audio_separator = types.ModuleType("audio_separator")
    audio_separator_separator = types.ModuleType("audio_separator.separator")
    audio_separator_separator.Separator = type("Separator", (), {})
    audio_separator.separator = audio_separator_separator
    monkeypatch.setitem(sys.modules, "audio_separator", audio_separator)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", audio_separator_separator)

    scipy = types.ModuleType("scipy")
    scipy_io = types.ModuleType("scipy.io")
    scipy_wavfile = types.ModuleType("scipy.io.wavfile")
    scipy_wavfile.write = lambda *args, **kwargs: None
    scipy_io.wavfile = scipy_wavfile
    scipy.io = scipy_io
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", scipy_io)
    monkeypatch.setitem(sys.modules, "scipy.io.wavfile", scipy_wavfile)

    rvc_audio = types.ModuleType("rvc_audio")
    rvc_audio.audio_to_bytes = lambda *args, **kwargs: b""
    rvc_audio.save_input_audio = lambda *args, **kwargs: None
    rvc_audio.load_input_audio = lambda *args, **kwargs: None
    rvc_audio.get_audio = lambda *args, **kwargs: (None, 0)
    monkeypatch.setitem(sys.modules, "rvc_audio", rvc_audio)

    rvc_utils = types.ModuleType("rvc_utils")
    rvc_utils.get_filenames = lambda *args, **kwargs: []
    rvc_utils.get_hash = lambda *args, **kwargs: ""
    rvc_utils.get_optimal_torch_device = lambda *args, **kwargs: "cpu"
    monkeypatch.setitem(sys.modules, "rvc_utils", rvc_utils)

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
    monkeypatch.setitem(sys.modules, "rvc_downloader", rvc_downloader)

    rvc_lib = types.ModuleType("lib")
    rvc_karafan = types.ModuleType("lib.karafan")
    rvc_lib.karafan = rvc_karafan
    monkeypatch.setitem(sys.modules, "lib", rvc_lib)
    monkeypatch.setitem(sys.modules, "lib.karafan", rvc_karafan)


@pytest.mark.unit
def test_plugin_loader_registers_the_complete_tts_node_surface(monkeypatch):
    _install_optional_dependency_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "tts_audio_suite_all_node_registration_test", REPO_ROOT / "nodes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    missing = sorted(EXPECTED_NODE_IDS - set(module.NODE_CLASS_MAPPINGS))
    assert not missing, f"nodes.py did not register expected node IDs: {missing}"
