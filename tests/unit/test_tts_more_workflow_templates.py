from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict[str, object]:
    path = ROOT / "example_workflows" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_tts_more_templates_have_stable_api_prompt_graphs() -> None:
    templates = {
        "TTS More - GPT-SoVITS API.json": ("TTSExternalGPTSovitsEngine", "gpt-sovits-local"),
        "TTS More - IndexTTS API.json": ("TTSExternalIndexTTSEngine", "indextts-local"),
        "TTS More - CosyVoice API.json": ("TTSExternalCosyVoiceEngine", "cosyvoice-local"),
    }

    for filename, (engine_class, resource_id) in templates.items():
        graph = _load(filename)
        assert set(graph) == {"1", "2", "3", "4"}
        assert graph["1"]["class_type"] == engine_class
        assert graph["1"]["inputs"]["resource_id"] == resource_id
        assert graph["2"]["class_type"] == "TTSExternalAudioAsset"
        assert graph["3"]["class_type"] == "UnifiedTTSTextNode"
        assert graph["3"]["inputs"]["TTS_engine"] == ["1", 0]
        assert graph["3"]["inputs"]["opt_narrator"] == ["2", 0]
        assert graph["4"]["class_type"] == "SaveAudio"
        assert graph["4"]["inputs"]["audio"] == ["3", 0]


def test_tts_more_templates_use_explicit_placeholders() -> None:
    graph = _load("TTS More - GPT-SoVITS API.json")
    assert graph["2"]["inputs"]["asset_id"] == "replace-with-uploaded-asset-id"
    assert isinstance(graph["3"]["inputs"]["text"], str)
    assert graph["3"]["inputs"]["text"].strip()
