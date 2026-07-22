"""Real target-branch lifecycle tests for wrapped external narrator voices."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import threading
import wave

import pytest
import torch

from api_bridge.assets import AssetInUseError, AudioAssetStore, pin_voice_asset
from api_bridge.runtime_registry import RuntimeHandle, RuntimeRegistry


ROOT = Path(__file__).parents[2]


def _wav_bytes() -> bytes:
    stream = io.BytesIO()
    with wave.open(stream, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0\0" * 160)
    return stream.getvalue()


def _load_node(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "nodes" / "unified" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine_type", ["gpt_sovits", "index_tts", "cosyvoice"])
@pytest.mark.parametrize("raises", [False, True])
def test_text_target_branches_pin_wrapped_external_voice_through_success_and_error(tmp_path, monkeypatch, engine_type, raises):
    module = _load_node("tts_text_node.py", f"wrapped_text_{engine_type}_{raises}")
    store = AudioAssetStore(tmp_path)
    asset = store.create("voice.wav", _wav_bytes())
    registry = RuntimeRegistry()
    started, finish = threading.Event(), threading.Event()

    class Engine:
        sample_rate = 32000
        _api_bridge_runtime_key = f"text:{engine_type}"

        def process_text(self, **kwargs):
            started.set(); assert finish.wait(timeout=3)
            if raises: raise RuntimeError("generation failed")
            return torch.ones(1, 32), {}

        def generate_tts_audio(self, **kwargs):
            started.set(); assert finish.wait(timeout=3)
            if raises: raise RuntimeError("generation failed")
            return ({"waveform": torch.ones(1, 1, 32), "sample_rate": 24000}, "ok")

    engine = Engine()
    registry.register(RuntimeHandle.create(engine._api_bridge_runtime_key, engine_type, "resource", "cpu", lambda: None))
    node = module.UnifiedTTSTextNode()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)
    monkeypatch.setattr(module, "pin_voice_asset", lambda voice: pin_voice_asset(voice, store=store))
    monkeypatch.setattr(node, "_create_proper_engine_node_instance", lambda _: engine)
    monkeypatch.setattr(node, "_get_voice_reference", lambda *_: (str(asset.path), {"waveform": torch.zeros(1, 1, 8), "sample_rate": 16000}, "ref", "external"))
    config = {"resource_id": "resource", "device": "cpu", "gpt_weight": "g", "sovits_weight": "s", "model_path": "m", "use_fp16": False}
    errors = []
    worker = threading.Thread(target=lambda: _call_text(node, engine_type, config, [{"asset_id": asset.asset_id}], errors))
    worker.start(); assert started.wait(timeout=3)
    with pytest.raises(AssetInUseError): store.delete(asset.asset_id)
    finish.set(); worker.join(timeout=3); assert not worker.is_alive()
    store.delete(asset.asset_id)


def _call_text(node, engine_type, config, voice, errors):
    try:
        node.generate_speech({"engine_type": engine_type, "config": config}, "hello", "none", 7, opt_narrator=voice)
    except Exception as exc:
        errors.append(exc)


@pytest.mark.parametrize("engine_type", ["index_tts", "cosyvoice"])
@pytest.mark.parametrize("raises", [False, True])
def test_srt_target_branches_pin_wrapped_external_voice_through_success_and_error(tmp_path, monkeypatch, engine_type, raises):
    module = _load_node("tts_srt_node.py", f"wrapped_srt_{engine_type}_{raises}")
    store = AudioAssetStore(tmp_path); asset = store.create("voice.wav", _wav_bytes())
    registry = RuntimeRegistry(); started, finish = threading.Event(), threading.Event()

    class Processor:
        def process_srt_content(self, **kwargs):
            started.set(); assert finish.wait(timeout=3)
            if raises: raise RuntimeError("generation failed")
            return ({"waveform": torch.ones(1, 1, 32), "sample_rate": 24000}, "ok", "timing", "srt")

    class Engine:
        _api_bridge_runtime_key = f"srt:{engine_type}"
        processor = Processor()

    engine = Engine(); registry.register(RuntimeHandle.create(engine._api_bridge_runtime_key, engine_type, "resource", "cpu", lambda: None))
    node = module.UnifiedTTSSRTNode()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)
    monkeypatch.setattr(module, "pin_voice_asset", lambda voice: pin_voice_asset(voice, store=store))
    monkeypatch.setattr(node, "_create_proper_engine_node_instance", lambda _: engine)
    srt_audio = None if engine_type == "cosyvoice" else {"waveform": torch.zeros(1, 1, 8), "sample_rate": 16000}
    monkeypatch.setattr(node, "_get_voice_reference", lambda *_: (str(asset.path), srt_audio, "ref", "external"))
    config = {"resource_id": "resource", "device": "cpu", "model_path": "m", "use_fp16": False}
    errors = []
    worker = threading.Thread(target=lambda: _call_srt(node, engine_type, config, [{"asset_id": asset.asset_id}], errors))
    worker.start(); assert started.wait(timeout=3)
    with pytest.raises(AssetInUseError): store.delete(asset.asset_id)
    finish.set(); worker.join(timeout=3); assert not worker.is_alive()
    store.delete(asset.asset_id)


def _call_srt(node, engine_type, config, voice, errors):
    try:
        node.generate_srt_speech({"engine_type": engine_type, "config": config}, "1\n00:00:00,000 --> 00:00:01,000\nHi", "none", 7, "natural", opt_narrator=voice)
    except Exception as exc:
        errors.append(exc)
