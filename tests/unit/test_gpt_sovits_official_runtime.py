import importlib.util
from pathlib import Path
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from engines.adapters.gpt_sovits_adapter import GPTSovitsAdapter
from engines.gpt_sovits import runtime as runtime_module
from engines.gpt_sovits.external_subprocess_runner import _configure_official_speaker_encoder
from utils.audio.cache import AudioCache


_RUNTIME_MODULE_NAMES = ("TTS_infer_pack.TTS", "TTS_infer_pack", "sv", "ERes2NetV2")
_MISSING_MODULE = object()


@pytest.fixture(autouse=True)
def _isolate_gpt_sovits_runtime_state():
    """Restore process-global checkout state even if a runtime assertion fails."""
    previous_modules = {
        module_name: sys.modules.get(module_name, _MISSING_MODULE)
        for module_name in _RUNTIME_MODULE_NAMES
    }
    runtime_module.reset_gpt_sovits_checkout_for_tests()
    try:
        yield
    finally:
        runtime_module.reset_gpt_sovits_checkout_for_tests()
        for module_name, previous_module in previous_modules.items():
            if previous_module is _MISSING_MODULE:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def test_gpt_sovits_cache_key_covers_every_generation_input():
    cache = AudioCache()
    base = {
        "gpt_weight": "gpt.ckpt",
        "sovits_weight": "sovits.pth",
        "ref_audio_path": "reference.wav",
        "ref_text": "reference text",
        "ref_lang": "zh",
        "text_lang": "zh",
        "text": "hello",
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
        "speed": 1.0,
        "how_to_cut": "凑四句一切",
        "seed": 7,
    }

    base_key = cache.generate_cache_key("gpt_sovits", **base)
    for name, value in {
        "gpt_weight": "other.ckpt",
        "sovits_weight": "other.pth",
        "ref_audio_path": "other.wav",
        "ref_text": "other text",
        "ref_lang": "en",
        "text_lang": "en",
        "text": "other",
        "top_k": 16,
        "top_p": 0.9,
        "temperature": 0.9,
        "speed": 1.1,
        "how_to_cut": "不切",
        "seed": 8,
    }.items():
        changed = {**base, name: value}
        assert cache.generate_cache_key("gpt_sovits", **changed) != base_key, name


def test_official_runner_can_bind_cwd_relative_speaker_encoder_to_registered_path(tmp_path, monkeypatch):
    speaker_encoder = tmp_path / "pretrained_eres2net.ckpt"
    speaker_encoder.write_bytes(b"checkpoint")
    module = SimpleNamespace(sv_path="relative/path.ckpt")
    monkeypatch.setitem(sys.modules, "sv", module)

    _configure_official_speaker_encoder(str(speaker_encoder))

    assert module.sv_path == str(speaker_encoder.resolve())


def test_native_engine_resolves_home_weights_when_comfy_directory_is_empty(tmp_path):
    home = tmp_path / "gpt-sovits"
    gpt_dir = home / "GPT_weights_v2"
    sovits_dir = home / "SoVITS_weights_v2"
    pretrained = home / "GPT_SoVITS" / "pretrained_models"
    gpt_dir.mkdir(parents=True)
    sovits_dir.mkdir()
    pretrained.mkdir(parents=True)
    (gpt_dir / "voice-e1.ckpt").touch()
    (sovits_dir / "voice_e1_s1.pth").touch()

    node_path = Path(__file__).parents[2] / "nodes" / "engines" / "gpt_sovits_engine_node.py"
    spec = importlib.util.spec_from_file_location("gpt_sovits_home_test_node", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.GPTSovitsEngineNode()

    weight_pair = node.INPUT_TYPES()["required"]["weight_pair"][1]["default"]
    engine_data, = node.create_engine_adapter(
        weight_pair=weight_pair,
        gpt_sovits_home=str(home),
        device="cpu",
        use_fp16=True,
    )

    config = engine_data["config"]
    assert config["gpt_weight"] == str(gpt_dir / "voice-e1.ckpt")
    assert config["sovits_weight"] == str(sovits_dir / "voice_e1_s1.pth")
    assert config["device"] == "cpu"
    assert config["use_fp16"] is False


def test_official_runtime_contract_and_lightweight_dependency_are_declared():
    adapter_source = Path(__file__).parents[2].joinpath("engines", "adapters", "gpt_sovits_adapter.py").read_text(encoding="utf-8")
    requirements = Path(__file__).parents[2].joinpath("requirements.txt").read_text(encoding="utf-8")

    assert 'importlib.import_module("TTS_infer_pack.TTS")' in adapter_source
    assert "self.runtime.run(inputs)" in adapter_source
    assert "inference_webui" not in adapter_source
    assert "wordsegment" in requirements


def test_nodes_module_registers_the_native_gpt_sovits_engine():
    nodes_path = Path(__file__).parents[2] / "nodes.py"
    spec = importlib.util.spec_from_file_location("gpt_sovits_nodes_mapping_test", nodes_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS["GPTSovitsEngineNode"] is module.GPTSovitsEngineNode


def test_second_checkout_is_rejected_without_rewriting_first_modules(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for checkout in (first, second):
        (checkout / "GPT_SoVITS" / "eres2net").mkdir(parents=True)

    first_context = runtime_module.configure_gpt_sovits_source(str(first))
    marker = object()
    sys.modules["TTS_infer_pack.TTS"] = marker

    with pytest.raises(RuntimeError, match="already bound"):
        runtime_module.configure_gpt_sovits_source(str(second))

    assert sys.modules["TTS_infer_pack.TTS"] is marker
    assert first_context.checkout_root == first.resolve()


def test_test_reset_removes_only_the_bound_checkout_import_paths(tmp_path):
    checkout = tmp_path / "checkout"
    package_root = checkout / "GPT_SoVITS"
    eres2net_root = package_root / "eres2net"
    eres2net_root.mkdir(parents=True)

    runtime_module.configure_gpt_sovits_source(str(checkout))

    injected_paths = {str(checkout), str(package_root), str(eres2net_root)}
    assert injected_paths.issubset(sys.path)

    runtime_module.reset_gpt_sovits_checkout_for_tests()

    assert not injected_paths.intersection(sys.path)
