import importlib.util
from pathlib import Path

import numpy as np
import torch

from engines.adapters.gpt_sovits_adapter import GPTSovitsAdapter
from utils.audio.cache import AudioCache


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


def test_adapter_uses_official_tts_runtime_and_converts_generator_output(monkeypatch, tmp_path):
    calls = {}

    class FakeConfig:
        def __init__(self, config):
            calls["config"] = config
            self.config = config

    class FakeTTS:
        def __init__(self, config):
            calls["runtime_config"] = config

        def run(self, inputs):
            calls["inputs"] = inputs
            yield 32000, np.array([0, 16384, -16384], dtype=np.int16)
            yield 32000, np.array([8192], dtype=np.int16)

    adapter = GPTSovitsAdapter()
    monkeypatch.setattr(adapter, "_import_official_runtime", lambda: (FakeConfig, FakeTTS))
    monkeypatch.setattr("engines.adapters.gpt_sovits_adapter.resolve_torch_device", lambda _: "cpu")

    adapter.initialize_engine(
        gpt_weight="gpt.ckpt",
        sovits_weight="sovits.pth",
        bert_path="bert",
        cnhubert_path="hubert",
        device="auto",
        use_fp16=True,
        gpt_sovits_home=str(tmp_path),
        version="v2Pro",
    )
    audio, sample_rate = adapter.generate(
        text="测试",
        text_lang="zh",
        ref_audio_path="reference.wav",
        ref_text="参考",
        ref_lang="zh",
        speed=1.2,
        top_k=9,
        top_p=0.8,
        temperature=0.7,
        how_to_cut="按标点符号切",
        seed=42,
    )

    assert calls["config"] == {
        "custom": {
            "device": "cpu",
            "is_half": False,
            "version": "v2Pro",
            "t2s_weights_path": "gpt.ckpt",
            "vits_weights_path": "sovits.pth",
            "bert_base_path": "bert",
            "cnhuhbert_base_path": "hubert",
        }
    }
    assert calls["inputs"] == {
        "text": "测试",
        "text_lang": "zh",
        "ref_audio_path": "reference.wav",
        "prompt_text": "参考",
        "prompt_lang": "zh",
        "top_k": 9,
        "top_p": 0.8,
        "temperature": 0.7,
        "text_split_method": "cut5",
        "speed_factor": 1.2,
        "seed": 42,
    }
    assert sample_rate == 32000
    assert audio.shape == (1, 4)
    assert torch.allclose(audio, torch.tensor([[0.0, 0.5, -0.5, 0.25]]))
    adapter.unload()
    assert adapter.runtime is None


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

    assert "from TTS_infer_pack.TTS import TTS, TTS_Config" in adapter_source
    assert "self.runtime.run(inputs)" in adapter_source
    assert "inference_webui" not in adapter_source
    assert "wordsegment" in requirements


def test_nodes_module_registers_the_native_gpt_sovits_engine():
    nodes_path = Path(__file__).parents[2] / "nodes.py"
    spec = importlib.util.spec_from_file_location("gpt_sovits_nodes_mapping_test", nodes_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS["GPTSovitsEngineNode"] is module.GPTSovitsEngineNode
