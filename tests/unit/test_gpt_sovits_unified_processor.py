import torch

import importlib.util
from pathlib import Path

from engines.processors.gpt_sovits_processor import GPTSovitsProcessor


class FakeAdapter:
    def __init__(self):
        self.initialized = None
        self.generated = None
        self.unload_calls = 0

    def initialize_engine(self, **kwargs):
        self.initialized = kwargs

    def generate(self, **kwargs):
        self.generated = kwargs
        return torch.ones(1, 3200), 32000

    def unload(self):
        self.unload_calls += 1


def test_processor_initializes_generates_and_unloads():
    adapter = FakeAdapter()
    processor = GPTSovitsProcessor(
        {
            "gpt_weight": "gpt.ckpt",
            "sovits_weight": "sovits.pth",
            "version": "v2ProPlus",
            "bert_path": "bert",
            "cnhubert_path": "hubert",
            "device": "cuda",
            "use_fp16": True,
            "text_language": "zh",
            "ref_language": "zh",
            "speed": 1.25,
            "top_k": 9,
            "top_p": 0.8,
            "temperature": 0.7,
            "how_to_cut": "按标点符号切",
        },
        adapter=adapter,
    )

    audio, info = processor.process_text(
        text="测试文本",
        speaker_audio={"audio_path": "reference.wav"},
        reference_text="参考文本",
        seed=7,
        return_info=True,
    )
    processor.cleanup()

    assert audio.shape[-1] == 3200
    assert "GPT-SoVITS" in info
    assert adapter.initialized == {
        "gpt_weight": "gpt.ckpt",
        "sovits_weight": "sovits.pth",
        "version": "v2ProPlus",
        "bert_path": "bert",
        "cnhubert_path": "hubert",
        "device": "cuda",
        "use_fp16": True,
    }
    assert adapter.generated == {
        "text": "测试文本",
        "text_lang": "zh",
        "ref_audio_path": "reference.wav",
        "ref_text": "参考文本",
        "ref_lang": "zh",
        "speed": 1.25,
        "top_k": 9,
        "top_p": 0.8,
        "temperature": 0.7,
        "how_to_cut": "按标点符号切",
        "seed": 7,
    }
    assert adapter.unload_calls == 1


def test_cleanup_is_idempotent_and_path_can_be_a_string():
    adapter = FakeAdapter()
    processor = GPTSovitsProcessor(
        {"gpt_weight": "gpt.ckpt", "sovits_weight": "sovits.pth"}, adapter=adapter
    )

    audio = processor.process_text(
        text="text",
        speaker_audio="reference.wav",
        reference_text="reference",
        seed=0,
    )
    processor.cleanup()
    processor.cleanup()

    assert audio.shape[-1] == 3200
    assert adapter.generated["ref_audio_path"] == "reference.wav"
    assert adapter.unload_calls == 1


def test_unified_node_passes_reference_path_text_and_seed_to_gpt_processor(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_text_node.py"
    spec = importlib.util.spec_from_file_location("gpt_sovits_unified_test_node", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeProcessor:
        def __init__(self):
            self.call = None

        def process_text(self, **kwargs):
            self.call = kwargs
            return torch.ones(1, 1600), "fake GPT-SoVITS info"

    processor = FakeProcessor()
    node = module.UnifiedTTSTextNode()
    monkeypatch.setattr(node, "_create_proper_engine_node_instance", lambda _: processor)
    monkeypatch.setattr(
        node,
        "_get_voice_reference",
        lambda *_: ("reference.wav", {"waveform": torch.zeros(1, 8)}, "参考文本", "narrator"),
    )

    _, info = node.generate_speech(
        {"engine_type": "gpt_sovits", "config": {"gpt_weight": "gpt.ckpt", "sovits_weight": "sovits.pth"}},
        text="测试文本",
        narrator_voice="none",
        seed=17,
    )

    assert "fake GPT-SoVITS info" in info
    assert processor.call == {
        "text": "测试文本",
        "speaker_audio": {"audio_path": "reference.wav"},
        "reference_text": "参考文本",
        "seed": 17,
        "return_info": True,
    }


def test_native_gpt_sovits_engine_node_is_registered():
    nodes_source = (Path(__file__).parents[2] / "nodes.py").read_text(encoding="utf-8")

    assert 'NODE_CLASS_MAPPINGS["GPTSovitsEngineNode"] = GPTSovitsEngineNode' in nodes_source
