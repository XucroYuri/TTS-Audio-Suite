import importlib.util
from pathlib import Path
import sys

import pytest

from api_bridge.models import TTSResource


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_PATH = REPO_ROOT / "nodes" / "api_bridge" / "resource_engine_nodes.py"
BRIDGE_SPEC = importlib.util.spec_from_file_location("api_bridge_resource_engine_nodes_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(BRIDGE_SPEC)
sys.modules[BRIDGE_SPEC.name] = bridge
BRIDGE_SPEC.loader.exec_module(bridge)


class FakeRegistry:
    def require(self, resource_id: str, engine: str) -> TTSResource:
        assert resource_id == "local-resource"
        common = {
            "resource_id": resource_id,
            "engine": engine,
            "source_root": Path("C:/source"),
        }
        if engine == "gpt_sovits":
            return TTSResource(
                **common,
                gpt_weight=Path("C:/gpt.ckpt"),
                sovits_weight=Path("C:/sovits.pth"),
                bert_path=Path("C:/bert"),
                cnhubert_path=Path("C:/cnhubert"),
                version="v2ProPlus",
            )
        return TTSResource(**common, model_dir=Path(f"C:/{engine}/model"))


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(bridge, "get_resource_registry", lambda: FakeRegistry())


@pytest.mark.unit
@pytest.mark.parametrize(
    "node_class",
    [
        bridge.ExternalGPTSovitsEngineNode,
        bridge.ExternalIndexTTSEngineNode,
        bridge.ExternalCosyVoiceEngineNode,
    ],
)
def test_bridge_nodes_accept_only_a_resource_id_as_model_identity(node_class):
    inputs = node_class.INPUT_TYPES()

    assert inputs["required"] == {"resource_id": ("STRING", {"default": ""})}
    forbidden = {"source_root", "model_path", "gpt_weight", "sovits_weight", "bert_path", "cnhubert_path"}
    assert forbidden.isdisjoint(inputs["required"])
    assert forbidden.isdisjoint(inputs.get("optional", {}))


@pytest.mark.unit
def test_gpt_bridge_resolves_private_resource_and_preserves_version(fake_registry):
    (engine,) = bridge.ExternalGPTSovitsEngineNode().create_engine("local-resource")

    assert engine["engine_type"] == "gpt_sovits"
    assert engine["adapter_class"] == "GPTSovitsAdapter"
    assert engine["config"] == {
        "resource_id": "local-resource",
        "gpt_weight": str(Path("C:/gpt.ckpt")),
        "sovits_weight": str(Path("C:/sovits.pth")),
        "bert_path": str(Path("C:/bert")),
        "cnhubert_path": str(Path("C:/cnhubert")),
        "gpt_sovits_home": str(Path("C:/source")),
        "version": "v2ProPlus",
        "device": "auto",
        "use_fp16": True,
        "text_language": "zh",
        "ref_language": "zh",
        "how_to_cut": "凑四句一切",
        "speed": 1.0,
        "top_k": 15,
        "top_p": 1.0,
        "temperature": 1.0,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("node_class", "engine", "adapter", "home_key"),
    [
        (bridge.ExternalIndexTTSEngineNode, "index_tts", "IndexTTSAdapter", "index_tts_home"),
        (bridge.ExternalCosyVoiceEngineNode, "cosyvoice", "CosyVoiceAdapter", "cosyvoice_home"),
    ],
)
def test_non_gpt_bridge_config_uses_current_processor_contract(
    fake_registry, node_class, engine, adapter, home_key
):
    (engine_data,) = node_class().create_engine("local-resource")

    config = engine_data["config"]
    assert engine_data["engine_type"] == engine
    assert engine_data["adapter_class"] == adapter
    assert config["resource_id"] == "local-resource"
    assert config["model_path"] == str(Path(f"C:/{engine}/model"))
    assert config[home_key] == str(Path("C:/source"))
    assert config["device"] == "auto"
    assert config["use_fp16"] is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ui_value", "expected"),
    [("auto", None), ("true", True), ("false", False)],
)
def test_index_bridge_normalizes_cuda_kernel_choice_for_the_processor(fake_registry, ui_value, expected):
    (engine_data,) = bridge.ExternalIndexTTSEngineNode().create_engine(
        "local-resource", use_cuda_kernel=ui_value
    )

    assert engine_data["config"]["use_cuda_kernel"] is expected


@pytest.mark.unit
def test_bridge_rejects_a_resource_for_the_wrong_engine():
    class WrongEngineRegistry:
        def require(self, resource_id: str, engine: str):
            raise ValueError(f"resource {resource_id} belongs to index_tts, not {engine}")

    original = bridge.get_resource_registry
    bridge.get_resource_registry = lambda: WrongEngineRegistry()
    try:
        with pytest.raises(ValueError, match="belongs to index_tts, not gpt_sovits"):
            bridge.ExternalGPTSovitsEngineNode().create_engine("local-resource")
    finally:
        bridge.get_resource_registry = original


@pytest.mark.unit
def test_external_engine_ids_are_registered_through_the_plugin_loader():
    spec = importlib.util.spec_from_file_location("api_bridge_nodes_mapping_test", REPO_ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_CLASS_MAPPINGS["TTSExternalGPTSovitsEngine"] is module.ExternalGPTSovitsEngineNode
    assert module.NODE_CLASS_MAPPINGS["TTSExternalIndexTTSEngine"] is module.ExternalIndexTTSEngineNode
    assert module.NODE_CLASS_MAPPINGS["TTSExternalCosyVoiceEngine"] is module.ExternalCosyVoiceEngineNode
