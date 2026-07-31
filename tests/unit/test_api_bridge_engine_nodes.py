import importlib.util
import json
from pathlib import Path
import sys
import types
import wave

import pytest

from api_bridge.models import TTSResource
from api_bridge.runtime_registry import RuntimeRegistry


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
def test_index_processor_passes_registered_checkout_to_adapter(monkeypatch):
    import engines.processors.index_tts_processor as processor_module

    initialized = {}

    class FakeAdapter:
        def initialize_engine(self, **kwargs):
            initialized.update(kwargs)

    monkeypatch.setattr(processor_module, "IndexTTSAdapter", FakeAdapter)
    monkeypatch.setattr(processor_module.IndexTTSProcessor, "_setup_character_parser", lambda self: None)

    processor_module.IndexTTSProcessor(
        {
            "model_path": "C:/index/model",
            "index_tts_home": "C:/index/source",
            "device": "cuda",
        }
    )

    assert initialized["model_path"] == "C:/index/model"
    assert initialized["index_tts_home"] == "C:/index/source"


@pytest.mark.unit
def test_index_adapter_builds_engine_for_registered_checkout(monkeypatch):
    import engines.adapters.index_tts_adapter as adapter_module

    initialized = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            initialized.update(kwargs)

    monkeypatch.setattr(adapter_module, "IndexTTSEngine", FakeEngine)
    adapter = adapter_module.IndexTTSAdapter.__new__(adapter_module.IndexTTSAdapter)
    adapter.engine = None

    adapter.initialize_engine(
        model_path="C:/index/model",
        index_tts_home="C:/index/source",
        device="cuda",
    )

    assert initialized["model_dir"] == "C:/index/model"
    assert initialized["source_root"] == "C:/index/source"


def _load_external_index_subprocess_module():
    path = REPO_ROOT / "engines" / "index_tts" / "external_subprocess.py"
    assert path.is_file(), "plugin-owned external IndexTTS subprocess adapter is missing"
    spec = importlib.util.spec_from_file_location("external_index_subprocess_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_external_index_runtime(tmp_path):
    source_root = tmp_path / "index-source"
    model_dir = source_root / "checkpoints"
    inference_module = source_root / "indextts" / "infer_v2.py"
    inference_module.parent.mkdir(parents=True)
    inference_module.write_text("class IndexTTS2: pass\n", encoding="utf-8")
    python_executable = source_root / ".venv" / "Scripts" / "python.exe"
    python_executable.parent.mkdir(parents=True)
    python_executable.touch()
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    voice_path = source_root / "voice.wav"
    with wave.open(str(voice_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x01\x00" * 160)
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    return source_root, model_dir, python_executable, voice_path, temp_root


@pytest.mark.unit
def test_external_index_subprocess_uses_checkout_venv_and_cleans_temp(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, python_executable, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    observed = {}

    class FakeProcess:
        pid = 4242
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs

        def communicate(self, timeout):
            observed["timeout"] = timeout
            manifest = json.loads(Path(observed["command"][2]).read_text(encoding="utf-8"))
            observed["manifest"] = manifest
            with wave.open(manifest["output_path"], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(22050)
                handle.writeframes(b"\x10\x00" * 220)
            return ("official stdout", "")

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=321.0,
        temp_root=temp_root,
    )

    sample_rate, samples = proxy.infer(
        spk_audio_prompt=str(voice_path),
        text="真实外部推理。",
        output_path=None,
        do_sample=True,
        temperature=0.8,
    )

    assert observed["command"][0] == str(python_executable)
    assert Path(observed["command"][1]).name == "external_subprocess_runner.py"
    assert observed["kwargs"]["cwd"] == str(source_root)
    assert observed["timeout"] == 321.0
    assert observed["manifest"]["source_root"] == str(source_root)
    assert observed["manifest"]["model_dir"] == str(model_dir)
    assert observed["manifest"]["inference"]["text"] == "真实外部推理。"
    assert sample_rate == 22050
    assert samples.shape == (220, 1)
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_index_subprocess_propagates_exit_stderr_and_cleans_temp(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)

    class FailedProcess:
        pid = 4343
        returncode = 4

        def __init__(self, command, **kwargs):
            pass

        def communicate(self, timeout=None):
            return ("", "official inference exploded")

    monkeypatch.setattr(module.subprocess, "Popen", FailedProcess)
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        temp_root=temp_root,
    )

    with pytest.raises(RuntimeError) as error:
        proxy.infer(spk_audio_prompt=str(voice_path), text="失败传播。", output_path=None)

    assert str(error.value) == "External IndexTTS subprocess exited 4: official inference exploded"
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_index_subprocess_terminates_tree_on_timeout_and_cleans_temp(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    terminated = []

    class TimedOutProcess:
        pid = 4444
        returncode = None

        def __init__(self, command, **kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise module.subprocess.TimeoutExpired("indextts", timeout)
            return ("", "child stopped")

    monkeypatch.setattr(module.subprocess, "Popen", TimedOutProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(lambda process: terminated.append(process.pid)),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.5,
        temp_root=temp_root,
    )

    with pytest.raises(TimeoutError, match="exceeded 0.5s"):
        proxy.infer(spk_audio_prompt=str(voice_path), text="超时清理。", output_path=None)

    assert terminated == [4444]
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_index_engine_uses_external_subprocess_for_registered_checkout(monkeypatch, tmp_path):
    import engines.index_tts.index_tts as engine_module

    source_root = tmp_path / "source"
    model_dir = tmp_path / "model"
    source_root.mkdir()
    model_dir.mkdir()
    created = {}

    class FakeExternalProxy:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(
        engine_module,
        "ExternalIndexTTSSubprocessProxy",
        FakeExternalProxy,
        raising=False,
    )
    engine = engine_module.IndexTTSEngine(
        model_dir=str(model_dir),
        source_root=str(source_root),
        device="cuda",
        use_fp16=True,
    )

    engine._ensure_model_loaded()

    assert isinstance(engine._tts_engine, FakeExternalProxy)
    assert created["source_root"] == str(source_root)
    assert created["model_dir"] == str(model_dir)
    assert str(created["device"]).startswith("cuda")
    assert created["use_fp16"] is True


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


def _load_unified_node(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "nodes" / "unified" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _force_cache_valid(monkeypatch):
    from utils.models import comfyui_model_wrapper

    monkeypatch.setattr(comfyui_model_wrapper, "is_engine_cache_valid", lambda _: True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [("resource_id", "gpt-resource-b"), ("version", "v3")],
)
def test_text_cache_recreates_gpt_processor_when_resource_identity_changes(
    monkeypatch, changed_key, changed_value
):
    module = _load_unified_node("tts_text_node.py", f"gpt_cache_key_test_{changed_key}")
    _force_cache_valid(monkeypatch)
    created = []

    class FakeProcessor:
        def __init__(self, config):
            self.config = config
            created.append(self)

        def update_config(self, config):
            self.config = config

    import engines.processors.gpt_sovits_processor as processor_module

    monkeypatch.setattr(processor_module, "GPTSovitsProcessor", FakeProcessor)
    node = module.UnifiedTTSTextNode()
    first = {
        "engine_type": "gpt_sovits",
        "config": {
            "resource_id": "gpt-resource-a",
            "version": "v2",
            "gpt_weight": "gpt.ckpt",
            "sovits_weight": "sovits.pth",
            "gpt_sovits_home": "C:/gpt",
        },
    }
    second = {"engine_type": "gpt_sovits", "config": dict(first["config"], **{changed_key: changed_value})}

    assert node._create_proper_engine_node_instance(first) is not node._create_proper_engine_node_instance(second)
    assert len(created) == 2


_INDEX_LOAD_KEYS = [
    ("resource_id", "index-resource-b"),
    ("model_path", "C:/index-b"),
    ("index_tts_home", "C:/source-b"),
    ("use_fp16", False),
    ("use_cuda_kernel", True),
    ("use_deepspeed", True),
    ("use_torch_compile", True),
    ("use_accel", True),
    ("low_vram", True),
]


def _index_engine_data(changed_key=None, changed_value=None):
    config = {
        "resource_id": "index-resource-a",
        "model_path": "C:/index-a",
        "index_tts_home": "C:/source-a",
        "device": "cpu",
        "use_fp16": True,
        "use_cuda_kernel": None,
        "use_deepspeed": False,
        "use_torch_compile": False,
        "use_accel": False,
        "low_vram": False,
    }
    if changed_key is not None:
        config[changed_key] = changed_value
    return {"engine_type": "index_tts", "config": config}


@pytest.mark.unit
@pytest.mark.parametrize(("changed_key", "changed_value"), _INDEX_LOAD_KEYS)
def test_text_cache_recreates_index_processor_for_each_load_identity(
    monkeypatch, changed_key, changed_value
):
    module = _load_unified_node("tts_text_node.py", f"index_text_cache_key_test_{changed_key}")
    _force_cache_valid(monkeypatch)
    created = []

    class FakeProcessor:
        def __init__(self, config):
            self.config = config
            created.append(self)

        def update_config(self, config):
            self.config = config

    import engines.processors.index_tts_processor as processor_module

    monkeypatch.setattr(processor_module, "IndexTTSProcessor", FakeProcessor)
    node = module.UnifiedTTSTextNode()

    assert node._create_proper_engine_node_instance(_index_engine_data()) is not node._create_proper_engine_node_instance(
        _index_engine_data(changed_key, changed_value)
    )
    assert len(created) == 2


@pytest.mark.unit
@pytest.mark.parametrize(("changed_key", "changed_value"), _INDEX_LOAD_KEYS)
def test_srt_cache_recreates_index_processor_for_each_load_identity(
    monkeypatch, changed_key, changed_value
):
    module = _load_unified_node("tts_srt_node.py", f"index_srt_cache_key_test_{changed_key}")
    _force_cache_valid(monkeypatch)
    created = []

    class FakeSRTProcessor:
        def __init__(self, wrapper, config):
            self.config = config
            created.append(self)

        def update_config(self, config):
            self.config = config

    fake_module = types.SimpleNamespace(IndexTTSSRTProcessor=FakeSRTProcessor)
    fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *_: fake_spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda _: fake_module)
    node = module.UnifiedTTSSRTNode()

    assert node._create_proper_engine_node_instance(_index_engine_data()) is not node._create_proper_engine_node_instance(
        _index_engine_data(changed_key, changed_value)
    )
    assert len(created) == 2


@pytest.mark.unit
def test_text_cache_recreates_cosyvoice_wrapper_when_resource_identity_changes(monkeypatch):
    module = _load_unified_node("tts_text_node.py", "cosy_text_resource_cache_key_test")
    _force_cache_valid(monkeypatch)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)
    node = module.UnifiedTTSTextNode()
    base = {
        "engine_type": "cosyvoice",
        "config": {
            "resource_id": "cosy-a",
            "model_path": "C:/cosy/model",
            "device": "cpu",
            "use_fp16": False,
        },
    }
    changed = {"engine_type": "cosyvoice", "config": dict(base["config"], resource_id="cosy-b")}

    assert node._create_proper_engine_node_instance(base) is not node._create_proper_engine_node_instance(changed)


@pytest.mark.unit
def test_srt_cache_recreates_cosyvoice_wrapper_when_resource_identity_changes(monkeypatch):
    module = _load_unified_node("tts_srt_node.py", "cosy_srt_resource_cache_key_test")
    _force_cache_valid(monkeypatch)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)

    class FakeSRTProcessor:
        def __init__(self, wrapper, config):
            self.config = config

        def update_config(self, config):
            self.config = config

        def cleanup(self):
            pass

    fake_module = types.SimpleNamespace(CosyVoiceSRTProcessor=FakeSRTProcessor)
    fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda _: None))
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *_: fake_spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda _: fake_module)
    node = module.UnifiedTTSSRTNode()
    base = {
        "engine_type": "cosyvoice",
        "config": {
            "resource_id": "cosy-a",
            "model_path": "C:/cosy/model",
            "device": "cpu",
            "use_fp16": False,
        },
    }
    changed = {"engine_type": "cosyvoice", "config": dict(base["config"], resource_id="cosy-b")}

    assert node._create_proper_engine_node_instance(base) is not node._create_proper_engine_node_instance(changed)
