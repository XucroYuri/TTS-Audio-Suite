import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
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


def _checkout_python(source_root: Path) -> Path:
    scripts_directory = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    return source_root / ".venv" / scripts_directory / executable_name


def _path_without_windows_extended_prefix(value: str) -> Path:
    if os.name != "nt":
        return Path(value)
    if value.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    if value.startswith("\\\\?\\"):
        return Path(value[4:])
    return Path(value)


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


@pytest.mark.unit
def test_registered_index_adapter_does_not_return_global_audio_cache(monkeypatch):
    import engines.adapters.index_tts_adapter as adapter_module

    generated = []
    cached = []

    class FakeExternalEngine:
        source_root = "C:/index/source"

        def generate(self, **kwargs):
            generated.append(kwargs)
            return adapter_module.torch.ones(1, 2205)

        def unload(self):
            pass

    class FakeAudioCache:
        def generate_cache_key(self, *args, **kwargs):
            return "existing-result"

        def get_cached_audio(self, cache_key):
            return (adapter_module.torch.zeros(1, 2205), 0.1)

        def cache_audio(self, *args):
            cached.append(args)

    adapter = adapter_module.IndexTTSAdapter.__new__(adapter_module.IndexTTSAdapter)
    adapter.engine = FakeExternalEngine()
    adapter.audio_cache = FakeAudioCache()

    audio = adapter.generate(text="must execute", seed=1)

    assert len(generated) == 1
    assert adapter_module.torch.count_nonzero(audio).item() == 2205
    assert cached == []


@pytest.mark.unit
def test_registered_cosyvoice_adapter_executes_external_engine_on_every_call():
    import engines.adapters.cosyvoice_adapter as adapter_module

    generated = []
    cache_reads = []
    cache_writes = []

    class FakeExternalEngine:
        source_root = "C:/cosy/source"

        def generate(self, **kwargs):
            generated.append(kwargs)
            return adapter_module.torch.full((1, 2400), float(len(generated)))

    class FakeAudioCache:
        def generate_cache_key(self, *args, **kwargs):
            return "coarse-existing-result"

        def get_cached_audio(self, cache_key):
            cache_reads.append(cache_key)
            return (adapter_module.torch.zeros(1, 2400), 0.1)

        def cache_audio(self, *args):
            cache_writes.append(args)

    adapter = adapter_module.CosyVoiceAdapter.__new__(adapter_module.CosyVoiceAdapter)
    adapter.engine = FakeExternalEngine()
    adapter.audio_cache = FakeAudioCache()
    adapter.model_variant = "standard"

    first = adapter.generate(text="must execute twice", mode="cross_lingual")
    second = adapter.generate(text="must execute twice", mode="cross_lingual")

    assert len(generated) == 2
    assert adapter_module.torch.count_nonzero(first == 1).item() == 2400
    assert adapter_module.torch.count_nonzero(second == 2).item() == 2400
    assert cache_reads == []
    assert cache_writes == []


@pytest.mark.unit
def test_gpt_adapter_uses_registered_checkout_runtime_without_inprocess_import(monkeypatch, tmp_path):
    import engines.adapters.gpt_sovits_adapter as adapter_module

    observed = {}

    class FakeExternalRuntime:
        source_root = tmp_path / "gpt-source"

        def __init__(self, **kwargs):
            observed.update(kwargs)

        def cleanup(self):
            pass

    monkeypatch.setattr(
        adapter_module,
        "ExternalGPTSovitsSubprocessProxy",
        FakeExternalRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        adapter_module.GPTSovitsAdapter,
        "_import_official_runtime",
        lambda self: (_ for _ in ()).throw(
            AssertionError("registered GPT-SoVITS must not import official modules in ComfyUI")
        ),
    )
    adapter = adapter_module.GPTSovitsAdapter()
    adapter.initialize_engine(
        gpt_weight="C:/gpt/s1.ckpt",
        sovits_weight="C:/gpt/s2.pth",
        bert_path="C:/gpt/bert",
        cnhubert_path="C:/gpt/cnhubert",
        device="cpu",
        use_fp16=False,
        gpt_sovits_home="C:/gpt/source",
        python_executable="D:/portable/python.exe",
        version="v2",
    )

    assert observed == {
        "source_root": "C:/gpt/source",
        "python_executable": "D:/portable/python.exe",
        "gpt_weight": "C:/gpt/s1.ckpt",
        "sovits_weight": "C:/gpt/s2.pth",
        "bert_path": "C:/gpt/bert",
        "cnhubert_path": "C:/gpt/cnhubert",
        "device": "cpu",
        "use_fp16": False,
        "version": "v2",
    }


@pytest.mark.unit
def test_gpt_adapter_uses_legacy_environment_checkout_when_home_is_omitted(monkeypatch, tmp_path):
    import engines.adapters.gpt_sovits_adapter as adapter_module
    from engines.gpt_sovits.runtime import reset_gpt_sovits_checkout_for_tests

    source_root = tmp_path / "legacy-gpt-source"
    (source_root / "GPT_SoVITS" / "eres2net").mkdir(parents=True)
    observed = {}

    class FakeExternalRuntime:
        def __init__(self, **kwargs):
            observed.update(kwargs)
            self.source_root = Path(kwargs["source_root"]).resolve()
            self.python_executable = _checkout_python(self.source_root)

        def cleanup(self):
            pass

    reset_gpt_sovits_checkout_for_tests()
    monkeypatch.setenv("GPT_SOVITS_PATH", str(source_root))
    monkeypatch.setattr(adapter_module, "ExternalGPTSovitsSubprocessProxy", FakeExternalRuntime)
    try:
        adapter_module.GPTSovitsAdapter().initialize_engine(
            gpt_weight="C:/gpt/s1.ckpt",
            sovits_weight="C:/gpt/s2.pth",
            bert_path="C:/gpt/bert",
            cnhubert_path="C:/gpt/cnhubert",
            device="cpu",
            use_fp16=False,
        )
    finally:
        reset_gpt_sovits_checkout_for_tests()

    assert observed["source_root"] == str(source_root.resolve())


@pytest.mark.unit
def test_gpt_adapter_requires_explicit_or_environment_checkout(monkeypatch):
    import engines.adapters.gpt_sovits_adapter as adapter_module
    from engines.gpt_sovits.runtime import reset_gpt_sovits_checkout_for_tests

    reset_gpt_sovits_checkout_for_tests()
    monkeypatch.delenv("GPT_SOVITS_PATH", raising=False)
    with pytest.raises(
        RuntimeError,
        match="gpt_sovits_home or GPT_SOVITS_PATH must point to an official GPT-SoVITS checkout",
    ):
        adapter_module.GPTSovitsAdapter().initialize_engine(
            gpt_weight="C:/gpt/s1.ckpt",
            sovits_weight="C:/gpt/s2.pth",
            bert_path="C:/gpt/bert",
            cnhubert_path="C:/gpt/cnhubert",
            device="cpu",
            use_fp16=False,
        )


@pytest.mark.unit
def test_gpt_character_profile_keeps_registered_checkout_and_interpreter(monkeypatch, tmp_path):
    import engines.adapters.gpt_sovits_adapter as adapter_module

    source_root = tmp_path / "registered-gpt-source"
    python_executable = tmp_path / "portable" / "python.exe"
    created = []

    class FakeExternalRuntime:
        def __init__(self, **kwargs):
            created.append(dict(kwargs))
            self.source_root = Path(kwargs["source_root"]).resolve()
            self.python_executable = Path(kwargs["python_executable"]).resolve()

        def run(self, _inputs):
            return 32000, adapter_module.np.full(320, len(created), dtype=adapter_module.np.int16)

        def cleanup(self):
            pass

    parser = types.SimpleNamespace(
        CHARACTER_TAG_PATTERN=types.SimpleNamespace(search=lambda _text: True),
        split_by_character=lambda _text, include_language=False: [("Alice", "profile text", None)],
    )
    monkeypatch.setattr(adapter_module, "ExternalGPTSovitsSubprocessProxy", FakeExternalRuntime)
    monkeypatch.setattr(adapter_module, "character_parser", parser)

    adapter = adapter_module.GPTSovitsAdapter()
    adapter.initialize_engine(
        gpt_weight="base-gpt.ckpt",
        sovits_weight="base-sovits.pth",
        bert_path="base-bert",
        cnhubert_path="base-cnhubert",
        device="cpu",
        use_fp16=False,
        gpt_sovits_home=str(source_root),
        python_executable=str(python_executable),
    )
    adapter._character_profiles = {
        "Alice": {
            "gpt_weight": "alice-gpt.ckpt",
            "sovits_weight": "alice-sovits.pth",
            "bert_path": "alice-bert",
            "cnhubert_path": "alice-cnhubert",
        }
    }

    waveform, sample_rate = adapter.generate(
        text="[Alice] profile text",
        ref_audio_path="voice.wav",
        ref_text="reference",
    )

    assert sample_rate == 32000
    assert waveform.shape == (1, 320)
    assert created[-1]["gpt_weight"] == "alice-gpt.ckpt"
    assert created[-1]["source_root"] == str(source_root)
    assert created[-1]["python_executable"] == str(python_executable)


@pytest.mark.unit
def test_gpt_bridge_forwards_private_interpreter_without_public_input(monkeypatch):
    resource = types.SimpleNamespace(
        resource_id="local-resource",
        engine="gpt_sovits",
        source_root=Path("C:/gpt/source"),
        gpt_weight=Path("C:/gpt/s1.ckpt"),
        sovits_weight=Path("C:/gpt/s2.pth"),
        bert_path=Path("C:/gpt/bert"),
        cnhubert_path=Path("C:/gpt/cnhubert"),
        python_executable=Path("D:/portable/python.exe"),
        version="v2",
    )
    registry = types.SimpleNamespace(require=lambda resource_id, engine: resource)
    monkeypatch.setattr(bridge, "get_resource_registry", lambda: registry)

    (engine,) = bridge.ExternalGPTSovitsEngineNode().create_engine("local-resource")

    assert engine["config"]["python_executable"] == str(Path("D:/portable/python.exe"))
    inputs = bridge.ExternalGPTSovitsEngineNode.INPUT_TYPES()
    assert "python_executable" not in inputs["required"]
    assert "python_executable" not in inputs.get("optional", {})


@pytest.mark.unit
def test_gpt_processor_forwards_private_interpreter_to_adapter():
    import engines.processors.gpt_sovits_processor as processor_module

    initialized = {}

    class FakeAdapter:
        def initialize_engine(self, **kwargs):
            initialized.update(kwargs)

    processor_module.GPTSovitsProcessor(
        {
            "gpt_weight": "C:/gpt/s1.ckpt",
            "sovits_weight": "C:/gpt/s2.pth",
            "gpt_sovits_home": "C:/gpt/source",
            "python_executable": "D:/portable/python.exe",
        },
        adapter=FakeAdapter(),
    )

    assert initialized["python_executable"] == "D:/portable/python.exe"


@pytest.mark.unit
def test_gpt_adapter_reuses_same_stateless_registered_runtime_proxy(monkeypatch, tmp_path):
    import engines.adapters.gpt_sovits_adapter as adapter_module

    created = []
    source_root = tmp_path / "gpt-source"

    class FakeExternalRuntime:
        def __init__(self, **kwargs):
            self.source_root = Path(kwargs["source_root"]).resolve()
            self.python_executable = _checkout_python(self.source_root)
            created.append(kwargs)

        def cleanup(self):
            pass

    monkeypatch.setattr(adapter_module, "ExternalGPTSovitsSubprocessProxy", FakeExternalRuntime)
    adapter = adapter_module.GPTSovitsAdapter()
    arguments = {
        "gpt_weight": "C:/gpt/s1.ckpt",
        "sovits_weight": "C:/gpt/s2.pth",
        "bert_path": "C:/gpt/bert",
        "cnhubert_path": "C:/gpt/cnhubert",
        "device": "cpu",
        "use_fp16": False,
        "gpt_sovits_home": str(source_root),
        "version": "v2",
    }

    adapter.initialize_engine(**arguments)
    adapter.initialize_engine(**arguments)

    assert len(created) == 1


@pytest.mark.unit
def test_registered_gpt_adapter_executes_one_shot_runtime_on_every_call():
    import engines.adapters.gpt_sovits_adapter as adapter_module

    runs = []
    cache_reads = []
    cache_writes = []

    class FakeExternalRuntime:
        source_root = "C:/gpt/source"

        def run(self, inputs):
            runs.append(inputs)
            return 32000, adapter_module.np.full(3200, len(runs), dtype=adapter_module.np.int16)

    class CoarseSharedCache:
        def generate_cache_key(self, *args, **kwargs):
            return "same-key"

        def get_cached_audio(self, cache_key):
            cache_reads.append(cache_key)
            return adapter_module.torch.zeros(1, 3200), 0.1

        def cache_audio(self, *args):
            cache_writes.append(args)

    adapter = adapter_module.GPTSovitsAdapter(audio_cache=CoarseSharedCache())
    adapter.runtime = FakeExternalRuntime()
    adapter._current_gpt_path = "C:/gpt/s1.ckpt"
    adapter._current_sovits_path = "C:/gpt/s2.pth"

    first, first_rate = adapter.generate(
        text="same request",
        ref_audio_path="C:/voice.wav",
        ref_text="reference",
        seed=7,
    )
    second, second_rate = adapter.generate(
        text="same request",
        ref_audio_path="C:/voice.wav",
        ref_text="reference",
        seed=7,
    )

    assert len(runs) == 2
    assert first_rate == second_rate == 32000
    assert adapter_module.torch.count_nonzero(first).item() == 3200
    assert adapter_module.torch.count_nonzero(second).item() == 3200
    assert cache_reads == []
    assert cache_writes == []


@pytest.mark.unit
def test_registered_cosyvoice_runtime_identities_cannot_share_waveform_cache():
    import engines.adapters.cosyvoice_adapter as adapter_module

    generated = []

    class FakeExternalEngine:
        def __init__(self, resource_id, source_root, model_dir, value):
            self.resource_id = resource_id
            self.source_root = source_root
            self.model_dir = model_dir
            self.value = value

        def generate(self, **kwargs):
            generated.append((self.resource_id, self.source_root, self.model_dir, kwargs))
            return adapter_module.torch.full((1, 2400), self.value)

    class CoarseSharedCache:
        def generate_cache_key(self, *args, **kwargs):
            return "same-key-without-runtime-identity"

        def get_cached_audio(self, cache_key):
            return (adapter_module.torch.full((1, 2400), 9.0), 0.1)

        def cache_audio(self, *args):
            raise AssertionError("registered external audio must not enter the waveform cache")

    shared_cache = CoarseSharedCache()
    adapters = []
    for identity in (
        ("cosy-a", "C:/cosy/a", "C:/cosy/a/model", 1.0),
        ("cosy-b", "D:/cosy/b", "D:/cosy/b/different-model", 2.0),
    ):
        adapter = adapter_module.CosyVoiceAdapter.__new__(adapter_module.CosyVoiceAdapter)
        adapter.engine = FakeExternalEngine(*identity)
        adapter.audio_cache = shared_cache
        adapter.model_variant = "standard"
        adapters.append(adapter)

    first = adapters[0].generate(text="same request", mode="cross_lingual")
    second = adapters[1].generate(text="same request", mode="cross_lingual")

    assert [item[:3] for item in generated] == [
        ("cosy-a", "C:/cosy/a", "C:/cosy/a/model"),
        ("cosy-b", "D:/cosy/b", "D:/cosy/b/different-model"),
    ]
    assert adapter_module.torch.count_nonzero(first == 1).item() == 2400
    assert adapter_module.torch.count_nonzero(second == 2).item() == 2400


@pytest.mark.unit
def test_unregistered_cosyvoice_adapter_preserves_waveform_cache():
    import engines.adapters.cosyvoice_adapter as adapter_module

    generated = []

    class FakeBundledEngine:
        source_root = None

        def generate(self, **kwargs):
            generated.append(kwargs)
            return adapter_module.torch.ones(1, 2400)

    class FakeAudioCache:
        def __init__(self):
            self.value = None
            self.writes = 0

        def generate_cache_key(self, *args, **kwargs):
            return "legacy-cache-key"

        def get_cached_audio(self, cache_key):
            return self.value

        def cache_audio(self, cache_key, audio, duration):
            self.writes += 1
            self.value = (audio, duration)

    cache = FakeAudioCache()
    adapter = adapter_module.CosyVoiceAdapter.__new__(adapter_module.CosyVoiceAdapter)
    adapter.engine = FakeBundledEngine()
    adapter.audio_cache = cache
    adapter.model_variant = "standard"

    first = adapter.generate(text="legacy cache stays", mode="cross_lingual")
    second = adapter.generate(text="legacy cache stays", mode="cross_lingual")

    assert len(generated) == 1
    assert cache.writes == 1
    assert first is second


def _load_external_index_subprocess_module():
    sys.modules["comfy.model_management"].processing_interrupted.return_value = False
    path = REPO_ROOT / "engines" / "index_tts" / "external_subprocess.py"
    assert path.is_file(), "plugin-owned external IndexTTS subprocess adapter is missing"
    spec = importlib.util.spec_from_file_location("external_index_subprocess_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_external_wait_returns_output_without_interrupt(monkeypatch):
    module = _load_external_index_subprocess_module()
    proxy = object.__new__(module.ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 1.0
    proxy.termination_grace_seconds = 0.2
    proxy.interrupt_check = lambda: False

    class Finished:
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

    assert proxy._communicate_with_control(Finished(), "IndexTTS") == ("ok", "")


def test_external_wait_interrupts_and_cleans_tree(monkeypatch):
    module = _load_external_index_subprocess_module()
    proxy = object.__new__(module.ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 10.0
    proxy.termination_grace_seconds = 0.2
    checks = iter((False, True))
    proxy.interrupt_check = lambda: next(checks, True)
    cleaned = []

    class Running:
        returncode = None

        def communicate(self, timeout=None):
            raise module.subprocess.TimeoutExpired("runner", timeout)

        def poll(self):
            return self.returncode

    def cleanup(process):
        process.returncode = -9
        cleaned.append(process)
        return "partial", "", "tree exited", True

    monkeypatch.setattr(proxy, "_cleanup_timed_out_process", cleanup)
    process = Running()
    with pytest.raises(InterruptedError, match="IndexTTS external subprocess interrupted"):
        proxy._communicate_with_control(process, "IndexTTS")
    assert cleaned == [process]


def test_external_wait_reports_cleanup_failure_instead_of_false_interrupt_success(monkeypatch):
    module = _load_external_index_subprocess_module()
    proxy = object.__new__(module.ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 10.0
    proxy.termination_grace_seconds = 0.01
    proxy.interrupt_check = lambda: True

    class Stuck:
        returncode = None

        def communicate(self, timeout=None):
            raise module.subprocess.TimeoutExpired("runner", timeout)

        def poll(self):
            return None

    monkeypatch.setattr(
        proxy,
        "_cleanup_timed_out_process",
        lambda process: ("", "", "process exit could not be verified", False),
    )
    with pytest.raises(RuntimeError, match="interruption cleanup failed"):
        proxy._communicate_with_control(Stuck(), "IndexTTS")


def test_external_wait_rejects_unverified_tree_when_windows_job_close_fails(monkeypatch):
    module = _load_external_index_subprocess_module()
    proxy = object.__new__(module.ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 10.0
    proxy.termination_grace_seconds = 0.2
    proxy.interrupt_check = lambda: True

    class ExitedParent:
        pid = 5252
        returncode = None

        def poll(self):
            return self.returncode

        @staticmethod
        def communicate(timeout=None):
            return "", ""

    process = ExitedParent()

    class BrokenJob:
        def close(self):
            process.returncode = -9
            raise OSError("job handle close failed")

    process._tts_windows_job = BrokenJob()
    monkeypatch.setattr(module.os, "name", "nt")

    with pytest.raises(RuntimeError, match="interruption cleanup failed"):
        proxy._communicate_with_control(process, "IndexTTS")


def test_external_wait_preserves_timeout_category(monkeypatch):
    module = _load_external_index_subprocess_module()
    proxy = object.__new__(module.ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 0.01
    proxy.termination_grace_seconds = 0.2
    proxy.interrupt_check = lambda: False

    class Running:
        returncode = None

        def communicate(self, timeout=None):
            raise module.subprocess.TimeoutExpired("runner", timeout)

    monkeypatch.setattr(
        proxy,
        "_cleanup_timed_out_process",
        lambda process: ("", "slow", "tree exited", True),
    )
    with pytest.raises(TimeoutError, match="exceeded 0.01s"):
        proxy._communicate_with_control(Running(), "IndexTTS")


def _prepare_external_index_runtime(tmp_path):
    source_root = tmp_path / "index-source"
    model_dir = source_root / "checkpoints"
    inference_module = source_root / "indextts" / "infer_v2.py"
    inference_module.parent.mkdir(parents=True)
    inference_module.write_text("class IndexTTS2: pass\n", encoding="utf-8")
    python_executable = _checkout_python(source_root)
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


def _load_external_cosyvoice_subprocess_module():
    path = REPO_ROOT / "engines" / "cosyvoice" / "external_subprocess.py"
    assert path.is_file(), "plugin-owned external CosyVoice subprocess adapter is missing"
    spec = importlib.util.spec_from_file_location("external_cosyvoice_subprocess_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_external_gpt_subprocess_module():
    path = REPO_ROOT / "engines" / "gpt_sovits" / "external_subprocess.py"
    assert path.is_file(), "plugin-owned external GPT-SoVITS subprocess adapter is missing"
    spec = importlib.util.spec_from_file_location("external_gpt_subprocess_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_external_gpt_runner_module():
    path = REPO_ROOT / "engines" / "gpt_sovits" / "external_subprocess_runner.py"
    assert path.is_file(), "plugin-owned external GPT-SoVITS subprocess runner is missing"
    spec = importlib.util.spec_from_file_location("external_gpt_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_external_gpt_runtime(tmp_path):
    source_root = tmp_path / "gpt-source"
    package_root = source_root / "GPT_SoVITS"
    official_module = package_root / "TTS_infer_pack" / "TTS.py"
    official_module.parent.mkdir(parents=True)
    official_module.write_text("class TTS: pass\nclass TTS_Config: pass\n", encoding="utf-8")
    (package_root / "eres2net").mkdir()
    python_executable = _checkout_python(source_root)
    python_executable.parent.mkdir(parents=True)
    python_executable.touch()
    gpt_weight = package_root / "pretrained_models" / "s1.ckpt"
    sovits_weight = package_root / "pretrained_models" / "s2.pth"
    bert_path = package_root / "pretrained_models" / "bert"
    cnhubert_path = package_root / "pretrained_models" / "cnhubert"
    gpt_weight.parent.mkdir(parents=True)
    gpt_weight.touch()
    sovits_weight.touch()
    bert_path.mkdir()
    cnhubert_path.mkdir()
    voice_path = source_root / "voice.wav"
    with wave.open(str(voice_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x01\x00" * 160)
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    return (
        source_root,
        python_executable,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    )


@pytest.mark.unit
def test_external_gpt_subprocess_preserves_registered_lineage_offline_and_cleans_temp(
    monkeypatch, tmp_path
):
    module = _load_external_gpt_subprocess_module()
    (
        source_root,
        python_executable,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    ) = _prepare_external_gpt_runtime(tmp_path)
    observed = {}

    class FakeProcess:
        pid = 6242
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            environment = kwargs["env"]
            for variable in ("TEMP", "TMP"):
                private_temp = _path_without_windows_extended_prefix(environment[variable])
                assert private_temp.is_dir(), f"{variable} missing before child start"
                assert private_temp.is_relative_to(temp_root.resolve())
                if os.name == "nt":
                    assert environment[variable].startswith("\\\\?\\")

        def communicate(self, timeout):
            observed["timeout"] = timeout
            manifest = json.loads(Path(observed["command"][2]).read_text(encoding="utf-8"))
            observed["manifest"] = manifest
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
        device="cuda",
        use_fp16=True,
        version="v2",
        timeout_seconds=321.0,
        temp_root=temp_root,
    )

    sample_rate, samples = proxy.run(
        {
            "text": "real external inference",
            "text_lang": "en",
            "ref_audio_path": str(voice_path),
            "prompt_text": "exact reference",
            "prompt_lang": "en",
            "seed": 7,
        }
    )

    assert observed["command"][0] == str(python_executable)
    assert Path(observed["command"][1]).name == "external_subprocess_runner.py"
    assert observed["kwargs"]["cwd"] == str(source_root)
    assert 0 < observed["timeout"] <= 0.25
    child_environment = observed["kwargs"]["env"]
    assert child_environment["TTS_AUDIO_SUITE_OFFLINE"] == "1"
    assert child_environment["HF_HUB_OFFLINE"] == "1"
    assert child_environment["PYTHONNOUSERSITE"] == "1"
    manifest = observed["manifest"]
    assert manifest["source_root"] == str(source_root)
    assert manifest["runtime_config_path"].startswith(str(temp_root))
    assert manifest["config"] == {
        "gpt_weight": str(gpt_weight),
        "sovits_weight": str(sovits_weight),
        "bert_path": str(bert_path),
        "cnhubert_path": str(cnhubert_path),
        "device": "cuda",
        "use_fp16": True,
        "version": "v2",
    }
    assert manifest["inference"]["prompt_text"] == "exact reference"
    assert sample_rate == 32000
    assert samples.shape == (320,)
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_gpt_subprocess_uses_explicit_registered_interpreter(tmp_path):
    module = _load_external_gpt_subprocess_module()
    (
        source_root,
        checkout_interpreter,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        _,
        temp_root,
    ) = _prepare_external_gpt_runtime(tmp_path)
    explicit_interpreter = tmp_path / "portable" / "python.exe"
    explicit_interpreter.parent.mkdir()
    explicit_interpreter.touch()

    proxy = module.ExternalGPTSovitsSubprocessProxy(
        source_root=source_root,
        gpt_weight=gpt_weight,
        sovits_weight=sovits_weight,
        bert_path=bert_path,
        cnhubert_path=cnhubert_path,
        python_executable=explicit_interpreter,
        device="cuda",
        use_fp16=True,
        version="v2",
        temp_root=temp_root,
    )

    assert proxy.python_executable == explicit_interpreter.resolve()
    assert proxy.python_executable != checkout_interpreter.resolve()


@pytest.mark.unit
def test_external_gpt_runner_imports_registered_source_and_writes_only_temp_config(
    monkeypatch, tmp_path
):
    module = _load_external_gpt_runner_module()
    (
        source_root,
        _,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    ) = _prepare_external_gpt_runtime(tmp_path)
    output_path = temp_root / "result.wav"
    runtime_config_path = temp_root / "runtime-config.yaml"
    manifest_path = temp_root / "request.json"
    payload = {
        "source_root": str(source_root),
        "output_path": str(output_path),
        "runtime_config_path": str(runtime_config_path),
        "config": {
            "gpt_weight": str(gpt_weight),
            "sovits_weight": str(sovits_weight),
            "bert_path": str(bert_path),
            "cnhubert_path": str(cnhubert_path),
            "device": "cuda",
            "use_fp16": True,
            "version": "v2",
        },
        "inference": {
            "text": "real runner inference",
            "text_lang": "en",
            "ref_audio_path": str(voice_path),
            "prompt_text": "exact reference",
            "prompt_lang": "en",
            "seed": 7,
        },
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    observed = {}

    package = types.ModuleType("TTS_infer_pack")
    package.__path__ = [str(source_root / "GPT_SoVITS" / "TTS_infer_pack")]
    official = types.ModuleType("TTS_infer_pack.TTS")
    official.__file__ = str(source_root / "GPT_SoVITS" / "TTS_infer_pack" / "TTS.py")

    class FakeConfig:
        def __init__(self, configs):
            observed["config"] = configs
            self.configs_path = "must-be-replaced"

    class FakeTTS:
        def __init__(self, config):
            observed["runtime_config_path"] = config.configs_path
            sock = module.socket.socket()
            try:
                sock.connect(("203.0.113.1", 443))
            except RuntimeError as exc:
                observed["network_error"] = str(exc)
            finally:
                sock.close()

        def run(self, inputs):
            observed["inference"] = inputs
            return 32000, module.np.full(320, 1000, dtype=module.np.int16)

    official.TTS_Config = FakeConfig
    official.TTS = FakeTTS
    monkeypatch.setitem(sys.modules, "TTS_infer_pack", package)
    monkeypatch.setitem(sys.modules, "TTS_infer_pack.TTS", official)

    assert module.main([str(manifest_path)]) == 0

    samples, sample_rate = module.soundfile.read(output_path, dtype="float32")
    assert sample_rate == 32000
    assert samples.size == 320
    assert observed["config"] == {
        "custom": {
            "device": "cuda",
            "is_half": True,
            "version": "v2",
            "t2s_weights_path": str(gpt_weight),
            "vits_weights_path": str(sovits_weight),
            "bert_base_path": str(bert_path),
            "cnhuhbert_base_path": str(cnhubert_path),
        }
    }
    assert observed["runtime_config_path"] == str(runtime_config_path)
    assert observed["inference"] == payload["inference"]
    assert "network access is disabled" in observed["network_error"]


def _load_external_cosyvoice_runner_module():
    path = REPO_ROOT / "engines" / "cosyvoice" / "external_subprocess_runner.py"
    assert path.is_file(), "plugin-owned external CosyVoice subprocess runner is missing"
    spec = importlib.util.spec_from_file_location("external_cosyvoice_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_external_cosyvoice_runtime(tmp_path):
    source_root = tmp_path / "cosyvoice-source"
    model_dir = source_root / "pretrained_models" / "CosyVoice-300M"
    inference_module = source_root / "cosyvoice" / "cli" / "cosyvoice.py"
    inference_module.parent.mkdir(parents=True)
    inference_module.write_text("def AutoModel(**kwargs): pass\n", encoding="utf-8")
    (source_root / "third_party" / "Matcha-TTS").mkdir(parents=True)
    python_executable = _checkout_python(source_root)
    python_executable.parent.mkdir(parents=True)
    python_executable.touch()
    model_dir.mkdir(parents=True)
    (model_dir / "cosyvoice.yaml").write_text("sample_rate: 22050\n", encoding="utf-8")
    for name in ("llm.pt", "flow.pt", "hift.pt", "campplus.onnx", "speech_tokenizer_v1.onnx"):
        (model_dir / name).touch()
    voice_path = source_root / "voice.wav"
    with wave.open(str(voice_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x01\x00" * 160)
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    return source_root, model_dir, python_executable, voice_path, temp_root


def _assert_registered_engine_interrupts_during_sliced_wait(
    monkeypatch,
    module,
    proxy,
    invoke_real_engine,
    temp_root,
    engine_label,
):
    """Exercise the registered public call while its runner is still active."""
    communicate_timeouts = []
    created_processes = []
    terminated_processes = []

    class RunningProcess:
        returncode = None

        def __init__(self, command, **kwargs):
            del command, kwargs
            created_processes.append(self)

        def communicate(self, timeout=None):
            communicate_timeouts.append(timeout)
            if self.returncode is None:
                raise module.subprocess.TimeoutExpired("runner", timeout)
            return "", ""

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def terminate_process_tree(process, grace_seconds):
        del grace_seconds
        terminated_processes.append(process)
        process.returncode = -9
        return "runner tree terminated"

    monkeypatch.setattr(module.subprocess, "Popen", RunningProcess)
    monkeypatch.setattr(
        type(proxy),
        "_terminate_process_tree",
        staticmethod(terminate_process_tree),
    )

    with pytest.raises(InterruptedError, match=engine_label):
        invoke_real_engine(proxy)

    assert communicate_timeouts
    assert all(0 < timeout <= 0.25 for timeout in communicate_timeouts)
    assert len(created_processes) == 1
    assert terminated_processes == [created_processes[0]]
    assert created_processes[0].returncode == -9
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_registered_index_interrupts_during_sliced_wait(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    checks = iter((False, True))
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=321.0,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
        interrupt_check=lambda: next(checks, True),
    )

    _assert_registered_engine_interrupts_during_sliced_wait(
        monkeypatch,
        module,
        proxy,
        lambda engine: engine.infer(
            spk_audio_prompt=str(voice_path),
            text="interrupt the external runner",
            output_path=None,
        ),
        temp_root,
        "IndexTTS",
    )


@pytest.mark.unit
def test_registered_gpt_sovits_interrupts_during_sliced_wait(monkeypatch, tmp_path):
    module = _load_external_gpt_subprocess_module()
    (
        source_root,
        python_executable,
        gpt_weight,
        sovits_weight,
        bert_path,
        cnhubert_path,
        voice_path,
        temp_root,
    ) = _prepare_external_gpt_runtime(tmp_path)
    checks = iter((False, True))
    proxy = module.ExternalGPTSovitsSubprocessProxy(
        source_root=source_root,
        gpt_weight=gpt_weight,
        sovits_weight=sovits_weight,
        bert_path=bert_path,
        cnhubert_path=cnhubert_path,
        device="cuda",
        use_fp16=True,
        version="v2",
        python_executable=python_executable,
        timeout_seconds=321.0,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
        interrupt_check=lambda: next(checks, True),
    )

    _assert_registered_engine_interrupts_during_sliced_wait(
        monkeypatch,
        module,
        proxy,
        lambda engine: engine.run(
            {
                "text": "interrupt the external runner",
                "text_lang": "en",
                "ref_audio_path": str(voice_path),
                "prompt_text": "reference",
                "prompt_lang": "en",
            }
        ),
        temp_root,
        "GPT-SoVITS",
    )


@pytest.mark.unit
def test_registered_cosyvoice_interrupts_during_sliced_wait(monkeypatch, tmp_path):
    module = _load_external_cosyvoice_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_cosyvoice_runtime(tmp_path)
    checks = iter((False, True))
    proxy = module.ExternalCosyVoiceSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        timeout_seconds=321.0,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
        interrupt_check=lambda: next(checks, True),
    )

    _assert_registered_engine_interrupts_during_sliced_wait(
        monkeypatch,
        module,
        proxy,
        lambda engine: list(
            engine.inference_cross_lingual(
                tts_text="interrupt the external runner",
                prompt_wav=str(voice_path),
            )
        ),
        temp_root,
        "CosyVoice",
    )


@pytest.mark.unit
def test_external_cosyvoice_subprocess_uses_checkout_venv_offline_and_cleans_temp(monkeypatch, tmp_path):
    module = _load_external_cosyvoice_subprocess_module()
    source_root, model_dir, python_executable, voice_path, temp_root = _prepare_external_cosyvoice_runtime(tmp_path)
    observed = {}

    class FakeProcess:
        pid = 5242
        returncode = 0

        def __init__(self, command, **kwargs):
            observed["command"] = command
            observed["kwargs"] = kwargs
            environment = kwargs["env"]
            for variable in ("TEMP", "TMP"):
                private_temp = _path_without_windows_extended_prefix(environment[variable])
                assert private_temp.is_dir(), f"{variable} missing before child start"
                assert private_temp.is_relative_to(temp_root.resolve())
                if os.name == "nt":
                    assert environment[variable].startswith("\\\\?\\")

        def communicate(self, timeout):
            observed["timeout"] = timeout
            manifest = json.loads(Path(observed["command"][2]).read_text(encoding="utf-8"))
            observed["manifest"] = manifest
            with wave.open(manifest["output_path"], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\x10\x00" * 240)
            return "official stdout", ""

    monkeypatch.setattr(module.subprocess, "Popen", FakeProcess)
    proxy = module.ExternalCosyVoiceSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        timeout_seconds=321.0,
        temp_root=temp_root,
    )

    outputs = list(
        proxy.inference_cross_lingual(
            tts_text="真实外部推理。",
            prompt_wav=str(voice_path),
            stream=False,
            speed=1.1,
            text_frontend=True,
        )
    )

    assert observed["command"][0] == str(python_executable)
    assert Path(observed["command"][1]).name == "external_subprocess_runner.py"
    assert observed["kwargs"]["cwd"] == str(source_root)
    assert 0 < observed["timeout"] <= 0.25
    child_environment = observed["kwargs"]["env"]
    assert child_environment["TTS_AUDIO_SUITE_OFFLINE"] == "1"
    assert child_environment["HF_HUB_OFFLINE"] == "1"
    assert child_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    for variable in ("TEMP", "TMP"):
        private_temp = _path_without_windows_extended_prefix(child_environment[variable])
        assert private_temp.is_relative_to(temp_root.resolve())
        if os.name == "nt":
            assert child_environment[variable].startswith("\\\\?\\")
    assert observed["manifest"]["mode"] == "cross_lingual"
    assert observed["manifest"]["text"] == "真实外部推理。"
    assert outputs[0]["tts_speech"].shape == (1, 240)
    assert proxy.sample_rate == 24000
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_cosyvoice_runner_uses_only_local_assets_and_selects_v1_loader(monkeypatch, tmp_path):
    module = _load_external_cosyvoice_runner_module()
    source_root, model_dir, _, voice_path, _ = _prepare_external_cosyvoice_runtime(tmp_path)
    output_path = tmp_path / "result.wav"
    manifest_path = tmp_path / "request.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "model_dir": str(model_dir),
                "output_path": str(output_path),
                "constructor": {"use_fp16": True, "load_trt": False, "load_vllm": False},
                "mode": "cross_lingual",
                "text": "You are a helpful assistant.<|endofprompt|>离线推理。",
                "prompt_wav": str(voice_path),
                "prompt_text": "",
                "instruct_text": "",
                "speed": 1.0,
                "stream": False,
                "text_frontend": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    observed = {"snapshot_calls": []}

    modelscope_module = types.ModuleType("modelscope")

    def snapshot_download(model_id, **kwargs):
        observed["snapshot_calls"].append((model_id, kwargs))
        return str(tmp_path / "cache")

    modelscope_module.snapshot_download = snapshot_download
    cosyvoice_package = types.ModuleType("cosyvoice")
    cosyvoice_package.__path__ = [str(source_root / "cosyvoice")]
    cli_package = types.ModuleType("cosyvoice.cli")
    cli_package.__path__ = [str(source_root / "cosyvoice" / "cli")]
    official_module = types.ModuleType("cosyvoice.cli.cosyvoice")
    official_module.__file__ = str(source_root / "cosyvoice" / "cli" / "cosyvoice.py")

    class FakeModel:
        sample_rate = 22050

        def inference_cross_lingual(self, **kwargs):
            observed["inference"] = kwargs
            yield {"tts_speech": module.torch.ones(1, 2205)}

    def auto_model(**kwargs):
        observed["constructor"] = kwargs
        import modelscope
        import socket

        modelscope.snapshot_download("pengzhendong/wetext")
        sock = socket.socket()
        try:
            sock.connect(("203.0.113.1", 443))
        except RuntimeError as exc:
            observed["network_error"] = str(exc)
        finally:
            sock.close()
        return FakeModel()

    official_module.AutoModel = auto_model
    monkeypatch.setitem(sys.modules, "modelscope", modelscope_module)
    monkeypatch.setitem(sys.modules, "cosyvoice", cosyvoice_package)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli_package)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", official_module)

    assert module.main([str(manifest_path)]) == 0

    samples, sample_rate = module.soundfile.read(output_path, dtype="float32")
    assert sample_rate == 24000
    assert samples.size > 0
    assert observed["snapshot_calls"] == [
        ("pengzhendong/wetext", {"local_files_only": True})
    ]
    assert "load_vllm" not in observed["constructor"]
    assert observed["inference"]["tts_text"] == "离线推理。"
    assert "network access is disabled" in observed["network_error"]


@pytest.mark.unit
def test_cosyvoice_processor_passes_registered_checkout_to_adapter(monkeypatch):
    import engines.processors.cosyvoice_processor as processor_module

    initialized = {}

    class FakeAdapter:
        def initialize_engine(self, **kwargs):
            initialized.update(kwargs)

        def get_sample_rate(self):
            return 24000

    monkeypatch.setattr(processor_module, "CosyVoiceAdapter", FakeAdapter)
    monkeypatch.setattr(processor_module.CosyVoiceProcessor, "_setup_character_parser", lambda self: None)

    processor_module.CosyVoiceProcessor(
        {
            "model_path": "C:/cosy/model",
            "cosyvoice_home": "C:/cosy/source",
            "device": "cuda",
        }
    )

    assert initialized["model_path"] == "C:/cosy/model"
    assert initialized["cosyvoice_home"] == "C:/cosy/source"



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
            environment = kwargs["env"]
            for variable in ("TEMP", "TMP"):
                private_temp = _path_without_windows_extended_prefix(environment[variable])
                assert private_temp.is_dir(), f"{variable} missing before child start"
                assert private_temp.is_relative_to(temp_root.resolve())
                if os.name == "nt":
                    assert environment[variable].startswith("\\\\?\\")

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
    assert observed["timeout"] == 0.25
    child_environment = observed["kwargs"]["env"]
    assert child_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    for variable in ("TEMP", "TMP"):
        private_temp = _path_without_windows_extended_prefix(child_environment[variable])
        assert private_temp.is_relative_to(temp_root.resolve())
        if os.name == "nt":
            assert child_environment[variable].startswith("\\\\?\\")
    pycache_prefix = _path_without_windows_extended_prefix(child_environment["PYTHONPYCACHEPREFIX"])
    assert pycache_prefix.is_relative_to(temp_root)
    assert not pycache_prefix.is_relative_to(source_root)
    assert _path_without_windows_extended_prefix(child_environment["NUMBA_CACHE_DIR"]).is_relative_to(temp_root)
    assert _path_without_windows_extended_prefix(child_environment["MPLCONFIGDIR"]).is_relative_to(temp_root)
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
            if self.returncode is None:
                raise module.subprocess.TimeoutExpired("indextts", timeout)
            return ("", "child stopped")

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(module.subprocess, "Popen", TimedOutProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(
            lambda process, grace: (
                terminated.append((process.pid, grace)),
                setattr(process, "returncode", -9),
            )
        ),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.5,
        termination_grace_seconds=0.25,
        temp_root=temp_root,
    )

    with pytest.raises(TimeoutError, match="exceeded 0.5s"):
        proxy.infer(spk_audio_prompt=str(voice_path), text="超时清理。", output_path=None)

    assert terminated[0][0] == 4444
    assert 0 < terminated[0][1] <= 0.25
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_index_timeout_falls_back_to_direct_kill_when_tree_kill_fails(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    instances = []

    class TreeKillFailedProcess:
        pid = 4545
        returncode = None

        def __init__(self, command, **kwargs):
            self.timeouts = []
            self.kill_calls = 0
            instances.append(self)

        def communicate(self, timeout=None):
            self.timeouts.append(timeout)
            if self.returncode is None:
                raise module.subprocess.TimeoutExpired("indextts", timeout)
            return ("", "direct child stopped")

        def poll(self):
            return self.returncode

        def kill(self):
            self.kill_calls += 1
            self.returncode = -9

    monkeypatch.setattr(module.subprocess, "Popen", TreeKillFailedProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(lambda process, grace: (_ for _ in ()).throw(OSError("taskkill denied"))),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.5,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
    )

    with pytest.raises(TimeoutError) as error:
        proxy.infer(spk_audio_prompt=str(voice_path), text="树终止失败。", output_path=None)

    assert "exceeded 0.5s" in str(error.value)
    assert "taskkill denied" in str(error.value)
    assert instances[0].kill_calls == 1
    assert 0 < instances[0].timeouts[0] <= 0.25
    assert 0 < instances[0].timeouts[-1] <= 0.21
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_index_timeout_cleanup_remains_bounded_when_process_will_not_exit(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    instances = []

    class StuckProcess:
        pid = 4646
        returncode = None

        def __init__(self, command, **kwargs):
            self.timeouts = []
            instances.append(self)

        def communicate(self, timeout=None):
            self.timeouts.append(timeout)
            raise module.subprocess.TimeoutExpired("indextts", timeout)

        def poll(self):
            return None

        def kill(self):
            raise PermissionError("direct kill denied")

    monkeypatch.setattr(module.subprocess, "Popen", StuckProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(lambda process, grace: None),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.5,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
    )

    with pytest.raises(TimeoutError) as error:
        proxy.infer(spk_audio_prompt=str(voice_path), text="无法退出。", output_path=None)

    assert "exceeded 0.5s" in str(error.value)
    assert "direct kill denied" in str(error.value)
    assert "cleanup communicate exceeded remaining" in str(error.value)
    assert "of 0.2s grace" in str(error.value)
    assert 0 < instances[0].timeouts[0] <= 0.25
    assert 0 < instances[0].timeouts[-1] <= 0.21
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_external_index_timeout_reports_when_cleanup_cannot_verify_process_exit(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)

    class UnverifiedExitProcess:
        pid = 4696
        returncode = None

        def __init__(self, command, **kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            raise module.subprocess.TimeoutExpired("indextts", timeout)

        def poll(self):
            return None

        def kill(self):
            pass

    monkeypatch.setattr(module.subprocess, "Popen", UnverifiedExitProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(lambda process, grace: None),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.5,
        termination_grace_seconds=0.2,
        temp_root=temp_root,
    )

    with pytest.raises(TimeoutError) as error:
        proxy.infer(spk_audio_prompt=str(voice_path), text="退出状态未知。", output_path=None)

    assert "process exit could not be verified" in str(error.value)
    assert not list(temp_root.iterdir())


@pytest.mark.unit
def test_windows_tree_kill_has_a_deadline_and_checks_taskkill_failure(monkeypatch):
    module = _load_external_index_subprocess_module()
    observed = {}

    class RunningProcess:
        pid = 4747

        @staticmethod
        def poll():
            return None

    def failed_taskkill(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return module.subprocess.CompletedProcess(command, 5, "", "access denied")

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.subprocess, "run", failed_taskkill)
    monkeypatch.setattr(
        module.psutil,
        "Process",
        lambda pid: (_ for _ in ()).throw(module.psutil.NoSuchProcess(pid)),
    )

    with pytest.raises(RuntimeError, match="access denied"):
        module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(RunningProcess(), 0.3)

    assert observed["command"] == ["taskkill", "/PID", "4747", "/T", "/F"]
    assert 0 < observed["kwargs"]["timeout"] <= 0.15


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree fallback contract")
def test_windows_tree_kill_falls_back_to_psutil_and_exits_real_parent_and_child(monkeypatch):
    module = _load_external_index_subprocess_module()
    parent_script = (
        "import subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    assert process.stdout is not None
    child_pid = int(process.stdout.readline().strip())
    real_run = subprocess.run

    def failed_taskkill(command, **kwargs):
        return subprocess.CompletedProcess(command, 5, "", "simulated taskkill denial")

    monkeypatch.setattr(module.subprocess, "run", failed_taskkill)
    started = time.monotonic()
    try:
        diagnostic = module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(
            process,
            1.5,
        )
        elapsed = time.monotonic() - started

        assert "simulated taskkill denial" in diagnostic
        assert elapsed < 1.8
        assert process.poll() is not None
        assert not module.psutil.pid_exists(child_pid)
    finally:
        if process.poll() is None or module.psutil.pid_exists(child_pid):
            real_run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3,
                check=False,
            )


@pytest.mark.unit
def test_windows_tree_kill_reports_stubborn_descendant_within_hard_deadline(monkeypatch):
    module = _load_external_index_subprocess_module()
    observed_waits = []

    class StubbornProcess:
        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive=False):
            assert recursive is False
            return [child] if self.pid == 4848 else []

        def create_time(self):
            return float(self.pid)

        @staticmethod
        def is_running():
            return True

        @staticmethod
        def terminate():
            pass

        @staticmethod
        def kill():
            pass

        @staticmethod
        def suspend():
            pass

        def wait(self, timeout=None):
            raise module.psutil.TimeoutExpired(timeout, pid=self.pid)

    root = StubbornProcess(4848)
    child = StubbornProcess(4949)

    class RunningPopen:
        pid = 4848

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.psutil, "Process", lambda pid: root)
    monkeypatch.setattr(module.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: module.subprocess.CompletedProcess(
            command,
            5,
            "",
            "simulated taskkill denial",
        ),
    )

    def wait_procs(processes, timeout):
        observed_waits.append(timeout)
        time.sleep(timeout)
        return [], list(processes)

    monkeypatch.setattr(module.psutil, "wait_procs", wait_procs)
    started = time.monotonic()
    with pytest.raises(RuntimeError) as error:
        module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(RunningPopen(), 0.2)
    elapsed = time.monotonic() - started

    assert "simulated taskkill denial" in str(error.value)
    assert "4949" in str(error.value)
    assert elapsed < 0.35
    assert sum(observed_waits) <= 0.21


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows late-descendant fallback contract")
def test_windows_fallback_catches_real_child_spawned_after_last_snapshot(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    spawn_signal = tmp_path / "spawn-late"
    late_pid_path = tmp_path / "late-pid"
    parent_script = (
        "import pathlib,subprocess,sys,time; "
        "signal=pathlib.Path(sys.argv[1]); late_path=pathlib.Path(sys.argv[2]); "
        "print('READY',flush=True); "
        "\nwhile not signal.exists(): time.sleep(0.005)\n"
        "late=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "late_path.write_text(str(late.pid)); time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_script, str(spawn_signal), str(late_pid_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "READY"
    real_psutil_process = module.psutil.Process
    real_run = subprocess.run

    root_process = real_psutil_process(process.pid)
    script_process = [
        candidate
        for candidate in [root_process, *root_process.children(recursive=True)]
        if str(spawn_signal) in " ".join(candidate.cmdline())
    ][-1]

    class ScriptProcessProxy:
        def __init__(self, delegate):
            self._delegate = delegate
            self.pid = delegate.pid

        def __getattr__(self, name):
            return getattr(self._delegate, name)

        def children(self, recursive=False):
            return [wrap_process(candidate) for candidate in self._delegate.children(recursive)]

        def _spawn_late(self):
            spawn_signal.touch()
            deadline = time.monotonic() + 0.75
            late_pid_text = ""
            while time.monotonic() < deadline:
                if late_pid_path.is_file():
                    late_pid_text = late_pid_path.read_text().strip()
                    if late_pid_text.isdecimal():
                        break
                time.sleep(0.005)
            assert late_pid_text.isdecimal(), (
                "controlled parent did not publish the late child PID; "
                f"root={process.pid} script={self.pid} "
                f"root_poll={process.poll()} script_running={self._delegate.is_running()} "
                f"signal_exists={spawn_signal.exists()} "
                f"late_pid_text={late_pid_text!r} "
                f"stderr={process.stderr.read() if process.poll() is not None else ''}"
            )

        def suspend(self):
            self._spawn_late()
            return self._delegate.suspend()

        def terminate(self):
            self._spawn_late()
            return self._delegate.terminate()

    script_proxy = ScriptProcessProxy(script_process)

    class RootProcessProxy:
        def __init__(self, delegate):
            self._delegate = delegate
            self.pid = delegate.pid

        def __getattr__(self, name):
            return getattr(self._delegate, name)

        def children(self, recursive=False):
            return [wrap_process(candidate) for candidate in self._delegate.children(recursive)]

    def wrap_process(candidate):
        return script_proxy if candidate.pid == script_process.pid else candidate

    root_proxy = script_proxy if script_process.pid == process.pid else RootProcessProxy(root_process)
    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(
        module,
        "psutil",
        types.SimpleNamespace(
            Process=lambda pid: root_proxy if pid == process.pid else real_psutil_process(pid),
            NoSuchProcess=module.psutil.NoSuchProcess,
            AccessDenied=module.psutil.AccessDenied,
            TimeoutExpired=module.psutil.TimeoutExpired,
            wait_procs=module.psutil.wait_procs,
            pid_exists=module.psutil.pid_exists,
        ),
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: module.subprocess.CompletedProcess(
            command,
            5,
            "",
            "simulated taskkill denial",
        ),
    )

    late_pid = None
    try:
        diagnostic = module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(process, 1.5)
        assert late_pid_path.is_file(), diagnostic
        late_pid = int(late_pid_path.read_text())

        assert "simulated taskkill denial" in diagnostic
        assert process.poll() is not None
        assert not module.psutil.pid_exists(late_pid)
    finally:
        for candidate_pid in (late_pid, process.pid):
            if candidate_pid and module.psutil.pid_exists(candidate_pid):
                real_run(
                    ["taskkill", "/PID", str(candidate_pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3,
                    check=False,
                )


@pytest.mark.unit
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object containment contract")
def test_windows_job_launch_contains_real_child_created_after_resume(tmp_path):
    module = _load_external_index_subprocess_module()
    spawn_signal = tmp_path / "spawn-contained-child"
    parent_script = (
        "import pathlib,subprocess,sys,time; signal=pathlib.Path(sys.argv[1]); "
        "\nwhile not signal.exists(): time.sleep(0.005)\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(child.pid,flush=True); time.sleep(60)"
    )
    process = module.ExternalIndexTTSSubprocessProxy._start_process(
        [sys.executable, "-c", parent_script, str(spawn_signal)],
        {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        },
    )
    real_run = subprocess.run
    child_pid = None
    try:
        spawn_signal.touch()
        assert process.stdout is not None
        child_pid = int(process.stdout.readline().strip())

        diagnostic = module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(process, 1.0)

        assert "Job Object" in diagnostic
        assert process.poll() is not None
        assert not module.psutil.pid_exists(child_pid)
    finally:
        for candidate_pid in (child_pid, process.pid):
            if candidate_pid and module.psutil.pid_exists(candidate_pid):
                real_run(
                    ["taskkill", "/PID", str(candidate_pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=3,
                    check=False,
                )


@pytest.mark.unit
def test_windows_broad_slow_fallback_obeys_one_hard_deadline(monkeypatch):
    module = _load_external_index_subprocess_module()

    class SlowProcess:
        def __init__(self, pid, children=()):
            self.pid = pid
            self._children = list(children)

        def children(self, recursive=False):
            time.sleep(0.005)
            return list(self._children)

        def create_time(self):
            time.sleep(0.005)
            return float(self.pid)

        def is_running(self):
            time.sleep(0.005)
            return True

        def suspend(self):
            time.sleep(0.005)

        def terminate(self):
            time.sleep(0.005)

        def kill(self):
            time.sleep(0.005)

    children = [SlowProcess(6100 + index) for index in range(8)]
    root = SlowProcess(6000, children)

    class RunningPopen:
        pid = 6000

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.psutil, "Process", lambda pid: root)
    monkeypatch.setattr(module.psutil, "wait_procs", lambda processes, timeout: ([], list(processes)))
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, **kwargs: module.subprocess.CompletedProcess(
            command,
            5,
            "",
            "simulated taskkill denial",
        ),
    )

    started = time.monotonic()
    with pytest.raises((RuntimeError, TimeoutError), match="deadline"):
        module.ExternalIndexTTSSubprocessProxy._terminate_process_tree(RunningPopen(), 0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.5


def _install_failing_temporary_directory(monkeypatch, module, message):
    real_temporary_directory = module.tempfile.TemporaryDirectory

    class FailingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            kwargs.pop("ignore_cleanup_errors", None)
            self._delegate = real_temporary_directory(*args, **kwargs)
            self.name = self._delegate.name

        def __enter__(self):
            return self.name

        def __exit__(self, exc_type, exc, traceback):
            self.cleanup()

        def cleanup(self):
            self._delegate.cleanup()
            raise OSError(message)

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", FailingTemporaryDirectory)


@pytest.mark.unit
def test_windows_private_temp_cleanup_retries_transient_directory_race(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    (temporary_path / "late-cache-write").write_bytes(b"cache")

    class TransientTemporaryDirectory:
        name = str(temporary_path)

        def cleanup(self):
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error

    real_rmtree = module.shutil.rmtree
    attempts = 0

    def retrying_rmtree(path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error
        return real_rmtree(path)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.shutil, "rmtree", retrying_rmtree)
    module._cleanup_temporary_directory(TransientTemporaryDirectory(), temporary_path)

    assert attempts == 2
    assert not temporary_path.exists()


@pytest.mark.unit
def test_external_index_success_surfaces_temporary_cleanup_failure(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    _install_failing_temporary_directory(monkeypatch, module, "cleanup locked")

    class SuccessfulProcess:
        pid = 5050
        returncode = 0

        def __init__(self, command, **kwargs):
            self.command = command

        def communicate(self, timeout=None):
            manifest = json.loads(Path(self.command[2]).read_text(encoding="utf-8"))
            with wave.open(manifest["output_path"], "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x01\x00" * 32)
            return "", ""

    monkeypatch.setattr(module.subprocess, "Popen", SuccessfulProcess)
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        temp_root=temp_root,
    )

    with pytest.raises(RuntimeError, match="temporary directory cleanup failed: cleanup locked"):
        proxy.infer(spk_audio_prompt=str(voice_path), text="成功后清理。", output_path=None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timeout", "expected_type", "primary_message"),
    [
        (False, RuntimeError, "official inference exploded"),
        (True, TimeoutError, "exceeded 0.1s"),
    ],
)
def test_external_index_primary_error_survives_temporary_cleanup_failure(
    monkeypatch,
    tmp_path,
    timeout,
    expected_type,
    primary_message,
):
    module = _load_external_index_subprocess_module()
    source_root, model_dir, _, voice_path, temp_root = _prepare_external_index_runtime(tmp_path)
    _install_failing_temporary_directory(monkeypatch, module, "cleanup locked")
    should_timeout = timeout

    class FailedProcess:
        pid = 5151
        returncode = 4

        def __init__(self, command, **kwargs):
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if should_timeout:
                raise module.subprocess.TimeoutExpired("indextts", timeout)
            return "", "official inference exploded"

        def poll(self):
            return self.returncode

    monkeypatch.setattr(module.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(
        module.ExternalIndexTTSSubprocessProxy,
        "_terminate_process_tree",
        staticmethod(lambda process, grace: ""),
    )
    proxy = module.ExternalIndexTTSSubprocessProxy(
        source_root=source_root,
        model_dir=model_dir,
        device="cuda:0",
        use_fp16=True,
        timeout_seconds=0.1 if timeout else 900,
        termination_grace_seconds=0.1,
        temp_root=temp_root,
    )

    with pytest.raises(expected_type) as error:
        proxy.infer(spk_audio_prompt=str(voice_path), text="主错误优先。", output_path=None)

    assert primary_message in str(error.value)
    assert "temporary directory cleanup failed: cleanup locked" in str(error.value)


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
def test_index_engine_preserves_complete_explicit_model_directory(monkeypatch, tmp_path):
    import engines.index_tts.index_tts as engine_module

    source_root = tmp_path / "source"
    model_dir = tmp_path / "model"
    source_root.mkdir()
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("model: local\n", encoding="utf-8")

    class FakeExternalProxy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(engine_module, "ExternalIndexTTSSubprocessProxy", FakeExternalProxy, raising=False)
    engine = engine_module.IndexTTSEngine(
        model_dir=str(model_dir), source_root=str(source_root), device="cpu", use_fp16=False
    )

    assert engine.model_dir == str(model_dir)


@pytest.mark.unit
@pytest.mark.parametrize("inference_fails", [False, True])
def test_index_engine_generate_does_not_create_unused_output_wav(monkeypatch, tmp_path, inference_fails):
    import engines.index_tts.index_tts as engine_module

    class FakeInference:
        def infer(self, **kwargs):
            if inference_fails:
                raise RuntimeError("official inference failed")
            return 22050, engine_module.np.ones((220, 1), dtype=engine_module.np.float32)

    engine = engine_module.IndexTTSEngine.__new__(engine_module.IndexTTSEngine)
    engine._tts_engine = FakeInference()
    engine._ensure_model_loaded = lambda: None
    monkeypatch.setattr(engine_module.folder_paths, "get_temp_directory", lambda: str(tmp_path))

    if inference_fails:
        with pytest.raises(RuntimeError, match="official inference failed"):
            engine.generate(text="失败路径", speaker_audio="voice.wav")
    else:
        audio = engine.generate(text="成功路径", speaker_audio="voice.wav")
        assert audio.shape == (1, 220)

    assert list(tmp_path.iterdir()) == []


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
    [
        ("resource_id", "gpt-resource-b"),
        ("version", "v3"),
        ("python_executable", "D:/portable/python.exe"),
    ],
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


@pytest.mark.unit
def test_registered_gpt_constructor_error_is_not_converted_to_silent_audio(monkeypatch):
    module = _load_unified_node("tts_text_node.py", "gpt_constructor_error_test")

    def fail_constructor(_engine_data):
        raise RuntimeError("External GPT-SoVITS subprocess exited 1: torchcodec mismatch")

    node = module.UnifiedTTSTextNode()
    monkeypatch.setattr(node, "_create_proper_engine_node_instance", fail_constructor)
    engine = {
        "engine_type": "gpt_sovits",
        "config": {"resource_id": "gpt-sovits-local"},
    }

    with pytest.raises(RuntimeError, match="torchcodec mismatch"):
        node.generate_speech(engine, "must fail terminally", "none", 0)


@pytest.mark.unit
def test_registered_gpt_processor_constructor_preserves_external_diagnostic(monkeypatch):
    module = _load_unified_node("tts_text_node.py", "gpt_processor_constructor_error_test")
    import engines.processors.gpt_sovits_processor as processor_module

    class FailedProcessor:
        def __init__(self, _config):
            raise RuntimeError("External GPT-SoVITS compatible interpreter is missing")

    monkeypatch.setattr(processor_module, "GPTSovitsProcessor", FailedProcessor)
    node = module.UnifiedTTSTextNode()
    engine = {
        "engine_type": "gpt_sovits",
        "config": {
            "resource_id": "gpt-sovits-local",
            "gpt_weight": "C:/gpt/s1.ckpt",
            "sovits_weight": "C:/gpt/s2.pth",
            "gpt_sovits_home": "C:/gpt/source",
        },
    }

    with pytest.raises(RuntimeError, match="compatible interpreter is missing"):
        node._create_proper_engine_node_instance(engine)


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
def test_text_node_forces_each_registered_runtime_prompt_to_execute():
    module = _load_unified_node("tts_text_node.py", "registered_runtime_is_changed_test")

    changed = module.UnifiedTTSTextNode.IS_CHANGED(
        TTS_engine=_index_engine_data(),
        text="must execute",
        narrator_voice="none",
        seed=1,
    )

    assert isinstance(changed, float)
    assert math.isnan(changed)


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
