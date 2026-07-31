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
    child_environment = observed["kwargs"]["env"]
    assert child_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    pycache_prefix = Path(child_environment["PYTHONPYCACHEPREFIX"])
    assert pycache_prefix.is_relative_to(temp_root)
    assert not pycache_prefix.is_relative_to(source_root)
    assert Path(child_environment["NUMBA_CACHE_DIR"]).is_relative_to(temp_root)
    assert Path(child_environment["MPLCONFIGDIR"]).is_relative_to(temp_root)
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

    assert terminated == [(4444, 0.25)]
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
            if len(self.timeouts) == 1:
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
    assert instances[0].timeouts[0] == 0.5
    assert 0 < instances[0].timeouts[1] <= 0.21
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
    assert instances[0].timeouts[0] == 0.5
    assert 0 < instances[0].timeouts[1] <= 0.21
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
            if self.calls == 1:
                raise module.subprocess.TimeoutExpired("indextts", timeout)
            return ("", "")

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
            assert recursive is True
            return [child] if self.pid == 4848 else []

        @staticmethod
        def is_running():
            return True

        @staticmethod
        def terminate():
            pass

        @staticmethod
        def kill():
            pass

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
            if should_timeout and self.calls == 1:
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
