"""CPU-only lease, direct-node and cleanup-fence regressions."""
from __future__ import annotations

import json
import io
from pathlib import Path
from types import SimpleNamespace
import sys
from contextlib import nullcontext
from urllib.error import HTTPError

import pytest

from api_bridge import gpu_coordination as gpu
from api_bridge import runtime_registry as runtimes


@pytest.fixture
def control(monkeypatch):
    events = []
    valid = [True]
    monkeypatch.setattr(gpu, "configuration", lambda: {"heartbeat_seconds": 0.1, "resource_group": "g"})
    client = SimpleNamespace(
        acquire_comfy=lambda *_a, **_k: events.append("acquire") or "lease",
        check_comfy=lambda _t: events.append("check") or valid[0],
        release_comfy=lambda _t, clean: events.append(("release", clean)) or clean,
    )
    monkeypatch.setattr(gpu, "CoordinatorClient", lambda _cfg: client)
    monkeypatch.setattr(gpu, "_gpu_memory_released", lambda: True)
    registry = runtimes.RuntimeRegistry()
    monkeypatch.setattr(runtimes, "get_runtime_registry", lambda: registry)
    class UserInterrupt(Exception):
        pass
    mm = SimpleNamespace(InterruptProcessingException=UserInterrupt, throw_exception_if_processing_interrupted=lambda: None)
    monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace(model_management=mm))
    return events, registry, valid, mm


@pytest.mark.parametrize("engine", ["gpt_sovits", "index_tts", "cosyvoice"])
def test_direct_workflow_gates_before_engine_creation_and_cleans_before_release(control, engine):
    events, registry, _valid, _mm = control
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            assert events[:2] == ["acquire", "check"]
            assert gpu.coordination_active()
            events.append("construct")
            registry.register(runtimes.RuntimeHandle.create("model", engine, "r", "cuda", lambda: events.append("unload")))
            with registry.lease("model"):
                events.append("infer")
            return "completed"
    data = {"engine_type": engine}
    if engine in {"index_tts", "cosyvoice"}:
        data["config"] = {f"{engine}_home": "operator-configured-checkout"}
    assert Node().generate(data) == "completed"
    assert events == ["acquire", "check", "construct", "infer", "unload", ("release", True)]
    assert not gpu.coordination_active()
    assert registry.status() == []


def test_native_priority_has_distinct_marker_after_cleanup(control):
    events, registry, _valid, _mm = control
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            registry.register(runtimes.RuntimeHandle.create("model", "gpt_sovits", "r", "cuda", lambda: events.append("unload")))
            with registry.lease("model"):
                gpu._current.guard.revoked.set()
                assert gpu.preemption_requested()
                gpu.check_gpu_interrupt()
    with pytest.raises(gpu.TTSMoreGPUPreempted, match="TTSMoreGPUPreempted"):
        Node().generate({"engine_type": "gpt_sovits"})
    assert events[-2:] == ["unload", ("release", True)]


@pytest.mark.parametrize("tree_verified", [True, False])
def test_external_child_preemption_uses_tree_cleanup_before_releasing_gpu(control, tree_verified):
    from engines.index_tts.external_subprocess import ExternalIndexTTSSubprocessProxy

    events, _registry, _valid, _mm = control
    proxy = ExternalIndexTTSSubprocessProxy.__new__(ExternalIndexTTSSubprocessProxy)
    proxy.timeout_seconds = 10
    proxy.interrupt_check = lambda: False
    proxy._cleanup_timed_out_process = lambda _p: events.append("tree_cleanup") or ("", "", "", tree_verified)
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            gpu._current.guard.revoked.set()
            proxy._communicate_with_control(object(), "IndexTTS")
    expected = gpu.TTSMoreGPUPreempted if tree_verified else gpu.TTSMoreGPUCleanupFailed
    with pytest.raises(expected):
        Node().generate({"engine_type": "index_tts", "config": {"index_tts_home": "operator-checkout"}})
    assert events[-2:] == ["tree_cleanup", ("release", tree_verified)]


@pytest.mark.parametrize("reason", ["busy", "unload", "heartbeat", "child", "cuda"])
def test_dirty_cleanup_retains_fence_and_never_becomes_retryable(control, monkeypatch, reason):
    events, registry, _valid, _mm = control
    if reason == "cuda":
        monkeypatch.setattr(gpu, "_gpu_memory_released", lambda: False)
    leaked = []
    def unload():
        if reason == "unload":
            raise RuntimeError("unload failed")
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            registry.register(runtimes.RuntimeHandle.create("model", "gpt_sovits", "r", "cuda", unload))
            if reason == "busy":
                leaked.append(registry.lease("model"))
            elif reason == "heartbeat":
                gpu._current.guard.unavailable.set()
            elif reason == "child":
                raise RuntimeError("child tree exit unconfirmed")
            return "audio"
    with pytest.raises(gpu.TTSMoreGPUCleanupFailed):
        Node().generate({"engine_type": "gpt_sovits"})
    assert events[-1] == ("release", False)
    assert not gpu.coordination_active()


def test_completed_fragment_wins_priority_race(control):
    events, _registry, _valid, _mm = control
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            gpu._current.guard.revoked.set()
            return "completed"
    assert Node().generate({"engine_type": "index_tts", "config": {"index_tts_home": "operator-checkout"}}) == "completed"
    assert events[-1] == ("release", True)


def test_cuda_allocator_verification_checks_every_device_without_using_gpu(monkeypatch):
    allocations = [0, 1]
    checked = []
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, device=lambda _i: nullcontext(), empty_cache=lambda: None, memory_allocated=lambda i: checked.append(i) or allocations[i])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert gpu._gpu_memory_released() is False
    assert checked == [0, 1]
    allocations[1] = 0
    assert gpu._gpu_memory_released() is True
    cuda.memory_allocated = lambda _i: (_ for _ in ()).throw(RuntimeError("allocator unavailable"))
    assert gpu._gpu_memory_released() is False


@pytest.mark.parametrize("status", [408, 409])
def test_control_http_timeout_is_waitable_and_other_errors_are_fatal(status):
    client = gpu.CoordinatorClient({"resource_group": "g", "coordinator_url": "http://127.0.0.1:9000", "token": "private-token-long"})
    client.opener = SimpleNamespace(open=lambda *_a, **_k: (_ for _ in ()).throw(HTTPError("url", status, "timeout", {}, io.BytesIO(b'{"error":{"code":"coordination_timeout"}}'))))
    with pytest.raises(gpu.TTSMoreGPUPreempted):
        client.acquire_comfy("holder", timeout=1)


def test_admission_waits_native_but_dirty_fence_stops_without_constructing(control, monkeypatch):
    events, _registry, _valid, _mm = control
    attempts = []
    def acquire(*_a, **_k):
        attempts.append(True)
        if len(attempts) == 1:
            raise gpu.TTSMoreGPUPreempted("native busy")
        return "lease"
    client = gpu.CoordinatorClient({})
    client.acquire_comfy = acquire
    client.snapshot = lambda: {"groups": {"g": {"state": "native_busy", "native_waiting": 1, "natives": {"n": {"fresh": True, "status": {"ready": True}}}}}}
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            events.append("construct")
            return "audio"
    data = {"engine_type": "index_tts", "config": {"index_tts_home": "operator-checkout"}}
    assert Node().generate(data) == "audio"
    assert len(attempts) == 2
    attempts.clear()
    client.snapshot = lambda: {"groups": {"g": {"comfy": {"cleanup_failed": True}}}}
    with pytest.raises(gpu.TTSMoreGPUCleanupFailed, match="fence"):
        Node().generate(data)
    assert len(attempts) == 1


def test_other_engines_fail_before_load_when_coordination_enabled(control):
    events, _registry, _valid, _mm = control
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            pytest.fail("must not construct an unsupported GPU engine")
    with pytest.raises(gpu.TTSMoreGPUCleanupFailed):
        Node().generate({"engine_type": "unsupported"})
    assert events == []


@pytest.mark.parametrize("engine", ["index_tts", "cosyvoice"])
def test_inprocess_engine_is_rejected_before_construction_only_when_enabled(control, monkeypatch, engine):
    events, _registry, _valid, _mm = control
    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            events.append("construct")
            return "audio"
    data = {"engine_type": engine, "config": {"model_path": "operator-model"}}
    with pytest.raises(gpu.TTSMoreGPUCleanupFailed, match="in-process"):
        Node().generate(data)
    assert events == []
    monkeypatch.setattr(gpu, "configuration", lambda: None)
    assert Node().generate(data) == "audio"
    assert events == ["construct"]


@pytest.mark.parametrize("phase", ["runtime_release", "initial_check"])
def test_cleanup_exceptions_stop_heartbeat_and_preserve_dirty_fence(control, monkeypatch, phase):
    events, registry, _valid, _mm = control
    original = RuntimeError("injected " + phase)
    guards = []
    guard_type = gpu._Guard

    def create_guard(*args):
        guard = guard_type(*args)
        guards.append(guard)
        return guard

    def fail():
        raise original

    monkeypatch.setattr(gpu, "_Guard", create_guard)
    if phase == "runtime_release":
        monkeypatch.setattr(registry, "release", fail)
    else:
        monkeypatch.setattr(gpu.CoordinatorClient({}), "check_comfy", lambda _token: fail())

    class Node:
        @gpu.coordinated_execution
        def generate(self, TTS_engine):
            assert phase != "initial_check", "a failed initial check must prevent engine construction"
            events.append("construct")
            return "audio"

    with pytest.raises(gpu.TTSMoreGPUCleanupFailed) as caught:
        Node().generate({"engine_type": "gpt_sovits"})
    assert caught.value.__cause__ is original
    assert len(guards) == 1
    assert guards[0].stop.is_set()
    assert not guards[0].thread.is_alive()
    assert events[-1] == ("release", False)
    assert not gpu.coordination_active()
    assert gpu._execution_lock.acquire(blocking=False)
    gpu._execution_lock.release()


def test_local_config_defaults_disabled_and_public_capability_has_no_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("TTS_AUDIO_SUITE_GPU_COORDINATION_CONFIG", raising=False)
    assert gpu.capabilities()["enabled"] is False
    config = tmp_path / "private.json"
    config.write_text(json.dumps({"enabled": True, "resource_group": "g", "groups": {"g": {"coordinator_url": "http://127.0.0.1:9010", "token": "private-test-secret"}}}), encoding="utf8")
    monkeypatch.setenv("TTS_AUDIO_SUITE_GPU_COORDINATION_CONFIG", str(config))
    assert gpu.capabilities() == {"enabled": True, "resource_group": "g", "protocol_version": 1}
    config.write_text(json.dumps({"enabled": True, "resource_group": "g", "groups": {"g": {"coordinator_url": "http://192.0.2.1:9010", "token": "private-test-secret"}}}), encoding="utf8")
    with pytest.raises(gpu.TTSMoreGPUCleanupFailed, match="Invalid local"):
        gpu.configuration()


def test_text_and_srt_entrypoints_are_wrapped_before_body():
    # Syntax/placement regression without importing GPU-heavy engines.
    root = Path(__file__).resolve().parents[2]
    for name in ["tts_text_node.py", "tts_srt_node.py"]:
        source = (root / "nodes" / "unified" / name).read_text(encoding="utf8")
        import ast
        owners = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in {"UnifiedTTSTextNode", "UnifiedTTSSRTNode"}]
        functions = [node for owner in owners for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in {"generate_speech", "generate_srt_speech"}]
        assert len(functions) == 1
        assert any(isinstance(item, ast.Name) and item.id == "coordinated_execution" for item in functions[0].decorator_list)
