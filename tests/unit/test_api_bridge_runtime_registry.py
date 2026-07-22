"""Unit coverage for API-bridge runtime ownership and deterministic release."""

from __future__ import annotations

import threading
import importlib.util
from pathlib import Path

import pytest
import torch

import api_bridge.runtime_registry as runtime_registry_module
from api_bridge.runtime_registry import RuntimeHandle, RuntimeRegistry, make_cache_identity, make_runtime_key


def test_runtime_key_uses_owner_and_hashed_cache_identity_without_path_leakage():
    first = make_cache_identity(
        "gpt_sovits",
        {"resource_id": "gpt-main", "model_path": r"J:\\models\\private\\voice.ckpt", "device": "cuda"},
    )
    second = make_cache_identity(
        "gpt_sovits",
        {"device": "cuda", "model_path": r"J:\\models\\private\\voice.ckpt", "resource_id": "gpt-main"},
    )

    assert first == second
    assert "voice.ckpt" not in first
    assert make_runtime_key("text", "node-a", first) == f"text:node-a:{first}"
    assert make_runtime_key("srt", "node-a", first) == f"srt:node-a:{first}"


@pytest.mark.parametrize(
    "stable_params",
    [
        {"unsafe": object()},
        {"unsafe": ("tuple",)},
        {1: "non-string-key"},
        {"unsafe": float("nan")},
    ],
)
def test_cache_identity_rejects_non_json_or_non_deterministic_values(stable_params):
    with pytest.raises(TypeError, match="JSON-safe"):
        make_cache_identity("gpt_sovits", stable_params)


def test_cache_identity_accepts_nested_json_values_deterministically():
    first = make_cache_identity(
        "cosyvoice",
        {"options": [None, True, 3, 1.25, "safe", {"nested": [False]}]},
    )
    second = make_cache_identity(
        "cosyvoice",
        {"options": [None, True, 3, 1.25, "safe", {"nested": [False]}]},
    )

    assert first == second


def test_unique_node_owner_tokens_prevent_same_configuration_replacement():
    registry = RuntimeRegistry()
    calls = []
    identity = make_cache_identity("gpt_sovits", {"resource_id": "gpt-main", "device": "cuda"})
    first_key = make_runtime_key("text", "node-a", identity)
    second_key = make_runtime_key("text", "node-b", identity)
    registry.register(RuntimeHandle.create(first_key, "gpt_sovits", "gpt-main", "cuda", lambda: calls.append("a")))
    registry.register(RuntimeHandle.create(second_key, "gpt_sovits", "gpt-main", "cuda", lambda: calls.append("b")))

    assert calls == []
    assert [item["runtime_key"] for item in registry.status()] == sorted([first_key, second_key])

    report = registry.release(runtime_key=first_key)

    assert report == {"released": [first_key], "busy": [], "errors": []}
    assert calls == ["a"]
    assert [item["runtime_key"] for item in registry.status()] == [second_key]


def test_release_is_idempotent_and_continues_after_failure():
    calls = []
    registry = RuntimeRegistry()
    registry.register(
        RuntimeHandle.create("text:gpt:a", "gpt_sovits", "voice-a", "cuda", lambda: calls.append("a"))
    )
    registry.register(
        RuntimeHandle.create(
            "text:index:b",
            "index_tts",
            "index",
            "cuda",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    )

    report = registry.release()
    second = registry.release()

    assert calls == ["a"]
    assert report == {
        "released": ["text:gpt:a"],
        "busy": [],
        "errors": [
            {
                "runtime_key": "text:index:b",
                "code": "runtime_unload_failed",
                "message": "Runtime cleanup failed; inspect server logs.",
            }
        ],
    }
    assert second == {"released": [], "busy": [], "errors": []}
    assert registry.status() == []


def test_status_is_sorted_safe_and_touch_updates_timestamp(monkeypatch):
    clock = iter((10.0, 10.0, 20.0))
    monkeypatch.setattr("api_bridge.runtime_registry.time.time", lambda: next(clock))
    registry = RuntimeRegistry()
    registry.register(RuntimeHandle.create("text:z", "cosyvoice", "cosy", "cpu", lambda: None))
    registry.register(RuntimeHandle.create("srt:a", "index_tts", "index", "cuda", lambda: None))

    before_touch = registry.status()
    registry.touch("text:z")
    status = registry.status()

    assert [item["runtime_key"] for item in status] == ["srt:a", "text:z"]
    assert status[1] == {
        "runtime_key": "text:z",
        "engine": "cosyvoice",
        "resource_id": "cosy",
        "device": "cpu",
        "loaded_at": 10.0,
        "last_used_at": 20.0,
        "active": 0,
        "busy": False,
    }
    assert before_touch[1]["last_used_at"] == 10.0
    assert all("unload" not in item and "config" not in item and "path" not in item for item in status)


def test_replacement_and_reentrant_unload_do_not_hold_registry_lock():
    registry = RuntimeRegistry()
    calls = []

    def old_unload():
        calls.append("old")
        registry.register(RuntimeHandle.create("nested", "cosyvoice", "nested", "cpu", lambda: calls.append("nested")))

    old = RuntimeHandle.create("text:same", "gpt_sovits", "voice", "cuda", old_unload)
    replacement = RuntimeHandle.create("text:same", "gpt_sovits", "voice", "cuda", lambda: calls.append("new"))
    registry.register(old)
    registry.register(replacement)

    assert calls == ["old"]
    assert [item["runtime_key"] for item in registry.status()] == ["nested", "text:same"]

    report = registry.release(runtime_key="text:same")

    assert report == {"released": ["text:same"], "busy": [], "errors": []}
    assert calls == ["old", "new"]
    assert [item["runtime_key"] for item in registry.status()] == ["nested"]


def test_concurrent_register_touch_and_filtered_release_are_safe():
    registry = RuntimeRegistry()
    errors = []
    start = threading.Barrier(21)

    def worker(index):
        try:
            start.wait()
            runtime_key = f"text:{index}"
            registry.register(RuntimeHandle.create(runtime_key, "index_tts", "shared" if index % 2 else "other", "cpu", lambda: None))
            registry.touch(runtime_key)
        except Exception as exc:  # pragma: no cover - assertion below is the contract
            errors.append(exc)

    def releaser():
        try:
            start.wait()
            for _ in range(10):
                registry.release(resource_id="shared")
        except Exception as exc:  # pragma: no cover - assertion below is the contract
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(20)]
    threads.append(threading.Thread(target=releaser))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    report = registry.release(resource_id="shared")

    assert errors == []
    assert report["errors"] == []
    assert report["released"] == sorted(report["released"])
    assert all(item["resource_id"] == "other" for item in registry.status())


def test_release_reports_busy_and_requires_retry_after_inflight_lease_completes():
    registry = RuntimeRegistry()
    started = threading.Event()
    finish_generation = threading.Event()
    calls = []
    runtime_key = "text:node-a:gpt"
    registry.register(RuntimeHandle.create(runtime_key, "gpt_sovits", "gpt-main", "cuda", lambda: calls.append("unload")))

    def synthesize():
        with registry.lease(runtime_key):
            started.set()
            assert finish_generation.wait(timeout=2)

    worker = threading.Thread(target=synthesize)
    worker.start()
    assert started.wait(timeout=2)

    report = registry.release(runtime_key=runtime_key)

    assert report == {
        "released": [],
        "busy": [
            {
                "runtime_key": runtime_key,
                "code": "runtime_busy",
                "message": "Runtime is in use; retry release later.",
            }
        ],
        "errors": [],
    }
    assert calls == []
    assert registry.status()[0]["active"] == 1
    assert registry.status()[0]["busy"] is True

    finish_generation.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert calls == []
    assert registry.status()[0]["active"] == 0

    retry = registry.release(runtime_key=runtime_key)

    assert retry == {"released": [runtime_key], "busy": [], "errors": []}
    assert calls == ["unload"]
    assert registry.status() == []


def test_global_test_reset_releases_old_registry_and_replaces_it_after_safe_failure():
    runtime_registry_module._reset_runtime_registry_for_tests()
    previous = runtime_registry_module.get_runtime_registry()
    previous.register(
        RuntimeHandle.create(
            "text:node-a:index",
            "index_tts",
            "index-main",
            "cpu",
            lambda: (_ for _ in ()).throw(RuntimeError(r"private J:\models\index")),
        )
    )

    report = runtime_registry_module._reset_runtime_registry_for_tests()

    assert report["reset"] is True
    assert report["errors"] == [
        {
            "runtime_key": "text:node-a:index",
            "code": "runtime_unload_failed",
            "message": "Runtime cleanup failed; inspect server logs.",
        }
    ]
    assert "private" not in repr(report)
    assert runtime_registry_module.get_runtime_registry() is not previous
    assert runtime_registry_module.get_runtime_registry().status() == []


def test_global_test_reset_refuses_to_replace_registry_while_runtime_is_busy():
    runtime_registry_module._reset_runtime_registry_for_tests()
    previous = runtime_registry_module.get_runtime_registry()
    previous.register(RuntimeHandle.create("text:busy", "gpt_sovits", "gpt", "cpu", lambda: None))
    lease = previous.lease("text:busy")

    report = runtime_registry_module._reset_runtime_registry_for_tests()

    assert report["reset"] is False
    assert report["busy"][0]["code"] == "runtime_busy"
    assert runtime_registry_module.get_runtime_registry() is previous

    lease.close()
    runtime_registry_module._reset_runtime_registry_for_tests()


def test_srt_target_runtime_has_its_own_cache_owner_and_exact_release(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_srt_node.py"
    spec = importlib.util.spec_from_file_location("runtime_registry_srt_test_node", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry, raising=False)

    class FakeRuntime:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1

    runtime = FakeRuntime()
    node = module.UnifiedTTSSRTNode()
    node._register_target_runtime("index-cache", "index_tts", {"resource_id": "index-main"}, "cpu", runtime)

    assert registry.status()[0]["runtime_key"].startswith("srt:")
    assert registry.status()[0]["runtime_key"].endswith(":index-cache")

    node._release_target_runtime("index-cache", "index_tts")

    assert runtime.cleanup_calls == 1
    assert registry.status() == []


def test_two_srt_nodes_with_same_target_config_keep_independent_runtime_owners(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_srt_node.py"
    spec = importlib.util.spec_from_file_location("runtime_registry_two_srt_owners", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)

    class FakeRuntime:
        def __init__(self):
            self.cleanup_calls = 0

        def cleanup(self):
            self.cleanup_calls += 1

    nodes = [module.UnifiedTTSSRTNode(), module.UnifiedTTSSRTNode()]
    runtimes = [FakeRuntime(), FakeRuntime()]
    for node, runtime in zip(nodes, runtimes):
        node._cached_engine_instances["same-cache"] = {"instance": runtime, "timestamp": 0.0}
        node._register_target_runtime(
            "same-cache", "cosyvoice", {"resource_id": "cosy-main"}, "cpu", runtime
        )

    status = registry.status()
    assert len(status) == 2
    assert status[0]["runtime_key"] != status[1]["runtime_key"]

    registry.release(runtime_key=status[0]["runtime_key"])

    assert sorted(runtime.cleanup_calls for runtime in runtimes) == [0, 1]
    assert len(registry.status()) == 1


def test_two_text_nodes_with_same_target_config_keep_independent_runtime_owners(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_text_node.py"
    spec = importlib.util.spec_from_file_location("runtime_registry_two_text_owners", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)

    created = []
    start = threading.Barrier(3)

    class FakeProcessor:
        def __init__(self, config):
            self.cleanup_calls = 0
            created.append(self)

        def cleanup(self):
            self.cleanup_calls += 1

    from engines.processors import gpt_sovits_processor

    monkeypatch.setattr(gpt_sovits_processor, "GPTSovitsProcessor", FakeProcessor)
    engine = {
        "engine_type": "gpt_sovits",
        "config": {
            "resource_id": "gpt-main",
            "device": "cpu",
            "gpt_weight": "gpt.ckpt",
            "sovits_weight": "sovits.pth",
        },
    }
    nodes = [module.UnifiedTTSTextNode(), module.UnifiedTTSTextNode()]
    errors = []

    def create_runtime(node):
        try:
            start.wait()
            node._create_proper_engine_node_instance(engine)
        except Exception as exc:  # pragma: no cover - assertion below is the contract
            errors.append(exc)

    workers = [threading.Thread(target=create_runtime, args=(node,)) for node in nodes]
    for worker in workers:
        worker.start()
    start.wait()
    for worker in workers:
        worker.join(timeout=3)

    status = registry.status()
    assert errors == []
    assert all(not worker.is_alive() for worker in workers)
    assert len(status) == 2
    assert status[0]["runtime_key"] != status[1]["runtime_key"]
    assert len(created) == 2

    registry.release(runtime_key=status[0]["runtime_key"])

    assert sorted(processor.cleanup_calls for processor in created) == [0, 1]
    assert len(registry.status()) == 1


def test_text_gpt_dispatch_holds_lease_for_full_generation_window(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_text_node.py"
    spec = importlib.util.spec_from_file_location("runtime_registry_text_dispatch", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)
    started = threading.Event()
    finish = threading.Event()
    created = []

    class FakeProcessor:
        sample_rate = 32000

        def __init__(self, config):
            self.cleanup_calls = 0
            created.append(self)

        def process_text(self, **kwargs):
            started.set()
            assert finish.wait(timeout=3)
            return torch.ones(1, 32), "generated"

        def cleanup(self):
            self.cleanup_calls += 1

    from engines.processors import gpt_sovits_processor

    monkeypatch.setattr(gpt_sovits_processor, "GPTSovitsProcessor", FakeProcessor)
    node = module.UnifiedTTSTextNode()
    monkeypatch.setattr(
        node,
        "_get_voice_reference",
        lambda *_: ("reference.wav", {"waveform": torch.zeros(1, 8)}, "reference", "narrator"),
    )
    engine = {
        "engine_type": "gpt_sovits",
        "config": {
            "resource_id": "gpt-main",
            "device": "cpu",
            "gpt_weight": "gpt.ckpt",
            "sovits_weight": "sovits.pth",
        },
    }
    results = []
    worker = threading.Thread(
        target=lambda: results.append(node.generate_speech(engine, "text", "none", 7))
    )
    worker.start()
    assert started.wait(timeout=3)
    runtime_key = registry.status()[0]["runtime_key"]

    report = registry.release(runtime_key=runtime_key)

    assert report["busy"][0]["code"] == "runtime_busy"
    assert created[0].cleanup_calls == 0
    finish.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert "generated" in results[0][1]
    assert registry.release(runtime_key=runtime_key)["released"] == [runtime_key]
    assert created[0].cleanup_calls == 1


def test_srt_cosy_dispatch_holds_lease_for_full_generation_window(monkeypatch):
    node_path = Path(__file__).parents[2] / "nodes" / "unified" / "tts_srt_node.py"
    spec = importlib.util.spec_from_file_location("runtime_registry_srt_dispatch", node_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = RuntimeRegistry()
    monkeypatch.setattr(module, "get_runtime_registry", lambda: registry)
    started = threading.Event()
    finish = threading.Event()
    created = []

    class FakeSRTProcessor:
        def __init__(self, wrapper, config):
            self.cleanup_calls = 0
            created.append(self)

        def process_srt_content(self, **kwargs):
            started.set()
            assert finish.wait(timeout=3)
            return (
                {"waveform": torch.ones(1, 1, 32), "sample_rate": 24000},
                "generated",
                "timing",
                "adjusted",
            )

        def cleanup(self):
            self.cleanup_calls += 1

    fake_module = type("FakeCosyModule", (), {"CosyVoiceSRTProcessor": FakeSRTProcessor})
    fake_spec = type("FakeSpec", (), {"loader": type("FakeLoader", (), {"exec_module": lambda self, module: None})()})
    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *_: fake_spec)
    monkeypatch.setattr(module.importlib.util, "module_from_spec", lambda _: fake_module)
    node = module.UnifiedTTSSRTNode()
    monkeypatch.setattr(node, "_get_voice_reference", lambda *_: (None, None, "", "narrator"))
    engine = {
        "engine_type": "cosyvoice",
        "config": {
            "resource_id": "cosy-main",
            "device": "cpu",
            "model_path": "C:/cosy/model",
            "use_fp16": False,
        },
    }
    results = []
    worker = threading.Thread(
        target=lambda: results.append(node.generate_srt_speech(engine, "1\n00:00:00,000 --> 00:00:01,000\nHi", "none", 7, "natural"))
    )
    worker.start()
    assert started.wait(timeout=3)
    runtime_key = registry.status()[0]["runtime_key"]

    report = registry.release(runtime_key=runtime_key)

    assert report["busy"][0]["code"] == "runtime_busy"
    assert created[0].cleanup_calls == 0
    finish.set()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert "generated" in results[0][1]
    assert registry.release(runtime_key=runtime_key)["released"] == [runtime_key]
    assert created[0].cleanup_calls == 1


def test_target_processor_cleanup_aliases_are_safe_before_model_initialization():
    from engines.processors.cosyvoice_processor import CosyVoiceProcessor
    from engines.processors.index_tts_processor import IndexTTSProcessor

    index_processor = object.__new__(IndexTTSProcessor)
    index_processor.adapter = None
    cosy_processor = object.__new__(CosyVoiceProcessor)
    cosy_processor.adapter = None

    assert index_processor.cleanup() is None
    assert index_processor.unload() is None
    assert cosy_processor.cleanup() is None
    assert cosy_processor.unload() is None
