"""Unit coverage for API-bridge runtime ownership and deterministic release."""

from __future__ import annotations

import threading
import importlib.util
from pathlib import Path

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
    assert make_runtime_key("text", first) == f"text:{first}"
    assert make_runtime_key("srt", first) == f"srt:{first}"


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
        "errors": [{"runtime_key": "text:index:b", "error": "boom"}],
    }
    assert second == {"released": [], "errors": []}
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

    assert report == {"released": ["text:same"], "errors": []}
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

    assert registry.status()[0]["runtime_key"] == "srt:index-cache"

    node._release_target_runtime("index-cache", "index_tts")

    assert runtime.cleanup_calls == 1
    assert registry.status() == []


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
