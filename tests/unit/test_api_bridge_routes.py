"""Contract coverage for versioned, non-execution API-bridge support routes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from aiohttp import web

from api_bridge.routes import (
    build_capabilities_payload,
    build_runtime_release_payload,
    is_asset_delete_blocked,
    register_api_bridge_routes,
)


class Registry:
    def capabilities(self):
        return [{"resource_id": "index-main", "engine": "index_tts", "ready": True}]


class RuntimeRegistry:
    def __init__(self, report=None, status=None):
        self.report = report or {"released": [], "busy": [], "errors": []}
        self._status = status or []
        self.calls = []

    def status(self):
        return self._status

    def release(self, **kwargs):
        self.calls.append(kwargs)
        return self.report


class PromptQueue:
    def __init__(self, pending=None, running=None):
        self.pending = [] if pending is None else pending
        self.running = [] if running is None else running

    def get_current_queue(self):
        return self.pending, self.running


class PromptServer:
    def __init__(self, queue):
        self.prompt_queue = queue


class AssetStore:
    max_bytes = 4

    def __init__(self):
        self.created = []
        self.deleted = []

    def create(self, filename, content):
        self.created.append((filename, content))
        return SimpleNamespace(asset_id="asset-id", sha256="a" * 64, size_bytes=len(content), path=SimpleNamespace(name="asset-id.wav"))

    def delete(self, asset_id):
        if asset_id == "missing":
            raise ValueError("unknown asset_id")
        self.deleted.append(asset_id)

    def require(self, asset_id):
        if asset_id == "missing":
            raise ValueError("unknown asset_id")


class UploadField:
    def __init__(self, name, filename, chunks):
        self.name = name
        self.filename = filename
        self._chunks = iter(chunks)

    async def read_chunk(self, size):
        return next(self._chunks, b"")


class UploadRequest:
    def __init__(self, fields):
        self.fields = iter(fields)

    async def multipart(self):
        request = self

        class Reader:
            async def next(self):
                return next(request.fields, None)

        return Reader()


def test_capabilities_payload_is_versioned_redacted_and_has_stable_nodes():
    payload = build_capabilities_payload(Registry(), plugin_version="5.5.2")

    assert payload == {
        "protocol_version": 1,
        "plugin_version": "5.5.2",
        "nodes": {
            "gpt_sovits": "TTSExternalGPTSovitsEngine",
            "index_tts": "TTSExternalIndexTTSEngine",
            "cosyvoice": "TTSExternalCosyVoiceEngine",
            "audio_asset": "TTSExternalAudioAsset",
            "text": "UnifiedTTSTextNode",
            "save_audio": "SaveAudio",
        },
        "resources": [{"resource_id": "index-main", "engine": "index_tts", "ready": True}],
    }
    assert "synthesize" not in json.dumps(payload)
    assert "path" not in json.dumps(payload)


def test_runtime_release_rejects_an_empty_or_unrecognized_payload():
    for payload in ({}, {"unknown": True}, {"runtime_key": ""}, {"all": False}):
        try:
            build_runtime_release_payload(payload)
        except ValueError:
            pass
        else:  # pragma: no cover - keeps each unsafe request a hard failure
            raise AssertionError(f"unsafe release payload was accepted: {payload}")


def test_runtime_release_accepts_one_explicit_target_or_explicit_all():
    assert build_runtime_release_payload({"runtime_key": "text:node:hash"}) == {
        "runtime_key": "text:node:hash",
        "resource_id": None,
    }
    assert build_runtime_release_payload({"resource_id": "gpt-main"}) == {
        "runtime_key": None,
        "resource_id": "gpt-main",
    }
    assert build_runtime_release_payload({"all": True}) == {"runtime_key": None, "resource_id": None}


def test_delete_guard_blocks_pending_prompt_or_busy_target_runtime():
    idle_prompt_server = PromptServer(PromptQueue())

    assert is_asset_delete_blocked(idle_prompt_server, RuntimeRegistry()) is False
    assert is_asset_delete_blocked(PromptServer(PromptQueue(pending=["prompt"])), RuntimeRegistry()) is True
    assert is_asset_delete_blocked(
        idle_prompt_server,
        RuntimeRegistry(status=[{"runtime_key": "text:node:hash", "busy": True}]),
    ) is True


def test_registered_support_routes_exclude_synthesis_and_preserve_registration_idempotency():
    routes = web.RouteTableDef()

    register_api_bridge_routes(routes, plugin_version="test")
    register_api_bridge_routes(routes, plugin_version="test")

    defined = {(item.method, item.path) for item in routes}
    assert defined == {
        ("GET", "/api/tts-audio-suite/v1/capabilities"),
        ("POST", "/api/tts-audio-suite/v1/assets/audio"),
        ("DELETE", "/api/tts-audio-suite/v1/assets/audio/{asset_id}"),
        ("GET", "/api/tts-audio-suite/v1/runtime/status"),
        ("POST", "/api/tts-audio-suite/v1/runtime/release"),
    }
    assert not any("synth" in path for _, path in defined)


def test_registered_runtime_release_handlers_keep_busy_and_error_contracts():
    async def run():
        routes = web.RouteTableDef()
        runtime_registry = RuntimeRegistry(
            report={"released": [], "busy": [{"code": "runtime_busy"}], "errors": []}
        )
        register_api_bridge_routes(
            routes,
            plugin_version="test",
            runtime_registry_getter=lambda: runtime_registry,
        )
        handler = next(item.handler for item in routes if item.path.endswith("/runtime/release"))

        class Request:
            async def json(self):
                return {"runtime_key": "text:node:hash"}

        response = await handler(Request())
        assert response.status == 409
        assert json.loads(response.body) == runtime_registry.report
        assert runtime_registry.calls == [{"runtime_key": "text:node:hash", "resource_id": None}]

    asyncio.run(run())


def test_registered_upload_stops_chunked_oversize_before_asset_store_create():
    async def run():
        routes = web.RouteTableDef()
        store = AssetStore()
        register_api_bridge_routes(routes, plugin_version="test", asset_store_getter=lambda: store)
        handler = next(item.handler for item in routes if item.path.endswith("/assets/audio") and item.method == "POST")

        response = await handler(UploadRequest([UploadField("audio", "voice.wav", [b"123", b"45"])]))

        assert response.status == 413
        assert json.loads(response.body) == {"error": "audio_too_large"}
        assert store.created == []

    asyncio.run(run())


def test_registered_upload_rejects_missing_duplicate_or_unknown_multipart_fields():
    async def run():
        routes = web.RouteTableDef()
        store = AssetStore()
        register_api_bridge_routes(routes, plugin_version="test", asset_store_getter=lambda: store)
        handler = next(item.handler for item in routes if item.path.endswith("/assets/audio") and item.method == "POST")

        for fields in (
            [],
            [UploadField("not-audio", "voice.wav", [b"12"])],
            [UploadField("audio", "voice.wav", [b"12"]), UploadField("audio", "second.wav", [b"34"])],
            [UploadField("audio", "voice.wav", [b"12"]), UploadField("extra", "extra.wav", [b"34"])],
        ):
            response = await handler(UploadRequest(fields))
            assert response.status == 400
            assert json.loads(response.body) == {"error": "invalid_audio_upload"}
        assert store.created == []

    asyncio.run(run())


def test_registered_upload_returns_only_safe_asset_metadata():
    async def run():
        routes = web.RouteTableDef()
        store = AssetStore()
        register_api_bridge_routes(routes, plugin_version="test", asset_store_getter=lambda: store)
        handler = next(item.handler for item in routes if item.path.endswith("/assets/audio") and item.method == "POST")

        response = await handler(UploadRequest([UploadField("audio", "voice.wav", [b"1234"])]))

        assert response.status == 201
        assert json.loads(response.body) == {
            "asset_id": "asset-id",
            "sha256": "a" * 64,
            "size_bytes": 4,
            "filename": "asset-id.wav",
        }
        assert store.created == [("voice.wav", b"1234")]

    asyncio.run(run())


def test_registered_delete_blocks_prompt_or_runtime_activity_and_hides_unknown_asset_details():
    async def run():
        routes = web.RouteTableDef()
        store = AssetStore()
        register_api_bridge_routes(
            routes,
            plugin_version="test",
            asset_store_getter=lambda: store,
            prompt_server=PromptServer(PromptQueue(pending=["queued"])),
        )
        handler = next(item.handler for item in routes if item.path.endswith("/{asset_id}"))

        blocked = await handler(SimpleNamespace(match_info={"asset_id": "asset-id"}))
        assert blocked.status == 409
        assert json.loads(blocked.body) == {"error": "asset_in_use"}

        blocked_unknown = await handler(SimpleNamespace(match_info={"asset_id": "missing"}))
        assert blocked_unknown.status == 404
        assert json.loads(blocked_unknown.body) == {"error": "unknown_asset"}

        routes = web.RouteTableDef()
        register_api_bridge_routes(
            routes,
            plugin_version="test",
            asset_store_getter=lambda: store,
            prompt_server=PromptServer(PromptQueue()),
        )
        handler = next(item.handler for item in routes if item.path.endswith("/{asset_id}"))
        missing = await handler(SimpleNamespace(match_info={"asset_id": "missing"}))
        assert missing.status == 404
        assert json.loads(missing.body) == {"error": "unknown_asset"}

        deleted = await handler(SimpleNamespace(match_info={"asset_id": "asset-id"}))
        assert deleted.status == 200
        assert json.loads(deleted.body) == {"asset_id": "asset-id", "deleted": True}
        assert store.deleted == ["asset-id"]

    asyncio.run(run())


def test_registered_runtime_release_returns_error_or_success_without_exception_details():
    async def run():
        for report, expected_status in (
            ({"released": [], "busy": [], "errors": [{"code": "runtime_unload_failed"}]}, 500),
            ({"released": ["text:node:hash"], "busy": [], "errors": []}, 200),
        ):
            routes = web.RouteTableDef()
            runtime_registry = RuntimeRegistry(report=report)
            register_api_bridge_routes(
                routes,
                plugin_version="test",
                runtime_registry_getter=lambda: runtime_registry,
            )
            handler = next(item.handler for item in routes if item.path.endswith("/runtime/release"))

            class Request:
                async def json(self):
                    return {"all": True}

            response = await handler(Request())
            assert response.status == expected_status
            assert json.loads(response.body) == report
            assert runtime_registry.calls == [{"runtime_key": None, "resource_id": None}]

    asyncio.run(run())
