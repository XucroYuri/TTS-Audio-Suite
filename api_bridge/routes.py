"""Versioned support routes for the TTS More / ComfyUI bridge.

These routes deliberately expose operational discovery and lifecycle helpers
only.  Graph submission and synthesis remain ComfyUI's own API contract.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from aiohttp import web

from . import BRIDGE_PROTOCOL_VERSION
from .assets import AssetInUseError, AssetQuotaError, get_audio_asset_store
from .resource_registry import get_resource_registry
from .runtime_registry import get_runtime_registry


LOGGER = logging.getLogger(__name__)

ROUTE_PREFIX = "/api/tts-audio-suite/v1"
MAX_RELEASE_VALUE_LENGTH = 512
_NODE_IDS = {
    "gpt_sovits": "TTSExternalGPTSovitsEngine",
    "index_tts": "TTSExternalIndexTTSEngine",
    "cosyvoice": "TTSExternalCosyVoiceEngine",
    "audio_asset": "TTSExternalAudioAsset",
    "text": "UnifiedTTSTextNode",
    "save_audio": "SaveAudio",
}


class _AudioTooLarge(ValueError):
    """Raised before upload bytes can grow beyond the per-file store limit."""


def build_capabilities_payload(registry: Any, *, plugin_version: str) -> dict[str, object]:
    """Return the public, path-free bridge discovery contract."""
    return {
        "protocol_version": BRIDGE_PROTOCOL_VERSION,
        "plugin_version": plugin_version,
        "nodes": dict(_NODE_IDS),
        "resources": registry.capabilities(),
    }


def build_runtime_release_payload(payload: object) -> dict[str, str | None]:
    """Validate a deliberately narrow runtime-release selector.

    An empty object must never mean "release every model".  Aggregate release
    therefore needs the explicit ``{"all": true}`` acknowledgement.
    """
    if not isinstance(payload, dict):
        raise ValueError("invalid_runtime_release")
    allowed_keys = {"runtime_key", "resource_id", "all"}
    if set(payload) - allowed_keys:
        raise ValueError("invalid_runtime_release")
    aggregate = payload.get("all", False)
    if not isinstance(aggregate, bool):
        raise ValueError("invalid_runtime_release")
    runtime_key = payload.get("runtime_key")
    resource_id = payload.get("resource_id")
    if runtime_key is not None and (not isinstance(runtime_key, str) or not runtime_key or len(runtime_key) > MAX_RELEASE_VALUE_LENGTH):
        raise ValueError("invalid_runtime_release")
    if resource_id is not None and (not isinstance(resource_id, str) or not resource_id or len(resource_id) > MAX_RELEASE_VALUE_LENGTH):
        raise ValueError("invalid_runtime_release")
    if aggregate:
        if set(payload) != {"all"}:
            raise ValueError("invalid_runtime_release")
        return {"runtime_key": None, "resource_id": None}
    if runtime_key is None and resource_id is None:
        raise ValueError("invalid_runtime_release")
    return {"runtime_key": runtime_key, "resource_id": resource_id}


def is_prompt_queue_active(prompt_server: Any | None) -> bool:
    """Return whether ComfyUI currently has a running or pending prompt.

    ``PromptQueue.get_current_queue`` is the stable public-in-practice shape
    used by ComfyUI's own queue endpoint.  The fallback supports older queue
    objects without holding their mutex or waiting on the event loop.
    """
    queue = getattr(prompt_server, "prompt_queue", None)
    if queue is None:
        return False
    try:
        get_current_queue = getattr(queue, "get_current_queue", None)
        if callable(get_current_queue):
            running, queued = get_current_queue()
            return bool(running) or bool(queued)
        return bool(getattr(queue, "queue", ())) or bool(getattr(queue, "currently_running", ()))
    except Exception:
        LOGGER.exception("Unable to inspect ComfyUI prompt queue before deleting bridge audio")
        return True


def is_asset_delete_blocked(prompt_server: Any | None, runtime_registry: Any) -> bool:
    """Fail closed while a graph or target runtime can still consume audio_path."""
    if is_prompt_queue_active(prompt_server):
        return True
    try:
        return any(bool(item.get("busy")) or int(item.get("active", 0)) > 0 for item in runtime_registry.status())
    except Exception:
        LOGGER.exception("Unable to inspect TTS runtime activity before deleting bridge audio")
        return True


def _delete_asset_while_idle(
    store: Any, asset_id: str, prompt_server: Any | None, runtime_registry: Any
) -> None:
    """Linearize bridge deletion with ComfyUI's real prompt-queue mutex.

    PromptQueue transitions pending work into ``currently_running`` while
    holding this same mutex, so a task cannot start between this inspection and
    ``store.delete``.  Missing queue mutexes fail closed in production.
    """
    queue = getattr(prompt_server, "prompt_queue", None)
    if queue is None:
        if prompt_server is not None:
            raise AssetInUseError("asset_in_use")
        store.known(asset_id)
        if is_asset_delete_blocked(None, runtime_registry):
            raise AssetInUseError("asset_in_use")
        store.delete(asset_id)
        return
    mutex = getattr(queue, "mutex", None)
    if mutex is None:
        raise AssetInUseError("asset_in_use")
    with mutex:
        # The official PromptQueue uses these two attributes under mutex. Do
        # not call get_current_queue() here: that re-enters its RLock and gives
        # no stronger guarantee than inspecting the protected state directly.
        store.known(asset_id)
        if bool(getattr(queue, "currently_running", ())) or bool(getattr(queue, "queue", ())):
            raise AssetInUseError("asset_in_use")
        if is_asset_delete_blocked(None, runtime_registry):
            raise AssetInUseError("asset_in_use")
        store.delete(asset_id)


def _no_store(response: web.Response) -> web.Response:
    response.headers["Cache-Control"] = "no-store"
    return response


async def _read_upload_content(field: Any, *, max_bytes: int) -> bytes:
    """Consume a multipart field with a hard bound before memory accumulation."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await field.read_chunk(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _AudioTooLarge("audio_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _route_exists(routes: Any, method: str, path: str) -> bool:
    return any(getattr(item, "method", None) == method and getattr(item, "path", None) == path for item in routes)


def _add_route(routes: Any, method: str, path: str, handler: Callable[..., Any]) -> None:
    if not _route_exists(routes, method, path):
        routes.route(method, path)(handler)


def _get_default_prompt_server() -> Any | None:
    try:
        from server import PromptServer

        return PromptServer.instance
    except Exception:
        LOGGER.exception("TTS bridge support routes could not inspect PromptServer")
        return None


def register_api_bridge_routes(
    routes: Any,
    *,
    plugin_version: str,
    registry_getter: Callable[[], Any] = get_resource_registry,
    asset_store_getter: Callable[[], Any] = get_audio_asset_store,
    runtime_registry_getter: Callable[[], Any] = get_runtime_registry,
    prompt_server: Any | None = None,
) -> None:
    """Add idempotent v1 support routes to ComfyUI's normal route registry."""
    active_prompt_server = _get_default_prompt_server() if prompt_server is None else prompt_server

    async def capabilities(request: web.Request) -> web.Response:
        try:
            return _no_store(web.json_response(build_capabilities_payload(registry_getter(), plugin_version=plugin_version)))
        except Exception:
            LOGGER.exception("Failed to build TTS bridge capabilities")
            return web.json_response({"error": "bridge_unavailable"}, status=500)

    async def upload_audio(request: web.Request) -> web.Response:
        try:
            if getattr(request, "content_type", "multipart/form-data") != "multipart/form-data":
                return web.json_response({"error": "unsupported_media_type"}, status=415)
            reader = await request.multipart()
            field = await reader.next()
            if field is None or field.name != "audio" or not field.filename:
                return web.json_response({"error": "invalid_audio_upload"}, status=400)
            store = await asyncio.to_thread(asset_store_getter)
            content = await _read_upload_content(field, max_bytes=store.max_bytes)
            if await reader.next() is not None:
                return web.json_response({"error": "invalid_audio_upload"}, status=400)
            asset = await asyncio.to_thread(store.create, field.filename, content)
            return web.json_response(
                {
                    "asset_id": asset.asset_id,
                    "sha256": asset.sha256,
                    "size_bytes": asset.size_bytes,
                    "filename": asset.path.name,
                },
                status=201,
            )
        except _AudioTooLarge:
            return web.json_response({"error": "audio_too_large"}, status=413)
        except AssetQuotaError:
            return web.json_response({"error": "asset_quota_exceeded"}, status=507)
        except (AssertionError, ValueError, web.HTTPBadRequest):
            return web.json_response({"error": "invalid_audio_upload"}, status=400)
        except web.HTTPRequestEntityTooLarge:
            return web.json_response({"error": "audio_too_large"}, status=413)
        except Exception:
            LOGGER.exception("Failed to upload TTS bridge audio asset")
            return web.json_response({"error": "bridge_unavailable"}, status=500)

    async def delete_audio(request: web.Request) -> web.Response:
        try:
            store = await asyncio.to_thread(asset_store_getter)
            runtime_registry = runtime_registry_getter()
            asset_id = request.match_info["asset_id"]
            await asyncio.to_thread(
                _delete_asset_while_idle, store, asset_id, active_prompt_server, runtime_registry
            )
            return web.json_response({"asset_id": asset_id, "deleted": True})
        except AssetInUseError:
            return web.json_response({"error": "asset_in_use"}, status=409)
        except ValueError:
            return web.json_response({"error": "unknown_asset"}, status=404)
        except Exception:
            LOGGER.exception("Failed to delete TTS bridge audio asset")
            return web.json_response({"error": "bridge_unavailable"}, status=500)

    async def runtime_status(request: web.Request) -> web.Response:
        try:
            return _no_store(web.json_response({"runtimes": runtime_registry_getter().status()}))
        except Exception:
            LOGGER.exception("Failed to read TTS bridge runtime status")
            return web.json_response({"error": "bridge_unavailable"}, status=500)

    async def runtime_release(request: web.Request) -> web.Response:
        try:
            selector = build_runtime_release_payload(await request.json())
        except (ValueError, web.HTTPBadRequest):
            return web.json_response({"error": "invalid_runtime_release"}, status=400)
        except Exception:
            LOGGER.exception("Failed to parse TTS bridge runtime release request")
            return web.json_response({"error": "invalid_runtime_release"}, status=400)
        try:
            report = await asyncio.to_thread(runtime_registry_getter().release, **selector)
            if report["errors"]:
                status = 500
            elif report["busy"]:
                status = 409
            else:
                status = 200
            return web.json_response(report, status=status)
        except Exception:
            LOGGER.exception("Failed to release TTS bridge runtime")
            return web.json_response({"error": "bridge_unavailable"}, status=500)

    _add_route(routes, "GET", f"{ROUTE_PREFIX}/capabilities", capabilities)
    _add_route(routes, "POST", f"{ROUTE_PREFIX}/assets/audio", upload_audio)
    _add_route(routes, "DELETE", f"{ROUTE_PREFIX}/assets/audio/{{asset_id}}", delete_audio)
    _add_route(routes, "GET", f"{ROUTE_PREFIX}/runtime/status", runtime_status)
    _add_route(routes, "POST", f"{ROUTE_PREFIX}/runtime/release", runtime_release)
