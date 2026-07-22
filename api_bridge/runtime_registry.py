"""Process-local ownership registry for API-bridge TTS runtime instances.

The registry deliberately stores only the runtime identity and small operational
metadata.  Model paths and engine configuration remain private to the owning
unified-node cache entry.
"""

from __future__ import annotations

import time
import hashlib
import json
import logging
import math
from dataclasses import dataclass
from threading import RLock
from typing import Callable


LOGGER = logging.getLogger(__name__)


def _validate_json_value(value: object, location: str) -> None:
    """Reject values whose serialization could depend on repr or memory address."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise TypeError(f"Cache identity values must be JSON-safe at {location}: non-finite float")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Cache identity values must be JSON-safe at {location}: non-string key")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise TypeError(
        f"Cache identity values must be JSON-safe at {location}: unsupported {type(value).__name__}"
    )


def make_cache_identity(engine: str, stable_params: dict[str, object]) -> str:
    """Build a deterministic, path-safe cache identity from load-time inputs."""
    if not isinstance(engine, str) or not isinstance(stable_params, dict):
        raise TypeError("Cache identity values must be JSON-safe: engine string and parameter mapping required")
    _validate_json_value(stable_params, "stable_params")
    encoded = json.dumps(
        {"engine": engine, "stable_params": stable_params},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{engine}_{hashlib.sha256(encoded).hexdigest()}"


def make_runtime_key(owner_type: str, owner_token: str, cache_identity: str) -> str:
    """Namespace a cache-owned runtime by node instance and cache identity."""
    return f"{owner_type}:{owner_token}:{cache_identity}"


@dataclass
class RuntimeHandle:
    """A target-engine runtime owned by a unified node cache entry."""

    runtime_key: str
    engine: str
    resource_id: str
    device: str
    unload: Callable[[], None]
    loaded_at: float
    last_used_at: float
    active: int = 0

    @classmethod
    def create(
        cls,
        runtime_key: str,
        engine: str,
        resource_id: str,
        device: str,
        unload: Callable[[], None],
    ) -> "RuntimeHandle":
        now = time.time()
        return cls(
            runtime_key=runtime_key,
            engine=engine,
            resource_id=resource_id,
            device=device,
            unload=unload,
            loaded_at=now,
            last_used_at=now,
        )


class RuntimeRegistry:
    """Register, inspect, and deterministically release target TTS runtimes.

    Unload callbacks execute after a handle is detached from the registry and
    outside the mutex. A lease protects a runtime during synthesis; release
    reports a stable busy outcome while a lease is active so callers can retry.
    """

    def __init__(self) -> None:
        self._handles: dict[str, RuntimeHandle] = {}
        self._lock = RLock()

    def register(self, handle: RuntimeHandle) -> None:
        """Register ``handle``, releasing a replaced instance after unlocking."""
        with self._lock:
            previous = self._handles.get(handle.runtime_key)
            if previous is not None and previous is not handle and previous.active:
                raise RuntimeError(
                    f"Cannot replace active runtime {handle.runtime_key}; synthesis is active"
                )
            self._handles[handle.runtime_key] = handle

        if previous is not None and previous is not handle:
            try:
                previous.unload()
            except Exception:
                # Replacement already owns the cache key.  A failing old-model
                # unload must not prevent the new runtime becoming available.
                LOGGER.exception("Failed to clean replaced TTS runtime %s", previous.runtime_key)

    def lease(self, runtime_key: str) -> "RuntimeLease":
        """Acquire an in-flight synthesis lease for a registered runtime."""
        with self._lock:
            handle = self._handles.get(runtime_key)
            if handle is None:
                raise RuntimeError(f"TTS runtime is unavailable: {runtime_key}")
            handle.active += 1
            handle.last_used_at = time.time()
        return RuntimeLease(self, handle)

    acquire = lease

    def _close_lease(self, handle: RuntimeHandle) -> None:
        with self._lock:
            if handle.active <= 0:
                return
            handle.active -= 1

    def touch(self, runtime_key: str) -> None:
        """Record use of a live handle without exposing cache internals."""
        with self._lock:
            handle = self._handles.get(runtime_key)
            if handle is not None:
                handle.last_used_at = time.time()

    def status(self) -> list[dict[str, object]]:
        """Return stable, path-free operational metadata for live runtimes."""
        with self._lock:
            handles = sorted(self._handles.values(), key=lambda value: value.runtime_key)
            return [
                {
                    "runtime_key": handle.runtime_key,
                    "engine": handle.engine,
                    "resource_id": handle.resource_id,
                    "device": handle.device,
                    "loaded_at": handle.loaded_at,
                    "last_used_at": handle.last_used_at,
                    "active": handle.active,
                    "busy": handle.active > 0,
                }
                for handle in handles
            ]

    def release(
        self,
        *,
        runtime_key: str | None = None,
        resource_id: str | None = None,
    ) -> dict[str, list[object]]:
        """Detach matching handles and invoke every unload callback once.

        Active handles remain registered and are reported as busy. Callers must
        retry after synthesis finishes. Idle handles are removed before callbacks,
        so repeated release calls are idempotent even when cleanup raises.
        """
        with self._lock:
            selected = sorted(
                (
                    handle
                    for handle in self._handles.values()
                    if (runtime_key is None or handle.runtime_key == runtime_key)
                    and (resource_id is None or handle.resource_id == resource_id)
                ),
                key=lambda handle: handle.runtime_key,
            )
            immediate: list[RuntimeHandle] = []
            busy: list[dict[str, str]] = []
            for handle in selected:
                if handle.active:
                    busy.append(
                        {
                            "runtime_key": handle.runtime_key,
                            "code": "runtime_busy",
                            "message": "Runtime is in use; retry release later.",
                        }
                    )
                else:
                    self._handles.pop(handle.runtime_key, None)
                    immediate.append(handle)

        released: list[str] = []
        errors: list[dict[str, str]] = []
        for handle in immediate:
            try:
                handle.unload()
                released.append(handle.runtime_key)
            except Exception:
                LOGGER.exception("Failed to clean TTS runtime %s", handle.runtime_key)
                errors.append(
                    {
                        "runtime_key": handle.runtime_key,
                        "code": "runtime_unload_failed",
                        "message": "Runtime cleanup failed; inspect server logs.",
                    }
                )
        return {"released": released, "busy": busy, "errors": errors}


class RuntimeLease:
    """One in-flight synthesis lease for a registered runtime handle."""

    def __init__(self, registry: RuntimeRegistry, handle: RuntimeHandle) -> None:
        self._registry = registry
        self._handle = handle
        self._closed = False

    def __enter__(self) -> "RuntimeLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._registry._close_lease(self._handle)


_runtime_registry = RuntimeRegistry()


def get_runtime_registry() -> RuntimeRegistry:
    """Return the process-local API-bridge runtime registry."""
    return _runtime_registry


def _reset_runtime_registry_for_tests() -> dict[str, object]:
    """Release and replace the singleton when no synthesis lease is active.

    This exists for deterministic test/module-reload isolation. Production hot
    reload must use the normal release API first; an active runtime is never
    abandoned because its callback still closes over the owning node cache.
    """
    global _runtime_registry
    previous = _runtime_registry
    report = previous.release()
    if report["busy"]:
        return {**report, "reset": False}
    _runtime_registry = RuntimeRegistry()
    return {**report, "reset": True}
