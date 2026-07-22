"""Process-local ownership registry for API-bridge TTS runtime instances.

The registry deliberately stores only the runtime identity and small operational
metadata.  Model paths and engine configuration remain private to the owning
unified-node cache entry.
"""

from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass
from threading import RLock
from typing import Callable


def make_cache_identity(engine: str, stable_params: dict[str, object]) -> str:
    """Build a deterministic, path-safe cache identity from load-time inputs."""
    encoded = json.dumps(
        {"engine": engine, "stable_params": stable_params},
        default=str,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"{engine}_{hashlib.sha256(encoded).hexdigest()}"


def make_runtime_key(owner: str, cache_identity: str) -> str:
    """Namespace a cache-owned target runtime to avoid Text/SRT collisions."""
    return f"{owner}:{cache_identity}"


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
    outside the mutex.  This makes release idempotent and allows callbacks to
    safely re-enter the registry without blocking cache management.
    """

    def __init__(self) -> None:
        self._handles: dict[str, RuntimeHandle] = {}
        self._lock = RLock()

    def register(self, handle: RuntimeHandle) -> None:
        """Register ``handle``, releasing a replaced instance after unlocking."""
        with self._lock:
            previous = self._handles.get(handle.runtime_key)
            self._handles[handle.runtime_key] = handle

        if previous is not None and previous is not handle:
            try:
                previous.unload()
            except Exception:
                # Replacement already owns the cache key.  A failing old-model
                # unload must not prevent the new runtime becoming available.
                pass

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

        Removing selected handles before callbacks means repeated release calls
        are empty even when an unload callback raises.
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
            for handle in selected:
                self._handles.pop(handle.runtime_key, None)

        released: list[str] = []
        errors: list[dict[str, str]] = []
        for handle in selected:
            try:
                handle.unload()
                released.append(handle.runtime_key)
            except Exception as exc:
                errors.append({"runtime_key": handle.runtime_key, "error": str(exc)})
        return {"released": released, "errors": errors}


_runtime_registry = RuntimeRegistry()


def get_runtime_registry() -> RuntimeRegistry:
    """Return the process-local API-bridge runtime registry."""
    return _runtime_registry
