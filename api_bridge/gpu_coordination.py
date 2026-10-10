"""Optional, operator-local GPU lease around target-engine node execution.

No native processes are inspected or stopped here. The coordinator owns idle
proof and native priority; this plugin owns its GPU lifetime and cleanup fence.
"""
from __future__ import annotations

from functools import wraps
import ipaddress
import json
import math
import os
from pathlib import Path
import threading
import uuid
from urllib.parse import urlsplit
from urllib.request import build_opener, ProxyHandler, HTTPRedirectHandler, Request
from urllib.error import HTTPError


class TTSMoreGPUPreempted(InterruptedError):
    pass


class TTSMoreGPUCleanupFailed(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class CoordinatorClient:
    """Small stdlib client for the shared v1 fixed-action protocol."""

    def __init__(self, config):
        parsed = urlsplit(config["coordinator_url"])
        if (parsed.scheme != "http" or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment or not ipaddress.ip_address(parsed.hostname).is_loopback):
            raise ValueError("GPU control endpoint must be literal loopback HTTP")
        self.url = config["coordinator_url"].rstrip("/") + "/v1/action"
        self.token = config["token"]
        self.group = config["resource_group"]
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def call(self, action, payload):
        body = json.dumps({"action": action, "payload": payload}, allow_nan=False).encode("utf-8")
        request = Request(self.url, data=body, method="POST", headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=max(2, payload.get("timeout", 0) + 2)) as response:
                raw = response.read(65537)
                if len(raw) > 65536:
                    raise RuntimeError("GPU coordinator response too large")
                result = json.loads(raw)
                if not isinstance(result, dict):
                    raise RuntimeError("Invalid GPU coordinator response")
                return result
        except HTTPError as exc:
            if exc.code in {408, 409}:
                raw = exc.read(65537)
                if len(raw) <= 65536 and (json.loads(raw).get("error") or {}).get("code") == "coordination_timeout":
                    raise TTSMoreGPUPreempted("Waiting for native GPU priority") from None
            raise RuntimeError("GPU coordinator rejected the operation") from None

    def acquire_comfy(self, holder, timeout):
        value = self.call("acquire_comfy", {"holder": holder, "timeout": timeout, "resource_group": self.group}).get("token")
        if not isinstance(value, str) or not value:
            raise RuntimeError("GPU coordinator lease token missing")
        return value

    def check_comfy(self, token):
        value = self.call("check_comfy", {"token": token}).get("valid")
        if not isinstance(value, bool):
            raise RuntimeError("GPU coordinator heartbeat invalid")
        return value

    def release_comfy(self, token, clean):
        value = self.call("release_comfy", {"token": token, "clean": clean}).get("released")
        if not isinstance(value, bool):
            raise RuntimeError("GPU coordinator release invalid")
        return value

    def snapshot(self):
        return self.call("snapshot", {"resource_group": self.group})


def configuration():
    path = os.environ.get("TTS_AUDIO_SUITE_GPU_COORDINATION_CONFIG", "").strip()
    if not path:
        return None
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if document.get("enabled", False) is not True:
            return None
        group = document["resource_group"]
        entry = document["groups"][group]
        if entry.get("enabled", True) is False:
            return None
        config = {**entry, "resource_group": group}
        token = config["token"]
        if (not isinstance(group, str) or not group or len(group) > 256
                or not isinstance(token, str) or not 16 <= len(token) <= 256
                or any(not (char.isascii() and (char.isalnum() or char in "._~+/=-")) for char in token)):
            raise ValueError("Invalid credentials")
        interval = float(config.get("heartbeat_seconds", 1.0))
        if not math.isfinite(interval) or not 0.1 <= interval <= 2:
            raise ValueError("Invalid heartbeat interval")
        config["heartbeat_seconds"] = interval
        CoordinatorClient(config)  # validate private endpoint before discovery/admission
        return config
    except Exception as exc:
        raise TTSMoreGPUCleanupFailed("Invalid local GPU coordination configuration") from exc


def capabilities():
    config = configuration()
    return {"enabled": bool(config), "protocol_version": 1, "resource_group": config["resource_group"] if config else None}


_current = threading.local()
_execution_lock = threading.Lock()


def coordination_active():
    return getattr(_current, "guard", None) is not None


def preemption_requested():
    guard = getattr(_current, "guard", None)
    return guard is not None and (guard.revoked.is_set() or guard.unavailable.is_set())


def check_gpu_interrupt():
    guard = getattr(_current, "guard", None)
    if guard is None:
        return
    if guard.unavailable.is_set():
        raise TTSMoreGPUCleanupFailed("GPU coordinator heartbeat unavailable")
    if guard.revoked.is_set():
        raise TTSMoreGPUPreempted("TTSMoreGPUPreempted: native GPU priority")


class _Guard:
    def __init__(self, client, token, interval):
        self.client, self.token, self.interval = client, token, interval
        self.stop, self.revoked, self.unavailable = threading.Event(), threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)

    def heartbeat(self):
        while not self.stop.wait(self.interval):
            try:
                if not self.client.check_comfy(self.token):
                    self.revoked.set()
                    return
            except Exception:
                self.unavailable.set()
                return


def _gpu_memory_released():
    """Verify all visible CUDA allocators after releasing registered runtimes.

    An unavailable allocator is unknown, never proof of a clean release. This
    does not gate other plugins; a dedicated TTS Comfy instance is required.
    """
    try:
        import gc
        import torch
        gc.collect()
        if not torch.cuda.is_available():
            # CPU-only operation still needs an observable zero allocator;
            # missing/broken CUDA APIs are handled as unknown below.
            return torch.cuda.device_count() == 0 and torch.cuda.memory_allocated() == 0
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()
            if torch.cuda.memory_allocated(index) != 0:
                return False
        return True
    except Exception:
        return False


def coordinated_execution(function):
    """Gate before any reference preprocessing/engine construction and release.

    Enabled coordination supports only engines with registered deterministic
    cleanup. Other GPU engines fail closed until they implement that contract.
    """
    @wraps(function)
    def execute(self, TTS_engine, *args, **kwargs):
        config = configuration()
        if config is None:
            return function(self, TTS_engine, *args, **kwargs)
        if TTS_engine.get("engine_type") not in {"gpt_sovits", "index_tts", "cosyvoice"}:
            raise TTSMoreGPUCleanupFailed("Engine lacks coordinated GPU cleanup")
        external_home = {"cosyvoice": "cosyvoice_home", "index_tts": "index_tts_home"}.get(TTS_engine.get("engine_type"))
        if external_home and not (TTS_engine.get("config") or {}).get(external_home):
            # Shared inference loops cannot cooperate with lease revocation.
            # External checkout proxies use the bounded child-tree controller.
            raise TTSMoreGPUCleanupFailed("Coordinated engine requires an external checkout; in-process execution cannot yield promptly")
        from .runtime_registry import get_runtime_registry
        from comfy import model_management
        client = CoordinatorClient(config)

        # Comfy usually executes serially; this lock also covers concurrent
        # direct callers and independent node instances in the same process.
        while not _execution_lock.acquire(timeout=0.25):
            model_management.throw_exception_if_processing_interrupted()
        token = None
        guard = None
        failure = None
        result = None
        clean = False

        def cleanup_failed(message, cause=None):
            nonlocal clean, failure
            clean = False
            if isinstance(failure, TTSMoreGPUCleanupFailed):
                # Retain the primary safety error and its diagnostic identity.
                failure.add_note(message)
            else:
                original = failure if failure is not None else cause
                failure = TTSMoreGPUCleanupFailed("TTSMoreGPUCleanupFailed: " + message)
                if original is not None:
                    failure.__cause__ = original

        try:
            while token is None:
                model_management.throw_exception_if_processing_interrupted()
                try:
                    token = client.acquire_comfy(f"suite:{uuid.uuid4().hex}", timeout=1)
                except TTSMoreGPUPreempted:
                    group = client.snapshot()["groups"][config["resource_group"]]
                    lease = group.get("comfy") or {}
                    if group.get("recovery_required") or group.get("cleanup_failed") or lease.get("cleanup_failed") or lease.get("reason") in {"comfy_cleanup_unconfirmed", "comfy_heartbeat_expired"}:
                        raise TTSMoreGPUCleanupFailed("GPU cleanup fence requires operator recovery")
                    natives = group.get("natives")
                    if not isinstance(natives, dict) or not natives or any(
                        item.get("fresh") is not True or not isinstance(item.get("status"), dict)
                        or (item["status"].get("ready") is not True and group.get("state") != "native_offloading")
                        for item in natives.values()
                    ):
                        raise TTSMoreGPUCleanupFailed("Native GPU ownership status is untrusted")
                    continue
            guard = _Guard(client, token, config["heartbeat_seconds"])
            _current.guard = guard
            if not client.check_comfy(token):
                guard.revoked.set()
            guard.thread.start()
            check_gpu_interrupt()
            result = function(self, TTS_engine, *args, **kwargs)
            # A completed segment remains valid even if priority changed during
            # its final CPU formatting; cleanup still precedes native approval.
            clean = True
        except BaseException as exc:
            failure = exc
            canonical = getattr(model_management, "InterruptProcessingException", ())
            clean = isinstance(exc, TTSMoreGPUPreempted) or (isinstance(canonical, type) and isinstance(exc, canonical))
            if token is None and not clean and not isinstance(exc, TTSMoreGPUCleanupFailed):
                failure = TTSMoreGPUCleanupFailed("GPU coordinator admission unavailable")
        finally:
            try:
                if token is not None:
                    try:
                        report = get_runtime_registry().release()
                        if report["busy"] or report["errors"]:
                            cleanup_failed("runtime release incomplete")
                    except BaseException as exc:
                        cleanup_failed("runtime release unconfirmed", exc)
                    try:
                        if not _gpu_memory_released():
                            cleanup_failed("CUDA allocation release unconfirmed")
                    except BaseException as exc:
                        cleanup_failed("CUDA allocation release unconfirmed", exc)
                if guard is not None:
                    try:
                        guard.stop.set()
                    except BaseException as exc:
                        cleanup_failed("heartbeat stop unconfirmed", exc)
                    try:
                        # Initial validation can fail before the thread starts.
                        if guard.thread.ident is not None or guard.thread.is_alive():
                            guard.thread.join(timeout=3)
                        if guard.thread.is_alive() or guard.unavailable.is_set():
                            cleanup_failed("heartbeat did not converge")
                    except BaseException as exc:
                        cleanup_failed("heartbeat exit unconfirmed", exc)
                if token is not None:
                    if not clean:
                        cleanup_failed("GPU cleanup fence retained")
                    try:
                        released = client.release_comfy(token, clean=clean)
                        if clean and not released:
                            cleanup_failed("GPU release unconfirmed")
                            client.release_comfy(token, clean=False)
                    except BaseException as exc:
                        cleanup_failed("GPU release unconfirmed", exc)
                        # Even a failed clean release must try to persist dirty
                        # ownership. Its failure cannot turn into a retry.
                        try:
                            client.release_comfy(token, clean=False)
                        except BaseException as dirty_exc:
                            cleanup_failed("dirty fence persistence unconfirmed", dirty_exc)
            finally:
                _current.guard = None
                _execution_lock.release()
        if failure is not None:
            raise failure
        return result
    return execute
