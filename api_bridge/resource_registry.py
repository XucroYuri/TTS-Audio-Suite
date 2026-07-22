from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import TTSResource

RESOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
ENGINES = {"gpt_sovits", "index_tts", "cosyvoice"}


class ResourceRegistry:
    def __init__(self, resources: dict[str, TTSResource]) -> None:
        self._resources = dict(resources)

    @classmethod
    def load(cls, path: Path) -> "ResourceRegistry":
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("version") != 1 or not isinstance(payload.get("resources"), dict):
            raise ValueError("resource registry must use version 1 and contain resources")
        resources: dict[str, TTSResource] = {}
        for resource_id, raw in payload["resources"].items():
            if not isinstance(resource_id, str) or not RESOURCE_ID.fullmatch(resource_id):
                raise ValueError(f"invalid resource_id: {resource_id!r}")
            if not isinstance(raw, dict) or raw.get("engine") not in ENGINES:
                raise ValueError(f"invalid engine for resource {resource_id}")
            resource = _build_resource(resource_id, raw)
            _validate_resource(resource)
            resources[resource_id] = resource
        return cls(resources)

    def require(self, resource_id: str, engine: str) -> TTSResource:
        resource = self._resources.get(resource_id)
        if resource is None:
            raise ValueError(f"unknown resource_id: {resource_id}")
        if resource.engine != engine:
            raise ValueError(f"resource {resource_id} belongs to {resource.engine}, not {engine}")
        return resource

    def capabilities(self) -> list[dict[str, object]]:
        return [
            {"resource_id": item.resource_id, "engine": item.engine, "ready": True}
            for item in sorted(self._resources.values(), key=lambda value: value.resource_id)
        ]


def _resolved(raw: dict[str, Any], key: str) -> Path | None:
    value = raw.get(key)
    return Path(str(value)).expanduser().resolve() if value else None


def _build_resource(resource_id: str, raw: dict[str, Any]) -> TTSResource:
    source_root = _resolved(raw, "source_root")
    if source_root is None:
        raise ValueError(f"source_root is required for {resource_id}")
    return TTSResource(
        resource_id=resource_id,
        engine=raw["engine"],
        source_root=source_root,
        model_dir=_resolved(raw, "model_dir"),
        gpt_weight=_resolved(raw, "gpt_weight"),
        sovits_weight=_resolved(raw, "sovits_weight"),
        bert_path=_resolved(raw, "bert_path"),
        cnhubert_path=_resolved(raw, "cnhubert_path"),
    )


def _require_path(path: Path | None, label: str, *, directory: bool = False) -> None:
    if path is None or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"missing {'directory' if directory else 'file'}: {label}")


def _validate_resource(resource: TTSResource) -> None:
    _require_path(resource.source_root, f"{resource.resource_id}.source_root", directory=True)
    if resource.engine == "gpt_sovits":
        _require_path(resource.gpt_weight, f"{resource.resource_id}.gpt_weight")
        _require_path(resource.sovits_weight, f"{resource.resource_id}.sovits_weight")
    else:
        _require_path(resource.model_dir, f"{resource.resource_id}.model_dir", directory=True)


_registry: ResourceRegistry | None = None


def get_resource_registry() -> ResourceRegistry:
    global _registry
    if _registry is None:
        configured = os.environ.get("TTS_AUDIO_SUITE_RESOURCES")
        if configured:
            path = Path(configured)
        else:
            import folder_paths

            path = Path(folder_paths.get_system_user_directory("tts_audio_suite")) / "resources.yaml"
        _registry = ResourceRegistry.load(path)
    return _registry


def reset_resource_registry_for_tests() -> None:
    global _registry
    _registry = None
