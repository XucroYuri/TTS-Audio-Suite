"""Bounded reference-audio assets for the public API bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Final
from uuid import uuid4

import soundfile


ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a"})
DEFAULT_MAX_BYTES: Final[int] = 64 * 1024 * 1024


@dataclass(frozen=True)
class AudioAsset:
    asset_id: str
    path: Path
    sha256: str
    size_bytes: int


class AudioAssetStore:
    """Own uploaded reference audio under one generated-name-only directory."""

    def __init__(self, root: Path, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.max_bytes = max_bytes
        self._assets: dict[str, AudioAsset] = {}

    def create(self, filename: str, content: bytes) -> AudioAsset:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError(f"unsupported audio extension: {suffix or '<none>'}")
        if not isinstance(content, bytes):
            raise ValueError("audio content must be bytes")
        if len(content) > self.max_bytes:
            raise ValueError(f"audio exceeds maximum size of {self.max_bytes} bytes")

        asset_id = uuid4().hex
        destination = self._destination(asset_id, suffix)
        try:
            with destination.open("xb") as handle:
                handle.write(content)
            self._validate_audio(destination)
        except Exception:
            if destination.exists() and not destination.is_symlink():
                destination.unlink()
            raise

        asset = AudioAsset(
            asset_id=asset_id,
            path=destination,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
        self._assets[asset_id] = asset
        return asset

    def require(self, asset_id: str) -> AudioAsset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        return asset

    def delete(self, asset_id: str) -> None:
        asset = self.require(asset_id)
        path = self._registered_path(asset)
        if path.exists():
            path.unlink()
        del self._assets[asset_id]

    def _destination(self, asset_id: str, suffix: str) -> Path:
        destination = self.root / f"{asset_id}{suffix}"
        if destination.parent != self.root or not destination.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("asset destination escapes the managed root")
        return destination

    def _registered_path(self, asset: AudioAsset) -> Path:
        path = asset.path
        if path.parent != self.root or path.is_symlink() or not path.resolve(strict=False).is_relative_to(self.root):
            raise ValueError("registered asset path is outside the managed root")
        return path

    @staticmethod
    def _validate_audio(path: Path) -> None:
        try:
            info = soundfile.info(path)
        except Exception as exc:
            raise ValueError("invalid audio content") from exc
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("invalid audio content")


_audio_asset_store: AudioAssetStore | None = None


def get_audio_asset_store() -> AudioAssetStore:
    global _audio_asset_store
    if _audio_asset_store is None:
        import folder_paths

        root = Path(folder_paths.get_input_directory()) / "tts-audio-suite"
        _audio_asset_store = AudioAssetStore(root)
    return _audio_asset_store


def reset_audio_asset_store_for_tests() -> None:
    global _audio_asset_store
    _audio_asset_store = None
