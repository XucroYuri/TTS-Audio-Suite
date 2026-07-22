"""Bounded, restart-safe reference-audio assets for the public API bridge."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from threading import Lock, RLock
from typing import Any, Final, Iterator
from uuid import uuid4

import soundfile


ALLOWED_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".flac", ".mp3", ".ogg", ".m4a"})
DEFAULT_MAX_BYTES: Final[int] = 64 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES: Final[int] = 512 * 1024 * 1024
DEFAULT_MAX_ASSETS: Final[int] = 128
_MANAGED_NAME = re.compile(r"^(?P<asset_id>[0-9a-f]{32})(?P<suffix>\.[a-z0-9]+)$")


class AssetInUseError(ValueError):
    """A known asset cannot be removed while a node has pinned it."""


class AssetQuotaError(ValueError):
    """A create would exceed the managed store's explicit quota."""


@dataclass(frozen=True)
class AudioAsset:
    asset_id: str
    path: Path
    sha256: str
    size_bytes: int
    valid: bool = True


@dataclass(frozen=True)
class AudioAssetSnapshot:
    asset: AudioAsset
    content: bytes


class AudioAssetStore:
    """Own generated-name-only assets beneath one ComfyUI input subdirectory.

    Existing managed UUID files are rebuilt on startup. Unknown files are never
    deleted automatically; they remain outside the bridge index and quota.
    """

    def __init__(
        self,
        root: Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
        max_assets: int = DEFAULT_MAX_ASSETS,
        owner_root: Path | None = None,
    ) -> None:
        if min(max_bytes, max_total_bytes, max_assets) <= 0:
            raise ValueError("asset limits must be positive")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        self.owner_root = (owner_root or self.root).resolve()
        if not self.root.is_relative_to(self.owner_root):
            raise ValueError("asset root is outside the ComfyUI input directory")
        self.max_bytes = max_bytes
        self.max_total_bytes = max_total_bytes
        self.max_assets = max_assets
        self._assets: dict[str, AudioAsset] = {}
        self._managed_paths: dict[str, tuple[Path, ...]] = {}
        self._managed_sizes: dict[Path, int] = {}
        self._conflicts: set[str] = set()
        self._file_count = 0
        self._pins: dict[str, int] = {}
        self._lock = RLock()
        self._total_bytes = 0
        self._rebuild_index()

    def create(self, filename: str, content: bytes) -> AudioAsset:
        with self._lock:
            suffix = Path(filename).suffix.lower()
            if suffix not in ALLOWED_EXTENSIONS:
                raise ValueError(f"unsupported audio extension: {suffix or '<none>'}")
            if not isinstance(content, bytes):
                raise ValueError("audio content must be bytes")
            if len(content) > self.max_bytes:
                raise ValueError(f"audio exceeds maximum size of {self.max_bytes} bytes")
            self._check_quota(len(content))
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
            asset = AudioAsset(asset_id, destination, hashlib.sha256(content).hexdigest(), len(content))
            self._assets[asset_id] = asset
            self._managed_paths[asset_id] = (destination,)
            self._managed_sizes[destination] = asset.size_bytes
            self._file_count += 1
            self._total_bytes += asset.size_bytes
            return asset

    def require(self, asset_id: str) -> AudioAsset:
        with self._lock:
            return self._require_unlocked(asset_id)

    def known(self, asset_id: str) -> AudioAsset:
        """Return registered metadata without decoding, for explicit cleanup."""
        with self._lock:
            return self._asset_unlocked(asset_id, allow_conflict=True)

    @contextmanager
    def lease(self, asset_id: str) -> Iterator[AudioAssetSnapshot]:
        """Provide a verified immutable byte snapshot for one node read."""
        with self._lock:
            asset = self._asset_unlocked(asset_id)
            content = self._read_verified_content(asset)
            yield AudioAssetSnapshot(asset=asset, content=content)

    @contextmanager
    def pin(self, asset_id: str) -> Iterator[AudioAsset]:
        """Pin an external reference through its real generation call."""
        with self._lock:
            asset = self._require_unlocked(asset_id)
            self._pins[asset_id] = self._pins.get(asset_id, 0) + 1
        try:
            yield asset
        finally:
            with self._lock:
                count = self._pins.get(asset_id, 0) - 1
                if count > 0:
                    self._pins[asset_id] = count
                else:
                    self._pins.pop(asset_id, None)

    def delete(self, asset_id: str) -> None:
        with self._lock:
            asset = self._assets.get(asset_id)
            if asset is None:
                raise ValueError(f"unknown asset_id: {asset_id}")
            if self._pins.get(asset_id, 0):
                raise AssetInUseError("asset_in_use")
            paths = self._managed_paths[asset_id]
            for path in paths:
                registered = self._registered_path(AudioAsset(asset_id, path, "", self._managed_sizes[path]))
                if registered.exists():
                    registered.unlink()
                self._total_bytes -= self._managed_sizes.pop(path)
                self._file_count -= 1
            del self._assets[asset_id]
            del self._managed_paths[asset_id]
            self._conflicts.discard(asset_id)

    def _rebuild_index(self) -> None:
        with self._lock:
            grouped: dict[str, list[Path]] = {}
            for path in self.root.iterdir():
                match = _MANAGED_NAME.fullmatch(path.name)
                if match is None or match.group("suffix") not in ALLOWED_EXTENSIONS:
                    continue
                if path.is_symlink() or not path.is_file() or not path.resolve(strict=False).is_relative_to(self.root):
                    continue
                asset_id = match.group("asset_id")
                grouped.setdefault(asset_id, []).append(path.resolve())
            for asset_id, paths in sorted(grouped.items()):
                paths = sorted(paths)
                records: list[tuple[Path, int, str, bool]] = []
                for path in paths:
                    size = path.stat().st_size
                    self._file_count += 1
                    self._managed_sizes[path] = size
                    self._total_bytes += size
                    if size > self.max_bytes:
                        records.append((path, size, "", False))
                        continue
                    with path.open("rb") as handle:
                        content = handle.read(self.max_bytes + 1)
                    stable = path.stat().st_size == size and len(content) == size
                    records.append((path, size, hashlib.sha256(content).hexdigest() if stable else "", stable and self._is_valid_audio(path)))
                primary, size, digest, valid = records[0]
                self._managed_paths[asset_id] = tuple(paths)
                if len(paths) > 1:
                    self._conflicts.add(asset_id)
                    valid = False
                self._assets[asset_id] = AudioAsset(asset_id, primary, digest, size, valid=valid)

    def _check_quota(self, incoming_bytes: int) -> None:
        if self._file_count >= self.max_assets or self._total_bytes + incoming_bytes > self.max_total_bytes:
            raise AssetQuotaError("asset_quota_exceeded")

    def _require_unlocked(self, asset_id: str) -> AudioAsset:
        asset = self._asset_unlocked(asset_id)
        self._read_verified_content(asset)
        return asset

    def _asset_unlocked(self, asset_id: str, *, allow_conflict: bool = False) -> AudioAsset:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise ValueError(f"unknown asset_id: {asset_id}")
        if asset_id in self._conflicts and not allow_conflict:
            raise ValueError("asset conflict")
        return asset

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

    def _read_verified_content(self, asset: AudioAsset) -> bytes:
        try:
            path = self._registered_path(asset)
            if not asset.valid:
                raise ValueError("invalid audio content")
            if not path.is_file():
                raise ValueError("asset is missing or tampered")
            with path.open("rb") as handle:
                content = handle.read(self.max_bytes + 1)
            if len(content) != asset.size_bytes or len(content) > self.max_bytes:
                raise ValueError("asset is missing or tampered")
            if hashlib.sha256(content).hexdigest() != asset.sha256:
                raise ValueError("asset is missing or tampered")
        except (OSError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        return content

    @staticmethod
    def _is_valid_audio(path: Path) -> bool:
        try:
            AudioAssetStore._validate_audio(path)
            return True
        except ValueError:
            return False

    @staticmethod
    def _validate_audio(path: Path) -> None:
        try:
            info = soundfile.info(path)
        except Exception as exc:
            raise ValueError("invalid audio content") from exc
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError("invalid audio content")


@contextmanager
def pin_voice_asset(voice: Any, *, store: AudioAssetStore | None = None) -> Iterator[None]:
    """Pin only bridge-owned voices; ordinary upstream voice dictionaries pass through."""
    seen: set[int] = set()
    while isinstance(voice, (list, tuple)):
        marker = id(voice)
        if marker in seen or len(voice) != 1:
            voice = None
            break
        seen.add(marker)
        voice = voice[0]
    asset_id = voice.get("asset_id") if isinstance(voice, dict) else None
    if not isinstance(asset_id, str) or not asset_id:
        with nullcontext():
            yield
        return
    with (store or get_audio_asset_store()).pin(asset_id):
        yield


_audio_asset_store: AudioAssetStore | None = None
_audio_asset_store_lock = Lock()


def _limit_from_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def get_audio_asset_store() -> AudioAssetStore:
    global _audio_asset_store
    with _audio_asset_store_lock:
        if _audio_asset_store is not None:
            return _audio_asset_store
        import folder_paths

        input_root = Path(folder_paths.get_input_directory()).resolve()
        _audio_asset_store = AudioAssetStore(
            input_root / "tts-audio-suite",
            owner_root=input_root,
            max_total_bytes=_limit_from_env("TTS_AUDIO_SUITE_ASSET_MAX_TOTAL_BYTES", DEFAULT_MAX_TOTAL_BYTES),
            max_assets=_limit_from_env("TTS_AUDIO_SUITE_ASSET_MAX_ASSETS", DEFAULT_MAX_ASSETS),
        )
        return _audio_asset_store


def reset_audio_asset_store_for_tests() -> None:
    global _audio_asset_store
    with _audio_asset_store_lock:
        _audio_asset_store = None
