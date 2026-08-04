"""Behavioral coverage for Windows external-runtime cleanup races."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_external_index_subprocess_module():
    path = REPO_ROOT / "engines" / "index_tts" / "external_subprocess.py"
    spec = importlib.util.spec_from_file_location("external_index_cleanup_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_windows_cleanup_retries_file_not_found_race(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    (temporary_path / "late-numba-cache").write_bytes(b"cache")

    class RacingTemporaryDirectory:
        def cleanup(self):
            error = OSError(3, "the system cannot find the path specified")
            error.winerror = 3
            raise error

    monkeypatch.setattr(module.os, "name", "nt")
    module._cleanup_temporary_directory(RacingTemporaryDirectory(), temporary_path)

    assert not temporary_path.exists()


@pytest.mark.unit
def test_windows_cleanup_keeps_non_transient_error_fail_closed(monkeypatch, tmp_path):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    (temporary_path / "sentinel").write_bytes(b"keep")

    class BrokenTemporaryDirectory:
        def cleanup(self):
            error = OSError(87, "the parameter is incorrect")
            error.winerror = 87
            raise error

    monkeypatch.setattr(module.os, "name", "nt")
    with pytest.raises(OSError, match="parameter is incorrect"):
        module._cleanup_temporary_directory(BrokenTemporaryDirectory(), temporary_path)

    assert temporary_path.exists()
    assert (temporary_path / "sentinel").read_bytes() == b"keep"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("callback_api", "error_number"),
    [("onexc", 3), ("onexc", 145), ("onerror", 3), ("onerror", 145)],
)
def test_windows_cleanup_rmtree_callback_handles_transient_directory_races(
    monkeypatch, tmp_path, callback_api, error_number
):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    late_cache_path = str(temporary_path / "late-numba-cache")
    (temporary_path / "late-numba-cache").write_bytes(b"cache")

    class RacingTemporaryDirectory:
        def cleanup(self):
            error = OSError(3, "the system cannot find the path specified")
            error.winerror = 3
            raise error

    real_rmtree = module.shutil.rmtree

    if callback_api == "onexc":

        def racing_rmtree(path, *, onexc=None):
            assert onexc is not None
            error = OSError(error_number, "the child disappeared during cleanup")
            error.winerror = error_number
            onexc(os.rmdir if error_number == 145 else os.unlink, late_cache_path, error)
            real_rmtree(path)

    else:

        def racing_rmtree(path, *, onerror=None):
            assert onerror is not None
            error = OSError(error_number, "the child disappeared during cleanup")
            error.winerror = error_number
            onerror(
                os.rmdir if error_number == 145 else os.unlink,
                late_cache_path,
                (OSError, error, None),
            )
            real_rmtree(path)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.shutil, "rmtree", racing_rmtree)
    module._cleanup_temporary_directory(RacingTemporaryDirectory(), temporary_path)

    assert not temporary_path.exists()


@pytest.mark.unit
def test_windows_cleanup_rmtree_callback_does_not_swallow_permission_error(
    monkeypatch, tmp_path
):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    sentinel_path = str(temporary_path / "sentinel")
    (temporary_path / "sentinel").write_bytes(b"keep")

    class BrokenTemporaryDirectory:
        def cleanup(self):
            error = OSError(5, "access is denied")
            error.winerror = 5
            raise error

    def broken_rmtree(path, *, onexc=None):
        assert onexc is not None
        error = OSError(5, "access is denied")
        error.winerror = 5
        onexc(os.unlink, sentinel_path, error)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.shutil, "rmtree", broken_rmtree)
    monkeypatch.setattr(module, "_WINDOWS_CLEANUP_RETRY_SECONDS", 0.0)

    with pytest.raises(OSError, match="access is denied"):
        module._cleanup_temporary_directory(BrokenTemporaryDirectory(), temporary_path)

    assert temporary_path.exists()
    assert (temporary_path / "sentinel").read_bytes() == b"keep"


@pytest.mark.unit
def test_windows_cleanup_allows_directory_not_empty_race_to_quiesce(
    monkeypatch, tmp_path
):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    (temporary_path / "late-numba-cache").write_bytes(b"cache")

    class FakeClock:
        def __init__(self):
            self.value = 0.0

        def monotonic(self):
            self.value += 1.0
            return self.value

        @staticmethod
        def sleep(_seconds):
            return None

    class RacingTemporaryDirectory:
        def cleanup(self):
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error

    real_rmtree = module.shutil.rmtree
    attempts = 0

    def delayed_rmtree(path):
        nonlocal attempts
        attempts += 1
        if attempts < 6:
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error
        return real_rmtree(path)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "time", FakeClock())
    monkeypatch.setattr(module.shutil, "rmtree", delayed_rmtree)
    module._cleanup_temporary_directory(RacingTemporaryDirectory(), temporary_path)

    assert attempts == 6
    assert not temporary_path.exists()


@pytest.mark.unit
def test_windows_cleanup_rejects_persistent_directory_not_empty_race(
    monkeypatch, tmp_path
):
    module = _load_external_index_subprocess_module()
    temporary_path = tmp_path / "private-child"
    temporary_path.mkdir()
    (temporary_path / "sentinel").write_bytes(b"keep")

    class FakeClock:
        def __init__(self):
            self.value = 0.0

        def monotonic(self):
            self.value += 1.0
            return self.value

        @staticmethod
        def sleep(_seconds):
            return None

    class RacingTemporaryDirectory:
        def cleanup(self):
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error

    attempts = 0

    def persistent_rmtree(_path):
        nonlocal attempts
        attempts += 1
        error = OSError(145, "directory not empty")
        error.winerror = 145
        raise error

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module, "time", FakeClock())
    monkeypatch.setattr(module.shutil, "rmtree", persistent_rmtree)
    with pytest.raises(OSError, match="directory not empty"):
        module._cleanup_temporary_directory(RacingTemporaryDirectory(), temporary_path)

    assert attempts == int(module._WINDOWS_DIRECTORY_NOT_EMPTY_RETRY_SECONDS)
    assert temporary_path.exists()
    assert (temporary_path / "sentinel").read_bytes() == b"keep"


@pytest.mark.unit
def test_windows_cleanup_retries_long_nested_paths_with_extended_prefix(
    monkeypatch, tmp_path
):
    module = _load_external_index_subprocess_module()
    long_raw_path = str(tmp_path) + ("\\" + ("cache-" + "x" * 32)) * 8 + "\\runner"

    class LongTemporaryPath:
        def __init__(self, raw_path):
            self.raw_path = raw_path
            self.exists_calls = 0

        def __fspath__(self):
            return self.raw_path

        def exists(self):
            self.exists_calls += 1
            return self.exists_calls == 1

    temporary_path = LongTemporaryPath(long_raw_path)

    class RacingTemporaryDirectory:
        def cleanup(self):
            error = OSError(145, "directory not empty")
            error.winerror = 145
            raise error

    captured_paths = []

    def capture_rmtree(path):
        captured_paths.append(path)

    monkeypatch.setattr(module.os, "name", "nt")
    monkeypatch.setattr(module.shutil, "rmtree", capture_rmtree)
    module._cleanup_temporary_directory(RacingTemporaryDirectory(), temporary_path)

    assert len(long_raw_path) > 260
    assert len(captured_paths) == 1
    assert isinstance(captured_paths[0], str)
    assert captured_paths[0].startswith("\\\\?\\")
    assert len(captured_paths[0]) > 260
    assert captured_paths[0] == module._private_child_path(long_raw_path)
