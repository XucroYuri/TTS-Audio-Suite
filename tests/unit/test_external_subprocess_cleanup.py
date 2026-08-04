"""Behavioral coverage for Windows external-runtime cleanup races."""

from __future__ import annotations

import importlib.util
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
