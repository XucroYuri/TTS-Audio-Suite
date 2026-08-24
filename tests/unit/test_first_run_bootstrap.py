import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.system import first_run_bootstrap as frb
from utils.system.dependency_checker import DependencyChecker


MISSING_ENGINE_ISSUES = {
    "rvc": [("faiss", "faiss-cpu"), ("torchcrepe", "torchcrepe")],
}


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect every cache artifact into a temp dir."""
    cache = tmp_path / ".cache"
    cache.mkdir()
    monkeypatch.setattr(frb, "_CACHE_DIR", cache)
    monkeypatch.setattr(frb, "_INSTALL_STATE_PATH", cache / "install_state.json")
    monkeypatch.setattr(frb, "_ACTIVE_MARKER_PATH", cache / "bootstrap_active.json")
    monkeypatch.setattr(frb, "_LOG_PATH", cache / "bootstrap.log")
    return cache


@pytest.fixture
def suspended_watchdog(monkeypatch):
    """Capture watchdog threads instead of running them."""
    started = []
    monkeypatch.setattr(
        frb.threading,
        "Thread",
        lambda target, args=(), daemon=False: started.append(target)
        or SimpleNamespace(start=lambda: None),
    )
    return started


@pytest.fixture
def fake_popen(monkeypatch):
    """Replace subprocess.Popen with a recorder returning a stub process."""
    spawned = []

    def record(*args, **kwargs):
        spawned.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(pid=424242, poll=lambda: None, wait=lambda timeout=None: 0)

    monkeypatch.setattr(frb.subprocess, "Popen", record)
    return spawned


@pytest.mark.unit
def test_pid_alive_distinguishes_live_and_dead_pids():
    assert frb._pid_alive(os.getpid()) is True
    assert frb._pid_alive(999999) is False
    assert frb._pid_alive(0) is False


@pytest.mark.unit
def test_timeout_seconds_reads_env_override(monkeypatch):
    monkeypatch.delenv("TTS_AUDIO_SUITE_BOOTSTRAP_TIMEOUT", raising=False)
    assert frb._timeout_seconds() == 1800
    monkeypatch.setenv("TTS_AUDIO_SUITE_BOOTSTRAP_TIMEOUT", "60")
    assert frb._timeout_seconds() == 60
    monkeypatch.setenv("TTS_AUDIO_SUITE_BOOTSTRAP_TIMEOUT", "nonsense")
    assert frb._timeout_seconds() == 1800


@pytest.mark.unit
def test_stale_marker_is_cleaned_and_repair_respawns(isolated_cache, suspended_watchdog, fake_popen):
    marker = isolated_cache / "bootstrap_active.json"
    marker.write_text(json.dumps({"pid": 999999}), encoding="utf-8")

    started = frb.start_background_repair(MISSING_ENGINE_ISSUES)

    assert started is True
    assert len(fake_popen) == 1
    assert fake_popen[0]["args"][0][1].endswith("install.py")
    assert json.loads(marker.read_text())["pid"] == 424242


@pytest.mark.unit
def test_live_marker_blocks_second_repair(isolated_cache, suspended_watchdog, fake_popen):
    marker = isolated_cache / "bootstrap_active.json"
    marker.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")

    started = frb.start_background_repair(MISSING_ENGINE_ISSUES)

    assert started is False
    assert fake_popen == []
    assert marker.exists()


@pytest.mark.unit
def test_disabled_auto_install_keeps_manual_guidance(isolated_cache, monkeypatch, capsys):
    monkeypatch.setenv("TTS_AUDIO_SUITE_AUTO_INSTALL", "0")

    started = frb.start_background_repair(MISSING_ENGINE_ISSUES)

    assert started is False
    output = capsys.readouterr().out
    assert "automatic repair is disabled" in output
    assert "install.py" in output


@pytest.mark.unit
def test_callback_preserves_default_output_and_skips_when_engines_healthy(isolated_cache, monkeypatch, capsys):
    spawned = []
    monkeypatch.setattr(frb, "start_background_repair", lambda issues: spawned.append(issues))
    monkeypatch.setattr(DependencyChecker, "check_engine_dependencies", staticmethod(lambda engine: []))

    frb.dependency_check_callback(["⚠️ Some TTS Audio Suite engines are not installed correctly:"])

    output = capsys.readouterr().out
    assert "📋 System Dependencies (background check):" in output
    assert spawned == []


@pytest.mark.unit
def test_callback_triggers_bootstrap_for_missing_engines(isolated_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(frb, "start_background_repair", lambda issues: calls.append(issues) or True)
    monkeypatch.setattr(
        DependencyChecker,
        "check_engine_dependencies",
        staticmethod(lambda engine: MISSING_ENGINE_ISSUES.get(engine, [])),
    )

    frb.dependency_check_callback(["⚠️ Some TTS Audio Suite engines are not installed correctly:"])

    assert len(calls) == 1
    assert set(calls[0]) == {"rvc"}


@pytest.mark.unit
def test_callback_never_spawns_when_installer_already_ran(isolated_cache, monkeypatch):
    isolated_cache.joinpath("install_state.json").write_text("{}", encoding="utf-8")
    calls = []
    monkeypatch.setattr(frb, "start_background_repair", lambda issues: calls.append(issues))
    monkeypatch.setattr(
        DependencyChecker,
        "check_engine_dependencies",
        staticmethod(lambda engine: MISSING_ENGINE_ISSUES.get(engine, [])),
    )

    frb.dependency_check_callback(["⚠️ Some TTS Audio Suite engines are not installed correctly:"])

    assert calls == []
