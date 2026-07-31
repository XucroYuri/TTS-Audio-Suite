import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INSTALL_SPEC = importlib.util.spec_from_file_location("tts_suite_install_test_module", REPO_ROOT / "install.py")
INSTALL_MODULE = importlib.util.module_from_spec(INSTALL_SPEC)
INSTALL_SPEC.loader.exec_module(INSTALL_MODULE)

from engines.dots_tts.dots_tts_engine import DotsTTSEngine
from utils.runtimes.profiles import get_runtime_profile


@pytest.mark.unit
def test_fish_source_restore_accepts_namespace_package(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    source_package = source_root / "fish_speech"
    (source_package / "inference_engine").mkdir(parents=True)
    (source_package / "inference_engine" / "__init__.py").write_text("", encoding="utf-8")

    site_root = tmp_path / "site-packages"
    target_package = site_root / "fish_speech"
    target_package.mkdir(parents=True)
    (target_package / "content_sequence.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(site_root), *sys.path])

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer._restore_fish_source_package(source_root) is True
    assert (target_package / "inference_engine" / "__init__.py").is_file()


@pytest.mark.unit
def test_module_available_finds_nested_module_without_importing_parent(tmp_path, monkeypatch):
    package_root = tmp_path / "dots_tts"
    package_root.mkdir()
    (package_root / "__init__.py").write_text(
        "raise RuntimeError('parent package must not be imported during availability probing')\n",
        encoding="utf-8",
    )
    (package_root / "runtime.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "dots_tts", raising=False)
    monkeypatch.delitem(sys.modules, "dots_tts.runtime", raising=False)

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer.module_available("dots_tts.runtime") is True
    assert "dots_tts" not in sys.modules


@pytest.mark.unit
def test_targets_profile_selects_only_api_bridge_target_engines(monkeypatch):
    monkeypatch.setenv("TTS_AUDIO_SUITE_INSTALL_PROFILE", "tts_more_targets")

    installer = INSTALL_MODULE.TTSAudioInstaller()

    assert installer.install_profile == "tts_more_targets"
    assert installer.active_engine_runtime_checks == (
        ("GPT-SoVITS API Bridge (configuration)", ("api_bridge.resource_registry", "nodes.api_bridge.resource_engine_nodes")),
        ("IndexTTS API Bridge (configuration)", ("api_bridge.resource_registry", "nodes.api_bridge.resource_engine_nodes")),
        ("CosyVoice API Bridge (configuration)", ("api_bridge.resource_registry", "nodes.api_bridge.resource_engine_nodes")),
    )
    assert installer.optional_engine_installers_enabled is False
    assert installer.active_core_module_checks == (("soundfile", "SoundFile"),)


@pytest.mark.unit
def test_targets_profile_summary_does_not_claim_generation_readiness(monkeypatch, capsys):
    monkeypatch.setenv("TTS_AUDIO_SUITE_INSTALL_PROFILE", "tts_more_targets")
    installer = INSTALL_MODULE.TTSAudioInstaller()
    installer.is_windows = True
    installer.engine_validation_results = {
        "GPT-SoVITS API Bridge (configuration)": True,
        "IndexTTS API Bridge (configuration)": True,
        "CosyVoice API Bridge (configuration)": True,
    }

    installer.print_installation_summary(True)

    output = capsys.readouterr().out
    assert "API Bridge checks prove configuration imports only" in output
    assert "API BRIDGE CONFIGURATION READY - VERIFY EXTERNAL RUNTIMES BEFORE SYNTHESIS" in output
    assert "READY TO USE TTS AUDIO SUITE IN COMFYUI!" not in output
    assert "IndexTTS-2 will use fallback text processing instead" not in output
    assert "all TTS generation will work fine" not in output


@pytest.mark.unit
def test_fresh_engine_probe_uses_a_subprocess(monkeypatch):
    installer = INSTALL_MODULE.TTSAudioInstaller()
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "OK\n", "")

    monkeypatch.setattr(INSTALL_MODULE.subprocess, "run", fake_run)

    assert installer.fresh_module_importable("api_bridge.resource_registry") is True
    assert calls[0][0][0] == sys.executable
    assert calls[0][0][1:3] == ["-c", "import importlib; importlib.import_module('api_bridge.resource_registry')"]


@pytest.mark.unit
def test_dependency_integrity_failure_is_reported(monkeypatch):
    installer = INSTALL_MODULE.TTSAudioInstaller()
    monkeypatch.setattr(
        INSTALL_MODULE.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "broken requirement"),
    )

    assert installer.dependency_integrity_ok() is False


@pytest.mark.unit
def test_targets_profile_exits_nonzero_when_pip_check_fails(monkeypatch):
    monkeypatch.setenv("TTS_AUDIO_SUITE_INSTALL_PROFILE", "tts_more_targets")
    installer = INSTALL_MODULE.TTSAudioInstaller()
    monkeypatch.setattr(INSTALL_MODULE, "TTSAudioInstaller", lambda: installer)
    monkeypatch.setattr(installer, "check_python_environment", lambda: True)
    monkeypatch.setattr(installer, "can_skip_dependency_installation", lambda: True)
    monkeypatch.setattr(installer, "check_version_conflicts", lambda: None)
    monkeypatch.setattr(installer, "validate_installation", lambda: True)
    monkeypatch.setattr(installer, "dependency_integrity_ok", lambda: False)
    monkeypatch.setattr(installer, "save_installation_state", lambda success: None)
    monkeypatch.setattr(installer, "print_installation_summary", lambda success: None)

    with pytest.raises(SystemExit) as error:
        INSTALL_MODULE.main()

    assert error.value.code == 1


@pytest.mark.unit
def test_dots_fallback_overrides_incomplete_tn_module_temporarily(monkeypatch):
    unrelated_tn = types.ModuleType("tn")
    monkeypatch.setitem(sys.modules, "tn", unrelated_tn)
    for name in (
        "tn.chinese",
        "tn.chinese.normalizer",
        "tn.english",
        "tn.english.normalizer",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with DotsTTSEngine._text_normalizer_compat():
        from tn.chinese.normalizer import Normalizer as ZhNormalizer
        from tn.english.normalizer import Normalizer as EnNormalizer

        assert ZhNormalizer().normalize("123") == "123"
        assert EnNormalizer().normalize("test") == "test"

    assert sys.modules["tn"] is unrelated_tn
    assert "tn.chinese" not in sys.modules
    assert "tn.english" not in sys.modules


@pytest.mark.unit
def test_vibevoice_runtime_avoids_webrtc_dependency_conflict():
    profile = get_runtime_profile("vibevoice_transformers4_shared")

    assert profile is not None
    assert "av" in profile.pip_packages
    assert not {
        "aiortc",
        "pyee",
        "dnspython",
        "ifaddr",
        "pylibsrtp",
        "pyopenssl",
    }.intersection(profile.pip_packages)


@pytest.mark.unit
@pytest.mark.parametrize(
    "profile_name",
    ("vibevoice_transformers4_shared", "qwen3_tts_transformers4_dedicated"),
)
def test_legacy_transformers_runtime_pins_compatible_kernels(profile_name):
    profile = get_runtime_profile(profile_name)

    assert profile is not None
    assert "kernels>=0.6.1,<=0.9" in profile.pip_packages
