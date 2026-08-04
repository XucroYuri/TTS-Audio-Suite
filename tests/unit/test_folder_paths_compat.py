from __future__ import annotations

from types import SimpleNamespace

from utils.compatibility.folder_paths_compat import ensure_system_user_directory


def test_folder_paths_compatibility_shim_uses_user_directory(tmp_path) -> None:
    module = SimpleNamespace(get_user_directory=lambda: str(tmp_path))

    assert ensure_system_user_directory(module, fallback_root=str(tmp_path / "fallback")) is True
    path = module.get_system_user_directory("tts_audio_suite")

    assert path == str(tmp_path / "tts_audio_suite")
    assert (tmp_path / "tts_audio_suite").is_dir()


def test_folder_paths_compatibility_shim_preserves_existing_helper(tmp_path) -> None:
    existing = lambda name="system": str(tmp_path / name)
    module = SimpleNamespace(get_system_user_directory=existing)

    assert ensure_system_user_directory(module, fallback_root=str(tmp_path / "fallback")) is False
    assert module.get_system_user_directory("tts_audio_suite") == str(tmp_path / "tts_audio_suite")
