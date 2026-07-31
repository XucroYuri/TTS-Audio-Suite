from pathlib import Path

import folder_paths
import pytest


@pytest.mark.unit
def test_tts_roots_follow_runtime_folder_registration(monkeypatch, tmp_path):
    primary = tmp_path / "primary" / "TTS"
    shared = tmp_path / "shared" / "TTS"
    primary.mkdir(parents=True)
    shared.mkdir(parents=True)
    registered = [str(primary)]
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda category: list(registered))

    from utils.models.tts_paths import get_tts_root_dirs

    assert get_tts_root_dirs() == [str(primary)]

    registered.append(str(shared))

    assert get_tts_root_dirs() == [str(primary), str(shared)]


@pytest.mark.unit
def test_tts_model_search_preserves_registered_root_priority(monkeypatch, tmp_path):
    roots = [tmp_path / "primary", tmp_path / "shared"]
    for root in roots:
        (root / "F5-TTS" / "voice").mkdir(parents=True)
        (root / "F5-TTS" / "voice" / "model.safetensors").write_bytes(b"model")
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda category: [str(root) for root in roots])

    from utils.models.tts_paths import find_tts_model_file, find_tts_model_subdir

    assert find_tts_model_subdir(str(Path("F5-TTS") / "voice")) == [
        str(root / "F5-TTS" / "voice") for root in roots
    ]
    assert find_tts_model_file(str(Path("F5-TTS") / "voice"), "model.safetensors") == [
        str(root / "F5-TTS" / "voice" / "model.safetensors") for root in roots
    ]
