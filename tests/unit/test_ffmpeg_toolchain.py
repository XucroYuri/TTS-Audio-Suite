import subprocess

import pytest

from utils.ffmpeg_utils import FFmpegUtils


@pytest.fixture(autouse=True)
def _reset_toolchain_cache():
    attributes = {
        "_ffmpeg_available": None,
        "_ffmpeg_path": None,
        "_ffprobe_available": None,
        "_ffprobe_path": None,
        "_check_performed": False,
        "_ffprobe_check_performed": False,
        "_toolchain_validated": False,
        "_toolchain_ok": False,
    }
    previous = {name: getattr(FFmpegUtils, name, None) for name in attributes}
    for name, value in attributes.items():
        setattr(FFmpegUtils, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(FFmpegUtils, name, value)


@pytest.mark.unit
def test_audio_toolchain_requires_ffprobe_even_when_ffmpeg_exists(monkeypatch):
    def fake_run(command, **_kwargs):
        if command[0] == "ffmpeg":
            return subprocess.CompletedProcess(command, 0)
        raise FileNotFoundError(command[0])

    monkeypatch.setattr("utils.ffmpeg_utils.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match=r"Missing: ffprobe"):
        FFmpegUtils.require_audio_toolchain(feature="F5-TTS reference audio preprocessing")


@pytest.mark.unit
def test_audio_toolchain_accepts_both_executables(monkeypatch):
    monkeypatch.setattr(
        "utils.ffmpeg_utils.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0),
    )

    FFmpegUtils.require_audio_toolchain(feature="F5-TTS reference audio preprocessing")
