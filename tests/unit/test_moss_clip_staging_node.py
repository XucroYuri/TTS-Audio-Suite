from __future__ import annotations

import importlib.util
from pathlib import Path


def test_moss_clip_staging_exposes_audio_writer() -> None:
    path = Path(__file__).parents[2] / "nodes" / "training" / "moss_clip_staging_node.py"
    spec = importlib.util.spec_from_file_location("moss_clip_staging_fallback_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert callable(module.wavfile.write)
