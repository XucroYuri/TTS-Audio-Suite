"""Import-path setup for an existing official GPT-SoVITS checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_gpt_sovits_source(project_root: str | None = None) -> None:
    """Make official GPT-SoVITS imports available without changing process cwd."""
    root = project_root or os.environ.get("GPT_SOVITS_PATH", "")
    if not root:
        return

    source_root = Path(root).expanduser().resolve()
    package_root = source_root / "GPT_SoVITS"
    for candidate in (source_root, package_root):
        candidate_text = str(candidate)
        if candidate.is_dir() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
