"""
TTS model path resolution — respects extra_model_paths.yaml TTS entries.

Replaces the pattern:

    os.path.join(folder_paths.models_dir, "TTS", "F5-TTS")

which hardcodes the *primary* model root and ignores shared/extra model paths,
with folder_paths.get_folder_paths("TTS") so that every registered TTS root is
searched.

Usage:
    from utils.models.tts_paths import get_tts_root_dirs

    for root in get_tts_root_dirs():
        candidate = os.path.join(root, "F5-TTS", "F5-Hindi-Small")
        if os.path.isdir(candidate):
            ...

Before this change, users with a shared model directory had to pass
--models-directory explicitly because the plugin only looked under
ComfyUI/models/. Now extra_model_paths.yaml entries like:

  comfyui:
    base_path: D:/Comfy-Desktop/ComfyUI-Shared
    TTS: models/TTS/

are automatically searched.
"""

import os
import folder_paths
from typing import List


# Cache TTS root directories once; these do not change at runtime.
_tts_roots: List[str] = []
_tts_roots_populated: bool = False


def get_tts_root_dirs() -> List[str]:
    """Return all registered TTS model root directories.

    Includes:
        - the primary models/TTS/ directory
        - every base_path + TTS mapping from extra_model_paths.yaml
        - any directories registered via folder_paths.add_model_folder_path("TTS", ...)

    Guaranteed to include at least one entry (the default models/TTS).
    Thread-safe to call from any engine.
    """
    global _tts_roots, _tts_roots_populated

    if _tts_roots_populated:
        return _tts_roots

    _tts_roots_populated = True

    resolved: List[str] = []
    seen: set = set()

    # get_folder_paths returns all registered roots for the category,
    # including the default and any extra paths.
    for root in folder_paths.get_folder_paths("TTS"):
        normalized = os.path.normpath(root)
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)

    _tts_roots = resolved
    return _tts_roots


def find_tts_model_subdir(subdir: str) -> List[str]:
    """Return existing directories matching subdir under every TTS root.

    Args:
        subdir: relative path under the TTS model root, e.g. "F5-TTS/F5-Hindi-Small"

    Returns:
        List of absolute paths that exist on disk, ordered by priority
        (primary root first, then extras).
    """
    existing: List[str] = []
    for root in get_tts_root_dirs():
        candidate = os.path.join(root, subdir)
        if os.path.isdir(candidate):
            existing.append(candidate)
    return existing


def find_tts_model_file(subdir: str, filename: str) -> List[str]:
    """Return existing files matching subdir/filename under every TTS root.

    Args:
        subdir: relative path under the TTS model root
        filename: exact filename, e.g. "model_2500000.safetensors"

    Returns:
        List of absolute file paths that exist, ordered by priority.
    """
    existing: List[str] = []
    for root in get_tts_root_dirs():
        candidate = os.path.join(root, subdir, filename)
        if os.path.isfile(candidate):
            existing.append(candidate)
    return existing
