"""Small compatibility helpers for ComfyUI folder_paths API drift."""

from __future__ import annotations

import os
from typing import Any


def ensure_system_user_directory(folder_paths: Any, *, fallback_root: str) -> bool:
    """Install the legacy plugin helper on ComfyUI versions that lack it.

    Older ComfyUI releases expose ``get_user_directory`` while the suite also
    supports the scoped ``get_system_user_directory(name)`` contract.  Keep the
    fallback local to the active ComfyUI user directory and return whether a
    shim was installed so callers can report compatibility without mutating an
    existing implementation.
    """

    if callable(getattr(folder_paths, "get_system_user_directory", None)):
        return False

    get_user_directory = getattr(folder_paths, "get_user_directory", None)
    base = get_user_directory() if callable(get_user_directory) else fallback_root

    def get_system_user_directory(name: str = "system") -> str:
        path = os.path.join(str(base), str(name))
        os.makedirs(path, exist_ok=True)
        return path

    setattr(folder_paths, "get_system_user_directory", get_system_user_directory)
    return True
