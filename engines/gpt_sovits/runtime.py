"""Checkout-bound import and configuration helpers for official GPT-SoVITS."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


_CHECKOUT_CWD_LOCK = threading.RLock()
_CHECKOUT_STATE_LOCK = threading.RLock()
_BOUND_CHECKOUT: Optional[Path] = None
_BOUND_IMPORT_PATHS: tuple[str, ...] = ()


@dataclass(frozen=True)
class GPTSovitsRuntimeContext:
    """Normalized absolute paths required by the official inference library."""

    checkout_root: Path
    package_root: Path
    eres2net_root: Path
    sv_path: Path


def configure_gpt_sovits_source(project_root: str | None = None) -> Optional[GPTSovitsRuntimeContext]:
    """Inject all official import roots and return the normalized checkout context."""
    root = project_root or os.environ.get("GPT_SOVITS_PATH", "")
    if not root:
        return None

    checkout_root = Path(root).expanduser().resolve()
    package_root = checkout_root / "GPT_SoVITS"
    eres2net_root = package_root / "eres2net"
    global _BOUND_CHECKOUT, _BOUND_IMPORT_PATHS
    with _CHECKOUT_STATE_LOCK:
        if _BOUND_CHECKOUT is None:
            _BOUND_CHECKOUT = checkout_root
        elif _BOUND_CHECKOUT != checkout_root:
            raise RuntimeError(
                f"GPT-SoVITS runtime is already bound to {_BOUND_CHECKOUT}; "
                f"refusing different checkout {checkout_root}"
            )
        import_paths: list[str] = []
        for candidate in (checkout_root, package_root, eres2net_root):
            candidate_text = str(candidate)
            if candidate.is_dir() and candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
                import_paths.append(candidate_text)
        if import_paths:
            _BOUND_IMPORT_PATHS = tuple(dict.fromkeys((*_BOUND_IMPORT_PATHS, *import_paths)))
    return GPTSovitsRuntimeContext(
        checkout_root=checkout_root,
        package_root=package_root,
        eres2net_root=eres2net_root,
        sv_path=package_root / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
    )


def reset_gpt_sovits_checkout_for_tests() -> None:
    """Clear checkout-only process state for isolated unit tests; never call in production."""
    global _BOUND_CHECKOUT, _BOUND_IMPORT_PATHS
    with _CHECKOUT_STATE_LOCK:
        for import_path in _BOUND_IMPORT_PATHS:
            while import_path in sys.path:
                sys.path.remove(import_path)
        _BOUND_CHECKOUT = None
        _BOUND_IMPORT_PATHS = ()


@contextmanager
def checkout_cwd(context: GPTSovitsRuntimeContext) -> Iterator[None]:
    """Temporarily satisfy official relative config setup without leaking cwd."""
    with _CHECKOUT_CWD_LOCK:
        original_cwd = os.getcwd()
        try:
            os.chdir(context.checkout_root)
            yield
        finally:
            os.chdir(original_cwd)
