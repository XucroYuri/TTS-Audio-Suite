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
    for candidate in (checkout_root, package_root, eres2net_root):
        candidate_text = str(candidate)
        if candidate.is_dir() and candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
    return GPTSovitsRuntimeContext(
        checkout_root=checkout_root,
        package_root=package_root,
        eres2net_root=eres2net_root,
        sv_path=package_root / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
    )


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
