"""Strict local-path validation for private resource configuration."""

from __future__ import annotations

from pathlib import Path


_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _is_reparse_point(path: Path) -> bool:
    """Return whether *path itself* is a link, junction, or other reparse point."""
    stat_result = path.lstat()
    return path.is_symlink() or bool(
        getattr(stat_result, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT
    )


def _reject_reparse_components(path: Path, label: str) -> None:
    current = Path(path.parts[0])
    for component in path.parts[1:]:
        current /= component
        if _is_reparse_point(current):
            raise ValueError(f"{label} must not traverse a reparse point: {current}")


def resolve_absolute_regular_file(value: str | Path, label: str) -> Path:
    """Resolve an absolute regular file without accepting a reparse-point path.

    Resource registries hold executable paths for a long time.  Rejecting a
    symbolic/junction path before resolving it prevents the configuration from
    silently changing target after validation.  The resolved file may live
    outside its model checkout; that is required for isolated runtimes.
    """
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    try:
        _reject_reparse_components(requested, label)
        resolved = requested.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} could not be resolved: {requested}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved
