from __future__ import annotations

"""Shared artifact metadata helpers."""

from pathlib import Path


def safe_file_size(path: str | None) -> int | None:
    """Return file size in bytes, or None when path is missing/unreadable."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    return int(file_path.stat().st_size)
