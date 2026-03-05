from __future__ import annotations

"""Shared artifact metadata helpers."""

import uuid
from pathlib import Path
from typing import Any


def safe_file_size(path: str | None) -> int | None:
    """Return file size in bytes, or None when path is missing/unreadable."""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    return int(file_path.stat().st_size)


def resolve_model_artifact_target(payload: dict[str, Any], artifacts_dir: Path) -> tuple[bool, str | None, Path | None]:
    """Resolve save-model intent and model artifact output location.

    Args:
        payload: Incoming train/distill request body.
        artifacts_dir: Framework artifact root directory.

    Returns:
        Tuple `(save_model, model_id, model_dir)`.
    """
    save_model = bool(payload.get("save_model", False))
    model_id = (payload.get("model_id") or str(uuid.uuid4())) if save_model else None
    model_dir = (artifacts_dir / str(model_id)) if model_id else None
    return save_model, model_id, model_dir
