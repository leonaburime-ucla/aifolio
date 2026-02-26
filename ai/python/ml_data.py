from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    from ml.file_util import load_tabular_file
except ModuleNotFoundError:  # pragma: no cover
    AI_ROOT = Path(__file__).resolve().parent.parent
    if str(AI_ROOT) not in sys.path:
        sys.path.append(str(AI_ROOT))
    from ml.file_util import load_tabular_file


ML_DATA_DIR = Path(__file__).resolve().parent.parent / "ml" / "data"
SUPPORTED_SUFFIXES = {".csv", ".xls", ".xlsx"}


def _load_config() -> dict[str, dict[str, Any]]:
    """Load the unified sources.json config (sources, targetColumn, task per dataset)."""
    config_path = ML_DATA_DIR / "sources.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return {str(k): v for k, v in payload.items() if isinstance(v, dict)}


def _dataset_files() -> list[Path]:
    if not ML_DATA_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in ML_DATA_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ],
        key=lambda item: item.name.lower(),
    )


def _to_manifest_entry(path: Path, config: dict[str, dict[str, Any]]) -> dict[str, Any]:
    info = config.get(path.name, {})
    entry: dict[str, Any] = {
        "id": path.name,
        "label": path.name,
        "format": path.suffix.lower().lstrip("."),
        "sizeBytes": path.stat().st_size,
        "files": {"data": path.name},
        "sources": info.get("sources", []),
    }
    if info.get("targetColumn"):
        entry["targetColumn"] = info["targetColumn"]
    if info.get("task"):
        entry["task"] = info["task"]
    return entry


def list_ml_datasets() -> list[dict[str, Any]]:
    config = _load_config()
    return [_to_manifest_entry(path, config) for path in _dataset_files()]


def resolve_ml_dataset_path(dataset_id: str) -> Path | None:
    datasets = _dataset_files()
    exact_name_match = {path.name: path for path in datasets}
    if dataset_id in exact_name_match:
        return exact_name_match[dataset_id]

    stem_matches = [path for path in datasets if path.stem == dataset_id]
    if len(stem_matches) == 1:
        return stem_matches[0]
    return None


def load_ml_dataset(
    dataset_id: str,
    row_limit: int | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    file_path = resolve_ml_dataset_path(dataset_id)
    if file_path is None:
        return {"status": "error", "error": "Dataset not found."}

    rows = load_tabular_file(file_path, sheet_name=sheet_name)
    total_rows = len(rows)
    if row_limit is not None and row_limit > 0:
        rows = rows[:row_limit]

    columns = list(rows[0].keys()) if rows else []
    config = _load_config()

    return {
        "status": "ok",
        "dataset": _to_manifest_entry(file_path, config),
        "columns": columns,
        "rows": rows,
        "rowCount": len(rows),
        "totalRowCount": total_rows,
        "dataPath": str(file_path),
    }

