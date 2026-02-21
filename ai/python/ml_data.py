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


def _load_sources() -> dict[str, list[str]]:
    sources_path = ML_DATA_DIR / "sources.json"
    if not sources_path.exists():
        return {}
    with sources_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [str(v) for v in value]
    return out


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


def _to_manifest_entry(path: Path, sources: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "id": path.name,
        "label": path.name,
        "format": path.suffix.lower().lstrip("."),
        "sizeBytes": path.stat().st_size,
        "files": {"data": path.name},
        "sources": sources.get(path.name, []),
    }


def list_ml_datasets() -> list[dict[str, Any]]:
    sources = _load_sources()
    return [_to_manifest_entry(path, sources) for path in _dataset_files()]


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
    sources = _load_sources()

    return {
        "status": "ok",
        "dataset": _to_manifest_entry(file_path, sources),
        "columns": columns,
        "rows": rows,
        "rowCount": len(rows),
        "totalRowCount": total_rows,
        "dataPath": str(file_path),
    }
