from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .contracts import parse_string_list_field, resolve_data_path_from_payload


def resolve_data_target(
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
) -> tuple[Path | None, str | None, tuple[int, dict[str, Any]] | None]:
    data_path, dataset_lookup_missing = resolve_data_path_from_payload(
        data_path=payload.get("data_path"),
        dataset_id=payload.get("dataset_id"),
        resolve_dataset_path=resolve_dataset_path,
    )
    if dataset_lookup_missing:
        return None, None, (404, {"status": "error", "error": "Dataset not found in ai/ml/data."})

    target_column = payload.get("target_column")
    if not data_path or not target_column:
        return None, None, (
            400,
            {
                "status": "error",
                "error": "target_column and either data_path or dataset_id are required.",
            },
        )
    return data_path, str(target_column), None


def parse_feature_columns(payload: dict[str, Any]) -> tuple[list[str], list[str], str | None]:
    try:
        exclude_columns = parse_string_list_field(payload.get("exclude_columns", []), "exclude_columns")
        date_columns = parse_string_list_field(payload.get("date_columns", []), "date_columns")
    except ValueError as exc:
        return [], [], str(exc)
    return exclude_columns, date_columns, None


def resolve_teacher_model_path(
    payload: dict[str, Any],
    artifacts_dir: Path,
    model_filename: str,
) -> str | None:
    teacher_model_path = payload.get("teacher_model_path")
    teacher_model_id = payload.get("teacher_model_id")
    if not teacher_model_path and teacher_model_id:
        teacher_model_path = str(artifacts_dir / str(teacher_model_id) / model_filename)
    return str(teacher_model_path) if teacher_model_path else None
