"""Dataset manifest and row-loading helpers for the data scientist agent."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import openpyxl
import xlrd


DatasetRowsLoader = Callable[[Path], list[dict[str, Any]]]


def _clean_header(value: Any) -> str:
    """Normalize spreadsheet and CSV header cells."""
    return str(value).replace("\ufeff", "").strip()


def _row_mapping(headers: list[str], row_values: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Build a row dictionary while skipping blank header cells."""
    return {
        headers[index]: value
        for index, value in enumerate(row_values)
        if index < len(headers) and headers[index]
    }


def load_manifest(sample_data_dir: Path) -> list[dict[str, Any]]:
    """Load dataset manifest entries from the sample data directory."""
    manifest_path = sample_data_dir / "datasets.json"
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_dataset_entry(dataset_id: str, manifest: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve a dataset manifest entry by ID."""
    for entry in manifest:
        if entry.get("id") == dataset_id:
            return entry
    return None


def detect_delimiter(handle: Any) -> str:
    """Detect a likely delimiter from a text handle."""
    sample = handle.read(2048)
    handle.seek(0)
    for delimiter in [",", ";", "\t"]:
        if sample.count(delimiter) > 0:
            return delimiter
    return ","


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows into dictionaries with normalized headers."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter=detect_delimiter(handle))
        rows = []
        for row in reader:
            rows.append({_clean_header(key): value for key, value in row.items()})
        return rows


def read_xlsx_rows(path: Path) -> list[dict[str, Any]]:
    """Read XLSX rows from the first worksheet."""
    workbook = openpyxl.load_workbook(path, read_only=True)
    sheet = workbook.active
    rows_iter = sheet.iter_rows(values_only=True)
    headers = next(rows_iter, None)
    if not headers:
        return []
    cleaned_headers = [_clean_header(header) for header in headers]
    rows = []
    for row in rows_iter:
        rows.append(_row_mapping(cleaned_headers, row))
    return rows


def read_xls_rows(path: Path) -> list[dict[str, Any]]:
    """Read XLS rows from the first worksheet."""
    workbook = xlrd.open_workbook(path.as_posix())
    sheet = workbook.sheet_by_index(0)
    headers = [_clean_header(value) for value in sheet.row_values(0)]
    rows = []
    for row_index in range(1, sheet.nrows):
        row_values = sheet.row_values(row_index)
        rows.append(_row_mapping(headers, row_values))
    return rows


DATASET_ROW_LOADERS: dict[str, DatasetRowsLoader] = {
    ".csv": read_csv_rows,
    ".xlsx": read_xlsx_rows,
    ".xls": read_xls_rows,
}


def load_dataset_rows(
    file_path: Path,
    row_loaders: dict[str, DatasetRowsLoader] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch row loading by file suffix."""
    loaders = row_loaders or DATASET_ROW_LOADERS
    loader = loaders.get(file_path.suffix.lower())
    if loader is None:
        raise ValueError("Unsupported file format.")
    return loader(file_path)
