from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import openpyxl

try:
    import xlrd  # type: ignore
except Exception:  # pragma: no cover
    xlrd = None


def _sniff_csv_dialect(sample: str) -> csv.Dialect:
    """Infer the CSV dialect from a sample, falling back to excel defaults."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _normalize_header_row(header: Any) -> list[str]:
    """Convert a worksheet header row into stripped column names."""
    return [str(value).strip() if value is not None else "" for value in header]


def _looks_like_generated_header(columns: list[str]) -> bool:
    """Detect synthetic X1..Xn/Y headers emitted by some exported datasets."""
    non_empty = [col for col in columns if col]
    return bool(non_empty) and all(re.fullmatch(r"X\d+", col) or col == "Y" for col in non_empty)


def _row_to_entry(columns: list[str], row: Any) -> dict[str, Any]:
    """Map a worksheet row to a dict while skipping blank column headers."""
    entry: dict[str, Any] = {}
    for index, column in enumerate(columns):
        if not column:
            continue
        entry[column] = row[index] if index < len(row) else None
    return entry


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = _sniff_csv_dialect(sample)
        reader = csv.DictReader(handle, dialect=dialect)
        return [dict(row) for row in reader]


def _read_xlsx(path: Path, sheet_name: str | None = None) -> list[dict[str, Any]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = workbook[sheet_name] if sheet_name else workbook.active

    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        return []

    columns = _normalize_header_row(header)
    start_from_row_index = 1
    if _looks_like_generated_header(columns):
        next_header = next(rows, None)
        if next_header is not None:
            columns = _normalize_header_row(next_header)
            start_from_row_index = 2

    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=start_from_row_index):
        _ = row_index  # keeps parity with xls logic if future debugging needs row numbers
        out.append(_row_to_entry(columns, row))
    return out


def _read_xls(path: Path, sheet_name: str | None = None) -> list[dict[str, Any]]:
    if xlrd is None:
        raise RuntimeError("xlrd is required to load .xls files")

    workbook = xlrd.open_workbook(path.as_posix())
    sheet = workbook.sheet_by_name(sheet_name) if sheet_name else workbook.sheet_by_index(0)
    if sheet.nrows == 0:
        return []

    columns = _normalize_header_row([sheet.cell_value(0, c) for c in range(sheet.ncols)])
    start_row = 1
    if _looks_like_generated_header(columns):
        if sheet.nrows > 1:
            columns = _normalize_header_row([sheet.cell_value(1, c) for c in range(sheet.ncols)])
            start_row = 2

    out: list[dict[str, Any]] = []
    for r in range(start_row, sheet.nrows):
        out.append(_row_to_entry(columns, [sheet.cell_value(r, c) for c in range(sheet.ncols)]))
    return out


def load_tabular_file(path: str | Path, sheet_name: str | None = None) -> list[dict[str, Any]]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return _read_csv(file_path)
    if suffix == ".xlsx":
        return _read_xlsx(file_path, sheet_name=sheet_name)
    if suffix == ".xls":
        return _read_xls(file_path, sheet_name=sheet_name)

    raise ValueError(f"Unsupported file extension: {suffix}. Expected .csv, .xls, or .xlsx")


# Common representations of missing data across CSV/Excel files.
# These are converted to NaN so they can be properly imputed later.
# Without this, "NA" would be treated as a categorical string value,
# creating garbage one-hot features in DictVectorizer.
_MISSING_VALUES = {"", "na", "n/a", "nan", "none", "null", "."}


def coerce_value(value: Any) -> Any:
    """Convert a cell value to a numeric type if possible, or NaN for missing values.

    Returns:
        - float("nan") for None or missing value strings (NA, N/A, etc.)
        - Original int/float for numeric values
        - float for string numbers ("123.45" -> 123.45)
        - Original string for non-numeric text (categorical values)
    """
    if value is None:
        return float("nan")
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if text.lower() in _MISSING_VALUES:
        return float("nan")

    try:
        number = float(text)
        return number
    except ValueError:
        # Non-numeric string = categorical value, keep as-is for one-hot encoding
        return text


def split_features_target(
    rows: list[dict[str, Any]],
    target_column: str,
) -> tuple[list[dict[str, Any]], list[Any]]:
    if not rows:
        raise ValueError("Dataset is empty")
    if not any(target_column in row for row in rows):
        raise ValueError(f"Target column '{target_column}' not found")

    x_rows: list[dict[str, Any]] = []
    y_raw: list[Any] = []

    for row in rows:
        if target_column not in row:
            continue
        y_raw.append(row[target_column])
        x_rows.append({k: coerce_value(v) for k, v in row.items() if k != target_column})

    return x_rows, y_raw
