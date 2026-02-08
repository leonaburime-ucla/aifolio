"""
Crypto dataset utilities for listing and fetching local data files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "crypto-data"


def list_datasets() -> List[str]:
    """
    Return dataset IDs (filenames without extension) for json/csv files.
    """
    if not DATA_DIR.exists():
        return []
    return sorted(
        [
            path.stem
            for path in DATA_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".csv"}
        ]
    )


def load_dataset(dataset_id: str) -> Tuple[bool, str]:
    """
    Load dataset content by ID, returning (found, content).
    """
    if not DATA_DIR.exists():
        return False, ""

    safe_name = Path(dataset_id).name
    json_path = DATA_DIR / f"{safe_name}.json"
    csv_path = DATA_DIR / f"{safe_name}.csv"

    if json_path.exists():
        return True, json_path.read_text(encoding="utf-8")
    if csv_path.exists():
        return True, csv_path.read_text(encoding="utf-8")
    return False, ""
