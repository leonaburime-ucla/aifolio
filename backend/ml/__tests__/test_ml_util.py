import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.ml_util import batch_indices, expand_date_columns, infer_task, parse_date_like


def test_infer_task_detects_classification_for_non_numeric_labels():
    assert infer_task(["a", "b", "a"]) == "classification"


def test_batch_indices_covers_all_rows():
    batches = batch_indices(10, 4)
    flattened = sorted(int(idx) for batch in batches for idx in batch)
    assert flattened == list(range(10))


def test_parse_date_like_handles_iso_and_invalid_values():
    assert parse_date_like("2024-01-02") is not None
    assert parse_date_like("not-a-date") is None


def test_expand_date_columns_adds_engineered_features():
    rows = [{"ts": "2024-01-02", "v": 1}]
    out = expand_date_columns(rows, date_columns=["ts"])
    assert "ts__year" in out[0]
    assert "ts__sin_month" in out[0]
