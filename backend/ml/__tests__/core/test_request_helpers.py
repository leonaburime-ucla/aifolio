import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.request_helpers import parse_feature_columns, resolve_data_target, resolve_teacher_model_path


def test_resolve_data_target_returns_404_for_missing_dataset():
    _, _, err = resolve_data_target(
        payload={"dataset_id": "missing", "target_column": "y"},
        resolve_dataset_path=lambda _id: None,
    )
    assert err is not None
    assert err[0] == 404


def test_parse_feature_columns_rejects_invalid_types():
    excludes, dates, error = parse_feature_columns({"exclude_columns": {"bad": 1}})
    assert excludes == []
    assert dates == []
    assert error is not None


def test_resolve_teacher_model_path_prefers_explicit_path_then_model_id(tmp_path):
    explicit = resolve_teacher_model_path({"teacher_model_path": "/tmp/model.pt"}, tmp_path, "x.pt")
    assert explicit == "/tmp/model.pt"

    resolved = resolve_teacher_model_path({"teacher_model_id": "abc"}, tmp_path, "bundle.pt")
    assert resolved == str(tmp_path / "abc" / "bundle.pt")
