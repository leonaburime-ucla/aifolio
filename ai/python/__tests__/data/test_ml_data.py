import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import backend.ml.data as ml_data


def test_load_config_handles_missing_and_invalid_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(ml_data, "ML_DATA_DIR", tmp_path / "missing")
    assert ml_data._load_config() == {}

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "sources.json").write_text(json.dumps(["bad"]), encoding="utf-8")
    monkeypatch.setattr(ml_data, "ML_DATA_DIR", tmp_path)
    assert ml_data._load_config() == {}


def test_list_ml_datasets_uses_config_metadata(tmp_path, monkeypatch):
    data_file = tmp_path / "sample.csv"
    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "sources.json").write_text(
        json.dumps(
            {
                "sample.csv": {
                    "targetColumn": "b",
                    "task": "regression",
                    "sources": ["x"],
                    "preprocessing": {"impute": "median"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ml_data, "ML_DATA_DIR", tmp_path)

    entries = ml_data.list_ml_datasets()
    assert len(entries) == 1
    assert entries[0]["targetColumn"] == "b"
    assert entries[0]["task"] == "regression"
    assert entries[0]["preprocessing"] == {"impute": "median"}


def test_resolve_ml_dataset_path_supports_exact_name_and_unique_stem(tmp_path, monkeypatch):
    first = tmp_path / "sample.csv"
    second = tmp_path / "other.xlsx"
    first.write_text("a,b\n1,2\n", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    monkeypatch.setattr(ml_data, "ML_DATA_DIR", tmp_path)

    assert ml_data.resolve_ml_dataset_path("sample.csv") == first
    assert ml_data.resolve_ml_dataset_path("sample") == first
    assert ml_data.resolve_ml_dataset_path("missing") is None


def test_load_ml_dataset_returns_manifest_rows_and_row_limit(tmp_path, monkeypatch):
    data_file = tmp_path / "sample.csv"
    data_file.write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "sources.json").write_text(json.dumps({"sample.csv": {"sources": ["x"]}}), encoding="utf-8")
    monkeypatch.setattr(ml_data, "ML_DATA_DIR", tmp_path)
    monkeypatch.setattr(
        ml_data,
        "load_tabular_file",
        lambda file_path, sheet_name=None: [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
    )

    payload = ml_data.load_ml_dataset("sample", row_limit=1, sheet_name="Sheet1")
    assert payload["status"] == "ok"
    assert payload["columns"] == ["a", "b"]
    assert payload["rowCount"] == 1
    assert payload["totalRowCount"] == 2
    assert payload["dataset"]["files"] == {"data": "sample.csv"}
    assert payload["dataPath"].endswith("sample.csv")


def test_load_ml_dataset_returns_not_found_for_unknown_dataset(monkeypatch):
    monkeypatch.setattr(ml_data, "resolve_ml_dataset_path", lambda _id: None)
    payload = ml_data.load_ml_dataset("missing")
    assert payload["status"] == "error"
