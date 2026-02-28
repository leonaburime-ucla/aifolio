import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from ml_data import list_ml_datasets
from langgraph_agents.data_scientist import _build_numeric_matrix


def test_ml_manifest_exposes_preprocessing_config():
    datasets = {entry["id"]: entry for entry in list_ml_datasets()}

    assert datasets["customer_churn_telco.csv"]["preprocessing"]["idColumns"] == ["customerID"]
    assert datasets["house_prices_ames.csv"]["preprocessing"]["idColumns"] == ["Order", "PID"]
    assert datasets["loan_default_credit_card_clients.xls"]["preprocessing"]["idColumns"] == ["ID"]
    assert datasets["sales_forecasting_walmart.csv"]["preprocessing"]["dateColumns"] == ["Date"]


def test_build_numeric_matrix_drops_configured_numeric_id_and_expands_dates():
    rows = [
        {"ID": 1001, "Date": "2024-01-15", "Feature": 1.5, "Target": 0},
        {"ID": 1002, "Date": "2024-02-20", "Feature": 2.5, "Target": 1},
        {"ID": 1003, "Date": "2024-03-10", "Feature": 3.5, "Target": 0},
    ]
    matrix, targets, feature_names = _build_numeric_matrix(
        rows,
        target_column="Target",
        preprocessing={"idColumns": ["ID"], "dateColumns": ["Date"]},
    )

    assert len(matrix) == 3
    assert len(targets) == 3
    assert "ID" not in feature_names
    assert "Feature" in feature_names
    assert "Date_year" in feature_names
    assert "Date_month" in feature_names
    assert "Date_day" in feature_names

