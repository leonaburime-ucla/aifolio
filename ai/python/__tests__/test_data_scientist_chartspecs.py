import sys
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from langgraph_agents.data_scientist import run_data_scientist_analysis


ALGORITHMS_WITH_DATASET = [
    # Decomposition & embeddings
    ("pca_transform", "sales_forecasting_walmart.csv"),
    ("incremental_pca", "sales_forecasting_walmart.csv"),
    ("truncated_svd", "sales_forecasting_walmart.csv"),
    ("nmf_decomposition", "sales_forecasting_walmart.csv"),
    ("fast_ica", "sales_forecasting_walmart.csv"),
    ("tsne_embedding", "sales_forecasting_walmart.csv"),
    # Classification
    ("random_forest_classification", "fraud_detection_phishing_websites.csv"),
    ("gradient_boosting_classification", "fraud_detection_phishing_websites.csv"),
    ("knn_classification", "fraud_detection_phishing_websites.csv"),
    ("naive_bayes_classification", "fraud_detection_phishing_websites.csv"),
    # Clustering
    ("kmeans_clustering", "sales_forecasting_walmart.csv"),
    ("minibatch_kmeans_clustering", "sales_forecasting_walmart.csv"),
    ("dbscan_clustering", "sales_forecasting_walmart.csv"),
    ("agglomerative_clustering", "sales_forecasting_walmart.csv"),
    ("gaussian_mixture_clustering", "sales_forecasting_walmart.csv"),
    # Regression (+ logistic per current orchestrator wiring)
    ("linear_regression", "sales_forecasting_walmart.csv"),
    ("ridge_regression", "sales_forecasting_walmart.csv"),
    ("lasso_regression", "sales_forecasting_walmart.csv"),
    ("elasticnet_regression", "sales_forecasting_walmart.csv"),
    ("svr_regression", "sales_forecasting_walmart.csv"),
    ("random_forest_regression", "sales_forecasting_walmart.csv"),
    ("gradient_boosting_regression", "sales_forecasting_walmart.csv"),
    ("pls_regression", "sales_forecasting_walmart.csv"),
    ("logistic_regression", "fraud_detection_phishing_websites.csv"),
]


@pytest.mark.parametrize(("tool_name", "dataset_id"), ALGORITHMS_WITH_DATASET)
def test_data_scientist_tool_returns_chartspec(tool_name: str, dataset_id: str):
    result = run_data_scientist_analysis(
        message=f"Run {tool_name}",
        dataset_id=dataset_id,
        planned_tool_calls=[{"tool_name": tool_name, "tool_args": {}, "chart_kind": "none"}],
        row_limit=120,
    )

    chart_spec = result.get("chartSpec")
    assert isinstance(chart_spec, list), f"{tool_name}: expected chartSpec list"
    assert len(chart_spec) > 0, f"{tool_name}: expected at least one chart"

    for chart in chart_spec:
        assert isinstance(chart, dict), f"{tool_name}: chart must be object"
        assert chart.get("type"), f"{tool_name}: missing chart type"
        assert chart.get("xKey"), f"{tool_name}: missing xKey"
        assert isinstance(chart.get("yKeys"), list) and chart.get("yKeys"), f"{tool_name}: missing yKeys"
        assert isinstance(chart.get("data"), list), f"{tool_name}: missing data array"

