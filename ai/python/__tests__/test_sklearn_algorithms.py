import math
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import pytest

from tools import sklearn_tools


def _regression_data():
    x = [[float(i), float(i % 3), float(i * 0.5), float((i % 5) - 2)] for i in range(1, 31)]
    y = [2.5 * row[0] - 1.2 * row[2] + 0.8 * row[1] + 3.0 for row in x]
    feature_names = ["f1", "f2", "f3", "f4"]
    return x, y, feature_names


def _classification_data():
    x = [[float(i), float(i % 4), float((i % 6) - 3)] for i in range(1, 41)]
    y = [1 if (row[0] + 0.5 * row[1] - row[2]) > 15 else 0 for row in x]
    feature_names = ["c1", "c2", "c3"]
    return x, y, feature_names


def _cluster_data():
    cluster_a = [[1.0 + 0.05 * i, 1.0 + 0.04 * i] for i in range(10)]
    cluster_b = [[5.0 + 0.03 * i, 5.0 + 0.02 * i] for i in range(10)]
    cluster_c = [[9.0 + 0.04 * i, 1.0 + 0.03 * i] for i in range(10)]
    return cluster_a + cluster_b + cluster_c


@pytest.mark.parametrize(
    ("tool_name", "tool_kwargs"),
    [
        ("pca_transform", {"n_components": 2}),
        ("incremental_pca", {"n_components": 2, "batch_size": 8}),
        ("truncated_svd", {"n_components": 2}),
        ("fast_ica", {"n_components": 2, "random_state": 42}),
        ("nmf_decomposition", {"n_components": 2, "random_state": 42}),
        ("tsne_embedding", {"n_components": 2, "perplexity": 8.0, "random_state": 42}),
    ],
)
def test_decomposition_algorithms_return_expected_shapes(tool_name, tool_kwargs):
    x, _, feature_names = _regression_data()
    fn = getattr(sklearn_tools, tool_name)
    payload = {"data": x, **tool_kwargs}
    if tool_name in {"pca_transform", "incremental_pca", "truncated_svd"}:
        payload["feature_names"] = feature_names
    result = fn(**payload)

    if tool_name == "tsne_embedding":
        embedding = result["embedding"]
        assert len(embedding) == len(x)
        assert len(embedding[0]) == 2
        return

    transformed = result["transformed"]
    assert len(transformed) == len(x)
    assert len(transformed[0]) == 2
    assert "components" in result
    assert len(result["components"]) == 2

    if tool_name == "nmf_decomposition":
        assert result["n_components"] == 2
        assert "reconstruction_err" in result
        assert result["data_shifted"] in {True, False}
    if tool_name in {"pca_transform", "incremental_pca", "truncated_svd"}:
        assert "explained_variance_ratio" in result
        assert len(result["explained_variance_ratio"]) == 2


@pytest.mark.parametrize(
    "tool_name",
    [
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elasticnet_regression",
        "svr_regression",
        "random_forest_regression",
        "gradient_boosting_regression",
        "pls_regression",
    ],
)
def test_regression_algorithms_run_end_to_end(tool_name):
    x, y, feature_names = _regression_data()
    fn = getattr(sklearn_tools, tool_name)

    kwargs = {"data": x, "target": y}
    if tool_name in {
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elasticnet_regression",
        "random_forest_regression",
        "gradient_boosting_regression",
        "pls_regression",
    }:
        kwargs["feature_names"] = feature_names
    if tool_name == "pls_regression":
        kwargs["n_components"] = 2

    result = fn(**kwargs)
    assert "predictions" in result
    assert len(result["predictions"]) == len(y)
    assert "r2_score" in result
    assert math.isfinite(float(result["r2_score"]))


@pytest.mark.parametrize(
    "tool_name",
    [
        "logistic_regression",
        "random_forest_classification",
        "gradient_boosting_classification",
        "knn_classification",
        "naive_bayes_classification",
    ],
)
def test_classification_algorithms_run_end_to_end(tool_name):
    x, y, feature_names = _classification_data()
    fn = getattr(sklearn_tools, tool_name)
    kwargs = {"data": x, "target": y}
    if tool_name in {
        "logistic_regression",
        "random_forest_classification",
        "gradient_boosting_classification",
    }:
        kwargs["feature_names"] = feature_names

    result = fn(**kwargs)
    assert "predictions" in result
    assert len(result["predictions"]) == len(y)
    assert "accuracy" in result
    accuracy = float(result["accuracy"])
    assert 0.0 <= accuracy <= 1.0


@pytest.mark.parametrize(
    ("tool_name", "extra_kwargs"),
    [
        ("kmeans_clustering", {"n_clusters": 3, "random_state": 42}),
        ("minibatch_kmeans_clustering", {"n_clusters": 3, "random_state": 42}),
        ("dbscan_clustering", {"eps": 0.7, "min_samples": 3}),
        ("agglomerative_clustering", {"n_clusters": 3}),
        ("spectral_clustering", {"n_clusters": 3, "random_state": 42}),
        ("gaussian_mixture_clustering", {"n_components": 3, "random_state": 42}),
        ("optics_clustering", {"min_samples": 3}),
    ],
)
def test_clustering_algorithms_return_labels(tool_name, extra_kwargs):
    x = _cluster_data()
    fn = getattr(sklearn_tools, tool_name)
    result = fn(data=x, **extra_kwargs)
    labels = result.get("labels")
    assert isinstance(labels, list)
    assert len(labels) == len(x)
