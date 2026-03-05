import math
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import pytest

from shared.tools import sklearn_tools


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


def test_list_available_tools_returns_non_empty_list():
    tools = sklearn_tools.list_available_tools()
    assert isinstance(tools, list)
    assert "linear_regression" in tools


def test_get_tools_schema_contains_expected_entries():
    schema = sklearn_tools.get_tools_schema()
    names = {entry.get("name") for entry in schema}
    assert "pca_transform" in names
    assert "logistic_regression" in names


@pytest.mark.parametrize(
    ("tool_name", "payload", "expected_key"),
    [
        ("standard_scaler", {"data": [[1.0, 2.0], [3.0, 4.0]]}, "scaled"),
        ("minmax_scaler", {"data": [[1.0, 2.0], [3.0, 4.0]]}, "scaled"),
        ("robust_scaler", {"data": [[1.0, 2.0], [3.0, 4.0], [100.0, 4.0]]}, "scaled"),
        ("one_hot_encoder", {"data": [["red"], ["blue"], ["red"]]}, "encoded"),
        ("polynomial_features", {"data": [[1.0, 2.0], [3.0, 4.0]], "degree": 2}, "transformed"),
        ("power_transformer", {"data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]}, "transformed"),
        (
            "quantile_transformer",
            {"data": [[1.0], [2.0], [3.0]], "n_quantiles": 50, "output_distribution": "normal"},
            "transformed",
        ),
        ("simple_imputer", {"data": [[1.0, float("nan")], [2.0, 3.0]]}, "transformed"),
    ],
)
def test_preprocessing_tools_return_expected_payload_shapes(tool_name, payload, expected_key):
    result = getattr(sklearn_tools, tool_name)(**payload)
    assert expected_key in result
    assert len(result[expected_key]) > 0


def test_label_encoder_returns_classes_and_encoded_values():
    result = sklearn_tools.label_encoder(labels=["cat", "dog", "cat"])
    assert result["classes"] == ["cat", "dog"]
    assert result["encoded"] == [0, 1, 0]


def test_quantile_transformer_clamps_quantiles_to_sample_count():
    result = sklearn_tools.quantile_transformer(data=[[1.0], [2.0], [3.0]], n_quantiles=10)
    assert result["n_quantiles"] == 3


def test_polynomial_features_expands_feature_count():
    result = sklearn_tools.polynomial_features(data=[[1.0, 2.0], [3.0, 4.0]], degree=2)
    assert len(result["transformed"][0]) == 5


def test_select_k_best_handles_chi2_by_shifting_negative_values():
    x, y, _ = _classification_data()
    result = sklearn_tools.select_k_best(data=x, target=y, k=2, score_func="chi2")
    assert result["k"] == 2
    assert result["data_shifted"] is True
    assert len(result["transformed"][0]) == 2


def test_select_from_model_returns_transformed_rows():
    x, y, _ = _regression_data()
    result = sklearn_tools.select_from_model(data=x, target=y, estimator="random_forest", task="regression")
    assert len(result["transformed"]) == len(x)
    assert result["task"] == "regression"


def test_rfe_returns_support_and_ranking():
    x, y, _ = _regression_data()
    result = sklearn_tools.rfe(data=x, target=y, estimator="linear", task="regression", n_features_to_select=2)
    assert len(result["support"]) == len(x[0])
    assert sum(result["support"]) == 2
    assert len(result["ranking"]) == len(x[0])


def test_rfecv_returns_cv_scores():
    x, y, _ = _regression_data()
    result = sklearn_tools.rfecv(data=x, target=y, estimator="linear", task="regression", cv=2)
    assert len(result["support"]) == len(x[0])
    assert result["n_features"] >= 1
    assert len(result["cv_scores"]) >= 1


def test_grid_search_cv_returns_best_params_and_score():
    x, y, _ = _regression_data()
    result = sklearn_tools.grid_search_cv(
        data=x,
        target=y,
        estimator="ridge",
        task="regression",
        param_grid={"alpha": [0.1, 1.0]},
        cv=2,
    )
    assert "alpha" in result["best_params"]
    assert math.isfinite(float(result["best_score"]))


def test_randomized_search_cv_returns_best_params_and_score():
    x, y, _ = _regression_data()
    result = sklearn_tools.randomized_search_cv(
        data=x,
        target=y,
        estimator="ridge",
        task="regression",
        param_distributions={"alpha": [0.1, 1.0], "fit_intercept": [True, False]},
        n_iter=2,
        cv=2,
    )
    assert "alpha" in result["best_params"]
    assert math.isfinite(float(result["best_score"]))


def test_cross_val_score_returns_scores_and_mean():
    x, y, _ = _regression_data()
    result = sklearn_tools.cross_val_score(data=x, target=y, estimator="ridge", task="regression", cv=2)
    assert len(result["scores"]) == 2
    assert math.isfinite(float(result["mean_score"]))


@pytest.mark.parametrize(
    ("tool_name", "payload", "expected_key"),
    [
        ("r2_score", {"y_true": [1.0, 2.0], "y_pred": [1.0, 2.0]}, "r2_score"),
        ("mean_squared_error", {"y_true": [1.0, 2.0], "y_pred": [1.5, 2.5]}, "mean_squared_error"),
        ("mean_absolute_error", {"y_true": [1.0, 2.0], "y_pred": [1.5, 2.5]}, "mean_absolute_error"),
        ("accuracy_score", {"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}, "accuracy_score"),
        ("f1_score", {"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}, "f1_score"),
        ("precision_score", {"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}, "precision_score"),
        ("recall_score", {"y_true": [0, 1, 1], "y_pred": [0, 1, 0]}, "recall_score"),
        ("auc_score", {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "auc"),
    ],
)
def test_metric_tools_return_numeric_values(tool_name, payload, expected_key):
    result = getattr(sklearn_tools, tool_name)(**payload)
    assert expected_key in result
    assert math.isfinite(float(result[expected_key]))


def test_roc_curve_returns_curve_arrays():
    result = sklearn_tools.roc_curve(y_true=[0, 0, 1, 1], y_score=[0.1, 0.4, 0.35, 0.8])
    assert len(result["fpr"]) == len(result["tpr"]) == len(result["thresholds"])


def test_precision_recall_curve_returns_curve_arrays():
    result = sklearn_tools.precision_recall_curve(y_true=[0, 0, 1, 1], y_score=[0.1, 0.4, 0.35, 0.8])
    assert len(result["precision"]) == len(result["recall"])
    assert len(result["thresholds"]) == len(result["precision"]) - 1


@pytest.mark.parametrize(
    ("tool_name", "payload", "expected_message"),
    [
        ("r2_score", {"y_true": [1.0, 2.0]}, "r2_score requires y_true, y_pred."),
        ("roc_curve", {"y_true": [0, 1]}, "roc_curve requires y_true, y_score."),
        ("auc_score", {"x": [0.0, 1.0]}, "auc_score requires x, y."),
    ],
)
def test_metric_tools_reject_missing_required_inputs(tool_name, payload, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        getattr(sklearn_tools, tool_name)(**payload)
