"""
Test all sklearn_tools functions with synthetic data.
Run: python test_sklearn_tools.py
"""

import numpy as np
from tools import sklearn_tools

# Generate synthetic data
np.random.seed(42)
N_SAMPLES = 100
N_FEATURES = 5

# Continuous features
X = np.random.randn(N_SAMPLES, N_FEATURES).tolist()
X_small = np.random.randn(10, 3).tolist()  # Small dataset for edge cases

# Regression target
y_reg = (np.random.randn(N_SAMPLES) * 10).tolist()

# Classification target (binary)
y_class_binary = np.random.randint(0, 2, N_SAMPLES).tolist()

# Classification target (multiclass)
y_class_multi = np.random.randint(0, 3, N_SAMPLES).tolist()

# Feature names
feature_names = [f"feature_{i}" for i in range(N_FEATURES)]
feature_names_small = [f"feat_{i}" for i in range(3)]

# Categorical data for encoders
categorical_data = [["a", "b"], ["c", "a"], ["b", "c"], ["a", "a"]]
labels = ["cat", "dog", "cat", "bird", "dog"]

# Data with missing values
X_missing = [[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]


def test_func(name, func, *args, **kwargs):
    """Run a test and report result."""
    try:
        result = func(*args, **kwargs)
        # Check for expected keys
        if isinstance(result, dict):
            print(f"  OK: {name} -> keys: {list(result.keys())[:5]}...")
        else:
            print(f"  OK: {name} -> type: {type(result)}")
        return True, result
    except Exception as e:
        print(f"  FAIL: {name} -> {type(e).__name__}: {e}")
        return False, None


def run_all_tests():
    results = {"passed": 0, "failed": 0, "errors": []}

    print("\n=== DECOMPOSITION & EMBEDDINGS ===")

    tests = [
        ("pca_transform", sklearn_tools.pca_transform, X, 3, feature_names),
        ("incremental_pca", sklearn_tools.incremental_pca, X, 3, None, feature_names),
        ("truncated_svd", sklearn_tools.truncated_svd, X, 3, feature_names),
        ("nmf_decomposition", sklearn_tools.nmf_decomposition, X),  # Has negative values - should auto-shift
        ("fast_ica", sklearn_tools.fast_ica, X, 3),
        ("tsne_embedding", sklearn_tools.tsne_embedding, X_small, 2, 5.0),  # Small perplexity for small data
    ]

    for name, func, *args in tests:
        ok, _ = test_func(name, func, *args)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== REGRESSION ===")

    tests = [
        ("linear_regression", sklearn_tools.linear_regression, X, y_reg, feature_names),
        ("ridge_regression", sklearn_tools.ridge_regression, X, y_reg, 1.0, feature_names),
        ("lasso_regression", sklearn_tools.lasso_regression, X, y_reg, 0.1, feature_names),
        ("elasticnet_regression", sklearn_tools.elasticnet_regression, X, y_reg, 0.1, 0.5, feature_names),
        ("svr_regression", sklearn_tools.svr_regression, X, y_reg),
        ("random_forest_regression", sklearn_tools.random_forest_regression, X, y_reg, 10, 42, feature_names),
        ("gradient_boosting_regression", sklearn_tools.gradient_boosting_regression, X, y_reg, 10, 0.1, 42, feature_names),
        ("pls_regression", sklearn_tools.pls_regression, X, y_reg, 2, feature_names),
    ]

    for name, func, *args in tests:
        ok, _ = test_func(name, func, *args)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== CLASSIFICATION ===")

    tests = [
        ("logistic_regression", sklearn_tools.logistic_regression, X, y_class_binary, 1.0, 100, feature_names),
        ("svc_classification", sklearn_tools.svc_classification, X, y_class_binary),
        ("random_forest_classification", sklearn_tools.random_forest_classification, X, y_class_multi, 10, 42, feature_names),
        ("gradient_boosting_classification", sklearn_tools.gradient_boosting_classification, X, y_class_multi, 10, 0.1, 42, feature_names),
        ("knn_classification", sklearn_tools.knn_classification, X, y_class_binary),
        ("naive_bayes_classification", sklearn_tools.naive_bayes_classification, X, y_class_binary),
    ]

    for name, func, *args in tests:
        ok, _ = test_func(name, func, *args)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    # LDA/QDA use kwargs
    print("  Testing lda_classification...")
    ok, _ = test_func("lda_classification", sklearn_tools.lda_classification, data=X, target=y_class_multi)
    results["passed" if ok else "failed"] += 1
    if not ok:
        results["errors"].append("lda_classification")

    print("  Testing qda_classification...")
    ok, _ = test_func("qda_classification", sklearn_tools.qda_classification, data=X, target=y_class_multi)
    results["passed" if ok else "failed"] += 1
    if not ok:
        results["errors"].append("qda_classification")

    print("\n=== CLUSTERING ===")

    tests = [
        ("kmeans_clustering", lambda: sklearn_tools.kmeans_clustering(data=X, n_clusters=3)),
        ("minibatch_kmeans_clustering", lambda: sklearn_tools.minibatch_kmeans_clustering(data=X, n_clusters=3)),
        ("dbscan_clustering", lambda: sklearn_tools.dbscan_clustering(data=X, eps=0.5, min_samples=3)),
        ("agglomerative_clustering", lambda: sklearn_tools.agglomerative_clustering(data=X, n_clusters=3)),
        ("spectral_clustering", lambda: sklearn_tools.spectral_clustering(data=X, n_clusters=3)),
        ("gaussian_mixture_clustering", lambda: sklearn_tools.gaussian_mixture_clustering(data=X, n_components=3)),
        ("optics_clustering", lambda: sklearn_tools.optics_clustering(data=X, min_samples=5)),
    ]

    for name, func in tests:
        ok, _ = test_func(name, func)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== PREPROCESSING ===")

    tests = [
        ("standard_scaler", lambda: sklearn_tools.standard_scaler(data=X)),
        ("standard_scale", lambda: sklearn_tools.standard_scale(data=X)),
        ("minmax_scaler", lambda: sklearn_tools.minmax_scaler(data=X)),
        ("robust_scaler", lambda: sklearn_tools.robust_scaler(data=X)),
        ("one_hot_encoder", lambda: sklearn_tools.one_hot_encoder(data=categorical_data)),
        ("label_encoder", lambda: sklearn_tools.label_encoder(labels=labels)),
        ("power_transformer", lambda: sklearn_tools.power_transformer(data=X)),
        ("quantile_transformer", lambda: sklearn_tools.quantile_transformer(data=X)),
        ("polynomial_features", lambda: sklearn_tools.polynomial_features(data=X_small, degree=2)),
        ("simple_imputer", lambda: sklearn_tools.simple_imputer(data=X_missing)),
    ]

    for name, func in tests:
        ok, _ = test_func(name, func)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== FEATURE SELECTION ===")

    tests = [
        ("select_k_best (f_regression)", lambda: sklearn_tools.select_k_best(data=X, target=y_reg, k=3, score_func="f_regression")),
        ("select_k_best (chi2)", lambda: sklearn_tools.select_k_best(data=X, target=y_class_binary, k=3, score_func="chi2")),
        ("select_from_model", lambda: sklearn_tools.select_from_model(data=X, target=y_reg, estimator="random_forest", task="regression")),
        ("rfe", lambda: sklearn_tools.rfe(data=X, target=y_reg, estimator="linear", task="regression", n_features_to_select=3)),
        ("rfecv", lambda: sklearn_tools.rfecv(data=X, target=y_reg, estimator="linear", task="regression", cv=3)),
    ]

    for name, func in tests:
        ok, _ = test_func(name, func)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== MODEL SELECTION ===")

    tests = [
        ("train_test_split", lambda: sklearn_tools.train_test_split(data=X, target=y_reg, test_size=0.2)),
        ("cross_val_score", lambda: sklearn_tools.cross_val_score(data=X, target=y_reg, estimator="ridge", task="regression", cv=3)),
        ("grid_search_cv", lambda: sklearn_tools.grid_search_cv(data=X, target=y_reg, estimator="ridge", task="regression", param_grid={"alpha": [0.1, 1.0]}, cv=3)),
        ("randomized_search_cv", lambda: sklearn_tools.randomized_search_cv(data=X, target=y_reg, estimator="ridge", task="regression", param_distributions={"alpha": [0.1, 0.5, 1.0]}, n_iter=2, cv=3)),
    ]

    for name, func in tests:
        ok, _ = test_func(name, func)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    print("\n=== METRICS ===")

    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    y_score = [0.1, 0.9, 0.4, 0.2, 0.8]

    tests = [
        ("r2_score", lambda: sklearn_tools.r2_score(y_true=y_reg[:10], y_pred=y_reg[10:20])),
        ("mean_squared_error", lambda: sklearn_tools.mean_squared_error(y_true=y_reg[:10], y_pred=y_reg[10:20])),
        ("mean_absolute_error", lambda: sklearn_tools.mean_absolute_error(y_true=y_reg[:10], y_pred=y_reg[10:20])),
        ("accuracy_score", lambda: sklearn_tools.accuracy_score(y_true=y_true, y_pred=y_pred)),
        ("f1_score", lambda: sklearn_tools.f1_score(y_true=y_true, y_pred=y_pred)),
        ("precision_score", lambda: sklearn_tools.precision_score(y_true=y_true, y_pred=y_pred)),
        ("recall_score", lambda: sklearn_tools.recall_score(y_true=y_true, y_pred=y_pred)),
        ("roc_curve", lambda: sklearn_tools.roc_curve(y_true=y_true, y_score=y_score)),
        ("precision_recall_curve", lambda: sklearn_tools.precision_recall_curve(y_true=y_true, y_score=y_score)),
    ]

    for name, func in tests:
        ok, _ = test_func(name, func)
        if ok:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(name)

    # auc_score needs fpr/tpr from roc_curve
    print("  Testing auc_score...")
    roc_result = sklearn_tools.roc_curve(y_true=y_true, y_score=y_score)
    ok, _ = test_func("auc_score", sklearn_tools.auc_score, x=roc_result["fpr"], y=roc_result["tpr"])
    results["passed" if ok else "failed"] += 1
    if not ok:
        results["errors"].append("auc_score")

    print("\n" + "=" * 50)
    print(f"RESULTS: {results['passed']} passed, {results['failed']} failed")
    if results["errors"]:
        print(f"FAILED: {', '.join(results['errors'])}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    run_all_tests()
