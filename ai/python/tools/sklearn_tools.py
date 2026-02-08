"""
Shared scikit-learn utilities for agent tooling.
Intentionally minimal so agents can import and extend as needed.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    OPTICS,
    SpectralClustering,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, FastICA, IncrementalPCA, NMF, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    chi2,
    f_regression,
    mutual_info_classif,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score as _accuracy_score,
    auc as _auc,
    f1_score as _f1_score,
    mean_absolute_error as _mean_absolute_error,
    mean_squared_error as _mean_squared_error,
    precision_recall_curve as _precision_recall_curve,
    precision_score as _precision_score,
    r2_score as _r2_score,
    recall_score as _recall_score,
    roc_curve as _roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score as _cross_val_score,
    train_test_split as _train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, SVR


def list_available_tools() -> List[str]:
    """
    Return the names of sklearn tool hooks exposed by this module.
    """
    return [
        "pca_transform",
        "train_test_split",
        "standard_scale",
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "elasticnet_regression",
        "svr_regression",
        "random_forest_regression",
        "gradient_boosting_regression",
        "pls_regression",
        "logistic_regression",
        # "svc_classification",  # Commented out: slow on large datasets
        "random_forest_classification",
        "gradient_boosting_classification",
        "knn_classification",
        "naive_bayes_classification",
        # "lda_classification",  # Commented out: can be slow
        # "qda_classification",  # Commented out: can be slow
        "kmeans_clustering",
        "minibatch_kmeans_clustering",
        "dbscan_clustering",
        "agglomerative_clustering",
        # "spectral_clustering",  # Commented out: slow on large datasets
        "gaussian_mixture_clustering",
        # "optics_clustering",  # Commented out: slow on large datasets
        "incremental_pca",
        "truncated_svd",
        "nmf_decomposition",
        "fast_ica",
        "tsne_embedding",
        "standard_scaler",
        "minmax_scaler",
        "robust_scaler",
        "one_hot_encoder",
        "label_encoder",
        "polynomial_features",
        "power_transformer",
        "quantile_transformer",
        "simple_imputer",
        "select_k_best",
        "select_from_model",
        "rfe",
        "rfecv",
        "grid_search_cv",
        "randomized_search_cv",
        "cross_val_score",
        "r2_score",
        "mean_squared_error",
        "mean_absolute_error",
        "accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score",
        "roc_curve",
        "auc_score",
        "precision_recall_curve",
    ]


def get_tools_schema() -> List[Dict[str, Any]]:
    """
    Build a lightweight schema catalog for all available tools.
    """
    tools = []
    for name in list_available_tools():
        fn = getattr(__import__(__name__), name, None)
        if fn is None:
            continue
        signature = inspect.signature(fn)
        params = []
        for param in signature.parameters.values():
            if param.name in {"args", "kwargs"}:
                continue
            annotation = (
                str(param.annotation)
                if param.annotation is not inspect._empty
                else "Any"
            )
            default = (
                None if param.default is inspect._empty else param.default
            )
            params.append(
                {
                    "name": param.name,
                    "type": annotation,
                    "default": default,
                }
            )
        tools.append(
            {
                "name": name,
                "params": params,
                "doc": (fn.__doc__ or "").strip(),
            }
        )
    return tools


def pca_transform(
    data: Sequence[Sequence[float]],
    n_components: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run PCA on numeric data.

    Args:
        data: List of rows, each row is a list of numeric features.
        n_components: Number of principal components to keep.
        feature_names: Optional names for original features.

    Returns:
        Dict with transformed points, components, and explained variance.
    """
    if not data:
        raise ValueError("pca_transform requires non-empty data.")

    matrix = np.array(data, dtype=float)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(matrix)

    feature_importance = _pca_feature_importance(
        pca.components_, pca.explained_variance_ratio_, feature_names
    )
    return {
        "transformed": transformed.tolist(),
        "components": pca.components_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_names": feature_names,
        "feature_importance": feature_importance,
    }


def _validate_xy(
    data: Sequence[Sequence[float]],
    target: Sequence[float] | Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    if not data:
        raise ValueError("data must be non-empty.")
    if not target:
        raise ValueError("target must be non-empty.")

    matrix = np.array(data, dtype=float)
    y = np.array(target)
    if matrix.shape[0] != y.shape[0]:
        raise ValueError("data and target must have the same number of rows.")
    return matrix, y


def _resolve_feature_names(
    feature_names: Optional[Sequence[str]],
    n_features: int,
) -> Optional[List[str]]:
    if feature_names is None:
        return None
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match data column count.")
    return list(feature_names)


def _validate_x(data: Sequence[Sequence[float]]) -> np.ndarray:
    if not data:
        raise ValueError("data must be non-empty.")
    return np.array(data, dtype=float)


def _estimator_from_name(name: str, task: str):
    if task == "regression":
        if name == "linear":
            return LinearRegression()
        if name == "ridge":
            return Ridge()
        if name == "lasso":
            return Lasso()
        if name == "elasticnet":
            return ElasticNet()
        if name == "svr":
            return SVR()
        if name == "random_forest":
            return RandomForestRegressor()
        if name == "gradient_boosting":
            return GradientBoostingRegressor()
    if task == "classification":
        if name == "logistic":
            return LogisticRegression(max_iter=1000)
        if name == "svc":
            return SVC(probability=True)
        if name == "random_forest":
            return RandomForestClassifier()
        if name == "gradient_boosting":
            return GradientBoostingClassifier()
        if name == "knn":
            return KNeighborsClassifier()
        if name == "naive_bayes":
            return GaussianNB()
    raise ValueError(f"Unsupported estimator: {name} for task {task}.")


def _pca_feature_importance(
    components: np.ndarray,
    explained_variance_ratio: np.ndarray,
    feature_names: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    n_features = components.shape[1]
    names = (
        list(feature_names)
        if feature_names is not None
        else [f"feature_{i}" for i in range(n_features)]
    )
    weighted = np.abs(components).T @ explained_variance_ratio
    total = weighted.sum()
    if total > 0:
        weighted = weighted / total
    ranked = [
        {"feature": name, "importance": float(score)}
        for name, score in zip(names, weighted, strict=False)
    ]
    ranked.sort(key=lambda item: item["importance"], reverse=True)
    return ranked


def train_test_split(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Split data into train/test sets.

    Args:
        data: List of rows.
        target: List of labels/targets.
        test_size: Fraction for test split.
        random_state: Seed.

    Returns:
        Dict with x_train, x_test, y_train, y_test.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    test_size = kwargs.get("test_size", 0.2)
    random_state = kwargs.get("random_state", 42)
    if data is None or target is None:
        raise ValueError("train_test_split requires data and target.")
    x = np.array(data, dtype=float)
    y = np.array(target)
    x_train, x_test, y_train, y_test = _train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return {
        "x_train": x_train.tolist(),
        "x_test": x_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }


def standard_scale(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    data = kwargs.get("data")
    if data is None:
        raise ValueError("standard_scale requires data.")
    matrix = np.array(data, dtype=float)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return {
        "scaled": scaled.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def linear_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    feature_names: Optional[Sequence[str]] = None,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Fit a linear regression model and return coefficients and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_) if fit_intercept else 0.0,
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "feature_names": names,
    }


def ridge_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    alpha: float = 1.0,
    feature_names: Optional[Sequence[str]] = None,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Fit a ridge regression model and return coefficients and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_) if fit_intercept else 0.0,
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "alpha": alpha,
        "feature_names": names,
    }


def lasso_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    alpha: float = 1.0,
    feature_names: Optional[Sequence[str]] = None,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Fit a lasso regression model and return coefficients and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_) if fit_intercept else 0.0,
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "alpha": alpha,
        "feature_names": names,
    }


def elasticnet_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    feature_names: Optional[Sequence[str]] = None,
    fit_intercept: bool = True,
) -> Dict[str, Any]:
    """
    Fit an elastic net regression model and return coefficients and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_) if fit_intercept else 0.0,
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "feature_names": names,
    }


def svr_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    kernel: str = "rbf",
    c: float = 1.0,
    epsilon: float = 0.1,
) -> Dict[str, Any]:
    """
    Fit an SVR model and return predictions and score.
    """
    matrix, y = _validate_xy(data, target)
    model = SVR(kernel=kernel, C=c, epsilon=epsilon)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "kernel": kernel,
        "c": c,
        "epsilon": epsilon,
    }


def random_forest_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    n_estimators: int = 100,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a random forest regressor and return feature importances and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "feature_importances": model.feature_importances_.tolist(),
        "n_estimators": n_estimators,
        "feature_names": names,
    }


def gradient_boosting_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a gradient boosting regressor and return feature importances and predictions.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "feature_importances": model.feature_importances_.tolist(),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "feature_names": names,
    }


def pls_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[float],
    n_components: int = 2,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a PLS regression model and return X scores and predictions.
    """
    matrix, y = _validate_xy(data, target)
    model = PLSRegression(n_components=n_components)
    model.fit(matrix, y)
    predictions = model.predict(matrix).ravel()
    names = _resolve_feature_names(feature_names, matrix.shape[1]) if feature_names else None
    x_scores = model.x_scores_
    x_loadings = model.x_loadings_
    # PLSRegression uses _x_mean/_x_std (leading underscore), not x_mean_/x_std_
    x_mean = getattr(model, "_x_mean", np.mean(matrix, axis=0))
    x_std = getattr(model, "_x_std", np.ones(matrix.shape[1]))
    x_std = np.where(x_std == 0, 1, x_std)
    x_centered = (matrix - x_mean) / x_std
    total_ss_x = float(np.sum(x_centered**2))
    explained_x = []
    prev_r2 = 0.0
    for k in range(1, min(n_components, x_scores.shape[1]) + 1):
        x_hat = x_scores[:, :k] @ x_loadings[:, :k].T
        sse = float(np.sum((x_centered - x_hat) ** 2))
        r2 = 1.0 - (sse / total_ss_x) if total_ss_x else 0.0
        explained_x.append(max(r2 - prev_r2, 0.0))
        prev_r2 = r2
    y_scores = getattr(model, "y_scores_", None)
    y_loadings = getattr(model, "y_loadings_", None)
    y_mean = getattr(model, "_y_mean", np.mean(y))
    y_std = getattr(model, "_y_std", 1.0)
    y_std = y_std if y_std != 0 else 1.0
    y_centered = (y - y_mean) / y_std
    total_ss_y = float(np.sum(y_centered**2))
    explained_y = []
    if y_scores is not None and y_loadings is not None:
        prev_r2 = 0.0
        for k in range(1, min(n_components, y_scores.shape[1]) + 1):
            y_hat = y_scores[:, :k] @ y_loadings[:, :k].T
            sse = float(np.sum((y_centered.reshape(-1, 1) - y_hat) ** 2))
            r2 = 1.0 - (sse / total_ss_y) if total_ss_y else 0.0
            explained_y.append(max(r2 - prev_r2, 0.0))
            prev_r2 = r2
    return {
        "x_scores": model.x_scores_.tolist(),
        "x_loadings": model.x_loadings_.tolist(),
        "predictions": predictions.tolist(),
        "r2_score": float(_r2_score(y, predictions)),
        "n_components": n_components,
        "feature_names": names,
        "explained_variance_ratio_x": explained_x,
        "explained_variance_ratio_y": explained_y,
    }


def logistic_regression(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
    c: float = 1.0,
    max_iter: int = 1000,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a logistic regression classifier and return predictions and probabilities.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = LogisticRegression(C=c, max_iter=max_iter)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix).tolist()
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities,
        "accuracy": float(_accuracy_score(y, predictions)),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
        "feature_names": names,
    }


def svc_classification(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
    kernel: str = "rbf",
    c: float = 1.0,
    probability: bool = True,
) -> Dict[str, Any]:
    """
    Fit an SVC classifier and return predictions and probabilities.
    """
    matrix, y = _validate_xy(data, target)
    model = SVC(kernel=kernel, C=c, probability=probability)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    response = {
        "predictions": predictions.tolist(),
        "accuracy": float(_accuracy_score(y, predictions)),
        "kernel": kernel,
        "c": c,
    }
    if probability:
        response["probabilities"] = model.predict_proba(matrix).tolist()
    return response


def random_forest_classification(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
    n_estimators: int = 100,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a random forest classifier and return predictions and feature importances.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "accuracy": float(_accuracy_score(y, predictions)),
        "feature_importances": model.feature_importances_.tolist(),
        "n_estimators": n_estimators,
        "feature_names": names,
    }


def gradient_boosting_classification(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Fit a gradient boosting classifier and return predictions and feature importances.
    """
    matrix, y = _validate_xy(data, target)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "accuracy": float(_accuracy_score(y, predictions)),
        "feature_importances": model.feature_importances_.tolist(),
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "feature_names": names,
    }


def knn_classification(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
    n_neighbors: int = 5,
    weights: str = "uniform",
) -> Dict[str, Any]:
    """
    Fit a KNN classifier and return predictions.
    """
    matrix, y = _validate_xy(data, target)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "accuracy": float(_accuracy_score(y, predictions)),
        "n_neighbors": n_neighbors,
        "weights": weights,
    }


def naive_bayes_classification(
    data: Sequence[Sequence[float]],
    target: Sequence[int],
) -> Dict[str, Any]:
    """
    Fit a Gaussian Naive Bayes classifier and return predictions.
    """
    matrix, y = _validate_xy(data, target)
    model = GaussianNB()
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    return {
        "predictions": predictions.tolist(),
        "accuracy": float(_accuracy_score(y, predictions)),
    }


def lda_classification(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Fit an LDA classifier and return predictions and probabilities.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    if data is None or target is None:
        raise ValueError("lda_classification requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = LinearDiscriminantAnalysis()
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix).tolist()
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities,
        "accuracy": float(_accuracy_score(y, predictions)),
    }


def qda_classification(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Fit a QDA classifier and return predictions and probabilities.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    if data is None or target is None:
        raise ValueError("qda_classification requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = QuadraticDiscriminantAnalysis()
    model.fit(matrix, y)
    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix).tolist()
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities,
        "accuracy": float(_accuracy_score(y, predictions)),
    }


def kmeans_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using KMeans.
    """
    data = kwargs.get("data")
    n_clusters = kwargs.get("n_clusters", 3)
    random_state = kwargs.get("random_state", 42)
    if data is None:
        raise ValueError("kmeans_clustering requires data.")
    matrix = _validate_x(data)
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(matrix)
    return {
        "labels": labels.tolist(),
        "cluster_centers": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_),
    }


def minibatch_kmeans_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using MiniBatch KMeans.
    """
    data = kwargs.get("data")
    n_clusters = kwargs.get("n_clusters", 3)
    batch_size = kwargs.get("batch_size", 100)
    random_state = kwargs.get("random_state", 42)
    if data is None:
        raise ValueError("minibatch_kmeans_clustering requires data.")
    matrix = _validate_x(data)
    model = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, random_state=random_state, n_init="auto"
    )
    labels = model.fit_predict(matrix)
    return {
        "labels": labels.tolist(),
        "cluster_centers": model.cluster_centers_.tolist(),
        "inertia": float(model.inertia_),
    }


def dbscan_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using DBSCAN.
    """
    data = kwargs.get("data")
    eps = kwargs.get("eps", 0.5)
    min_samples = kwargs.get("min_samples", 5)
    if data is None:
        raise ValueError("dbscan_clustering requires data.")
    matrix = _validate_x(data)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(matrix)
    return {
        "labels": labels.tolist(),
        "eps": eps,
        "min_samples": min_samples,
    }


def agglomerative_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using Agglomerative Clustering.
    """
    data = kwargs.get("data")
    n_clusters = kwargs.get("n_clusters", 3)
    linkage = kwargs.get("linkage", "ward")
    if data is None:
        raise ValueError("agglomerative_clustering requires data.")
    matrix = _validate_x(data)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(matrix)
    return {"labels": labels.tolist(), "n_clusters": n_clusters, "linkage": linkage}


def spectral_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using Spectral Clustering.
    """
    data = kwargs.get("data")
    n_clusters = kwargs.get("n_clusters", 3)
    random_state = kwargs.get("random_state", 42)
    if data is None:
        raise ValueError("spectral_clustering requires data.")
    matrix = _validate_x(data)
    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(matrix)
    return {"labels": labels.tolist(), "n_clusters": n_clusters}


def gaussian_mixture_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using Gaussian Mixture Models.
    """
    data = kwargs.get("data")
    n_components = kwargs.get("n_components", 3)
    random_state = kwargs.get("random_state", 42)
    if data is None:
        raise ValueError("gaussian_mixture_clustering requires data.")
    matrix = _validate_x(data)
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(matrix)
    return {
        "labels": labels.tolist(),
        "n_components": n_components,
        "weights": model.weights_.tolist(),
    }


def optics_clustering(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Cluster data using OPTICS.
    """
    data = kwargs.get("data")
    min_samples = kwargs.get("min_samples", 5)
    if data is None:
        raise ValueError("optics_clustering requires data.")
    matrix = _validate_x(data)
    model = OPTICS(min_samples=min_samples)
    labels = model.fit_predict(matrix)
    return {"labels": labels.tolist(), "min_samples": min_samples}


def incremental_pca(
    data: Sequence[Sequence[float]],
    n_components: int = 2,
    batch_size: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run Incremental PCA on numeric data.
    """
    if not data:
        raise ValueError("incremental_pca requires non-empty data.")
    matrix = np.array(data, dtype=float)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    transformed = ipca.fit_transform(matrix)
    feature_importance = _pca_feature_importance(
        ipca.components_, ipca.explained_variance_ratio_, names
    )
    return {
        "transformed": transformed.tolist(),
        "components": ipca.components_.tolist(),
        "explained_variance_ratio": ipca.explained_variance_ratio_.tolist(),
        "feature_names": names,
        "feature_importance": feature_importance,
    }


def truncated_svd(
    data: Sequence[Sequence[float]],
    n_components: int = 2,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run Truncated SVD on numeric data.
    """
    if not data:
        raise ValueError("truncated_svd requires non-empty data.")
    matrix = np.array(data, dtype=float)
    names = _resolve_feature_names(feature_names, matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components)
    transformed = svd.fit_transform(matrix)
    feature_importance = _pca_feature_importance(
        svd.components_, svd.explained_variance_ratio_, names
    )
    return {
        "transformed": transformed.tolist(),
        "components": svd.components_.tolist(),
        "explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
        "feature_names": names,
        "feature_importance": feature_importance,
    }


def nmf_decomposition(
    data: Sequence[Sequence[float]],
    n_components: int = 2,
    init: str = "nndsvda",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run NMF decomposition on non-negative data.
    """
    if not data:
        raise ValueError("nmf_decomposition requires non-empty data.")
    matrix = np.array(data, dtype=float)
    # NMF requires non-negative data; shift if needed
    min_val = matrix.min()
    shifted = False
    if min_val < 0:
        matrix = matrix - min_val
        shifted = True
    model = NMF(n_components=n_components, init=init, random_state=random_state)
    transformed = model.fit_transform(matrix)
    return {
        "transformed": transformed.tolist(),
        "components": model.components_.tolist(),
        "reconstruction_err": float(model.reconstruction_err_),
        "n_components": n_components,
        "data_shifted": shifted,
        "shift_amount": float(-min_val) if shifted else 0.0,
    }


def fast_ica(
    data: Sequence[Sequence[float]],
    n_components: int = 2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run FastICA on numeric data.
    """
    if not data:
        raise ValueError("fast_ica requires non-empty data.")
    matrix = np.array(data, dtype=float)
    model = FastICA(n_components=n_components, random_state=random_state)
    transformed = model.fit_transform(matrix)
    return {
        "transformed": transformed.tolist(),
        "components": model.components_.tolist(),
        "mixing": model.mixing_.tolist() if model.mixing_ is not None else None,
        "n_components": n_components,
    }


def tsne_embedding(
    data: Sequence[Sequence[float]],
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run t-SNE to compute an embedding for visualization.
    """
    if not data:
        raise ValueError("tsne_embedding requires non-empty data.")
    matrix = np.array(data, dtype=float)
    # Perplexity must be less than n_samples; auto-adjust if needed
    n_samples = matrix.shape[0]
    adjusted_perplexity = min(perplexity, max(1.0, n_samples - 1))
    model = TSNE(n_components=n_components, perplexity=adjusted_perplexity, random_state=random_state)
    transformed = model.fit_transform(matrix)
    return {
        "embedding": transformed.tolist(),
        "n_components": n_components,
        "perplexity": adjusted_perplexity,
    }


def standard_scaler(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Scale data with StandardScaler.
    """
    data = kwargs.get("data")
    if data is None:
        raise ValueError("standard_scaler requires data.")
    matrix = _validate_x(data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return {
        "scaled": scaled.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def minmax_scaler(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Scale data with MinMaxScaler.
    """
    data = kwargs.get("data")
    if data is None:
        raise ValueError("minmax_scaler requires data.")
    matrix = _validate_x(data)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(matrix)
    return {
        "scaled": scaled.tolist(),
        "data_min": scaler.data_min_.tolist(),
        "data_max": scaler.data_max_.tolist(),
    }


def robust_scaler(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Scale data with RobustScaler.
    """
    data = kwargs.get("data")
    if data is None:
        raise ValueError("robust_scaler requires data.")
    matrix = _validate_x(data)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)
    return {
        "scaled": scaled.tolist(),
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
    }


def one_hot_encoder(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    One-hot encode categorical features.
    """
    data = kwargs.get("data")
    if data is None:
        raise ValueError("one_hot_encoder requires data.")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    transformed = encoder.fit_transform(data)
    return {
        "encoded": transformed.tolist(),
        "categories": [cats.tolist() for cats in encoder.categories_],
    }


def label_encoder(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Encode labels to integers.
    """
    labels = kwargs.get("labels")
    if labels is None:
        raise ValueError("label_encoder requires labels.")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return {"encoded": encoded.tolist(), "classes": encoder.classes_.tolist()}


def polynomial_features(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Generate polynomial and interaction features.
    """
    data = kwargs.get("data")
    degree = kwargs.get("degree", 2)
    include_bias = kwargs.get("include_bias", False)
    if data is None:
        raise ValueError("polynomial_features requires data.")
    matrix = _validate_x(data)
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    transformed = poly.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "degree": degree}


def power_transformer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Apply power transform to make data more Gaussian-like.
    """
    data = kwargs.get("data")
    method = kwargs.get("method", "yeo-johnson")
    if data is None:
        raise ValueError("power_transformer requires data.")
    matrix = _validate_x(data)
    transformer = PowerTransformer(method=method)
    transformed = transformer.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "method": method}


def quantile_transformer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Apply quantile transform to map data to a uniform/normal distribution.
    """
    data = kwargs.get("data")
    output_distribution = kwargs.get("output_distribution", "uniform")
    n_quantiles = kwargs.get("n_quantiles", 1000)
    if data is None:
        raise ValueError("quantile_transformer requires data.")
    matrix = _validate_x(data)
    # n_quantiles must be <= n_samples
    n_quantiles = min(n_quantiles, matrix.shape[0])
    transformer = QuantileTransformer(
        output_distribution=output_distribution,
        n_quantiles=n_quantiles,
        random_state=42,
    )
    transformed = transformer.fit_transform(matrix)
    return {
        "transformed": transformed.tolist(),
        "output_distribution": output_distribution,
        "n_quantiles": n_quantiles,
    }


def simple_imputer(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Impute missing values.
    """
    from sklearn.impute import SimpleImputer

    data = kwargs.get("data")
    strategy = kwargs.get("strategy", "mean")
    if data is None:
        raise ValueError("simple_imputer requires data.")
    matrix = np.array(data, dtype=float)
    imputer = SimpleImputer(strategy=strategy)
    transformed = imputer.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "strategy": strategy}


def select_k_best(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Select top features based on a score function.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    k = kwargs.get("k", 5)
    score_func = kwargs.get("score_func", "f_regression")
    if data is None or target is None:
        raise ValueError("select_k_best requires data and target.")
    matrix, y = _validate_xy(data, target)
    # Ensure k doesn't exceed number of features
    k = min(k, matrix.shape[1])
    funcs = {
        "f_regression": f_regression,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
    }
    # chi2 requires non-negative data; shift if using chi2
    shifted = False
    if score_func == "chi2":
        min_val = matrix.min()
        if min_val < 0:
            matrix = matrix - min_val
            shifted = True
    selector = SelectKBest(score_func=funcs.get(score_func, f_regression), k=k)
    transformed = selector.fit_transform(matrix, y)
    return {
        "transformed": transformed.tolist(),
        "scores": selector.scores_.tolist(),
        "k": k,
        "score_func": score_func,
        "data_shifted": shifted,
    }


def select_from_model(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Select features based on model feature importances.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "random_forest")
    task = kwargs.get("task", "regression")
    if data is None or target is None:
        raise ValueError("select_from_model requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    selector = SelectFromModel(model)
    transformed = selector.fit_transform(matrix, y)
    return {"transformed": transformed.tolist(), "estimator": estimator, "task": task}


def rfe(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Recursive feature elimination.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "linear")
    task = kwargs.get("task", "regression")
    n_features_to_select = kwargs.get("n_features_to_select")
    if data is None or target is None:
        raise ValueError("rfe requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector.fit(matrix, y)
    return {
        "support": selector.support_.tolist(),
        "ranking": selector.ranking_.tolist(),
        "n_features_to_select": selector.n_features_to_select,
    }


def rfecv(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Recursive feature elimination with cross-validation.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "linear")
    task = kwargs.get("task", "regression")
    cv = kwargs.get("cv", 5)
    if data is None or target is None:
        raise ValueError("rfecv requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    selector = RFECV(model, cv=cv)
    selector.fit(matrix, y)
    return {
        "support": selector.support_.tolist(),
        "ranking": selector.ranking_.tolist(),
        "n_features": int(selector.n_features_),
        "cv_scores": selector.cv_results_["mean_test_score"].tolist(),
    }


def grid_search_cv(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Grid search over estimator hyperparameters.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    param_grid = kwargs.get("param_grid")
    cv = kwargs.get("cv", 5)
    if data is None or target is None or param_grid is None:
        raise ValueError("grid_search_cv requires data, target, and param_grid.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    search = GridSearchCV(model, param_grid=param_grid, cv=cv)
    search.fit(matrix, y)
    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }


def randomized_search_cv(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Randomized search over estimator hyperparameters.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    param_distributions = kwargs.get("param_distributions")
    n_iter = kwargs.get("n_iter", 10)
    cv = kwargs.get("cv", 5)
    if data is None or target is None or param_distributions is None:
        raise ValueError("randomized_search_cv requires data, target, and param_distributions.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    search = RandomizedSearchCV(
        model, param_distributions=param_distributions, n_iter=n_iter, cv=cv, random_state=42
    )
    search.fit(matrix, y)
    return {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }


def cross_val_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute cross-validation scores for an estimator.
    """
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    cv = kwargs.get("cv", 5)
    if data is None or target is None:
        raise ValueError("cross_val_score requires data and target.")
    matrix, y = _validate_xy(data, target)
    model = _estimator_from_name(estimator, task)
    scores = _cross_val_score(model, matrix, y, cv=cv)
    return {"scores": scores.tolist(), "mean_score": float(scores.mean())}


def r2_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute R2 score.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    if y_true is None or y_pred is None:
        raise ValueError("r2_score requires y_true and y_pred.")
    return {"r2_score": float(_r2_score(y_true, y_pred))}


def mean_squared_error(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute mean squared error.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    if y_true is None or y_pred is None:
        raise ValueError("mean_squared_error requires y_true and y_pred.")
    return {"mean_squared_error": float(_mean_squared_error(y_true, y_pred))}


def mean_absolute_error(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute mean absolute error.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    if y_true is None or y_pred is None:
        raise ValueError("mean_absolute_error requires y_true and y_pred.")
    return {"mean_absolute_error": float(_mean_absolute_error(y_true, y_pred))}


def accuracy_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute accuracy score.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    if y_true is None or y_pred is None:
        raise ValueError("accuracy_score requires y_true and y_pred.")
    return {"accuracy_score": float(_accuracy_score(y_true, y_pred))}


def f1_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute F1 score.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    average = kwargs.get("average", "binary")
    if y_true is None or y_pred is None:
        raise ValueError("f1_score requires y_true and y_pred.")
    return {"f1_score": float(_f1_score(y_true, y_pred, average=average))}


def precision_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute precision score.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    average = kwargs.get("average", "binary")
    if y_true is None or y_pred is None:
        raise ValueError("precision_score requires y_true and y_pred.")
    return {"precision_score": float(_precision_score(y_true, y_pred, average=average))}


def recall_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute recall score.
    """
    y_true = kwargs.get("y_true")
    y_pred = kwargs.get("y_pred")
    average = kwargs.get("average", "binary")
    if y_true is None or y_pred is None:
        raise ValueError("recall_score requires y_true and y_pred.")
    return {"recall_score": float(_recall_score(y_true, y_pred, average=average))}


def roc_curve(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute ROC curve.
    """
    y_true = kwargs.get("y_true")
    y_score = kwargs.get("y_score")
    if y_true is None or y_score is None:
        raise ValueError("roc_curve requires y_true and y_score.")
    fpr, tpr, thresholds = _roc_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}


def auc_score(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute area under the curve.
    """
    x = kwargs.get("x")
    y = kwargs.get("y")
    if x is None or y is None:
        raise ValueError("auc_score requires x and y.")
    return {"auc": float(_auc(x, y))}


def precision_recall_curve(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Compute precision-recall curve.
    """
    y_true = kwargs.get("y_true")
    y_score = kwargs.get("y_score")
    if y_true is None or y_score is None:
        raise ValueError("precision_recall_curve requires y_true and y_score.")
    precision, recall, thresholds = _precision_recall_curve(y_true, y_score)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }
