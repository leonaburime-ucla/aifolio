"""Shared validation and estimator helpers for sklearn tool wrappers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR


REGRESSION_ESTIMATORS = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "svr": SVR,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

CLASSIFICATION_ESTIMATORS = {
    "logistic": lambda: LogisticRegression(max_iter=1000),
    "svc": lambda: SVC(probability=True),
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "knn": KNeighborsClassifier,
    "naive_bayes": GaussianNB,
}


def validate_xy(
    data: Sequence[Sequence[float]],
    target: Sequence[float] | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired feature/target arrays and coerce them to numpy."""
    if not data:
        raise ValueError("data must be non-empty.")
    if not target:
        raise ValueError("target must be non-empty.")

    matrix = np.array(data, dtype=float)
    y = np.array(target)
    if matrix.shape[0] != y.shape[0]:
        raise ValueError("data and target must have the same number of rows.")
    return matrix, y


def validate_x(data: Sequence[Sequence[float]]) -> np.ndarray:
    """Validate feature rows and coerce them to numpy."""
    if not data:
        raise ValueError("data must be non-empty.")
    return np.array(data, dtype=float)


def resolve_feature_names(
    feature_names: Optional[Sequence[str]],
    n_features: int,
) -> Optional[list[str]]:
    """Validate optional feature names against matrix width."""
    if feature_names is None:
        return None
    if len(feature_names) != n_features:
        raise ValueError("feature_names length must match data column count.")
    return list(feature_names)


def estimator_from_name(name: str, task: str):
    """Build a supported sklearn estimator by shorthand name."""
    if task == "regression":
        factory = REGRESSION_ESTIMATORS.get(name)
        if factory is not None:
            return factory()
    if task == "classification":
        factory = CLASSIFICATION_ESTIMATORS.get(name)
        if factory is not None:
            return factory()
    raise ValueError(f"Unsupported estimator: {name} for task {task}.")
