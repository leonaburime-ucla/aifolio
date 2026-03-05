import sys
from pathlib import Path

import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.tools.sklearn_tools import shared


def test_validate_xy_converts_to_numpy_arrays():
    matrix, target = shared.validate_xy([[1, 2], [3, 4]], [0, 1])
    assert matrix.shape == (2, 2)
    assert target.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("data", "target", "message"),
    [
        ([], [1], "data must be non-empty"),
        ([[1]], [], "target must be non-empty"),
        ([[1], [2]], [1], "same number of rows"),
    ],
)
def test_validate_xy_rejects_invalid_inputs(data, target, message):
    with pytest.raises(ValueError, match=message):
        shared.validate_xy(data, target)


def test_validate_x_converts_to_numpy_array():
    matrix = shared.validate_x([[1, 2], [3, 4]])
    assert matrix.shape == (2, 2)


def test_validate_x_requires_non_empty_data():
    with pytest.raises(ValueError, match="data must be non-empty"):
        shared.validate_x([])


def test_resolve_feature_names_returns_list_and_validates_width():
    assert shared.resolve_feature_names(("a", "b"), 2) == ["a", "b"]
    assert shared.resolve_feature_names(None, 2) is None
    with pytest.raises(ValueError, match="feature_names length must match"):
        shared.resolve_feature_names(["a"], 2)


@pytest.mark.parametrize(
    ("name", "task", "expected_type"),
    [
        ("linear", "regression", LinearRegression),
        ("ridge", "regression", Ridge),
        ("lasso", "regression", Lasso),
        ("elasticnet", "regression", ElasticNet),
        ("svr", "regression", SVR),
        ("random_forest", "regression", RandomForestRegressor),
        ("gradient_boosting", "regression", GradientBoostingRegressor),
        ("logistic", "classification", LogisticRegression),
        ("svc", "classification", SVC),
        ("random_forest", "classification", RandomForestClassifier),
        ("gradient_boosting", "classification", GradientBoostingClassifier),
        ("knn", "classification", KNeighborsClassifier),
        ("naive_bayes", "classification", GaussianNB),
    ],
)
def test_estimator_from_name_builds_supported_estimators(name, task, expected_type):
    estimator = shared.estimator_from_name(name, task)
    assert isinstance(estimator, expected_type)


def test_estimator_from_name_rejects_unsupported_names():
    with pytest.raises(ValueError, match="Unsupported estimator: bad for task regression"):
        shared.estimator_from_name("bad", "regression")

    with pytest.raises(ValueError, match="Unsupported estimator: linear for task classification"):
        shared.estimator_from_name("linear", "classification")
