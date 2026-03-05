"""Feature-selection and model-selection wrappers for sklearn tools."""

from __future__ import annotations

from typing import Any

from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, chi2, f_regression, mutual_info_classif
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score as _cross_val_score

from .shared import estimator_from_name, validate_xy


def select_k_best(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    k = kwargs.get("k", 5)
    score_func = kwargs.get("score_func", "f_regression")
    if data is None or target is None:
        raise ValueError("select_k_best requires data and target.")
    matrix, y = validate_xy(data, target)
    k = min(k, matrix.shape[1])
    funcs = {
        "f_regression": f_regression,
        "chi2": chi2,
        "mutual_info_classif": mutual_info_classif,
    }
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


def select_from_model(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "random_forest")
    task = kwargs.get("task", "regression")
    if data is None or target is None:
        raise ValueError("select_from_model requires data and target.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    selector = SelectFromModel(model)
    transformed = selector.fit_transform(matrix, y)
    return {"transformed": transformed.tolist(), "estimator": estimator, "task": task}


def rfe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "linear")
    task = kwargs.get("task", "regression")
    n_features_to_select = kwargs.get("n_features_to_select")
    if data is None or target is None:
        raise ValueError("rfe requires data and target.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector.fit(matrix, y)
    return {
        "support": selector.support_.tolist(),
        "ranking": selector.ranking_.tolist(),
        "n_features_to_select": selector.n_features_to_select,
    }


def rfecv(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "linear")
    task = kwargs.get("task", "regression")
    cv = kwargs.get("cv", 5)
    if data is None or target is None:
        raise ValueError("rfecv requires data and target.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    selector = RFECV(model, cv=cv)
    selector.fit(matrix, y)
    return {
        "support": selector.support_.tolist(),
        "ranking": selector.ranking_.tolist(),
        "n_features": int(selector.n_features_),
        "cv_scores": selector.cv_results_["mean_test_score"].tolist(),
    }


def grid_search_cv(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    param_grid = kwargs.get("param_grid")
    cv = kwargs.get("cv", 5)
    if data is None or target is None or param_grid is None:
        raise ValueError("grid_search_cv requires data, target, and param_grid.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    search = GridSearchCV(model, param_grid=param_grid, cv=cv)
    search.fit(matrix, y)
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}


def randomized_search_cv(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    param_distributions = kwargs.get("param_distributions")
    n_iter = kwargs.get("n_iter", 10)
    cv = kwargs.get("cv", 5)
    if data is None or target is None or param_distributions is None:
        raise ValueError("randomized_search_cv requires data, target, and param_distributions.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=n_iter, cv=cv, random_state=42)
    search.fit(matrix, y)
    return {"best_params": search.best_params_, "best_score": float(search.best_score_)}


def cross_val_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    target = kwargs.get("target")
    estimator = kwargs.get("estimator", "ridge")
    task = kwargs.get("task", "regression")
    cv = kwargs.get("cv", 5)
    if data is None or target is None:
        raise ValueError("cross_val_score requires data and target.")
    matrix, y = validate_xy(data, target)
    model = estimator_from_name(estimator, task)
    scores = _cross_val_score(model, matrix, y, cv=cv)
    return {"scores": scores.tolist(), "mean_score": float(scores.mean())}
