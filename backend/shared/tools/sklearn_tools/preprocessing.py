"""Preprocessing wrappers for sklearn tools."""

from __future__ import annotations

from typing import Any

import numpy as np
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

from .shared import validate_x


def standard_scaler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    if data is None:
        raise ValueError("standard_scaler requires data.")
    matrix = validate_x(data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return {"scaled": scaled.tolist(), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}


def minmax_scaler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    if data is None:
        raise ValueError("minmax_scaler requires data.")
    matrix = validate_x(data)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(matrix)
    return {"scaled": scaled.tolist(), "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist()}


def robust_scaler(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    if data is None:
        raise ValueError("robust_scaler requires data.")
    matrix = validate_x(data)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(matrix)
    return {"scaled": scaled.tolist(), "center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()}


def one_hot_encoder(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    if data is None:
        raise ValueError("one_hot_encoder requires data.")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    transformed = encoder.fit_transform(data)
    return {"encoded": transformed.tolist(), "categories": [cats.tolist() for cats in encoder.categories_]}


def label_encoder(*args: Any, **kwargs: Any) -> dict[str, Any]:
    labels = kwargs.get("labels")
    if labels is None:
        raise ValueError("label_encoder requires labels.")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    return {"encoded": encoded.tolist(), "classes": encoder.classes_.tolist()}


def polynomial_features(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    degree = kwargs.get("degree", 2)
    include_bias = kwargs.get("include_bias", False)
    if data is None:
        raise ValueError("polynomial_features requires data.")
    matrix = validate_x(data)
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    transformed = poly.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "degree": degree}


def power_transformer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    method = kwargs.get("method", "yeo-johnson")
    if data is None:
        raise ValueError("power_transformer requires data.")
    matrix = validate_x(data)
    transformer = PowerTransformer(method=method)
    transformed = transformer.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "method": method}


def quantile_transformer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    data = kwargs.get("data")
    output_distribution = kwargs.get("output_distribution", "uniform")
    n_quantiles = kwargs.get("n_quantiles", 1000)
    if data is None:
        raise ValueError("quantile_transformer requires data.")
    matrix = validate_x(data)
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


def simple_imputer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from sklearn.impute import SimpleImputer

    data = kwargs.get("data")
    strategy = kwargs.get("strategy", "mean")
    if data is None:
        raise ValueError("simple_imputer requires data.")
    matrix = np.array(data, dtype=float)
    imputer = SimpleImputer(strategy=strategy)
    transformed = imputer.fit_transform(matrix)
    return {"transformed": transformed.tolist(), "strategy": strategy}
