from __future__ import annotations

"""
Shared preprocessing helpers for tabular ML runtimes.

All helpers here are pure numpy operations and framework-agnostic.
"""

import numpy as np


def impute_train_test_non_finite(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Impute NaN/inf values using training-set column medians.
    """
    col_medians = np.nanmedian(np.where(np.isfinite(x_train_np), x_train_np, np.nan), axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    for col_idx in range(x_train_np.shape[1]):
        train_mask = ~np.isfinite(x_train_np[:, col_idx])
        x_train_np[train_mask, col_idx] = col_medians[col_idx]
        test_mask = ~np.isfinite(x_test_np[:, col_idx])
        x_test_np[test_mask, col_idx] = col_medians[col_idx]
    return x_train_np, x_test_np, col_medians


def impute_non_finite_with_reference_medians(
    x_np: np.ndarray,
    medians: np.ndarray,
) -> np.ndarray:
    """
    Impute NaN/inf values in inference rows using reference medians.
    """
    for col_idx in range(x_np.shape[1]):
        mask = ~np.isfinite(x_np[:, col_idx])
        x_np[mask, col_idx] = medians[col_idx]
    return x_np
