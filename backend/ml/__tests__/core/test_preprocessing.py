import sys
from pathlib import Path

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.preprocessing import impute_non_finite_with_reference_medians, impute_train_test_non_finite


def test_impute_train_test_non_finite_uses_train_medians():
    train = np.array([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    test = np.array([[np.inf, np.nan]], dtype=np.float32)
    train_out, test_out, medians = impute_train_test_non_finite(train, test)

    assert medians.tolist() == [2.0, 5.0]
    assert np.isfinite(train_out).all()
    assert np.isfinite(test_out).all()


def test_impute_non_finite_with_reference_medians_applies_columnwise():
    values = np.array([[np.nan, np.inf], [2.0, 3.0]], dtype=np.float32)
    out = impute_non_finite_with_reference_medians(values, np.array([7.0, 8.0], dtype=np.float32))
    assert out[0, 0] == 7.0
    assert out[0, 1] == 8.0
