import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.runtime_imports import import_runtime_trainer


def test_import_runtime_trainer_returns_module_when_present():
    module, error = import_runtime_trainer("json")
    assert module is not None
    assert error is None


def test_import_runtime_trainer_returns_error_when_missing():
    module, error = import_runtime_trainer("definitely_missing_package_12345")
    assert module is None
    assert error is not None
