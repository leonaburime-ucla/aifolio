import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.mode_catalog import allowed_modes_error, unsupported_distill_mode_error


def test_allowed_modes_error_formats_modes():
    msg = allowed_modes_error(["a", "b"])
    assert msg == "training_mode must be 'a', 'b'."


def test_unsupported_distill_mode_error_contains_mode_and_hint():
    msg = unsupported_distill_mode_error("x", "'mlp_dense'")
    assert "x" in msg
    assert "mlp_dense" in msg
