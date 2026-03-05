import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import ml.tensorflow as tensorflow_shim


def test_build_arg_parser_accepts_expected_required_flags():
    parser = tensorflow_shim._build_arg_parser()
    args = parser.parse_args(["--data", "d.csv", "--target", "y"])
    assert args.data == "d.csv"
    assert args.target == "y"


def test_distill_impl_passes_through_to_runtime(monkeypatch):
    monkeypatch.setattr(tensorflow_shim, "distill_model_from_file", lambda **kwargs: (kwargs["k"], "ok"))
    assert tensorflow_shim._distill_model_from_file_impl(k=1) == (1, "ok")
