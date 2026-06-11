import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import ml.pytorch as pytorch_shim
import ml.cli.pytorch as pytorch_cli


def test_legacy_module_reexports_cli_entrypoints():
    assert pytorch_shim._build_arg_parser is pytorch_cli._build_arg_parser
    assert pytorch_shim._distill_model_from_file_impl is pytorch_cli._distill_model_from_file_impl
    assert pytorch_shim.main is pytorch_cli.main


def test_build_arg_parser_accepts_expected_required_flags():
    parser = pytorch_cli._build_arg_parser()
    args = parser.parse_args(["--data", "d.csv", "--target", "y"])
    assert args.data == "d.csv"
    assert args.target == "y"


def test_distill_impl_passes_through_to_runtime(monkeypatch):
    monkeypatch.setattr(pytorch_cli, "distill_model_from_file", lambda **kwargs: (kwargs["k"], "ok"))
    assert pytorch_cli._distill_model_from_file_impl(k=1) == (1, "ok")
