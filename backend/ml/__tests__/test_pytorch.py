import sys
from pathlib import Path
from types import SimpleNamespace

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import ml.pytorch as pytorch_module
from ml.core.types import Metrics


def test_main_builds_config_and_calls_runtime(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        pytorch_module,
        "_build_arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                data="data.csv",
                target="y",
                task="classification",
                test_size=0.3,
                seed=7,
                epochs=12,
                batch_size=4,
                lr=0.02,
                hidden_dim=16,
                hidden_layers=3,
                dropout=0.25,
                sheet="Sheet1",
                save_dir=None,
            )
        ),
    )
    monkeypatch.setattr(
        pytorch_module,
        "train_model_from_file",
        lambda **kwargs: (
            captured.setdefault("train_kwargs", kwargs),
            (
                "bundle",
                Metrics(
                    task="classification",
                    train_loss=0.1,
                    test_loss=0.2,
                    test_metric_name="accuracy",
                    test_metric_value=0.9,
                ),
            ),
        )[1],
    )
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: None)

    pytorch_module.main()

    cfg = captured["train_kwargs"]["cfg"]
    assert captured["train_kwargs"]["data_path"] == "data.csv"
    assert captured["train_kwargs"]["sheet_name"] == "Sheet1"
    assert cfg.target_column == "y"
    assert cfg.hidden_dim == 16
