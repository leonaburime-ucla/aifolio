import sys
from pathlib import Path
from types import SimpleNamespace

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import ml.tensorflow as tensorflow_module
from ml.core.types import Metrics


def test_main_builds_config_and_calls_runtime(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        tensorflow_module,
        "_build_arg_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                data="data.csv",
                target="y",
                task="regression",
                test_size=0.25,
                seed=9,
                epochs=8,
                batch_size=2,
                lr=0.01,
                hidden_dim=32,
                hidden_layers=2,
                dropout=0.15,
                sheet=None,
                save_dir=None,
            )
        ),
    )
    monkeypatch.setattr(
        tensorflow_module,
        "train_model_from_file",
        lambda **kwargs: (
            captured.setdefault("train_kwargs", kwargs),
            (
                "bundle",
                Metrics(
                    task="regression",
                    train_loss=0.1,
                    test_loss=0.2,
                    test_metric_name="rmse",
                    test_metric_value=1.3,
                ),
            ),
        )[1],
    )
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: None)

    tensorflow_module.main()

    cfg = captured["train_kwargs"]["cfg"]
    assert captured["train_kwargs"]["data_path"] == "data.csv"
    assert captured["train_kwargs"]["sheet_name"] is None
    assert cfg.target_column == "y"
    assert cfg.task == "regression"
