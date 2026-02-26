from __future__ import annotations

"""Legacy compatibility shim for PyTorch runtime paths."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .core.types import Metrics, ModelBundle, TrainingConfig
from .frameworks.pytorch.handlers import (
    _handle_distill_request_impl,
    handle_distill_request,
    handle_train_request,
)
from .frameworks.pytorch.trainer import (
    distill_model_from_file,
    load_bundle,
    predict_rows,
    save_bundle,
    train_model_from_file,
)


def _distill_model_from_file_impl(**kwargs: Any) -> tuple[ModelBundle, Metrics]:
    return distill_model_from_file(**kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/test a PyTorch model on CSV/XLS/XLSX tabular data")
    parser.add_argument("--data", required=True, help="Path to .csv, .xls, or .xlsx file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=None, help="Directory to save model bundle + metrics")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = TrainingConfig(
        target_column=args.target,
        task=args.task,
        test_size=args.test_size,
        random_seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )

    bundle, metrics = train_model_from_file(
        data_path=args.data,
        cfg=cfg,
        sheet_name=args.sheet,
    )

    print(json.dumps(asdict(metrics), indent=2))

    if args.save_dir:
        model_path = save_bundle(bundle, args.save_dir, metrics)
        print(f"Saved bundle to {model_path}")


if __name__ == "__main__":
    main()
