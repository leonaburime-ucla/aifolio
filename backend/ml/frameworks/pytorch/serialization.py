from __future__ import annotations

from pathlib import Path

import torch

from ...core.types import Metrics, ModelBundle
from .models import build_model, model_dropout, model_hidden_dim, model_num_hidden_layers, model_training_mode


def save_bundle(bundle: ModelBundle, output_dir: str | Path, metrics: Metrics | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "state_dict": bundle.model.state_dict(),
        "task": bundle.task,
        "target_column": bundle.target_column,
        "input_dim": bundle.input_dim,
        "output_dim": bundle.output_dim,
        "class_names": bundle.class_names,
        "vectorizer": bundle.vectorizer,
        "scaler": bundle.scaler,
        "feature_medians": bundle.feature_medians,
        "label_encoder": bundle.label_encoder,
        "target_scaler": bundle.target_scaler,
        "model_config": {
            "training_mode": model_training_mode(bundle.model),
            "hidden_dim": model_hidden_dim(bundle.model),
            "num_hidden_layers": model_num_hidden_layers(bundle.model),
            "dropout": model_dropout(bundle.model),
        },
    }

    model_path = out / "model_bundle.pt"
    torch.save(payload, model_path)

    if metrics is not None:
        import json
        from dataclasses import asdict

        (out / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return model_path


def load_bundle(path: str | Path, map_location: str = "cpu") -> ModelBundle:
    payload = torch.load(Path(path), map_location=map_location)
    model_cfg = payload.get("model_config", {})

    model = build_model(
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        training_mode=model_cfg.get("training_mode", "mlp_dense"),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])

    return ModelBundle(
        model=model,
        task=payload["task"],
        vectorizer=payload["vectorizer"],
        scaler=payload["scaler"],
        feature_medians=payload.get("feature_medians"),
        label_encoder=payload.get("label_encoder"),
        target_scaler=payload.get("target_scaler"),
        target_column=payload["target_column"],
        input_dim=int(payload["input_dim"]),
        output_dim=int(payload["output_dim"]),
        class_names=payload.get("class_names"),
    )
