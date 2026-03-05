from __future__ import annotations

from dataclasses import asdict
import json
import pickle
from pathlib import Path

import tensorflow as tf

from ...core.types import Metrics, ModelBundle


def save_bundle(bundle: ModelBundle, output_dir: str | Path, metrics: Metrics | None = None) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "model_bundle.keras"
    bundle.model.save(model_path)

    payload = {
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
        "model_config": bundle.model_config,
    }

    with (out / "bundle_meta.pkl").open("wb") as handle:
        pickle.dump(payload, handle)

    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    return model_path


def load_bundle(path: str | Path, map_location: str = "cpu") -> ModelBundle:
    _ = map_location
    model_path = Path(path)
    out = model_path.parent
    meta_path = out / "bundle_meta.pkl"
    if not meta_path.exists():
        raise ValueError("TensorFlow bundle metadata not found.")

    with meta_path.open("rb") as handle:
        payload = pickle.load(handle)

    model = tf.keras.models.load_model(model_path)

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
        model_config=payload.get("model_config", {}),
    )
