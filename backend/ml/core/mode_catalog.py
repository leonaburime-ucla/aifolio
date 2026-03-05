from __future__ import annotations

from typing import Iterable

PYTORCH_ALLOWED_TRAINING_MODES: set[str] = {
    "mlp_dense",
    "linear_glm_baseline",
    "tabresnet",
    "imbalance_aware",
    "calibrated_classifier",
    "tree_teacher_distillation",
}

PYTORCH_UNSUPPORTED_DISTILL_MODES: set[str] = {
    "imbalance_aware",
    "calibrated_classifier",
    "tree_teacher_distillation",
}

TENSORFLOW_ALLOWED_TRAINING_MODES: set[str] = {
    "mlp_dense",
    "linear_glm_baseline",
    "wide_and_deep",
    "imbalance_aware",
    "quantile_regression",
    "calibrated_classifier",
    "entity_embeddings",
    "autoencoder_head",
    "multi_task_learning",
    "time_aware_tabular",
}

TENSORFLOW_UNSUPPORTED_DISTILL_MODES: set[str] = {
    "imbalance_aware",
    "quantile_regression",
    "calibrated_classifier",
    "entity_embeddings",
    "autoencoder_head",
    "multi_task_learning",
    "time_aware_tabular",
}


def allowed_modes_error(allowed_modes: Iterable[str]) -> str:
    quoted = ", ".join(f"'{mode}'" for mode in allowed_modes)
    return f"training_mode must be {quoted}."


def unsupported_distill_mode_error(mode: str, supported_hint: str) -> str:
    return (
        f"Distillation is not yet supported for training_mode '{mode}'. "
        f"Use {supported_hint}."
    )
