from __future__ import annotations

from typing import Any, Callable


def parse_train_numeric(payload: dict[str, Any], default_epochs: int = 500) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return {
            "test_size": float(payload.get("test_size", 0.2)),
            "epochs": int(payload.get("epochs", default_epochs)),
            "batch_size": int(payload.get("batch_size", 64)),
            "learning_rate": float(payload.get("learning_rate", 1e-3)),
            "training_mode": str(payload.get("training_mode", "mlp_dense")),
            "hidden_dim": int(payload.get("hidden_dim", 128)),
            "num_hidden_layers": int(payload.get("num_hidden_layers", 2)),
            "dropout": float(payload.get("dropout", 0.1)),
            "random_seed": int(payload.get("random_seed", 42)),
        }, None
    except (TypeError, ValueError):
        return None, "Invalid numeric training parameters."


def parse_distill_numeric(payload: dict[str, Any], default_epochs: int = 60) -> tuple[dict[str, Any] | None, str | None]:
    try:
        hidden_dim = int(payload.get("student_hidden_dim")) if payload.get("student_hidden_dim") is not None else None
        num_hidden_layers = (
            int(payload.get("student_num_hidden_layers")) if payload.get("student_num_hidden_layers") is not None else None
        )
        student_dropout = float(payload.get("student_dropout")) if payload.get("student_dropout") is not None else None
        return {
            "test_size": float(payload.get("test_size", 0.2)),
            "epochs": int(payload.get("epochs", default_epochs)),
            "batch_size": int(payload.get("batch_size", 64)),
            "learning_rate": float(payload.get("learning_rate", 1e-3)),
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "student_dropout": student_dropout,
            "temperature": float(payload.get("temperature", 2.5)),
            "alpha": float(payload.get("alpha", 0.5)),
            "random_seed": int(payload.get("random_seed", 42)),
            "training_mode": str(payload.get("training_mode", "mlp_dense")),
        }, None
    except (TypeError, ValueError):
        return None, "Invalid numeric distillation parameters."


def hidden_config_error(training_mode: str, hidden_dim: int, num_hidden_layers: int, dropout: float) -> str | None:
    if training_mode == "linear_glm_baseline":
        return None
    if not 8 <= hidden_dim <= 500:
        return "hidden_dim must be between 8 and 500."
    if not 1 <= num_hidden_layers <= 15:
        return "num_hidden_layers must be between 1 and 15."
    if not 0 <= dropout <= 0.9:
        return "dropout must be between 0 and 0.9."
    return None


def runtime_unavailable_response(framework: str, details: str | None) -> tuple[int, dict[str, Any]]:
    lower = framework.lower()
    package = "torch" if lower == "pytorch" else "tensorflow"
    return (
        503,
        {
            "status": "error",
            "error": f"{framework} runtime is unavailable in this Python environment.",
            "details": details,
            "hint": f"Activate ai/.venv or install {package} in the interpreter running the server.",
        },
    )


def compute_distill_stats(
    *,
    teacher_model_for_stats: Any,
    student_model: Any,
    teacher_model_size_bytes: int | None,
    student_model_size_bytes: int | None,
    parameter_count_fn: Callable[[Any], int],
    serialized_size_fn: Callable[[Any], int | None],
) -> dict[str, int | float | None]:
    teacher_param_count = parameter_count_fn(teacher_model_for_stats)
    student_param_count = parameter_count_fn(student_model)
    param_saved_count = teacher_param_count - student_param_count
    param_saved_percent = (float(param_saved_count) / float(teacher_param_count)) * 100.0 if teacher_param_count > 0 else None

    if teacher_model_size_bytes is None:
        teacher_model_size_bytes = serialized_size_fn(teacher_model_for_stats)
    if student_model_size_bytes is None:
        student_model_size_bytes = serialized_size_fn(student_model)

    size_saved_bytes = (
        teacher_model_size_bytes - student_model_size_bytes
        if teacher_model_size_bytes is not None and student_model_size_bytes is not None
        else None
    )
    size_saved_percent = (
        (float(size_saved_bytes) / float(teacher_model_size_bytes)) * 100.0
        if size_saved_bytes is not None and teacher_model_size_bytes and teacher_model_size_bytes > 0
        else None
    )

    return {
        "teacher_model_size_bytes": teacher_model_size_bytes,
        "student_model_size_bytes": student_model_size_bytes,
        "size_saved_bytes": size_saved_bytes,
        "size_saved_percent": size_saved_percent,
        "teacher_param_count": teacher_param_count,
        "student_param_count": student_param_count,
        "param_saved_count": param_saved_count,
        "param_saved_percent": param_saved_percent,
    }
