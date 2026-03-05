from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.request_prep import prepare_distill_request, prepare_train_request


def _resolver(dataset_id: str) -> Path | None:
    if dataset_id == "ok":
        return Path("/tmp/data.csv")
    return None


def test_prepare_train_request_requires_target_and_data():
    prepared, error = prepare_train_request(
        payload={},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error == (
        400,
        {
            "status": "error",
            "error": "target_column and either data_path or dataset_id are required.",
        },
    )


def test_prepare_train_request_builds_cfg_for_valid_payload():
    prepared, error = prepare_train_request(
        payload={
            "dataset_id": "ok",
            "target_column": "target",
            "training_mode": "mlp",
            "epochs": 10,
            "save_model": True,
            "model_id": "m1",
        },
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense", "linear_glm_baseline"},
    )
    assert error is None
    assert prepared is not None
    assert prepared.cfg.training_mode == "mlp_dense"
    assert prepared.model_id == "m1"
    assert prepared.model_dir == Path("/tmp/artifacts/m1")


def test_prepare_train_request_surfaces_list_and_numeric_errors(monkeypatch):
    monkeypatch.setattr("ml.core.request_prep.parse_feature_columns", lambda payload: ([], [], "bad list field"))
    prepared, error = prepare_train_request(
        payload={"dataset_id": "ok", "target_column": "target"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "bad list field"})

    monkeypatch.setattr("ml.core.request_prep.parse_feature_columns", lambda payload: ([], [], None))
    monkeypatch.setattr("ml.core.request_prep.parse_train_numeric", lambda payload: (None, "bad numeric field"))
    prepared, error = prepare_train_request(
        payload={"dataset_id": "ok", "target_column": "target"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "bad numeric field"})


def test_prepare_train_request_rejects_bounds_mode_and_hidden_config():
    prepared, error = prepare_train_request(
        payload={"dataset_id": "ok", "target_column": "target", "test_size": 1.2},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "test_size must be > 0 and < 1."})

    prepared, error = prepare_train_request(
        payload={"dataset_id": "ok", "target_column": "target", "training_mode": "unknown"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error is not None
    assert error[1]["error"] == "training_mode must be 'mlp_dense'."

    prepared, error = prepare_train_request(
        payload={
            "dataset_id": "ok",
            "target_column": "target",
            "training_mode": "mlp_dense",
            "hidden_dim": 0,
            "num_hidden_layers": 0,
            "dropout": 2.0,
        },
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        allowed_training_modes={"mlp_dense"},
    )
    assert prepared is None
    assert error is not None
    assert error[1]["status"] == "error"


def test_prepare_distill_request_requires_teacher_reference():
    prepared, error = prepare_distill_request(
        payload={"dataset_id": "ok", "target_column": "target"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense"},
        unsupported_distill_modes=set(),
        unsupported_distill_hint="mlp_dense",
    )
    assert prepared is None
    assert error == (
        400,
        {
            "status": "error",
            "error": "teacher_run_id, teacher_model_path, or teacher_model_id is required.",
        },
    )


def test_prepare_distill_request_rejects_unsupported_mode():
    prepared, error = prepare_distill_request(
        payload={
            "dataset_id": "ok",
            "target_column": "target",
            "teacher_model_path": "/tmp/teacher.pt",
            "training_mode": "tabresnet",
        },
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense", "tabresnet"},
        unsupported_distill_modes={"tabresnet"},
        unsupported_distill_hint="mlp_dense",
    )
    assert prepared is None
    assert error is not None
    status, body = error
    assert status == 400
    assert body["status"] == "error"
    assert "tabresnet" in body["error"]


def test_prepare_distill_request_surfaces_list_numeric_and_bounds_errors(monkeypatch):
    monkeypatch.setattr("ml.core.request_prep.parse_feature_columns", lambda payload: ([], [], "bad list field"))
    prepared, error = prepare_distill_request(
        payload={"dataset_id": "ok", "target_column": "target", "teacher_model_path": "/tmp/teacher.pt"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense"},
        unsupported_distill_modes=set(),
        unsupported_distill_hint="mlp_dense",
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "bad list field"})

    monkeypatch.setattr("ml.core.request_prep.parse_feature_columns", lambda payload: ([], [], None))
    monkeypatch.setattr("ml.core.request_prep.parse_distill_numeric", lambda payload: (None, "bad numeric field"))
    prepared, error = prepare_distill_request(
        payload={"dataset_id": "ok", "target_column": "target", "teacher_model_path": "/tmp/teacher.pt"},
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense"},
        unsupported_distill_modes=set(),
        unsupported_distill_hint="mlp_dense",
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "bad numeric field"})

    monkeypatch.undo()
    prepared, error = prepare_distill_request(
        payload={
            "dataset_id": "ok",
            "target_column": "target",
            "teacher_model_path": "/tmp/teacher.pt",
            "temperature": 0,
        },
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense"},
        unsupported_distill_modes=set(),
        unsupported_distill_hint="mlp_dense",
    )
    assert prepared is None
    assert error == (400, {"status": "error", "error": "temperature must be > 0 and <= 20."})


def test_prepare_distill_request_builds_cfg_with_default_student_shape_values():
    prepared, error = prepare_distill_request(
        payload={
            "dataset_id": "ok",
            "target_column": "target",
            "teacher_model_id": "teacher-1",
            "training_mode": "mlp",
            "save_model": True,
            "model_id": "student-1",
        },
        resolve_dataset_path=_resolver,
        artifacts_dir=Path("/tmp/artifacts"),
        artifact_filename="model_bundle.pt",
        allowed_training_modes={"mlp_dense"},
        unsupported_distill_modes=set(),
        unsupported_distill_hint="mlp_dense",
    )
    assert error is None
    assert prepared is not None
    assert prepared.teacher_model_path == "/tmp/artifacts/teacher-1/model_bundle.pt"
    assert prepared.save_model is True
    assert prepared.model_dir == Path("/tmp/artifacts/student-1")
    assert prepared.cfg.training_mode == "mlp_dense"
    assert prepared.cfg.hidden_dim == 128
    assert prepared.cfg.num_hidden_layers == 2
    assert prepared.cfg.dropout == 0.1
