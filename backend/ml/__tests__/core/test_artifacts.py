from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.artifacts import resolve_model_artifact_target


def test_resolve_model_artifact_target_without_save_model():
    save_model, model_id, model_dir = resolve_model_artifact_target(payload={}, artifacts_dir=Path("/tmp/artifacts"))
    assert save_model is False
    assert model_id is None
    assert model_dir is None


def test_resolve_model_artifact_target_with_explicit_model_id():
    save_model, model_id, model_dir = resolve_model_artifact_target(
        payload={"save_model": True, "model_id": "abc123"},
        artifacts_dir=Path("/tmp/artifacts"),
    )
    assert save_model is True
    assert model_id == "abc123"
    assert model_dir == Path("/tmp/artifacts/abc123")


def test_resolve_model_artifact_target_generates_model_id_when_missing():
    save_model, model_id, model_dir = resolve_model_artifact_target(
        payload={"save_model": True},
        artifacts_dir=Path("/tmp/artifacts"),
    )
    assert save_model is True
    assert isinstance(model_id, str)
    assert len(model_id) > 0
    assert model_dir == Path(f"/tmp/artifacts/{model_id}")
