import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.frameworks.tensorflow import handlers


def test_train_handler_returns_503_when_runtime_missing(monkeypatch):
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (None, "missing runtime module"))

    status, body = handlers.handle_train_request(
        payload={},
        resolve_dataset_path=lambda _: None,
        artifacts_dir=Path("/tmp"),
    )

    assert status == 503
    assert body["status"] == "error"
    assert "TensorFlow" in body["error"]
    assert "Activate backend/.venv" in body["hint"]


def test_train_handler_maps_trainer_exception_to_400(monkeypatch):
    class _Trainer:
        @staticmethod
        def train_model_from_file(**_kwargs):
            raise RuntimeError("trainer exploded")

    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (_Trainer, None))

    status, body = handlers.handle_train_request(
        payload={"data_path": "/tmp/data.csv", "target_column": "target", "epochs": 1},
        resolve_dataset_path=lambda _: Path("/tmp/data.csv"),
        artifacts_dir=Path("/tmp"),
    )

    assert status == 400
    assert body == {"status": "error", "error": "trainer exploded"}


def test_distill_handler_returns_503_when_runtime_missing(monkeypatch):
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (None, "missing runtime module"))

    status, body = handlers.handle_distill_request(
        payload={},
        resolve_dataset_path=lambda _: None,
        artifacts_dir=Path("/tmp"),
    )

    assert status == 503
    assert body["status"] == "error"
    assert "TensorFlow" in body["error"]
    assert "Activate backend/.venv" in body["hint"]
