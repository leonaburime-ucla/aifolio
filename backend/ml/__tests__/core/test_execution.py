from dataclasses import dataclass
from pathlib import Path
import sys

AI_ROOT = Path(__file__).resolve().parents[3]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.core.execution import execute_distill_request, execute_train_request
from ml.core.request_prep import PreparedDistillRequest, PreparedTrainRequest
from ml.core.types import TrainingConfig


@dataclass
class _Metrics:
    loss: float


class _Model:
    def __init__(self, params: int):
        self.params = params


class _Bundle:
    def __init__(self, params: int = 10):
        self.model = _Model(params)
        self.input_dim = 4
        self.output_dim = 1


class _Trainer:
    @staticmethod
    def train_model_from_file(**_kwargs):
        return _Bundle(), _Metrics(loss=0.1)

    @staticmethod
    def distill_model_from_file(**_kwargs):
        return _Bundle(params=6), _Metrics(loss=0.2)

    @staticmethod
    def save_bundle(bundle, model_dir, metrics):
        return Path(model_dir) / "model_bundle.pt"

    @staticmethod
    def load_bundle(path: str):
        return _Bundle(params=12)


def _cfg() -> TrainingConfig:
    return TrainingConfig(
        target_column="target",
        training_mode="mlp_dense",
        task="auto",
        test_size=0.2,
        random_seed=42,
        epochs=1,
        batch_size=16,
        learning_rate=0.01,
        hidden_dim=32,
        num_hidden_layers=2,
        dropout=0.1,
    )


def test_execute_train_request_success_envelope():
    prepared = PreparedTrainRequest(
        data_path="/tmp/data.csv",
        exclude_columns=[],
        date_columns=[],
        cfg=_cfg(),
        save_model=True,
        model_id="m1",
        model_dir=Path("/tmp/m1"),
    )

    status, body = execute_train_request(
        runtime_trainer=_Trainer,
        prepared=prepared,
        payload={},
        store_bundle=lambda bundle: "run-1",
    )

    assert status == 200
    assert body["status"] == "ok"
    assert body["run_id"] == "run-1"
    assert body["model_id"] == "m1"
    assert body["metrics"] == {"loss": 0.1}


def test_execute_distill_request_returns_404_for_missing_teacher_run():
    prepared = PreparedDistillRequest(
        data_path="/tmp/data.csv",
        exclude_columns=[],
        date_columns=[],
        cfg=_cfg(),
        numeric={"temperature": 2.0, "alpha": 0.5, "hidden_dim": 32, "num_hidden_layers": 2, "student_dropout": 0.1},
        teacher_run_id="missing",
        teacher_model_path="/tmp/teacher.pt",
        save_model=False,
        model_id=None,
        model_dir=None,
    )

    status, body = execute_distill_request(
        runtime_trainer=_Trainer,
        prepared=prepared,
        payload={},
        store_bundle=lambda bundle: "run-2",
        load_in_memory_bundle=lambda run_id: None,
        parameter_count_fn=lambda model: model.params,
        serialized_size_fn=lambda model: 100,
    )

    assert status == 404
    assert body == {"status": "error", "error": "Teacher run not found or expired."}


def test_execute_distill_request_success_envelope():
    prepared = PreparedDistillRequest(
        data_path="/tmp/data.csv",
        exclude_columns=[],
        date_columns=[],
        cfg=_cfg(),
        numeric={"temperature": 2.0, "alpha": 0.5, "hidden_dim": 32, "num_hidden_layers": 2, "student_dropout": 0.1},
        teacher_run_id="teacher-1",
        teacher_model_path="/tmp/teacher.pt",
        save_model=False,
        model_id="d1",
        model_dir=None,
    )

    status, body = execute_distill_request(
        runtime_trainer=_Trainer,
        prepared=prepared,
        payload={},
        store_bundle=lambda bundle: "run-2",
        load_in_memory_bundle=lambda run_id: _Bundle(params=12),
        parameter_count_fn=lambda model: model.params,
        serialized_size_fn=lambda model: 120,
    )

    assert status == 200
    assert body["status"] == "ok"
    assert body["run_id"] == "run-2"
    assert body["teacher_param_count"] == 12
    assert body["student_param_count"] == 6
