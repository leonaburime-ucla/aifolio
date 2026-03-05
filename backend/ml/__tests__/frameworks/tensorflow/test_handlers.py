import sys
from pathlib import Path

import numpy as np

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.frameworks.tensorflow import handlers


class _FakeModel:
    def count_params(self) -> int:
        return 17

    def get_weights(self):
        return [np.ones((2, 2), dtype=np.float32), np.ones((3,), dtype=np.float32)]


class _BrokenModel:
    def get_weights(self):
        raise RuntimeError("boom")


def test_parameter_count_uses_count_params():
    assert handlers._parameter_count(_FakeModel()) == 17


def test_serialized_model_size_sums_weight_bytes():
    assert handlers._serialized_model_size_bytes(_FakeModel()) == (4 * 4) + (3 * 4)


def test_serialized_model_size_returns_none_when_weights_fail():
    assert handlers._serialized_model_size_bytes(_BrokenModel()) is None


def test_store_and_load_in_memory_bundle_delegate_registry(monkeypatch):
    calls = {}

    class _Registry:
        def store(self, bundle):
            calls["store"] = bundle
            return "run-456"

        def load(self, run_id):
            calls["load"] = run_id
            return {"bundle": True}

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())

    assert handlers._store_in_memory_bundle({"bundle": True}) == "run-456"
    assert handlers._load_in_memory_bundle("run-456") == {"bundle": True}
    assert calls == {"store": {"bundle": True}, "load": "run-456"}


def test_handle_train_request_returns_prep_error(monkeypatch):
    prep_error = (422, {"status": "error", "error": "bad request"})
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (object(), None))
    monkeypatch.setattr(handlers, "prepare_train_request", lambda **kwargs: (None, prep_error))

    assert handlers.handle_train_request({}, lambda _dataset_id: None, Path("/tmp")) == prep_error


def test_handle_train_request_delegates_to_execute_train_request(monkeypatch):
    calls = {}

    def _fake_execute_train_request(**kwargs):
        calls["kwargs"] = kwargs
        run_id = kwargs["store_bundle"]({"bundle": True})
        calls["run_id"] = run_id
        return 200, {"status": "ok"}

    class _Registry:
        def store(self, bundle):
            calls["bundle"] = bundle
            return "run-456"

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: ("runtime", None))
    monkeypatch.setattr(handlers, "prepare_train_request", lambda **kwargs: ("prepared", None))
    monkeypatch.setattr(handlers, "execute_train_request", _fake_execute_train_request)

    status, body = handlers.handle_train_request({}, lambda _dataset_id: None, Path("/tmp"))
    assert (status, body) == (200, {"status": "ok"})
    assert calls["kwargs"]["runtime_trainer"] == "runtime"
    assert calls["kwargs"]["prepared"] == "prepared"
    assert calls["run_id"] == "run-456"
    assert calls["bundle"] == {"bundle": True}


def test_handle_distill_request_returns_prep_error(monkeypatch):
    prep_error = (422, {"status": "error", "error": "bad request"})
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (object(), None))
    monkeypatch.setattr(handlers, "prepare_distill_request", lambda **kwargs: (None, prep_error))

    assert handlers.handle_distill_request({}, lambda _dataset_id: None, Path("/tmp")) == prep_error


def test_handle_distill_request_delegates_to_execute_distill_request(monkeypatch):
    calls = {}

    def _fake_execute_distill_request(**kwargs):
        calls["kwargs"] = kwargs
        run_id = kwargs["store_bundle"]({"bundle": True})
        reloaded = kwargs["load_in_memory_bundle"](run_id)
        calls["parameter_count"] = kwargs["parameter_count_fn"](_FakeModel())
        calls["size_bytes"] = kwargs["serialized_size_fn"](_FakeModel())
        return 200, {"status": "ok", "reloaded": reloaded}

    class _Registry:
        def store(self, bundle):
            calls["stored_bundle"] = bundle
            return "run-789"

        def load(self, run_id):
            calls["loaded_run_id"] = run_id
            return {"bundle": "loaded"}

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: ("runtime", None))
    monkeypatch.setattr(handlers, "prepare_distill_request", lambda **kwargs: ("prepared", None))
    monkeypatch.setattr(handlers, "execute_distill_request", _fake_execute_distill_request)

    status, body = handlers.handle_distill_request({}, lambda _dataset_id: None, Path("/tmp"))
    assert (status, body) == (200, {"status": "ok", "reloaded": {"bundle": "loaded"}})
    assert calls["kwargs"]["runtime_trainer"] == "runtime"
    assert calls["kwargs"]["prepared"] == "prepared"
    assert calls["stored_bundle"] == {"bundle": True}
    assert calls["loaded_run_id"] == "run-789"
    assert calls["parameter_count"] == 17
    assert calls["size_bytes"] == (4 * 4) + (3 * 4)
