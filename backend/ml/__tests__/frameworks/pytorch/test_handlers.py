import sys
from pathlib import Path

import torch

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.frameworks.pytorch import handlers


class _FakeParam:
    def __init__(self, size: int):
        self._size = size

    def numel(self) -> int:
        return self._size


class _FakeModel:
    def parameters(self):
        return [_FakeParam(3), _FakeParam(5)]


def test_parameter_count_sums_trainable_parameters():
    assert handlers._parameter_count(_FakeModel()) == 8


def test_store_and_load_in_memory_bundle_delegate_registry(monkeypatch):
    calls = {}

    class _Registry:
        def store(self, bundle):
            calls["store"] = bundle
            return "run-123"

        def load(self, run_id):
            calls["load"] = run_id
            return {"bundle": True}

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())

    assert handlers._store_in_memory_bundle({"bundle": True}) == "run-123"
    assert handlers._load_in_memory_bundle("run-123") == {"bundle": True}
    assert calls == {"store": {"bundle": True}, "load": "run-123"}


def test_serialized_model_size_bytes_returns_none_when_state_dict_fails():
    class _BrokenModel:
        def state_dict(self):
            raise RuntimeError("boom")

    assert handlers._serialized_model_size_bytes(_BrokenModel()) is None


def test_serialized_model_size_bytes_serializes_torch_model():
    model = torch.nn.Linear(2, 1)
    assert handlers._serialized_model_size_bytes(model) is not None


def test_handle_train_request_returns_prep_error(monkeypatch):
    prep_error = (422, {"status": "error", "error": "bad request"})
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: (object(), None))
    monkeypatch.setattr(handlers, "prepare_train_request", lambda **kwargs: (None, prep_error))

    assert handlers.handle_train_request({}, lambda _dataset_id: None, Path("/tmp")) == prep_error


def test_handle_train_request_delegates_to_execute_train_request(monkeypatch):
    calls = {}

    def _fake_execute_train_request(**kwargs):
        calls["kwargs"] = kwargs
        stored_run_id = kwargs["store_bundle"]({"bundle": True})
        calls["stored_run_id"] = stored_run_id
        return 200, {"status": "ok"}

    class _Registry:
        def store(self, bundle):
            calls["bundle"] = bundle
            return "run-123"

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: ("runtime", None))
    monkeypatch.setattr(handlers, "prepare_train_request", lambda **kwargs: ("prepared", None))
    monkeypatch.setattr(handlers, "execute_train_request", _fake_execute_train_request)

    status, body = handlers.handle_train_request({}, lambda _dataset_id: None, Path("/tmp"))
    assert (status, body) == (200, {"status": "ok"})
    assert calls["kwargs"]["runtime_trainer"] == "runtime"
    assert calls["kwargs"]["prepared"] == "prepared"
    assert calls["stored_run_id"] == "run-123"
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
        calls["parameter_count"] = kwargs["parameter_count_fn"](_ModelWithParams())
        calls["size_bytes"] = kwargs["serialized_size_fn"](torch.nn.Linear(2, 1))
        return 200, {"status": "ok", "reloaded": reloaded}

    class _Registry:
        def store(self, bundle):
            calls["stored_bundle"] = bundle
            return "run-456"

        def load(self, run_id):
            calls["loaded_run_id"] = run_id
            return {"bundle": "loaded"}

    class _ModelWithParams:
        def parameters(self):
            return [_FakeParam(2), _FakeParam(3)]

    monkeypatch.setattr(handlers, "_BUNDLE_REGISTRY", _Registry())
    monkeypatch.setattr(handlers, "_runtime_trainer", lambda: ("runtime", None))
    monkeypatch.setattr(handlers, "prepare_distill_request", lambda **kwargs: ("prepared", None))
    monkeypatch.setattr(handlers, "execute_distill_request", _fake_execute_distill_request)

    status, body = handlers.handle_distill_request({}, lambda _dataset_id: None, Path("/tmp"))
    assert (status, body) == (200, {"status": "ok", "reloaded": {"bundle": "loaded"}})
    assert calls["kwargs"]["runtime_trainer"] == "runtime"
    assert calls["kwargs"]["prepared"] == "prepared"
    assert calls["stored_bundle"] == {"bundle": True}
    assert calls["loaded_run_id"] == "run-456"
    assert calls["parameter_count"] == 5
    assert calls["size_bytes"] is not None
