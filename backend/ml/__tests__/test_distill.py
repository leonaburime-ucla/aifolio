import sys
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

import ml.distill as distill
from ml.frameworks.pytorch import handlers as pytorch_handlers
from ml.frameworks.pytorch import trainer as pytorch_trainer
from ml.frameworks.tensorflow import handlers as tensorflow_handlers
from ml.frameworks.tensorflow import trainer as tensorflow_trainer


def test_registry_store_and_load_roundtrip():
    registry = distill.InMemoryBundleRegistry[str](ttl_seconds=60, max_items=10)
    run_id = registry.store("bundle")
    assert registry.load(run_id) == "bundle"


def test_registry_prunes_when_over_capacity():
    registry = distill.InMemoryBundleRegistry[str](ttl_seconds=60, max_items=1)
    first = registry.store("a")
    _second = registry.store("b")
    assert registry.load(first) is None


def test_registry_prunes_expired_items_and_refreshes_hot_entry(monkeypatch):
    times = iter([100.0, 100.0, 120.0, 121.0])
    monkeypatch.setattr(distill.time, "time", lambda: next(times))

    registry = distill.InMemoryBundleRegistry[str](ttl_seconds=10, max_items=2)
    run_id = registry.store("bundle")
    assert registry.load(run_id) is None


def test_distill_model_from_file_routes_to_pytorch(monkeypatch):
    monkeypatch.setattr(
        pytorch_trainer,
        "distill_model_from_file",
        lambda **kwargs: ("pytorch", kwargs["sentinel"]),
    )

    assert distill.distill_model_from_file("pytorch", sentinel=7) == ("pytorch", 7)


def test_distill_model_from_file_routes_to_tensorflow(monkeypatch):
    monkeypatch.setattr(
        tensorflow_trainer,
        "distill_model_from_file",
        lambda **kwargs: ("tensorflow", kwargs["sentinel"]),
    )

    assert distill.distill_model_from_file("tensorflow", sentinel=9) == ("tensorflow", 9)


def test_distill_model_from_file_rejects_unsupported_framework():
    try:
        distill.distill_model_from_file("unknown")
    except ValueError as exc:
        assert "Unsupported framework" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported distillation framework")


def test_handle_distill_request_returns_unsupported_for_unknown_framework():
    status, payload = distill.handle_distill_request("unknown", {}, lambda _id: None, Path("."))
    assert status == 400
    assert payload["status"] == "error"


def test_handle_distill_request_routes_to_pytorch(monkeypatch):
    expected = (200, {"status": "ok", "framework": "pytorch"})
    monkeypatch.setattr(
        pytorch_handlers,
        "handle_distill_request",
        lambda payload, resolve_dataset_path, artifacts_dir: expected,
    )

    assert distill.handle_distill_request("pytorch", {}, lambda _id: None, Path(".")) == expected


def test_handle_distill_request_routes_to_tensorflow(monkeypatch):
    expected = (200, {"status": "ok", "framework": "tensorflow"})
    monkeypatch.setattr(
        tensorflow_handlers,
        "handle_distill_request",
        lambda payload, resolve_dataset_path, artifacts_dir: expected,
    )

    assert distill.handle_distill_request("tensorflow", {}, lambda _id: None, Path(".")) == expected
