from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar


TBundle = TypeVar("TBundle")


@dataclass
class DistillationShrinkStats:
    teacher_size_bytes: int | None
    student_size_bytes: int | None
    size_saved_bytes: int | None
    size_saved_percent: float | None
    teacher_param_count: int | None
    student_param_count: int | None
    param_saved_count: int | None
    param_saved_percent: float | None


class InMemoryBundleRegistry(Generic[TBundle]):
    """
    Stores teacher/student bundles in process memory for short-lived distillation flows.
    This avoids requiring persistent artifact files between train -> distill calls.
    """

    def __init__(self, ttl_seconds: int = 900, max_items: int = 128) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.max_items = int(max_items)
        self._items: dict[str, tuple[float, TBundle]] = {}

    def _prune(self, now: float | None = None) -> None:
        current = now if now is not None else time.time()
        expired = [
            key
            for key, (created_at, _bundle) in self._items.items()
            if current - created_at > self.ttl_seconds
        ]
        for key in expired:
            self._items.pop(key, None)

        while len(self._items) > self.max_items:
            oldest = min(self._items.items(), key=lambda entry: entry[1][0])[0]
            self._items.pop(oldest, None)

    def store(self, bundle: TBundle) -> str:
        run_id = str(uuid.uuid4())
        self._prune()
        self._items[run_id] = (time.time(), bundle)
        return run_id

    def load(self, run_id: str) -> TBundle | None:
        self._prune()
        value = self._items.get(run_id)
        if value is None:
            return None
        # Touch access timestamp to keep hot entries alive during a distill sequence.
        self._items[run_id] = (time.time(), value[1])
        return value[1]


def distill_model_from_file(framework: str, **kwargs: Any) -> Any:
    """
    Framework-routed distillation entrypoint used by backend modules.
    """
    if framework == "pytorch":
        try:
            from . import pytorch as mod
        except ImportError:  # pragma: no cover
            import pytorch as mod  # type: ignore
        return mod._distill_model_from_file_impl(**kwargs)
    if framework == "tensorflow":
        try:
            from . import tensorflow as mod
        except ImportError:  # pragma: no cover
            import tensorflow as mod  # type: ignore
        return mod._distill_model_from_file_impl(**kwargs)
    raise ValueError(f"Unsupported framework: {framework}")


def handle_distill_request(
    framework: str,
    payload: dict[str, Any],
    resolve_dataset_path: Callable[[str], Path | None],
    artifacts_dir: Path,
) -> tuple[int, dict[str, Any]]:
    """
    Framework-routed request handler entrypoint used by backend modules.
    """
    if framework == "pytorch":
        try:
            from . import pytorch as mod
        except ImportError:  # pragma: no cover
            import pytorch as mod  # type: ignore
        return mod._handle_distill_request_impl(payload, resolve_dataset_path, artifacts_dir)
    if framework == "tensorflow":
        try:
            from . import tensorflow as mod
        except ImportError:  # pragma: no cover
            import tensorflow as mod  # type: ignore
        return mod._handle_distill_request_impl(payload, resolve_dataset_path, artifacts_dir)
    return 400, {"status": "error", "error": f"Unsupported framework: {framework}"}
