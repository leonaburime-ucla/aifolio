from __future__ import annotations

"""Runtime import helpers for optional ML framework trainer modules."""

from importlib import import_module
from typing import Any


def import_runtime_trainer(module_path: str) -> tuple[Any | None, str | None]:
    """Import runtime trainer module and return structured import result.

    Args:
        module_path: Absolute import path for framework trainer module.

    Returns:
        Tuple `(module, error)` where one side is `None`.
    """
    try:
        return import_module(module_path), None
    except ModuleNotFoundError as exc:
        return None, str(exc)
