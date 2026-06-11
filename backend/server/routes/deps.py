from __future__ import annotations

import sys
from typing import TypeVar

T = TypeVar("T")


def resolve_http_override(name: str, fallback: T) -> T:
    """Return a monkeypatched `server.http` dependency when available.

    Route handlers resolve dependencies lazily so the compatibility facade
    in `server.http` can keep existing tests and external callers working.
    """

    http_module = sys.modules.get("server.http")
    if http_module is not None:
        candidate = getattr(http_module, name, None)
        if candidate is not None:
            return candidate
    return fallback
