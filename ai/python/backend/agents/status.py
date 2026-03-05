"""
Lightweight agent status tracking for UI reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class AgentStatus:
    total_requests: int = 0
    last_request_at: Optional[float] = None
    last_model: Optional[str] = None
    last_error: Optional[str] = None
    last_latency_ms: Optional[float] = None


_STATUS = AgentStatus()


def record_run(model_id: str, latency_ms: float, error: Optional[str] = None) -> None:
    """
    Update the shared status tracker.
    """
    _STATUS.total_requests += 1
    _STATUS.last_request_at = time.time()
    _STATUS.last_model = model_id
    _STATUS.last_latency_ms = latency_ms
    _STATUS.last_error = error


def get_status() -> Dict[str, Any]:
    """
    Return a JSON-serializable snapshot of the current status.
    """
    return asdict(_STATUS)
