import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from backend.agents import status


def test_record_run_updates_global_status():
    status._STATUS = status.AgentStatus()
    status.record_run("model-a", latency_ms=12.5, error=None)

    snapshot = status.get_status()
    assert snapshot["total_requests"] == 1
    assert snapshot["last_model"] == "model-a"
    assert snapshot["last_latency_ms"] == 12.5
