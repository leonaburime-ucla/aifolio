import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from agents import langsmith


def test_configure_langsmith_returns_false_without_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert langsmith.configure_langsmith() is False


def test_configure_langsmith_sets_compat_env_vars(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)

    assert langsmith.configure_langsmith() is True
    assert langsmith.os.environ["LANGSMITH_TRACING"] == "true"
    assert langsmith.os.environ["LANGSMITH_PROJECT"] == "AIfolio"
    assert langsmith.os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert langsmith.os.environ["LANGCHAIN_PROJECT"] == "AIfolio"


def test_safe_serialize_handles_datetime_and_nested():
    out = langsmith._safe_serialize({"t": datetime(2024, 1, 1, tzinfo=timezone.utc), "x": [1, {"y": 2}], "z": (3, 4)})
    assert isinstance(out["t"], str)
    assert out["x"][1]["y"] == 2
    assert out["z"] == [3, 4]


def test_run_to_dict_supports_model_dump_dict_and_dunder_dict():
    class _WithModelDump:
        def model_dump(self):
            return {"id": "model-dump"}

    class _WithDict:
        def dict(self):
            return {"id": "dict"}

    class _Plain:
        def __init__(self):
            self.id = "plain"

    assert langsmith._run_to_dict(_WithModelDump()) == {"id": "model-dump"}
    assert langsmith._run_to_dict(_WithDict()) == {"id": "dict"}
    assert langsmith._run_to_dict({"id": "raw"}) == {"id": "raw"}
    assert langsmith._run_to_dict(_Plain()) == {"id": "plain"}


def test_parse_timestamp_handles_datetime_z_suffix_and_invalid_values():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert langsmith._parse_timestamp(now) is now
    assert langsmith._parse_timestamp("2024-01-01T00:00:00Z") == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert langsmith._parse_timestamp("") is None
    assert langsmith._parse_timestamp("not-a-time") is None
    assert langsmith._parse_timestamp(123) is None


def test_latency_ms_computes_from_iso_strings():
    ms = langsmith._latency_ms("2024-01-01T00:00:00+00:00", "2024-01-01T00:00:01+00:00")
    assert ms == 1000.0


def test_latency_ms_returns_none_when_timestamps_are_invalid():
    assert langsmith._latency_ms("bad", "2024-01-01T00:00:01+00:00") is None


def test_extract_usage_prefers_usage_dict():
    usage = langsmith._extract_usage({"usage": {"total_tokens": 9}})
    assert usage == {"total_tokens": 9}


def test_extract_usage_supports_extra_and_fallback_token_fields():
    assert langsmith._extract_usage({"extra": {"usage_metadata": {"total_tokens": 5}}}) == {"total_tokens": 5}
    assert langsmith._extract_usage({"prompt_tokens": 1, "completion_tokens": 2}) == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
    }
    assert langsmith._extract_usage({}) == {}


def test_extract_cost_and_resolve_project_name():
    assert langsmith._extract_cost({"total_cost": 1.0, "cost": 0.5, "ignored": 1}) == {"total_cost": 1.0, "cost": 0.5}
    assert langsmith._resolve_project_name("explicit") == "explicit"


def test_status_from_run():
    assert langsmith._status_from_run({"error": "x"}) == "error"
    assert langsmith._status_from_run({"end_time": None}) == "running"
    assert langsmith._status_from_run({"end_time": "2024-01-01T00:00:01+00:00"}) == "success"


def test_get_trace_report_requires_trace_id():
    assert langsmith.get_trace_report("") == {"status": "error", "error": "trace_id is required."}


def test_collect_retry_events_only_keeps_retry_related_items():
    events = langsmith._collect_retry_events(
        [
            {"id": "1", "name": "call-a", "run_type": "tool", "events": [{"name": "retry-start"}]},
            {"id": "2", "name": "call-b", "run_type": "tool", "events": [{"name": "complete"}]},
        ]
    )

    assert len(events) == 1
    assert events[0]["run_id"] == "1"


def test_collect_retry_events_skips_non_list_and_non_dict_events():
    events = langsmith._collect_retry_events(
        [
            {"id": "1", "name": "call-a", "run_type": "tool", "events": "bad"},
            {"id": "2", "name": "call-b", "run_type": "tool", "events": ["bad", {"message": "Retry later"}]},
        ]
    )
    assert len(events) == 1
    assert events[0]["run_id"] == "2"


def test_build_trace_report_returns_not_found_when_no_runs():
    report = langsmith._build_trace_report(
        trace_id="trace-1",
        project="proj",
        runs=[],
        client=object(),
        include_raw=False,
    )
    assert report == {
        "status": "error",
        "error": "Trace not found or inaccessible in the selected project.",
        "trace_id": "trace-1",
        "project": "proj",
    }


def test_build_trace_report_aggregates_model_tool_and_retry_data():
    root_run = SimpleNamespace(
        id="root-1",
        parent_run_id=None,
        name="coordinator",
        run_type="chain",
        start_time="2024-01-01T00:00:00+00:00",
        end_time="2024-01-01T00:00:02+00:00",
        total_cost=1.25,
        inputs={"prompt": "hi"},
        outputs={"message": "ok"},
        tenant_id="tenant-1",
        session_id="project-1",
    )
    model_run = {
        "id": "llm-1",
        "parent_run_id": "root-1",
        "name": "generative-model",
        "run_type": "llm",
        "start_time": "2024-01-01T00:00:00+00:00",
        "end_time": "2024-01-01T00:00:01+00:00",
        "usage": {"total_tokens": 12},
        "total_cost": 0.5,
        "inputs": {"messages": ["hi"]},
        "outputs": {"text": "hello"},
    }
    tool_run = {
        "id": "tool-1",
        "parent_run_id": "root-1",
        "name": "load_dataset",
        "run_type": "tool",
        "start_time": "2024-01-01T00:00:01+00:00",
        "end_time": "2024-01-01T00:00:02+00:00",
        "events": [{"name": "retry-start"}],
        "error": "boom",
    }

    class _FakeClient:
        def get_run_url(self, run, project_name):
            assert run is root_run
            assert project_name == "proj"
            return "https://smith.example/root-1"

    report = langsmith._build_trace_report(
        trace_id="trace-1",
        project="proj",
        runs=[root_run, model_run, tool_run],
        client=_FakeClient(),
        include_raw=True,
    )

    assert report["status"] == "ok"
    assert report["summary"]["status"] == "error"
    assert report["summary"]["model_call_count"] == 1
    assert report["summary"]["tool_call_count"] == 1
    assert report["summary"]["retry_count"] == 1
    assert report["summary"]["total_tokens"] == 12
    assert report["summary"]["total_cost"] == 0.5
    assert report["root"]["url"] == "https://smith.example/root-1"
    assert report["errors"] == [
        {"id": "tool-1", "name": "load_dataset", "run_type": "tool", "error": "boom"}
    ]
    assert report["retries"][0]["run_id"] == "tool-1"
    assert report["raw"]["root_run"]["id"] == "root-1"


def test_build_trace_report_uses_cost_and_url_fallbacks_when_client_lookup_fails():
    root_run = {
        "id": "root-1",
        "parent_run_id": None,
        "name": "coordinator",
        "run_type": "chain",
        "start_time": "2024-01-01T00:00:00+00:00",
        "end_time": "2024-01-01T00:00:02+00:00",
        "total_cost": 2.25,
        "tenant_id": "tenant-1",
        "session_id": "project-1",
    }

    class _ExplodingClient:
        def get_run_url(self, run, project_name):
            raise RuntimeError("boom")

    report = langsmith._build_trace_report(
        trace_id="trace-1",
        project="proj",
        runs=[root_run],
        client=_ExplodingClient(),
        include_raw=False,
    )

    assert report["summary"]["total_cost"] == 2.25
    assert report["root"]["url"] == "https://smith.langchain.com/o/tenant-1/projects/p/project-1/r/root-1?poll=true"
    assert "raw" not in report


def test_get_trace_report_returns_error_when_sdk_unavailable(monkeypatch):
    monkeypatch.setattr(langsmith, "Client", None)
    assert langsmith.get_trace_report("trace-1") == {
        "status": "error",
        "error": "langsmith SDK is not available in this environment.",
        "trace_id": "trace-1",
    }


def test_get_trace_report_builds_report_from_client_runs(monkeypatch):
    captured = {}

    class _Client:
        def list_runs(self, *, trace_id, project_name, limit):
            captured["list_runs"] = {"trace_id": trace_id, "project_name": project_name, "limit": limit}
            return [{"id": "root-1", "parent_run_id": None, "name": "root", "run_type": "chain"}]

        def get_run_url(self, run, project_name):
            return "https://smith.example/root-1"

    monkeypatch.setattr(langsmith, "Client", _Client)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.setenv("LANGCHAIN_PROJECT", "env-project")

    report = langsmith.get_trace_report("trace-1", include_raw=False)
    assert captured["list_runs"] == {"trace_id": "trace-1", "project_name": "env-project", "limit": 100}
    assert report["status"] == "ok"
    assert report["root"]["url"] == "https://smith.example/root-1"
