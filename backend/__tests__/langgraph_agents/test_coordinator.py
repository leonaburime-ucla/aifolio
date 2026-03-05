import json
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from agents import coordinator


def test_load_dataset_metadata_returns_empty_when_manifest_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(coordinator, "DATASETS_MANIFEST_PATH", tmp_path / "missing.json")
    assert coordinator._load_dataset_metadata("any") == {}


def test_load_dataset_metadata_reads_context_and_names_excerpt(monkeypatch, tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    names = sample_dir / "dataset.names"
    names.write_text("line1\nline2", encoding="utf-8")

    manifest = sample_dir / "datasets.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "id": "d1",
                    "metadata": {
                        "context": "ctx",
                        "files": {"names": "dataset.names"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(coordinator, "SAMPLE_DATA_DIR", sample_dir)
    monkeypatch.setattr(coordinator, "DATASETS_MANIFEST_PATH", manifest)

    meta = coordinator._load_dataset_metadata("d1")
    assert meta["context"] == "ctx"
    assert "line1" in meta["excerpts"]["names"]


def test_coordinator_agent_builds_conversation_history_and_returns_response(monkeypatch):
    captured = {}

    def _pipeline(state):
        captured["state"] = state
        return {"response": {"message": "ok", "chartSpec": None, "findings": []}}

    monkeypatch.setattr(coordinator, "_coordinator_pipeline", _pipeline)

    result = coordinator.coordinator_agent(
        {
            "message": "analyze",
            "dataset_id": "d1",
            "model": "m1",
            "messages": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
            ],
        }
    )

    assert result["message"] == "ok"
    assert captured["state"]["conversation_history"] == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_parse_conversation_history_drops_empty_messages():
    history = coordinator._parse_conversation_history(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": ""},
            {"content": "defaults to user"},
        ]
    )
    assert history == [
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "defaults to user"},
    ]


def test_normalize_charts_accepts_dict_and_invalid_values():
    assert coordinator._normalize_charts({"mark": "bar"}) == [{"mark": "bar"}]
    assert coordinator._normalize_charts("bad") == []


def test_resolve_dataset_label_prefers_chart_metadata():
    label = coordinator._resolve_dataset_label(
        "fallback-id",
        {"chartSpec": [{"meta": {"datasetLabel": "Readable Label"}}]},
    )
    assert label == "Readable Label"
    assert coordinator._resolve_dataset_label("fallback-id", {"chartSpec": []}) == "fallback-id"


def test_build_langsmith_metadata_matches_disabled_payload():
    payload = coordinator._build_langsmith_metadata()
    assert payload["enabled"] is False
    assert "disabled" in payload["note"].lower()


def test_build_response_payload_normalizes_findings_and_charts():
    response = coordinator._build_response_payload(
        {"dataset_label": "Housing"},
        {"message": "tool output", "chartSpec": {"mark": "line"}},
        {"analyst_summary": "summary", "findings": "bad-shape"},
    )
    assert response["chartSpec"] == [{"mark": "line"}]
    assert response["findings"] == []
    assert response["data"]["dataset"] == "Housing"
    assert "[Data Scientist] tool output" in response["message"]


def test_data_scientist_node_uses_defaults_and_updates_state(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "run_data_scientist_analysis",
        lambda **kwargs: {"message": "done", "chartSpec": [{"meta": {"datasetLabel": "Cars"}}]},
    )
    result = coordinator._data_scientist_node({"user_message": "analyze", "dataset_id": "cars"})
    assert result["dataset_label"] == "Cars"
    assert result["data_scientist_result"]["message"] == "done"


def test_analyst_node_normalizes_single_chart_and_loads_metadata(monkeypatch):
    captured = {}
    monkeypatch.setattr(coordinator, "_load_dataset_metadata", lambda dataset_id: {"context": "ctx"})

    def _interpret_analysis(**kwargs):
        captured.update(kwargs)
        return {"analyst_summary": "summary", "findings": ["f1"]}

    monkeypatch.setattr(coordinator, "interpret_analysis", _interpret_analysis)
    result = coordinator._analyst_node(
        {
            "user_message": "question",
            "dataset_id": "d1",
            "dataset_label": "Readable",
            "data_scientist_result": {"message": "tool", "chartSpec": {"mark": "bar"}},
            "conversation_history": [{"role": "user", "content": "u1"}],
        }
    )

    assert result["analyst_result"]["findings"] == ["f1"]
    assert captured["charts"] == [{"mark": "bar"}]
    assert captured["dataset_metadata"] == {"context": "ctx"}


def test_format_response_node_attaches_response_payload():
    result = coordinator._format_response_node(
        {
            "dataset_id": "d1",
            "dataset_label": "Dataset",
            "data_scientist_result": {"message": "tool", "chartSpec": []},
            "analyst_result": {"analyst_summary": "summary", "findings": ["f1"]},
        }
    )
    assert result["response"]["findings"] == ["f1"]
    assert result["response"]["data"]["dataset"] == "Dataset"


def test_coordinator_pipeline_runs_nodes_in_sequence(monkeypatch):
    monkeypatch.setattr(
        coordinator,
        "_data_scientist_node",
        lambda state: {**state, "data_scientist_result": {"message": "tool", "chartSpec": []}},
    )
    monkeypatch.setattr(
        coordinator,
        "_analyst_node",
        lambda state: {**state, "analyst_result": {"analyst_summary": "summary", "findings": []}},
    )
    monkeypatch.setattr(
        coordinator,
        "_format_response_node",
        lambda state: {**state, "response": {"message": "ok", "data": {"langsmith": state["langsmith"]}}},
    )
    result = coordinator._coordinator_pipeline({"user_message": "u"})
    assert result["response"]["data"]["langsmith"]["enabled"] is False


def test_load_dataset_metadata_returns_empty_for_bad_json(monkeypatch, tmp_path):
    manifest = tmp_path / "datasets.json"
    manifest.write_text("{bad json", encoding="utf-8")
    monkeypatch.setattr(coordinator, "DATASETS_MANIFEST_PATH", manifest)
    assert coordinator._load_dataset_metadata("d1") == {}
