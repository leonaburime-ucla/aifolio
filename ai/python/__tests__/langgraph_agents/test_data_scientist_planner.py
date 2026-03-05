import json
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from backend.agents.data_scientist import planner


def test_build_plan_prompt_includes_recent_history_context():
    prompt = planner.build_plan_prompt(
        message="Run PCA",
        tools_schema=[{"name": "pca_transform", "params": [], "doc": "Principal components"}],
        dataset={"columns": ["feature_a"], "targetColumn": None, "task": "unsupervised"},
        conversation_history=[{"role": "user", "content": "earlier analysis"}],
    )

    assert "Previous conversation" in prompt
    assert "Run PCA" in prompt
    assert "pca_transform" in prompt


def test_resolve_tool_plan_repairs_invalid_tool_names():
    responses = iter(
        [
            '{"summary":"draft","tool_calls":[{"tool_name":"bad_tool","tool_args":{},"chart_kind":"pca"}]}',
            '{"summary":"repaired","tool_calls":[{"tool_name":"pca_transform","tool_args":{"n_components":2},"chart_kind":"pca"}]}',
        ]
    )

    plan = planner.resolve_tool_plan(
        message="Run PCA",
        tools_schema=[{"name": "pca_transform", "params": [], "doc": "Principal components"}],
        dataset={"columns": ["feature_a"], "targetColumn": None, "task": "unsupervised"},
        conversation_history=None,
        invoke_prompt=lambda _prompt: next(responses),
        parser=json.loads,
    )

    assert plan["summary"] == "repaired"
    assert plan["tool_calls"] == [
        {"tool_name": "pca_transform", "tool_args": {"n_components": 2}, "chart_kind": "pca"}
    ]


def test_parse_planner_response_returns_structured_error_when_json_is_invalid():
    parsed = planner.parse_planner_response(
        "not-json",
        parser=lambda _raw: {},
        failure_summary="Failed to plan tools.",
        stage="plan",
    )

    assert parsed["summary"] == "Failed to plan tools."
    assert parsed["planner_error"] == {"stage": "plan", "raw": "not-json"}
