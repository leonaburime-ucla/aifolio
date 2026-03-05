import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.agui import (
    _build_enforced_pytorch_actions,
    _is_pure_tab_switch_intent,
    _resolve_ag_ui_tab_target,
    _resolve_navigation_target,
)


def test_resolve_ag_ui_tab_target_supports_switch_language():
    assert _resolve_ag_ui_tab_target("switch to pytorch") == "pytorch"
    assert _resolve_ag_ui_tab_target("please switch to tensorflow tab") == "tensorflow"


def test_resolve_navigation_target_supports_switch_language():
    assert _resolve_navigation_target("switch to agentic research") == "/agentic-research"


def test_enforced_pytorch_actions_filters_training_tools_without_training_intent():
    actions = [
        {"name": "start_pytorch_training_runs", "args": {}},
        {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
    ]
    result = _build_enforced_pytorch_actions(
        latest_user_text="switch to pytorch",
        has_tool_messages_after_latest_user=False,
        action_calls=actions,
        available_tool_names={"start_pytorch_training_runs", "switch_ag_ui_tab"},
    )

    assert result == [{"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}}]


def test_enforced_pytorch_actions_keeps_form_patch_without_training_intent():
    actions = [
        {"name": "set_pytorch_form_fields", "args": {"fields": {"hidden_dims": [128, 64, 32]}}},
        {"name": "start_pytorch_training_runs", "args": {}},
    ]
    result = _build_enforced_pytorch_actions(
        latest_user_text="set hidden dims to 128,64,32",
        has_tool_messages_after_latest_user=False,
        action_calls=actions,
        available_tool_names={"set_pytorch_form_fields", "start_pytorch_training_runs"},
    )

    assert result == [{"name": "set_pytorch_form_fields", "args": {"fields": {"hidden_dims": [128, 64, 32]}}}]


def test_enforced_pytorch_actions_keeps_training_when_user_asked_to_train():
    actions = []
    result = _build_enforced_pytorch_actions(
        latest_user_text="switch to pytorch and train",
        has_tool_messages_after_latest_user=False,
        action_calls=actions,
        available_tool_names={"switch_ag_ui_tab", "start_pytorch_training_runs", "set_pytorch_form_fields"},
    )

    names = [item["name"] for item in result]
    assert "switch_ag_ui_tab" in names
    assert "start_pytorch_training_runs" in names
    assert "set_pytorch_form_fields" in names


def test_enforced_pytorch_actions_supports_serial_switch_then_sweep_then_train():
    result = _build_enforced_pytorch_actions(
        latest_user_text="switch to pytorch and run sweep with autodistill then train",
        has_tool_messages_after_latest_user=False,
        action_calls=[],
        available_tool_names={"switch_ag_ui_tab", "set_pytorch_form_fields", "start_pytorch_training_runs"},
    )

    names = [item["name"] for item in result]
    assert "switch_ag_ui_tab" in names
    assert "set_pytorch_form_fields" in names
    assert "start_pytorch_training_runs" in names
    assert names.index("set_pytorch_form_fields") < names.index("start_pytorch_training_runs")


def test_pure_tab_switch_detection_for_fast_path():
    assert _is_pure_tab_switch_intent("switch to pytorch")
    assert not _is_pure_tab_switch_intent("switch to pytorch and set hidden dims 128,64,32")
