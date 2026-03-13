import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.agui_runtime import actions


def test_normalize_action_calls_filters_invalid_items():
    raw = [
        {"name": " switch_ag_ui_tab ", "args": {"tab": "pytorch"}},
        {"name": "", "args": {}},
        {"name": "start_pytorch_training_runs", "args": "not-a-dict"},
        "bad",
    ]

    assert actions.normalize_action_calls(raw) == [
        {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
        {"name": "start_pytorch_training_runs", "args": {}},
    ]


def test_detect_pytorch_training_intent_sets_expected_flags():
    wants_train, wants_sweep, wants_auto_distill = actions.detect_pytorch_training_intent(
        "switch to pytorch and run sweep with autodistill then train"
    )

    assert wants_train is True
    assert wants_sweep is True
    assert wants_auto_distill is True


def test_has_phrase_matches_regex():
    assert actions._has_phrase("please train model", r"\btrain\b") is True
    assert actions._has_phrase("hello world", r"\btrain\b") is False


def test_detect_pytorch_training_intent_returns_false_for_blank_text():
    assert actions.detect_pytorch_training_intent("   ") == (False, False, False)


def test_build_enforced_pytorch_actions_filters_training_tools_without_intent():
    logs = []
    result = actions.build_enforced_pytorch_actions(
        latest_user_text="show me the chart",
        has_tool_messages_after_latest_user=False,
        action_calls=[
            {"name": "start_pytorch_training_runs", "args": {}},
            {"name": "switch_ag_ui_tab", "args": {"tab": "charts"}},
        ],
        available_tool_names={"start_pytorch_training_runs", "switch_ag_ui_tab"},
        debug_log=lambda event, **meta: logs.append((event, meta)),
    )
    assert result == [{"name": "switch_ag_ui_tab", "args": {"tab": "charts"}}]
    assert logs[0][0] == "stream.actions.training_filtered"


def test_build_enforced_pytorch_actions_returns_existing_actions_after_tool_messages():
    action_calls = [{"name": "start_pytorch_training_runs", "args": {}}]
    result = actions.build_enforced_pytorch_actions(
        latest_user_text="train pytorch",
        has_tool_messages_after_latest_user=True,
        action_calls=action_calls,
        available_tool_names={"start_pytorch_training_runs"},
    )
    assert result == action_calls


def test_build_enforced_pytorch_actions_inserts_required_actions_and_dedupes():
    result = actions.build_enforced_pytorch_actions(
        latest_user_text="switch to pytorch and run sweep with autodistill then train",
        has_tool_messages_after_latest_user=False,
        action_calls=[
            {"name": "start_pytorch_training_runs", "args": {}},
            {"name": "start_pytorch_training_runs", "args": {}},
            {"name": "unknown", "args": {}},
        ],
        available_tool_names={"switch_ag_ui_tab", "set_pytorch_form_fields", "start_pytorch_training_runs"},
    )
    assert result == [
        {"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": True, "auto_distill": True}}},
        {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
        {"name": "start_pytorch_training_runs", "args": {}},
    ]


def test_build_enforced_pytorch_actions_forces_run_sweep_false_for_simple_train():
    result = actions.build_enforced_pytorch_actions(
        latest_user_text="train pytorch model",
        has_tool_messages_after_latest_user=False,
        action_calls=[],
        available_tool_names={"set_pytorch_form_fields", "start_pytorch_training_runs"},
    )
    assert result == [
        {"name": "set_pytorch_form_fields", "args": {"fields": {"run_sweep": False}}},
        {"name": "start_pytorch_training_runs", "args": {}},
    ]


def test_build_enforced_pytorch_actions_updates_existing_form_patch():
    result = actions.build_enforced_pytorch_actions(
        latest_user_text="train pytorch model",
        has_tool_messages_after_latest_user=False,
        action_calls=[{"name": "set_pytorch_form_fields", "args": {"fields": {"auto_distill": True}}}],
        available_tool_names={"set_pytorch_form_fields", "start_pytorch_training_runs"},
    )
    assert result[0]["args"]["fields"] == {"auto_distill": True, "run_sweep": False}


def test_normalize_ml_tab_actions_rewrites_generic_tools_and_merges_setter_aliases():
    result = actions.normalize_ml_tab_actions(
        [
            {
                "name": "set_active_ml_form_fields",
                "args": {
                    "fields": {
                        "dataset": "fraud detection",
                        "model_type": "TabResNet",
                        "batch_size": [33, 40],
                    }
                },
            },
            {
                "name": "set_pytorch_form_fields",
                "args": {
                    "fields": {
                        "model_architecture": "TabResNet",
                        "hidden_dim": [64, 96],
                        "dropout": [0.1, 0.2],
                    }
                },
            },
            {"name": "start_active_ml_training_runs", "args": {}},
        ],
        active_tab="pytorch",
    )

    assert result == [
        {
            "name": "set_pytorch_form_fields",
            "args": {
                "fields": {
                    "dataset_id": "fraud_detection_phishing_websites.csv",
                    "training_mode": "tabresnet",
                    "batch_sizes": [33, 40],
                    "hidden_dims": [64, 96],
                    "dropouts": [0.1, 0.2],
                }
            },
        },
        {"name": "start_pytorch_training_runs", "args": {}},
    ]


def test_normalize_ml_tab_actions_preserves_serial_multi_tool_chain():
    result = actions.normalize_ml_tab_actions(
        [
            {"name": "randomize_active_ml_form_fields", "args": {"value_count": 1}},
            {"name": "start_active_ml_training_runs", "args": {}},
            {"name": "start_active_ml_training_runs", "args": {}},
        ],
        active_tab="tensorflow",
    )

    assert result == [
        {"name": "randomize_tensorflow_form_fields", "args": {"value_count": 1}},
        {"name": "start_tensorflow_training_runs", "args": {}},
    ]
