import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.agui_runtime import ml_actions


def test_normalize_lookup_token_standardizes_freeform_strings():
    assert ml_actions._normalize_lookup_token(" Wide & Deep ") == "wide_and_deep"


def test_normalize_training_mode_value_maps_known_aliases_and_non_strings():
    assert ml_actions._normalize_training_mode_value("TabResNet") == "tabresnet"
    assert ml_actions._normalize_training_mode_value("wide and deep model") == "wide_and_deep"
    sentinel = object()
    assert ml_actions._normalize_training_mode_value(sentinel) is sentinel


def test_normalize_dataset_id_value_maps_known_aliases_and_keeps_unknown():
    assert (
        ml_actions._normalize_dataset_id_value("fraud detection")
        == "fraud_detection_phishing_websites.csv"
    )
    assert ml_actions._normalize_dataset_id_value("custom.csv") == "custom.csv"
    assert ml_actions._normalize_dataset_id_value("   ") == "   "
    sentinel = object()
    assert ml_actions._normalize_dataset_id_value(sentinel) is sentinel


def test_normalize_ml_form_fields_canonicalizes_aliases():
    result = ml_actions.normalize_ml_form_fields(
        {
            "dataset": "house prices",
            "architecture": "wide and deep",
            "testSize": [0.25, 0.3],
            "hidden_units": [128, 256],
            "setSweepValues": True,
            "autoDistill": True,
        }
    )

    assert result == {
        "dataset_id": "house_prices_ames.csv",
        "training_mode": "wide_and_deep",
        "test_sizes": [0.25, 0.3],
        "hidden_dims": [128, 256],
        "set_sweep_values": True,
        "auto_distill": True,
        "run_sweep": True,
    }


def test_normalize_ml_tab_actions_returns_input_for_non_ml_tab():
    actions = [{"name": "add_chart_spec", "args": {"chartSpec": {"id": "chart-1"}}}]
    assert ml_actions.normalize_ml_tab_actions(actions, active_tab="charts") == actions


def test_normalize_ml_tab_actions_merges_duplicate_tensorflow_setters_and_target_changes():
    result = ml_actions.normalize_ml_tab_actions(
        [
            {
                "name": "set_active_ml_form_fields",
                "args": {"fields": {"datasetId": "house prices", "model_type": "wide and deep"}},
            },
            {
                "name": "set_tensorflow_form_fields",
                "args": {"fields": {"hiddenDims": [128, 256], "batch_size": [32, 64]}},
            },
            {"name": "change_active_ml_target_column", "args": {"target_column": "price"}},
            {"name": "change_tensorflow_target_column", "args": {"target_column": "sale_price"}},
            {"name": "randomize_active_ml_form_fields", "args": {"value_count": 1}},
            {"name": "randomize_tensorflow_form_fields", "args": {"value_count": 1}},
            {"name": "start_active_ml_training_runs", "args": {}},
            {"name": "start_tensorflow_training_runs", "args": {}},
        ],
        active_tab="tensorflow",
    )

    assert result == [
        {
            "name": "set_tensorflow_form_fields",
            "args": {
                "fields": {
                    "dataset_id": "house_prices_ames.csv",
                    "training_mode": "wide_and_deep",
                    "hidden_dims": [128, 256],
                    "batch_sizes": [32, 64],
                }
            },
        },
        {"name": "change_tensorflow_target_column", "args": {"target_column": "sale_price"}},
        {"name": "randomize_tensorflow_form_fields", "args": {"value_count": 1}},
        {"name": "start_tensorflow_training_runs", "args": {}},
    ]


def test_normalize_ml_tab_actions_skips_blank_names_and_dedupes_other_actions():
    result = ml_actions.normalize_ml_tab_actions(
        [
            {"name": "", "args": {}},
            {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
            {"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}},
        ],
        active_tab="pytorch",
    )

    assert result == [{"name": "switch_ag_ui_tab", "args": {"tab": "pytorch"}}]


def test_has_explicit_sweep_intent_requires_sweep_language():
    assert ml_actions.has_explicit_sweep_intent("") is False
    assert ml_actions.has_explicit_sweep_intent("set sweep values on") is True
    assert ml_actions.has_explicit_sweep_intent("run a hyperparameter sweep") is True
    assert (
        ml_actions.has_explicit_sweep_intent(
            "set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256"
        )
        is False
    )


def test_strip_implicit_sweep_flags_removes_truthy_flags_without_explicit_sweep():
    result = ml_actions.strip_implicit_sweep_flags(
        [
            {
                "name": "set_tensorflow_form_fields",
                "args": {
                    "fields": {
                        "training_mode": "wide_and_deep",
                        "test_sizes": [0.25, 0.3],
                        "run_sweep": True,
                        "set_sweep_values": True,
                    }
                },
            }
        ],
        active_tab="tensorflow",
        latest_user_text="Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.",
    )

    assert result == [
        {
            "name": "set_tensorflow_form_fields",
            "args": {
                "fields": {
                    "training_mode": "wide_and_deep",
                    "test_sizes": [0.25, 0.3],
                }
            },
        }
    ]


def test_strip_implicit_sweep_flags_preserves_explicit_sweep_and_false_values():
    explicit_result = ml_actions.strip_implicit_sweep_flags(
        [
            {
                "name": "set_pytorch_form_fields",
                "args": {"fields": {"run_sweep": True, "set_sweep_values": True}},
            }
        ],
        active_tab="pytorch",
        latest_user_text="Switch the algorithm to calibrated classifier and set sweep values on.",
    )
    assert explicit_result[0]["args"]["fields"] == {
        "run_sweep": True,
        "set_sweep_values": True,
    }

    false_result = ml_actions.strip_implicit_sweep_flags(
        [
            {
                "name": "set_pytorch_form_fields",
                "args": {"fields": {"run_sweep": False}},
            }
        ],
        active_tab="pytorch",
        latest_user_text="train pytorch model",
    )
    assert false_result[0]["args"]["fields"] == {"run_sweep": False}


def test_strip_implicit_sweep_flags_returns_input_for_non_ml_tab_and_non_dict_fields():
    chart_actions = [{"name": "add_chart_spec", "args": {"chartSpec": {"id": "chart-1"}}}]
    assert (
        ml_actions.strip_implicit_sweep_flags(
            chart_actions,
            active_tab="charts",
            latest_user_text="show a chart",
        )
        == chart_actions
    )

    malformed_fields_result = ml_actions.strip_implicit_sweep_flags(
        [
            {
                "name": "set_tensorflow_form_fields",
                "args": {"fields": "not-a-dict"},
            }
        ],
        active_tab="tensorflow",
        latest_user_text="set test sizes to 0.25 and 0.3",
    )
    assert malformed_fields_result == [
        {
            "name": "set_tensorflow_form_fields",
            "args": {"fields": "not-a-dict"},
        }
    ]
