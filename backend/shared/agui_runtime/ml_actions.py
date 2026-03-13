from __future__ import annotations

"""ML-tab action normalization helpers for AG-UI tool planning/emission."""

import json
import re
from typing import Any

ML_TAB_TOOL_NAME_ALIASES: dict[str, dict[str, str]] = {
    "pytorch": {
        "set_active_ml_form_fields": "set_pytorch_form_fields",
        "change_active_ml_target_column": "change_pytorch_target_column",
        "randomize_active_ml_form_fields": "randomize_pytorch_form_fields",
        "start_active_ml_training_runs": "start_pytorch_training_runs",
    },
    "tensorflow": {
        "set_active_ml_form_fields": "set_tensorflow_form_fields",
        "change_active_ml_target_column": "change_tensorflow_target_column",
        "randomize_active_ml_form_fields": "randomize_tensorflow_form_fields",
        "start_active_ml_training_runs": "start_tensorflow_training_runs",
    },
}

ML_FORM_FIELD_ALIASES: dict[str, str] = {
    "dataset": "dataset_id",
    "datasetId": "dataset_id",
    "epochs": "epoch_values",
    "epoch": "epoch_values",
    "epochValues": "epoch_values",
    "batch_size": "batch_sizes",
    "batchSize": "batch_sizes",
    "batchSizes": "batch_sizes",
    "learning_rate": "learning_rates",
    "learningRate": "learning_rates",
    "learningRates": "learning_rates",
    "test_size": "test_sizes",
    "testSize": "test_sizes",
    "testSizes": "test_sizes",
    "hidden_dim": "hidden_dims",
    "hiddenDim": "hidden_dims",
    "hiddenDims": "hidden_dims",
    "hidden_units": "hidden_dims",
    "hiddenUnits": "hidden_dims",
    "num_hidden_layer": "num_hidden_layers",
    "numHiddenLayer": "num_hidden_layers",
    "numHiddenLayers": "num_hidden_layers",
    "dropout": "dropouts",
    "model": "training_mode",
    "model_type": "training_mode",
    "modelType": "training_mode",
    "algorithm": "training_mode",
    "architecture": "training_mode",
    "model_architecture": "training_mode",
    "trainingMode": "training_mode",
    "targetColumn": "target_column",
    "runSweep": "run_sweep",
    "autoDistill": "auto_distill",
    "setSweepValues": "set_sweep_values",
}

TRAINING_MODE_ALIASES: dict[str, str] = {
    "neural_net": "mlp_dense",
    "neural_network": "mlp_dense",
    "neural_net_dense": "mlp_dense",
    "dense_neural_net": "mlp_dense",
    "dense_network": "mlp_dense",
    "tab_resnet": "tabresnet",
    "residual_mlp": "tabresnet",
    "wide_deep": "wide_and_deep",
    "wide_and_deep_model": "wide_and_deep",
}

DATASET_ID_ALIASES: dict[str, str] = {
    "customer_churn": "customer_churn_telco.csv",
    "customer_churn_telco": "customer_churn_telco.csv",
    "customer_churn_telco_csv": "customer_churn_telco.csv",
    "churn": "customer_churn_telco.csv",
    "fraud_detection": "fraud_detection_phishing_websites.csv",
    "fraud_detection_csv": "fraud_detection_phishing_websites.csv",
    "fraud": "fraud_detection_phishing_websites.csv",
    "fraud_detection_phishing_websites": "fraud_detection_phishing_websites.csv",
    "fraud_detection_phishing_websites_csv": "fraud_detection_phishing_websites.csv",
    "house_prices": "house_prices_ames.csv",
    "house_prices_csv": "house_prices_ames.csv",
    "house_prices_ames": "house_prices_ames.csv",
    "house_prices_ames_csv": "house_prices_ames.csv",
}


def _normalize_lookup_token(value: str) -> str:
    """Normalize freeform strings for alias lookup."""
    return (
        value.strip()
        .lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _normalize_training_mode_value(value: Any) -> Any:
    """Canonicalize common training-mode aliases."""
    if not isinstance(value, str):
        return value
    token = re.sub(r"[^a-z0-9_]+", "_", _normalize_lookup_token(value)).strip("_")
    return TRAINING_MODE_ALIASES.get(token, token or value)


def normalize_dataset_id_value(value: Any) -> Any:
    """Canonicalize known dataset aliases to dataset ids."""
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed:
        return value
    token = re.sub(r"[^a-z0-9_]+", "_", _normalize_lookup_token(trimmed)).strip("_")
    return DATASET_ID_ALIASES.get(token, trimmed)


def normalize_ml_form_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize ML patch aliases to the bridge-understood field schema."""
    normalized = dict(fields)
    for alias, canonical in ML_FORM_FIELD_ALIASES.items():
        if canonical in normalized or alias not in normalized:
            continue
        normalized[canonical] = normalized.pop(alias)

    if "training_mode" in normalized:
        normalized["training_mode"] = _normalize_training_mode_value(normalized["training_mode"])
    if "dataset_id" in normalized:
        normalized["dataset_id"] = normalize_dataset_id_value(normalized["dataset_id"])
    if "set_sweep_values" in normalized and "run_sweep" not in normalized:
        normalized["run_sweep"] = normalized["set_sweep_values"]
    return normalized


def has_explicit_sweep_intent(text: str) -> bool:
    """Return whether the user explicitly asked to enable/configure sweep."""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    patterns = [
        r"\bhyperparameter\s+sweep\b",
        r"\brun\s+sweep\b",
        r"\bsweep\s+values?\b",
        r"\bset\s+sweep\b",
        r"\bturn\s+sweep\b",
        r"\benable\s+sweep\b",
        r"\bsweep\b",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def normalize_ml_tab_actions(
    action_calls: list[dict[str, Any]],
    *,
    active_tab: str,
) -> list[dict[str, Any]]:
    """Normalize and merge ML actions for one tab while preserving serial order."""
    tool_aliases = ML_TAB_TOOL_NAME_ALIASES.get(active_tab)
    if not tool_aliases:
        return action_calls

    set_tool_name = tool_aliases["set_active_ml_form_fields"]
    change_target_tool_name = tool_aliases["change_active_ml_target_column"]
    randomize_tool_name = tool_aliases["randomize_active_ml_form_fields"]
    start_tool_name = tool_aliases["start_active_ml_training_runs"]

    normalized_actions: list[dict[str, Any]] = []
    for item in action_calls:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        rewritten_name = tool_aliases.get(name, name)
        args = item.get("args")
        normalized_args = dict(args) if isinstance(args, dict) else {}
        if rewritten_name == set_tool_name:
            patch_candidate = normalized_args.get("fields")
            patch = patch_candidate if isinstance(patch_candidate, dict) else normalized_args
            normalized_actions.append(
                {
                    "name": rewritten_name,
                    "args": {"fields": normalize_ml_form_fields(dict(patch))},
                }
            )
            continue
        normalized_actions.append({"name": rewritten_name, "args": normalized_args})

    merged: list[dict[str, Any]] = []
    setter_index: int | None = None
    target_index: int | None = None
    randomize_seen = False
    start_seen = False
    dedupe_keys: set[str] = set()

    for item in normalized_actions:
        name = item["name"]
        args = item["args"]
        if name == set_tool_name:
            if setter_index is None:
                setter_index = len(merged)
                merged.append(item)
            else:
                existing_fields = merged[setter_index].setdefault("args", {}).setdefault("fields", {})
                next_fields = args.get("fields", {})
                if isinstance(existing_fields, dict) and isinstance(next_fields, dict):
                    existing_fields.update(next_fields)
            continue
        if name == change_target_tool_name:
            if target_index is None:
                target_index = len(merged)
                merged.append(item)
            else:
                merged[target_index] = item
            continue
        if name == randomize_tool_name:
            if randomize_seen:
                continue
            randomize_seen = True
        if name == start_tool_name:
            if start_seen:
                continue
            start_seen = True

        key = json.dumps({"name": name, "args": args}, sort_keys=True, ensure_ascii=False)
        if key in dedupe_keys:
            continue
        dedupe_keys.add(key)
        merged.append(item)

    return merged


def strip_implicit_sweep_flags(
    action_calls: list[dict[str, Any]],
    *,
    active_tab: str,
    latest_user_text: str,
) -> list[dict[str, Any]]:
    """Remove model-inferred sweep enablement unless the user explicitly asked.

    This preserves explicit `False` values, which are used by deterministic
    training flows to force a non-sweep run from current UI state.
    """
    if not action_calls or has_explicit_sweep_intent(latest_user_text):
        return action_calls

    tool_aliases = ML_TAB_TOOL_NAME_ALIASES.get(active_tab)
    if not tool_aliases:
        return action_calls
    set_tool_name = tool_aliases["set_active_ml_form_fields"]

    stripped: list[dict[str, Any]] = []
    for item in action_calls:
        name = str(item.get("name") or "").strip()
        if name != set_tool_name:
            stripped.append(item)
            continue
        args = dict(item.get("args") or {})
        fields = args.get("fields")
        if not isinstance(fields, dict):
            stripped.append(item)
            continue
        next_fields = dict(fields)
        if next_fields.get("run_sweep") is True:
            next_fields.pop("run_sweep", None)
        if next_fields.get("set_sweep_values") is True:
            next_fields.pop("set_sweep_values", None)
        stripped.append({"name": name, "args": {**args, "fields": next_fields}})

    return stripped


def _normalize_dataset_id_value(value: Any) -> Any:
    """Backward-compatible alias for tests/imports."""
    return normalize_dataset_id_value(value)
