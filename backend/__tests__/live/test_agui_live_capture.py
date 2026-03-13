from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

LIVE_BASE_URL = os.getenv("AGUI_LIVE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
CAPTURE_DIR = PROJECT_ROOT / "tmp" / "agui-live-captures"
DEFAULT_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_DATASET_ID = "customer_churn_telco.csv"

CHART_TOOLS = [
    {"name": "switch_ag_ui_tab", "description": "Switch AG-UI tabs.", "parameters": {}},
    {"name": "add_chart_spec", "description": "Add a chart spec to the charts canvas.", "parameters": {}},
    {"name": "clear_charts", "description": "Clear rendered charts.", "parameters": {}},
]

PYTORCH_TOOLS = [
    {"name": "switch_ag_ui_tab", "description": "Switch AG-UI tabs.", "parameters": {}},
    {"name": "set_active_ml_form_fields", "description": "Update active ML form fields.", "parameters": {}},
    {"name": "change_active_ml_target_column", "description": "Change the active ML target column.", "parameters": {}},
    {"name": "randomize_active_ml_form_fields", "description": "Randomize active ML form fields.", "parameters": {}},
    {"name": "start_active_ml_training_runs", "description": "Start active ML training runs.", "parameters": {}},
    {"name": "set_pytorch_form_fields", "description": "Patch PyTorch form fields.", "parameters": {}},
    {"name": "change_pytorch_target_column", "description": "Change the PyTorch target column.", "parameters": {}},
    {"name": "randomize_pytorch_form_fields", "description": "Randomize PyTorch form fields.", "parameters": {}},
    {"name": "start_pytorch_training_runs", "description": "Start PyTorch training runs.", "parameters": {}},
    {"name": "train_pytorch_model", "description": "Start one PyTorch training run.", "parameters": {}},
]

TENSORFLOW_TOOLS = [
    {"name": "switch_ag_ui_tab", "description": "Switch AG-UI tabs.", "parameters": {}},
    {"name": "set_active_ml_form_fields", "description": "Update active ML form fields.", "parameters": {}},
    {"name": "change_active_ml_target_column", "description": "Change the active ML target column.", "parameters": {}},
    {"name": "randomize_active_ml_form_fields", "description": "Randomize active ML form fields.", "parameters": {}},
    {"name": "start_active_ml_training_runs", "description": "Start active ML training runs.", "parameters": {}},
    {"name": "set_tensorflow_form_fields", "description": "Patch TensorFlow form fields.", "parameters": {}},
    {"name": "change_tensorflow_target_column", "description": "Change the TensorFlow target column.", "parameters": {}},
    {"name": "randomize_tensorflow_form_fields", "description": "Randomize TensorFlow form fields.", "parameters": {}},
    {"name": "start_tensorflow_training_runs", "description": "Start TensorFlow training runs.", "parameters": {}},
    {"name": "train_tensorflow_model", "description": "Start one TensorFlow training run.", "parameters": {}},
]

AGENTIC_RESEARCH_TOOLS = [
    {"name": "switch_ag_ui_tab", "description": "Switch AG-UI tabs.", "parameters": {}},
    {"name": "add_chart_spec", "description": "Add a chart spec to the charts canvas.", "parameters": {}},
    {"name": "clear_charts", "description": "Clear rendered charts.", "parameters": {}},
    {"name": "ar-add_chart_spec", "description": "Add a chart spec to Agentic Research.", "parameters": {}},
    {"name": "ar-clear_charts", "description": "Clear Agentic Research charts.", "parameters": {}},
    {"name": "ar-set_active_dataset", "description": "Select the active research dataset.", "parameters": {}},
]


@dataclass
class LiveTurnResult:
    scenario: str
    prompt: str
    active_tab: str
    thread_id: str
    run_id: str
    request_payload: dict[str, Any]
    response_status: int
    events: list[dict[str, Any]]
    raw_lines: list[str]
    tool_call_names: list[str]
    assistant_transport_text: str
    assistant_payload: dict[str, Any] | None
    assistant_message: str
    contamination_flags: dict[str, bool]


def _backend_is_reachable() -> bool:
    try:
        response = httpx.get(f"{LIVE_BASE_URL}/health", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _build_payload(
    prompt: str,
    *,
    active_tab: str,
    thread_id: str,
    run_id: str,
    tools: list[dict[str, Any]],
    prior_messages: list[dict[str, Any]] | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
) -> dict[str, Any]:
    messages = list(prior_messages or [])
    messages.append(
        {
            "id": f"user-{run_id}",
            "role": "user",
            "content": prompt,
        }
    )

    return {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "forwardedProps": {},
        "messages": messages,
        "tools": tools,
        "context": [
            {"description": "ag_ui_active_tab", "value": json.dumps(active_tab)},
            {"description": "ag_ui_selected_model_id", "value": json.dumps(DEFAULT_MODEL_ID)},
            {"description": "agentic_research_selected_dataset_id", "value": json.dumps(dataset_id)},
        ],
    }


def _read_sse(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]], list[str]]:
    with httpx.stream(
        "POST",
        f"{LIVE_BASE_URL}/agui",
        json=payload,
        timeout=httpx.Timeout(180.0, connect=10.0),
    ) as response:
        raw_lines: list[str] = []
        events: list[dict[str, Any]] = []
        for line in response.iter_lines():
            if line is None:
                continue
            raw_lines.append(line)
            if not line.startswith("data: "):
                continue
            body = line[len("data: ") :]
            try:
                decoded = json.loads(body)
            except json.JSONDecodeError:
                decoded = {"type": "UNPARSEABLE", "raw": body}
            events.append(decoded)
        return response.status_code, events, raw_lines


def _collect_assistant_transport_text(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in events:
        if event.get("type") == "TEXT_MESSAGE_CONTENT":
            delta = event.get("delta")
            if isinstance(delta, str):
                parts.append(delta)
    return "".join(parts)


def _parse_assistant_payload(transport_text: str) -> dict[str, Any] | None:
    if not transport_text.strip():
        return None
    try:
        parsed = json.loads(transport_text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _collect_tool_call_names(events: list[dict[str, Any]]) -> list[str]:
    return [
        str(event.get("toolCallName") or "")
        for event in events
        if event.get("type") == "TOOL_CALL_START" and str(event.get("toolCallName") or "").strip()
    ]


def _collect_tool_call_args(events: list[dict[str, Any]]) -> list[str]:
    return [
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "TOOL_CALL_ARGS" and str(event.get("delta") or "").strip()
    ]


def _build_contamination_flags(assistant_message: str) -> dict[str, bool]:
    normalized = assistant_message.lower()
    return {
        "mentions_data_scientist": "data scientist" in normalized,
        "mentions_analyst": "analyst" in normalized,
        "mentions_telco_dataset": "customer_churn_telco.csv" in normalized or "telco" in normalized,
        "mentions_agentic_research_dataset": "dataset" in normalized and "research" in normalized,
        "mentions_pca": "pca" in normalized,
        "mentions_sklearn": "sklearn" in normalized,
    }


def _assistant_history_message(
    turn: LiveTurnResult,
    *,
    message_id: str,
) -> dict[str, Any]:
    content = turn.assistant_transport_text or turn.assistant_message
    return {
        "id": message_id,
        "role": "assistant",
        "content": content,
    }


def _write_capture(result: LiveTurnResult) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    capture_path = CAPTURE_DIR / f"{timestamp}-{result.scenario}-{result.run_id}.json"
    capture_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return capture_path


def _run_turn(
    scenario: str,
    prompt: str,
    *,
    active_tab: str,
    thread_id: str,
    run_id: str,
    tools: list[dict[str, Any]],
    prior_messages: list[dict[str, Any]] | None = None,
) -> tuple[LiveTurnResult, Path]:
    payload = _build_payload(
        prompt,
        active_tab=active_tab,
        thread_id=thread_id,
        run_id=run_id,
        tools=tools,
        prior_messages=prior_messages,
    )
    status_code, events, raw_lines = _read_sse(payload)
    transport_text = _collect_assistant_transport_text(events)
    assistant_payload = _parse_assistant_payload(transport_text)
    assistant_message = ""
    if assistant_payload and isinstance(assistant_payload.get("message"), str):
        assistant_message = assistant_payload["message"]
    else:
        assistant_message = transport_text

    result = LiveTurnResult(
        scenario=scenario,
        prompt=prompt,
        active_tab=active_tab,
        thread_id=thread_id,
        run_id=run_id,
        request_payload=payload,
        response_status=status_code,
        events=events,
        raw_lines=raw_lines,
        tool_call_names=_collect_tool_call_names(events),
        assistant_transport_text=transport_text,
        assistant_payload=assistant_payload,
        assistant_message=assistant_message,
        contamination_flags=_build_contamination_flags(assistant_message),
    )
    return result, _write_capture(result)


def _assert_protocol_shape(turn: LiveTurnResult) -> None:
    event_types = [event.get("type") for event in turn.events]
    assert turn.response_status == 200
    assert "RUN_STARTED" in event_types
    assert "TEXT_MESSAGE_CONTENT" in event_types
    assert event_types[-1] == "RUN_FINISHED"


def _assert_no_cross_surface_contamination(turn: LiveTurnResult) -> None:
    assert turn.contamination_flags["mentions_data_scientist"] is False
    assert turn.contamination_flags["mentions_analyst"] is False
    assert turn.contamination_flags["mentions_telco_dataset"] is False
    assert turn.contamination_flags["mentions_sklearn"] is False


def _assert_tool_calls_are_tab_scoped(
    turn: LiveTurnResult,
    *,
    allowed_tool_names: set[str],
) -> None:
    assert set(turn.tool_call_names).issubset(allowed_tool_names)
    assert len(turn.tool_call_names) == len(set(turn.tool_call_names))


def _run_serial_sequence(
    scenario: str,
    *,
    active_tab: str,
    tools: list[dict[str, Any]],
    prompts: list[str],
) -> list[tuple[LiveTurnResult, Path]]:
    prior_messages: list[dict[str, Any]] = []
    thread_id = f"live-{scenario}"
    results: list[tuple[LiveTurnResult, Path]] = []
    for index, prompt in enumerate(prompts, start=1):
        turn, capture_path = _run_turn(
            scenario,
            prompt,
            active_tab=active_tab,
            thread_id=thread_id,
            run_id=f"turn-{index}",
            tools=tools,
            prior_messages=prior_messages,
        )
        results.append((turn, capture_path))
        prior_messages.extend(
            [
                {"id": f"user-turn-{index}", "role": "user", "content": prompt},
                _assistant_history_message(turn, message_id=f"assistant-turn-{index}"),
            ]
        )
    return results


@pytest.mark.skipif(not _backend_is_reachable(), reason="Live backend is not reachable on AGUI_LIVE_BASE_URL.")
def test_live_capture_agentic_research_then_charts(capfd: pytest.CaptureFixture[str]) -> None:
    thread_id = "live-seq-ar-to-charts"

    turn_one, turn_one_path = _run_turn(
        "agentic-research-then-charts",
        "run pca",
        active_tab="agentic-research",
        thread_id=thread_id,
        run_id="turn-1",
        tools=AGENTIC_RESEARCH_TOOLS,
    )
    _assert_protocol_shape(turn_one)

    turn_two_history = [
        {"id": "user-turn-1", "role": "user", "content": turn_one.prompt},
        _assistant_history_message(turn_one, message_id="assistant-turn-1"),
    ]
    turn_two, turn_two_path = _run_turn(
        "agentic-research-then-charts",
        "make a scatter chart comparing bitcoin and ethereum returns over the last 30 days",
        active_tab="charts",
        thread_id=thread_id,
        run_id="turn-2",
        tools=CHART_TOOLS,
        prior_messages=turn_two_history,
    )
    _assert_protocol_shape(turn_two)

    print(
        json.dumps(
            {
                "scenario": "agentic-research-then-charts",
                "turn_one_capture": str(turn_one_path),
                "turn_one_tools": turn_one.tool_call_names,
                "turn_one_flags": turn_one.contamination_flags,
                "turn_one_message_preview": turn_one.assistant_message[:300],
                "turn_two_capture": str(turn_two_path),
                "turn_two_tools": turn_two.tool_call_names,
                "turn_two_flags": turn_two.contamination_flags,
                "turn_two_message_preview": turn_two.assistant_message[:300],
            },
            indent=2,
        )
    )

    captured = capfd.readouterr()
    assert "agentic-research-then-charts" in captured.out


@pytest.mark.skipif(not _backend_is_reachable(), reason="Live backend is not reachable on AGUI_LIVE_BASE_URL.")
def test_live_capture_charts_then_charts(capfd: pytest.CaptureFixture[str]) -> None:
    thread_id = "live-seq-charts-to-charts"

    turn_one, turn_one_path = _run_turn(
        "charts-then-charts",
        "show a line chart of Manhattan vs London vs Paris average rent since 2000 as a share of average salary",
        active_tab="charts",
        thread_id=thread_id,
        run_id="turn-1",
        tools=CHART_TOOLS,
    )
    _assert_protocol_shape(turn_one)

    turn_two_history = [
        {"id": "user-turn-1", "role": "user", "content": turn_one.prompt},
        _assistant_history_message(turn_one, message_id="assistant-turn-1"),
    ]
    turn_two, turn_two_path = _run_turn(
        "charts-then-charts",
        "make a scatter chart comparing bitcoin and ethereum returns over the last 30 days",
        active_tab="charts",
        thread_id=thread_id,
        run_id="turn-2",
        tools=CHART_TOOLS,
        prior_messages=turn_two_history,
    )
    _assert_protocol_shape(turn_two)

    print(
        json.dumps(
            {
                "scenario": "charts-then-charts",
                "turn_one_capture": str(turn_one_path),
                "turn_one_tools": turn_one.tool_call_names,
                "turn_one_flags": turn_one.contamination_flags,
                "turn_one_message_preview": turn_one.assistant_message[:300],
                "turn_two_capture": str(turn_two_path),
                "turn_two_tools": turn_two.tool_call_names,
                "turn_two_flags": turn_two.contamination_flags,
                "turn_two_message_preview": turn_two.assistant_message[:300],
            },
            indent=2,
        )
    )

    captured = capfd.readouterr()
    assert "charts-then-charts" in captured.out


@pytest.mark.skipif(not _backend_is_reachable(), reason="Live backend is not reachable on AGUI_LIVE_BASE_URL.")
def test_live_capture_agentic_research_dataset_switch_only(capfd: pytest.CaptureFixture[str]) -> None:
    turn, capture_path = _run_turn(
        "agentic-research-dataset-switch-only",
        "Change the dataset to fraud detection.",
        active_tab="agentic-research",
        thread_id="live-agentic-research-dataset-switch-only",
        run_id="turn-1",
        tools=AGENTIC_RESEARCH_TOOLS,
    )
    _assert_protocol_shape(turn)

    tool_args = _collect_tool_call_args(turn.events)

    print(
        json.dumps(
            {
                "scenario": "agentic-research-dataset-switch-only",
                "capture": str(capture_path),
                "tools": turn.tool_call_names,
                "tool_args": tool_args,
                "message_preview": turn.assistant_message[:300],
                "flags": turn.contamination_flags,
            },
            indent=2,
        )
    )

    assert turn.tool_call_names == ["ar-set_active_dataset"]
    assert turn.contamination_flags["mentions_data_scientist"] is False
    assert turn.contamination_flags["mentions_analyst"] is False

    captured = capfd.readouterr()
    assert "agentic-research-dataset-switch-only" in captured.out


@pytest.mark.skipif(not _backend_is_reachable(), reason="Live backend is not reachable on AGUI_LIVE_BASE_URL.")
def test_live_capture_pytorch_serial_commands(capfd: pytest.CaptureFixture[str]) -> None:
    prompts = [
        "Use the fraud detection dataset. Switch the training algorithm from neural net to TabResNet. Set batch sizes to 33 and 40, hidden dims to 64 and 96, and dropouts to 0.1 and 0.2.",
        "Change from customer churn to fraud detection. Set task to classification, choose a different target column, set test sizes to 0.2 and 0.3, and start training runs.",
        "Randomize PyTorch form fields with one value each, keep the current algorithm, and start training runs.",
        "Switch the algorithm to calibrated classifier and set sweep values on.",
    ]
    results = _run_serial_sequence(
        "pytorch-serial-commands",
        active_tab="pytorch",
        tools=PYTORCH_TOOLS,
        prompts=prompts,
    )

    allowed_tool_names = {tool["name"] for tool in PYTORCH_TOOLS}
    summary: list[dict[str, Any]] = []
    for turn, capture_path in results:
        _assert_protocol_shape(turn)
        _assert_no_cross_surface_contamination(turn)
        _assert_tool_calls_are_tab_scoped(turn, allowed_tool_names=allowed_tool_names)
        assert turn.tool_call_names
        summary.append(
            {
                "run_id": turn.run_id,
                "capture": str(capture_path),
                "tools": turn.tool_call_names,
                "message_preview": turn.assistant_message[:240],
            }
        )

    assert any("start_pytorch_training_runs" in turn.tool_call_names for turn, _ in results)
    assert any(
        {"set_pytorch_form_fields", "set_active_ml_form_fields"} & set(turn.tool_call_names)
        for turn, _ in results
    )

    print(json.dumps({"scenario": "pytorch-serial-commands", "turns": summary}, indent=2))
    captured = capfd.readouterr()
    assert "pytorch-serial-commands" in captured.out


@pytest.mark.skipif(not _backend_is_reachable(), reason="Live backend is not reachable on AGUI_LIVE_BASE_URL.")
def test_live_capture_tensorflow_serial_commands(capfd: pytest.CaptureFixture[str]) -> None:
    prompts = [
        "Use the house prices dataset. Switch the training algorithm from neural net to wide and deep. Set test sizes to 0.25 and 0.3, batch sizes to 32 and 64, and hidden dims to 128 and 256.",
        "Change from customer churn to house prices. Set task to regression, set epochs to 20 and 40, and start training runs.",
        "Randomize TensorFlow form fields with one value each, and keep the current algorithm.",
        "Switch the algorithm to entity embeddings, and turn auto-distill on.",
    ]
    results = _run_serial_sequence(
        "tensorflow-serial-commands",
        active_tab="tensorflow",
        tools=TENSORFLOW_TOOLS,
        prompts=prompts,
    )

    allowed_tool_names = {tool["name"] for tool in TENSORFLOW_TOOLS}
    summary: list[dict[str, Any]] = []
    for turn, capture_path in results:
        _assert_protocol_shape(turn)
        _assert_no_cross_surface_contamination(turn)
        _assert_tool_calls_are_tab_scoped(turn, allowed_tool_names=allowed_tool_names)
        assert turn.tool_call_names
        summary.append(
            {
                "run_id": turn.run_id,
                "capture": str(capture_path),
                "tools": turn.tool_call_names,
                "message_preview": turn.assistant_message[:240],
            }
        )

    assert any("start_tensorflow_training_runs" in turn.tool_call_names for turn, _ in results)
    assert any(
        {"set_tensorflow_form_fields", "set_active_ml_form_fields"} & set(turn.tool_call_names)
        for turn, _ in results
    )

    print(json.dumps({"scenario": "tensorflow-serial-commands", "turns": summary}, indent=2))
    captured = capfd.readouterr()
    assert "tensorflow-serial-commands" in captured.out
