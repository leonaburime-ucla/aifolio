import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = PYTHON_ROOT.parent
for path in (str(PYTHON_ROOT), str(AI_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import ml_data
from langgraph_agents import data_scientist
svc = data_scientist._service


class _Result:
    def __init__(self, content=None, tool_calls=None, additional_kwargs=None):
        self.content = content
        self.tool_calls = tool_calls
        self.additional_kwargs = additional_kwargs or {}


def test_normalize_tool_args_preserves_invalid_json_as_raw():
    assert data_scientist._normalize_tool_args("{not-json}") == {"_raw": "{not-json}"}


def test_build_gemini_agent_binds_tools_when_provided(monkeypatch):
    captured = {}

    class _LLM:
        def __init__(self, model, temperature):
            captured["model"] = model
            captured["temperature"] = temperature

        def bind_tools(self, tools):
            captured["tools"] = tools
            return {"bound_tools": tools}

    monkeypatch.setattr(svc, "ChatGoogleGenerativeAI", _LLM)
    agent = data_scientist.build_gemini_agent("m1", tools=["tool"])
    assert agent == {"bound_tools": ["tool"]}
    assert captured == {"model": "m1", "temperature": 0.3, "tools": ["tool"]}


def test_get_sklearn_tools_returns_router():
    assert data_scientist.get_sklearn_tools() == [data_scientist.sklearn_tool_router]


def test_get_tools_catalog_uses_sklearn_metadata(monkeypatch):
    monkeypatch.setattr(svc.sklearn_tools, "list_available_tools", lambda: ["pca_transform"])
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: {"pca_transform": {"params": []}})
    catalog = data_scientist.get_tools_catalog()
    assert catalog == {
        "tool_router": "sklearn_tool_router",
        "available_tools": ["pca_transform"],
        "tool_schemas": {"pca_transform": {"params": []}},
    }


def test_get_data_scientist_agent_caches_by_model(monkeypatch):
    built = []
    svc._DS_AGENT_CACHE.clear()
    monkeypatch.setattr(svc, "build_gemini_agent", lambda model_id, tools: built.append((model_id, tools)) or {"model": model_id})
    first = data_scientist.get_data_scientist_agent("m1")
    second = data_scientist.get_data_scientist_agent("m1")
    assert first is second
    assert built == [("m1", data_scientist.get_sklearn_tools())]


def test_data_scientist_agent_placeholder_raises():
    with pytest.raises(NotImplementedError, match="not implemented"):
        data_scientist.data_scientist_agent()


def test_run_data_scientist_invokes_cached_agent(monkeypatch):
    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(content=f"ran:{messages[0].content}")

    monkeypatch.setattr(svc, "get_data_scientist_agent", lambda model_id=None: _LLM())
    assert data_scientist.run_data_scientist("hello", model_id="m1") == "ran:hello"


def test_demo_pca_payload_and_transform(monkeypatch):
    payload = data_scientist.get_demo_pca_payload()
    assert payload["message"] == data_scientist.DEFAULT_PCA_MESSAGE
    assert len(payload["data"]) == 6
    assert payload["feature_names"] == ["feature_a", "feature_b", "feature_c", "feature_d"]

    captured = {}
    monkeypatch.setattr(svc, "run_data_scientist_tool", lambda **kwargs: captured.update(kwargs) or {"status": "ok"})
    result = data_scientist.run_demo_pca_transform(n_components=3)
    assert result == {"status": "ok"}
    assert captured["tool_name"] == "pca_transform"
    assert captured["tool_args"]["n_components"] == 3


def test_build_tool_call_prompt_includes_router_and_args():
    prompt = data_scientist._build_tool_call_prompt(
        "Run PCA",
        "pca_transform",
        {"n_components": 2},
        {"tool_router": "sklearn_tool_router", "available_tools": ["pca_transform"], "tool_schemas": {"pca_transform": {"params": []}}},
    )
    assert "Tool router: sklearn_tool_router" in prompt
    assert '"tool_name": "pca_transform"' in prompt
    assert '"n_components": 2' in prompt


def test_select_tool_call_prefers_router_or_named_tool():
    tool_calls = [{"name": "other"}, {"tool": "sklearn_tool_router"}]
    assert data_scientist._select_tool_call(tool_calls, "pca_transform") == {"tool": "sklearn_tool_router"}
    assert data_scientist._select_tool_call([{"name": "pca_transform"}], "pca_transform") == {"name": "pca_transform"}
    assert data_scientist._select_tool_call([], "pca_transform") is None


def test_stringify_content_parts_and_json_helpers():
    assert data_scientist._stringify_content_parts([{"text": "a"}, {"content": "b"}, "c"]) == "a\nb\nc"
    assert data_scientist._strip_json_fence("```json\n{\"a\": 1}\n```") == '{"a": 1}'
    assert data_scientist._json_object_slice("prefix {\"a\": 1} suffix") == '{"a": 1}'
    assert data_scientist._json_object_slice("plain") == ""


def test_extract_tool_calls_prefers_attribute_then_additional_kwargs():
    assert data_scientist._extract_tool_calls(_Result(tool_calls=[{"name": "x"}])) == [{"name": "x"}]
    assert data_scientist._extract_tool_calls(_Result(additional_kwargs={"tool_calls": [{"name": "y"}]})) == [{"name": "y"}]
    assert data_scientist._extract_tool_calls(_Result()) == []


def test_validate_tool_plan_filters_unknown_tools():
    plan = {
        "summary": "mixed",
        "tool_calls": [
            {"tool_name": "pca_transform", "tool_args": {}},
            {"tool_name": "not_a_real_tool", "tool_args": {}},
        ],
    }

    validated = data_scientist._validate_tool_plan(plan, ["pca_transform"])
    assert validated["tool_calls"] == [{"tool_name": "pca_transform", "tool_args": {}}]


def test_sklearn_tool_router_returns_unknown_tool_error():
    result = data_scientist.sklearn_tool_router("not_a_real_tool", {})
    assert result["status"] == "error"
    assert "Unknown tool" in result["error"]


def test_sklearn_tool_router_handles_missing_attr_not_implemented_and_success(monkeypatch):
    monkeypatch.setattr(svc.sklearn_tools, "list_available_tools", lambda: ["ghost", "implemented", "pending", "broken"])
    monkeypatch.setattr(svc.sklearn_tools, "implemented", lambda **kwargs: {"ok": kwargs}, raising=False)
    monkeypatch.setattr(svc.sklearn_tools, "pending", lambda **kwargs: (_ for _ in ()).throw(NotImplementedError("later")), raising=False)
    monkeypatch.setattr(svc.sklearn_tools, "broken", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")), raising=False)

    assert data_scientist.sklearn_tool_router("ghost", {}) == {"status": "error", "error": "Tool not found: ghost"}
    assert data_scientist.sklearn_tool_router("implemented", {"x": 1}) == {"status": "ok", "result": {"ok": {"x": 1}}}
    assert data_scientist.sklearn_tool_router("pending", {}) == {"status": "error", "error": "later", "tool": "pending"}
    broken = data_scientist.sklearn_tool_router("broken", {})
    assert broken["status"] == "error"
    assert broken["tool"] == "broken"


def test_run_data_scientist_tool_returns_error_when_llm_emits_no_matching_call(monkeypatch):
    class _LLM:
        def invoke(self, messages):
            return _Result(content="no tool calls", tool_calls=[{"name": "other_tool"}])

    monkeypatch.setattr(svc, "get_data_scientist_agent", lambda model_id=None: _LLM())
    monkeypatch.setattr(
        svc,
        "get_tools_catalog",
        lambda: {"tool_router": "sklearn_tool_router", "available_tools": ["pca_transform"], "tool_schemas": {}},
    )

    result = data_scientist.run_data_scientist_tool("Run PCA", "pca_transform", {"n_components": 2})
    assert result["status"] == "error"
    assert result["tool_calls"] == [{"name": "other_tool"}]


def test_run_data_scientist_tool_routes_through_router(monkeypatch):
    class _LLM:
        def invoke(self, messages):
            return _Result(tool_calls=[{"name": "sklearn_tool_router", "args": {"tool_name": "pca_transform", "tool_args": {"n_components": 2}}}])

    monkeypatch.setattr(svc, "get_data_scientist_agent", lambda model_id=None: _LLM())
    monkeypatch.setattr(
        svc,
        "get_tools_catalog",
        lambda: {"tool_router": "sklearn_tool_router", "available_tools": ["pca_transform"], "tool_schemas": {}},
    )
    monkeypatch.setattr(svc, "sklearn_tool_router", lambda tool_name, tool_args: {"status": "ok", "result": {"name": tool_name, "args": tool_args}})

    result = data_scientist.run_data_scientist_tool("Run PCA", "pca_transform", {"n_components": 2})
    assert result["status"] == "ok"
    assert result["result"] == {"name": "pca_transform", "args": {"n_components": 2}}


def test_run_data_scientist_tool_executes_direct_tool_when_model_ignores_router(monkeypatch):
    class _LLM:
        def invoke(self, messages):
            return _Result(tool_calls=[{"name": "pca_transform", "args": '{"n_components": 3}'}])

    monkeypatch.setattr(svc, "get_data_scientist_agent", lambda model_id=None: _LLM())
    monkeypatch.setattr(
        svc,
        "get_tools_catalog",
        lambda: {"tool_router": "sklearn_tool_router", "available_tools": ["pca_transform"], "tool_schemas": {}},
    )
    monkeypatch.setattr(svc.sklearn_tools, "pca_transform", lambda **kwargs: {"n_components": kwargs["n_components"]})

    result = data_scientist.run_data_scientist_tool("Run PCA", "pca_transform", {"n_components": 2})
    assert result["status"] == "ok"
    assert result["tool_args"] == {"n_components": 3}
    assert result["result"] == {"n_components": 3}


def test_parse_json_response_extracts_fenced_json():
    raw = """```json
    {"summary":"ok","tool_calls":[]}
    ```"""

    parsed = data_scientist._parse_json_response(raw)
    assert parsed == {"summary": "ok", "tool_calls": []}


def test_parse_json_response_handles_lists_embedded_json_and_invalid_text():
    parsed = data_scientist._parse_json_response([{"text": "prefix "}, {"content": '{"summary":"ok"}'}])
    assert parsed == {"summary": "ok"}
    assert data_scientist._parse_json_response("plain text") == {}


def test_load_dataset_rows_rejects_unknown_suffix(tmp_path):
    path = tmp_path / "dataset.unsupported"
    path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        data_scientist._load_dataset_rows(path)


def test_resolve_tool_plan_repairs_invalid_tool_names(monkeypatch):
    llm = SimpleNamespace()
    responses = iter(
        [
            '{"summary":"draft","tool_calls":[{"tool_name":"bad_tool","tool_args":{},"chart_kind":"pca"}]}',
            '{"summary":"repaired","tool_calls":[{"tool_name":"pca_transform","tool_args":{"n_components":2},"chart_kind":"pca"}]}',
        ]
    )
    llm.invoke = lambda _messages: SimpleNamespace(content=next(responses))
    monkeypatch.setattr(data_scientist._service, "get_data_scientist_agent", lambda model_id=None: llm)

    plan = data_scientist._resolve_tool_plan(
        message="Run PCA",
        tools_schema=[{"name": "pca_transform", "params": [], "doc": "Principal components"}],
        dataset={"columns": ["feature_a"], "targetColumn": None, "task": "unsupervised"},
        model_id="fake-model",
    )

    assert plan["summary"] == "repaired"
    assert plan["tool_calls"][0]["tool_name"] == "pca_transform"


def test_load_dataset_rows_dispatches_spreadsheet_suffix(monkeypatch, tmp_path):
    monkeypatch.setattr(
        data_scientist.ds_datasets,
        "DATASET_ROW_LOADERS",
        {
            ".xlsx": lambda _path: [{"kind": "xlsx"}],
            ".xls": lambda _path: [{"kind": "xls"}],
        },
    )

    assert data_scientist._load_dataset_rows(tmp_path / "sheet.xlsx") == [{"kind": "xlsx"}]
    assert data_scientist._load_dataset_rows(tmp_path / "sheet.xls") == [{"kind": "xls"}]


def test_load_sample_dataset_handles_missing_entry_file_and_success(monkeypatch, tmp_path):
    monkeypatch.setattr(svc, "_resolve_dataset_entry", lambda dataset_id: None)
    assert data_scientist.load_sample_dataset("missing") == {"status": "error", "error": "Dataset not found."}

    monkeypatch.setattr(svc, "_resolve_dataset_entry", lambda dataset_id: {"id": dataset_id, "metadata": {"files": {}}})
    assert data_scientist.load_sample_dataset("d1") == {"status": "error", "error": "Dataset file missing."}

    monkeypatch.setattr(
        svc,
        "_resolve_dataset_entry",
        lambda dataset_id: {"id": dataset_id, "metadata": {"files": {"data": "data.csv"}}},
    )
    monkeypatch.setattr(svc, "SAMPLE_DATA_DIR", tmp_path)
    assert data_scientist.load_sample_dataset("d1") == {"status": "error", "error": "Dataset file not found."}

    path = tmp_path / "data.csv"
    path.write_text("feature,target\n1,yes\n", encoding="utf-8")
    monkeypatch.setattr(svc, "_load_dataset_rows", lambda file_path: [{"feature": 1, "target": "yes"}])
    result = data_scientist.load_sample_dataset("d1")
    assert result["status"] == "ok"
    assert result["columns"] == ["feature", "target"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        (" 3.5 ", 3.5),
        ("bad", None),
        (7, 7.0),
    ],
)
def test_coerce_float_handles_mixed_values(value, expected):
    assert data_scientist._coerce_float(value) == expected


def test_try_parse_date_supports_multiple_formats_and_invalid_text():
    assert data_scientist._try_parse_date("2024-01-15").year == 2024
    assert data_scientist._try_parse_date("15/01/2024").day == 15
    assert data_scientist._try_parse_date("2024-01-15T10:00:00Z").hour == 10
    assert data_scientist._try_parse_date("not-a-date") is None


def test_looks_like_id_column_matches_expected_aliases():
    assert data_scientist._looks_like_id_column("customerID") is True
    assert data_scientist._looks_like_id_column("recordid") is True
    assert data_scientist._looks_like_id_column("feature_name") is False


def test_fallback_tool_mapping_and_alias_resolution_helpers():
    tool_calls = data_scientist._build_fallback_tool_calls("run pca and kmeans and pca")
    assert [call["tool_name"] for call in tool_calls] == ["pca_transform", "kmeans_clustering"]
    assert [call["chart_kind"] for call in tool_calls] == ["pca", "clustering"]

    assert data_scientist._infer_chart_kind_for_tool("pls_regression") == "plsr"
    assert data_scientist._infer_chart_kind_for_tool("naive_bayes_classification") == "classification"
    assert data_scientist._infer_chart_kind_for_tool("linear_regression") == "regression"

    assert data_scientist._resolve_tool_name("gmm", None) == "gaussian_mixture_clustering"
    assert data_scientist._resolve_tool_name(None, "pca") == "pca_transform"
    assert data_scientist._resolve_tool_name(None, "plsr") == "pls_regression"
    assert data_scientist._resolve_tool_name(None, "other") is None


def test_build_numeric_matrix_handles_empty_rows_and_unsupervised_mode():
    assert data_scientist._build_numeric_matrix([]) == ([], [], [])

    matrix, targets, feature_names = data_scientist._build_numeric_matrix(
        [{"feature": 1, "category": "A"}, {"feature": 2, "category": "B"}],
        target_column=None,
    )
    assert len(matrix) == 2
    assert targets == []
    assert "feature" in feature_names


def test_build_numeric_matrix_drops_high_cardinality_and_encodes_string_targets():
    rows = [
        {"ID": 1, "Code": "A1", "Category": "A", "Feature": 1.0, "Target": "yes"},
        {"ID": 2, "Code": "A2", "Category": "A", "Feature": 2.0, "Target": "no"},
        {"ID": 3, "Code": "A3", "Category": "B", "Feature": 3.0, "Target": "yes"},
    ]
    matrix, targets, feature_names = data_scientist._build_numeric_matrix(
        rows,
        target_column="Target",
        preprocessing={"highCardinalityThreshold": 2},
    )
    assert len(matrix) == 3
    assert set(targets) == {0.0, 1.0}
    assert "Feature" in feature_names
    assert all(not name.startswith("Code=") for name in feature_names)
    assert all("ID" not in name for name in feature_names)


def test_build_numeric_matrix_imputes_non_finite_values():
    matrix, targets, feature_names = data_scientist._build_numeric_matrix(
        [
            {"Feature": "nan", "Target": 1},
            {"Feature": "2.0", "Target": 0},
        ],
        target_column="Target",
    )
    assert matrix[0][0] == 2.0
    assert targets == [1.0, 0.0]


def test_chart_builders_cover_empty_and_optional_paths():
    assert data_scientist._build_loadings_chart("PCA", [], ["x"]) is None
    loadings = data_scientist._build_loadings_chart("PCA Loadings", [[1.0, 2.0]], ["feature"], "desc")
    assert loadings["description"] == "desc"
    assert loadings["data"][0]["pc2"] == 2.0

    regression = data_scientist._build_regression_chart([0.1, 0.2], [1.0], "Predictions")
    assert regression["data"] == [{"id": "pred-1", "actual": 1.0, "predicted": 0.1}]

    assert data_scientist._transpose_components([]) == []
    assert data_scientist._transpose_components([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]

    assert data_scientist._build_feature_importance_chart("Importance", [], []) is None
    feature_chart = data_scientist._build_feature_importance_chart("Importance", ["a", "b"], [0.5], "desc")
    assert feature_chart["description"] == "desc"
    assert feature_chart["data"][1]["importance"] == 0

    assert data_scientist._build_embedding_chart("Embed", []) is None
    embedding = data_scientist._build_embedding_chart("Embed", [[1.0, 2.0], [3.0]], labels=[1], description="desc")
    assert embedding["description"] == "desc"
    assert embedding["data"][0]["cluster"] == 1
    assert embedding["data"][1]["y"] == 0

    assert data_scientist._build_cluster_distribution_chart("Clusters", []) is None
    cluster_chart = data_scientist._build_cluster_distribution_chart("Clusters", [-1, 1, 1], "desc")
    assert cluster_chart["description"] == "desc"
    assert cluster_chart["data"][0]["cluster"] == "Noise"


def test_planner_helpers_delegate_to_planner_module(monkeypatch):
    monkeypatch.setattr(svc.ds_planner, "format_conversation_history", lambda history: "history-text")
    assert data_scientist._format_ds_conversation_history([{"role": "user", "content": "hi"}]) == "history-text"

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(content="planner-response")

    monkeypatch.setattr(svc, "get_data_scientist_agent", lambda model_id=None: _LLM())
    assert data_scientist._invoke_planner_prompt("prompt", "m1") == "planner-response"

    monkeypatch.setattr(svc.ds_planner, "build_plan_prompt", lambda *args: "plan-prompt")
    monkeypatch.setattr(svc.ds_planner, "parse_planner_response", lambda raw, parser, **kwargs: {"raw": raw, "stage": kwargs["stage"]})
    monkeypatch.setattr(svc, "_invoke_planner_prompt", lambda prompt, model_id: f"invoked:{prompt}:{model_id}")
    assert data_scientist._plan_tools_with_llm("msg", [{"name": "pca"}], {"columns": []}, "m1") == {
        "raw": "invoked:plan-prompt:m1",
        "stage": "plan",
    }

    monkeypatch.setattr(svc.ds_planner, "build_repair_prompt", lambda *args: "repair-prompt")
    assert data_scientist._repair_tool_plan({"summary": "bad"}, "msg", [{"name": "pca"}], "m1") == {
        "raw": "invoked:repair-prompt:m1",
        "stage": "repair",
    }


def test_run_data_scientist_analysis_returns_not_found_for_missing_dataset(monkeypatch):
    monkeypatch.setattr(ml_data, "load_ml_dataset", lambda dataset_id, row_limit=None: {"status": "error"})
    result = data_scientist.run_data_scientist_analysis("analyze", "missing")
    assert result == {"message": "Dataset not found.", "chartSpec": None}


def test_run_data_scientist_analysis_returns_debug_when_no_plan_and_no_fallback(monkeypatch):
    monkeypatch.setattr(
        ml_data,
        "load_ml_dataset",
        lambda dataset_id, row_limit=None: {
            "status": "ok",
            "dataset": {"label": "Demo"},
            "rows": [{"feature": 1.0}],
            "columns": ["feature"],
        },
    )
    monkeypatch.setattr(svc, "_resolve_tool_plan", lambda *args, **kwargs: {"summary": "Need more detail.", "tool_calls": [], "planner_error": "bad-json"})
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: [])

    result = data_scientist.run_data_scientist_analysis("show me something", "demo")
    assert result == {"message": "Need more detail.", "chartSpec": None, "debug": "bad-json"}


def test_run_data_scientist_analysis_uses_fallback_dispatch_and_attaches_meta(monkeypatch):
    monkeypatch.setattr(
        ml_data,
        "load_ml_dataset",
        lambda dataset_id, row_limit=None: {
            "status": "ok",
            "dataset": {"label": "Demo Dataset"},
            "rows": [{"feature": 1.0}, {"feature": 2.0}],
            "columns": ["feature"],
        },
    )
    monkeypatch.setattr(svc, "_resolve_tool_plan", lambda *args, **kwargs: {"summary": "none", "tool_calls": []})
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: [])
    monkeypatch.setattr(
        svc.sklearn_tools,
        "pca_transform",
        lambda **kwargs: {"components": [[1.0, 0.0], [0.0, 1.0]], "explained_variance_ratio": [0.6, 0.3]},
    )

    result = data_scientist.run_data_scientist_analysis("please run pca", "demo")
    assert result["message"].startswith("Running requested analysis from deterministic fallback mapping.")
    assert result["chartSpec"][0]["meta"]["datasetLabel"] == "Demo Dataset"
    assert result["chartSpec"][0]["meta"]["chartKind"] == "loadings_scatter"


@pytest.mark.parametrize(
    ("tool_name", "expected_note"),
    [
        ("pls_regression", "Target column is missing for PLSR."),
        ("linear_regression", "Target column is missing for regression."),
        ("logistic_regression", "Target column is missing for classification."),
        ("random_forest_classification", "Target column is missing for classification."),
        ("gradient_boosting_classification", "Target column is missing for classification."),
        ("knn_classification", "Target column is missing for classification."),
        ("naive_bayes_classification", "Target column is missing for classification."),
    ],
)
def test_run_data_scientist_analysis_reports_missing_target_for_supervised_tools(monkeypatch, tool_name, expected_note):
    monkeypatch.setattr(
        ml_data,
        "load_ml_dataset",
        lambda dataset_id, row_limit=None: {
            "status": "ok",
            "dataset": {"label": "Demo"},
            "rows": [{"feature": 1.0}, {"feature": 2.0}],
            "columns": ["feature"],
        },
    )
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: [])

    result = data_scientist.run_data_scientist_analysis(
        message=f"run {tool_name}",
        dataset_id="demo",
        planned_tool_calls=[{"tool_name": tool_name, "tool_args": {}, "chart_kind": "none"}],
    )
    assert result["chartSpec"] is None
    assert expected_note in result["nonChartResponse"]["notes"]


def test_run_data_scientist_analysis_handles_alias_and_unwired_tool(monkeypatch):
    monkeypatch.setattr(
        ml_data,
        "load_ml_dataset",
        lambda dataset_id, row_limit=None: {
            "status": "ok",
            "dataset": {"label": "Demo"},
            "rows": [{"feature": 1.0}, {"feature": 2.0}],
            "columns": ["feature"],
        },
    )
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: [])
    monkeypatch.setattr(
        svc.sklearn_tools,
        "pca_transform",
        lambda **kwargs: {"components": [[1.0, 0.0], [0.0, 1.0]], "explained_variance_ratio": [0.7, 0.2], "r2_score": 0.9},
    )

    result = data_scientist.run_data_scientist_analysis(
        message="run aliases",
        dataset_id="demo",
        planned_tool_calls=[
            {"tool_name": None, "tool_args": {}, "chart_kind": "pca"},
            {"tool_name": "mystery_tool", "tool_args": {}, "chart_kind": "none"},
        ],
    )
    assert result["chartSpec"][0]["meta"]["chartKind"] == "loadings_scatter"
    assert "Tool mystery_tool is not wired yet." in result["nonChartResponse"]["notes"]
    assert "pca_transform r2: 0.9" in result["nonChartResponse"]["metrics"]


def test_run_data_scientist_analysis_samples_large_tsne_inputs(monkeypatch):
    rows = [{"feature": float(index), "target": index % 2} for index in range(600)]
    monkeypatch.setattr(
        ml_data,
        "load_ml_dataset",
        lambda dataset_id, row_limit=None: {
            "status": "ok",
            "dataset": {"label": "Large", "targetColumn": "target"},
            "rows": rows,
            "columns": ["feature", "target"],
        },
    )
    monkeypatch.setattr(svc.sklearn_tools, "get_tools_schema", lambda: [])
    monkeypatch.setattr(
        svc.sklearn_tools,
        "tsne_embedding",
        lambda **kwargs: {
            "embedding": [[0.0, 1.0], [1.0, 0.0]],
            "perplexity": kwargs["perplexity"],
        },
    )

    result = data_scientist.run_data_scientist_analysis(
        message="run tsne",
        dataset_id="large",
        planned_tool_calls=[{"tool_name": "tsne_embedding", "tool_args": {}, "chart_kind": "embedding"}],
    )
    assert "sampled 500 of 600 rows" in result["chartSpec"][0]["description"]
