import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

import shared.google_gemini as gg


def test_ensure_google_api_key_in_env_uses_gemini_alias(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert gg.ensure_google_api_key_in_env() == "k"


def test_ensure_google_api_key_in_env_prefers_google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "google")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini")
    assert gg.ensure_google_api_key_in_env() == "google"


def test_ensure_google_api_key_in_env_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        gg.ensure_google_api_key_in_env()
    except ValueError as exc:
        assert "Missing GOOGLE_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when no API key is configured")


def test_normalize_model_id_maps_alias_and_unknown():
    assert gg.normalize_model_id("gemini-3-flash") == "gemini-3-flash-preview"
    assert gg.normalize_model_id("unknown") == gg.DEFAULT_MODEL_ID
    assert gg.normalize_model_id(None) == gg.DEFAULT_MODEL_ID


def test_get_model_caches_instances(monkeypatch):
    gg._MODEL_CACHE.clear()

    class _Model:
        def __init__(self, model, temperature):
            self.model = model
            self.temperature = temperature

    monkeypatch.setattr(gg, "ChatGoogleGenerativeAI", _Model)
    first = gg.get_model("gemini-2.5-pro")
    second = gg.get_model("gemini-2.5-pro")
    assert first is second


def test_configure_gemini_client_is_idempotent(monkeypatch):
    clients = []
    gg._GENAI_CONFIGURED = False
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(gg, "ensure_google_api_key_in_env", lambda: "key")
    monkeypatch.setattr(gg.genai, "Client", lambda api_key: clients.append(api_key) or {"api_key": api_key})

    assert gg.configure_gemini_client() == "key"
    assert gg.configure_gemini_client() == "key"
    assert clients == ["key"]


def test_get_genai_client_uses_configured_key(monkeypatch):
    monkeypatch.setattr(gg, "configure_gemini_client", lambda: "configured-key")
    monkeypatch.setattr(gg.genai, "Client", lambda api_key: {"api_key": api_key})
    assert gg.get_genai_client() == {"api_key": "configured-key"}


def test_get_available_models_returns_curated_list():
    models = gg.get_available_models()
    assert {"id": gg.DEFAULT_MODEL_ID, "label": gg.AVAILABLE_MODELS[gg.DEFAULT_MODEL_ID]} in models


def test_list_gemini_models_filters_to_supported_curated_models(monkeypatch):
    class _Model:
        def __init__(self, name, display_name, methods):
            self.name = name
            self.display_name = display_name
            self.supported_generation_methods = methods

    class _Models:
        def list(self):
            return [
                _Model("models/gemini-2.5-pro", "Pro", ["generateContent"]),
                _Model("models/not-curated", "Skip", ["generateContent"]),
                _Model("models/gemini-3-pro-preview", "Preview", ["embedContent"]),
            ]

    class _Client:
        models = _Models()

    monkeypatch.setattr(gg, "get_genai_client", lambda: _Client())
    models = gg.list_gemini_models()
    assert models == [{"id": "gemini-2.5-pro", "label": gg.AVAILABLE_MODELS["gemini-2.5-pro"]}]


def test_list_gemini_models_falls_back_to_curated_list_on_error(monkeypatch):
    monkeypatch.setattr(gg, "get_genai_client", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    models = gg.list_gemini_models()
    assert models == gg.get_available_models()


def test_resolve_default_model_id_prefers_curated_default():
    assert gg.resolve_default_model_id([{"id": gg.DEFAULT_MODEL_ID}]) == gg.DEFAULT_MODEL_ID
    assert gg.resolve_default_model_id([{"id": "gemini-2.5-pro"}]) == "gemini-2.5-pro"
    assert gg.resolve_default_model_id([]) == gg.DEFAULT_MODEL_ID
