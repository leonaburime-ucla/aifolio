"""
Google Gemini helpers used across the server and agent layers.
Centralizes model metadata, client configuration, and discovery logic.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI


AVAILABLE_MODELS: Dict[str, str] = {
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    # "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gemini-3-pro-preview": "Gemini 3 Pro Preview",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
}

DEFAULT_MODEL_ID = "gemini-3-flash-preview"
MODEL_ID_ALIASES: Dict[str, str] = {
    "gemini-3-flash": "gemini-3-flash-preview",
}

_MODEL_CACHE: Dict[str, ChatGoogleGenerativeAI] = {}
_GENAI_CONFIGURED = False


def ensure_google_api_key_in_env() -> str:
    """
    Ensure GOOGLE_API_KEY is set in the environment.
    Returns the resolved key to confirm what was used.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        return google_key
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
        return gemini_key
    raise ValueError("Missing GOOGLE_API_KEY or GEMINI_API_KEY in environment")


def configure_gemini_client() -> str:
    """
    Configure the google-generativeai client with the API key.
    Safe to call multiple times.
    """
    global _GENAI_CONFIGURED
    if _GENAI_CONFIGURED:
        return os.environ["GOOGLE_API_KEY"]
    api_key = ensure_google_api_key_in_env()
    genai.Client(api_key=api_key)
    _GENAI_CONFIGURED = True
    return api_key


def get_genai_client() -> genai.Client:
    """
    Return a configured GenAI client.
    """
    api_key = configure_gemini_client()
    return genai.Client(api_key=api_key)


def get_available_models() -> List[Dict[str, str]]:
    """
    Return the curated model list for UI selection.
    """
    return [{"id": model_id, "label": label} for model_id, label in AVAILABLE_MODELS.items()]


def list_gemini_models() -> List[Dict[str, str]]:
    """
    Return the intersection of supported Gemini models and the curated list.
    Falls back to the curated list if discovery fails.
    """
    try:
        client = get_genai_client()
        curated_ids = set(AVAILABLE_MODELS.keys())
        discovered: List[Dict[str, str]] = []
        for model in client.models.list():
            supported = getattr(model, "supported_generation_methods", None)
            if supported is None and isinstance(model, dict):
                supported = model.get("supported_generation_methods")
            if supported is not None and "generateContent" not in supported:
                continue
            model_name = getattr(model, "name", None) or (
                model.get("name") if isinstance(model, dict) else None
            )
            if not model_name:
                continue
            model_id = model_name.replace("models/", "")
            if model_id not in curated_ids:
                continue
            display_name = getattr(model, "display_name", None) or (
                model.get("display_name") if isinstance(model, dict) else None
            )
            label = AVAILABLE_MODELS.get(model_id, display_name or model_id)
            discovered.append({"id": model_id, "label": label})
        if discovered:
            return discovered
    except Exception:
        pass
    return get_available_models()


def resolve_default_model_id(models: List[Dict[str, str]]) -> str:
    """
    Pick a default model ID based on the curated default and discovery results.
    """
    if any(model["id"] == DEFAULT_MODEL_ID for model in models):
        return DEFAULT_MODEL_ID
    return models[0]["id"] if models else DEFAULT_MODEL_ID


def normalize_model_id(model_id: Optional[str]) -> str:
    """
    Normalize incoming model IDs to supported, curated IDs.
    """
    raw = (model_id or "").strip()
    if not raw:
        return DEFAULT_MODEL_ID
    normalized = MODEL_ID_ALIASES.get(raw, raw)
    if normalized in AVAILABLE_MODELS:
        return normalized
    return DEFAULT_MODEL_ID


def get_model(model_id: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """
    Build or reuse a LangChain Gemini model by ID.
    Defaults to DEFAULT_MODEL_ID when no ID is provided.
    """
    resolved = normalize_model_id(model_id)
    if resolved not in _MODEL_CACHE:
        _MODEL_CACHE[resolved] = ChatGoogleGenerativeAI(
            model=resolved,
            temperature=0.3,
        )
    return _MODEL_CACHE[resolved]
