"""
LangSmith telemetry setup helpers.
"""

from __future__ import annotations

import os


def configure_langsmith() -> bool:
    """
    Enable LangSmith tracing when credentials are available.
    Returns True if tracing was enabled.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "AIfolio")
    return True
