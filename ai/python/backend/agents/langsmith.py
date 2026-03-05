"""Compatibility module for ``backend.agents.langsmith`` imports."""

import sys

from langgraph_agents import langsmith as _legacy_module

sys.modules[__name__] = _legacy_module
