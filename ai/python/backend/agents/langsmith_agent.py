"""Compatibility module for ``backend.agents.langsmith_agent`` imports."""

import sys

from langgraph_agents import langsmith_agent as _legacy_module

sys.modules[__name__] = _legacy_module
