"""Compatibility module for ``backend.agents.analyst`` imports."""

import sys

from langgraph_agents import analyst as _legacy_module

sys.modules[__name__] = _legacy_module
