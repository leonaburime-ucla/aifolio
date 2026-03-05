"""Compatibility module for ``backend.agents.coordinator`` imports."""

import sys

from langgraph_agents import coordinator as _legacy_module

sys.modules[__name__] = _legacy_module
