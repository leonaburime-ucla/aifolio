"""Compatibility module for ``backend.agents.data_scientist.planner`` imports."""

import sys

from langgraph_agents.data_scientist import planner as _legacy_module

sys.modules[__name__] = _legacy_module
