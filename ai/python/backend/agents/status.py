"""Compatibility module for ``backend.agents.status`` imports."""

import sys

from langgraph_agents import status as _legacy_module

sys.modules[__name__] = _legacy_module
