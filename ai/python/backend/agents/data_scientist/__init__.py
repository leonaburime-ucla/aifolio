"""Compatibility exports for ``backend.agents.data_scientist``."""

import sys

from langgraph_agents import data_scientist as _legacy_module

sys.modules[__name__] = _legacy_module
