"""Compatibility module for ``backend.agents.data_scientist.dataset_io`` imports."""

import sys

from langgraph_agents.data_scientist import dataset_io as _legacy_module

sys.modules[__name__] = _legacy_module
