"""Compatibility module for ``backend.server.ml`` imports."""

import sys

import server_ml as _legacy_module

sys.modules[__name__] = _legacy_module
