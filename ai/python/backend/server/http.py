"""Compatibility module for ``backend.server.http`` imports."""

import sys

import server as _legacy_module

sys.modules[__name__] = _legacy_module
