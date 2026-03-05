"""Compatibility module for ``backend.ml.data`` imports."""

import sys

import ml_data as _legacy_module

sys.modules[__name__] = _legacy_module
