"""Server-layer compatibility exports under ``backend.server``."""

from server import app
from server_ml import framework_status, run_predict_endpoint, run_training_or_distill_endpoint

__all__ = [
    "app",
    "framework_status",
    "run_training_or_distill_endpoint",
    "run_predict_endpoint",
]

