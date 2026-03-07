"""Vercel Python API entrypoint.

Vercel detects Python serverless functions under `api/`.
Expose the FastAPI `app` object from the backend server module.
"""

from server.http import app

