"""Vercel Python entrypoint.

This exposes the FastAPI `app` object so Vercel can serve backend routes.
"""

from server.http import app

