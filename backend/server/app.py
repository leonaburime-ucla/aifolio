from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.core import router as core_router
from server.routes.ml_framework import router as ml_framework_router


def _resolve_cors_origins() -> list[str]:
    default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    configured = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    frontend_url = os.getenv("FRONTEND_URL", "").strip()
    dynamic_origins = [
        origin.strip()
        for origin in [*configured.split(","), frontend_url]
        if origin and origin.strip()
    ]
    return list(dict.fromkeys([*default_origins, *dynamic_origins]))


def create_app() -> FastAPI:
    app = FastAPI(title="AI Portfolio", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(core_router)
    app.include_router(ml_framework_router)
    return app


app = create_app()
