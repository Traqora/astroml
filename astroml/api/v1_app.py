"""Standalone v1 inference app without the database-backed batch scheduler."""

from __future__ import annotations

from fastapi import FastAPI

from astroml.api.routers.v1_inference import router

app = FastAPI(
    title="AstroML V1 Inference API",
    version="1.0.0",
    description="Local health, readiness, and baseline prediction endpoints.",
)
app.include_router(router)