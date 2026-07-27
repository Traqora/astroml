"""LLM Health and Provider Status API."""
from __future__ import annotations

from fastapi import APIRouter

from astroml.llm.health import check_all_providers, check_provider_health

router = APIRouter(prefix="/api/v1/llm", tags=["llm-health"])


@router.get("/health")
async def llm_health():
    result = await check_all_providers()
    return result


@router.get("/health/{provider_name}")
async def llm_provider_health(provider_name: str):
    result = await check_provider_health(provider_name)
    return result
