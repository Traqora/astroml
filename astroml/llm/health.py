"""LLM Provider health checks."""
import asyncio
import os
import time
from typing import Any, Dict

import aiohttp

PROVIDER_ENDPOINTS = {
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "method": "GET",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/messages",
        "method": "HEAD",
        "headers": lambda key: {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/status",
        "method": "GET",
        "headers": lambda key: {"Authorization": f"Bearer {key}"},
    },
}


def _get_api_key(provider_name: str) -> str:
    env_key = f"{provider_name.upper()}_API_KEY"
    return os.getenv(env_key, "")


async def check_provider_health(
    provider_name: str, timeout: float = 5.0
) -> Dict[str, Any]:
    start = time.perf_counter()
    if provider_name not in PROVIDER_ENDPOINTS:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "provider": provider_name,
            "status": "unknown",
            "latency_ms": round(latency_ms, 2),
            "error": "Provider not supported for health checks",
        }

    api_key = _get_api_key(provider_name)
    if not api_key:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "provider": provider_name,
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 2),
            "error": "API key not configured",
        }

    config = PROVIDER_ENDPOINTS[provider_name]

    try:
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=config["method"],
                url=config["url"],
                headers=config["headers"](api_key),
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                latency_ms = (time.perf_counter() - start) * 1000
                healthy = 200 <= response.status < 300
                return {
                    "provider": provider_name,
                    "status": "healthy" if healthy else "unhealthy",
                    "latency_ms": round(latency_ms, 2),
                    "http_status": response.status,
                }
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "provider": provider_name,
            "status": "unhealthy",
            "latency_ms": round(latency_ms, 2),
            "error": str(e),
        }


async def check_all_providers() -> Dict[str, Any]:
    providers = list(PROVIDER_ENDPOINTS.keys())
    results = await asyncio.gather(
        *(check_provider_health(p) for p in providers),
        return_exceptions=True,
    )

    provider_statuses = {}
    for result in results:
        if isinstance(result, Exception):
            provider_statuses["unknown"] = {
                "provider": "unknown",
                "status": "unhealthy",
                "latency_ms": 0,
                "error": str(result),
            }
        else:
            provider_statuses[result["provider"]] = result

    all_healthy = all(
        r.get("status") == "healthy" for r in provider_statuses.values()
    )
    return {
        "overall_status": "healthy" if all_healthy else "degraded",
        "providers": provider_statuses,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
