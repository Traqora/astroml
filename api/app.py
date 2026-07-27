"""AstroML REST API — main FastAPI application.

Wires together all routers:
  - /api/v1/transactions      (Issue #248)
  - /api/v1/fraud/*           (Issue #254)
  - /api/v1/accounts/*        (Issue #247)
  - /api/v1/monitoring/*      (Issue #256)
  - /api/v1/loyalty/*         (Issue #255)
  - /api/v1/models/*          (Issue #237)
  - /api/v1/auth/*            (Issue #240)
  - /api/v1/ws/*              (Issue #239)
  - /api/v1/mentorship/*      (Contributors)

Usage
-----
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.auth.middleware import AuthMiddleware
from api.audit_middleware import AuditLoggingMiddleware
from api.config import settings
from api.database import get_async_session_factory
from api.middleware.csp import CSPMiddleware
from api.middleware.https import HSTSMiddleware, HTTPSRedirectMiddleware
from api.tracing import setup_tracing
from api.validation_middleware import ValidationMiddleware
from api.versioning import VersionMiddleware
from astroml.utils.logging import set_correlation_id, get_correlation_id, clear_correlation_id
from api.routers import (
    accounts_router,
    audit_router,
    auth_router,
    backup_router,
    chat_router,
    compliance_router,
    contact_router,
    contributors_router,
    discussions_router,
    errors_router,
    faq_router,
    feedback_router,
    fraud_router,
    loyalty_router,
    llm_health_router,
    mentorship_router,
    models_router,
    monitoring_router,
    notifications_router,
    onboarding_router,
    rate_limit_router,
    transactions_router,
    validation_router,
    voice_router,
    ws_router,
    streaming_router,
    cost_router,
    llm_usage_router,
    llm_cache_metrics_router,
    llm_router,
    llm_metrics_router,
    search_router,
    stream_router,
    reports_router,
    alerts_router,
    query_router,
)
from api.routers import (
    llm_router,
    query_router,
    explanations_router,
    agents_router,
)



from api.routers.monitoring import record_latency


from api.routers.ws import poll_and_broadcast_transactions
from api.websocket.llm import router as ws_llm_router
from astroml.llm import metrics as _llm_metrics
from api.routers import health
from api.routers import healthz
from api.routers import admin
from astroml.observability.health import readiness_state
from astroml.observability.metrics import observe_http_request, render_latest

from strawberry.fastapi import GraphQLRouter
from api.graphql.schema import schema
from api.graphql.context import get_graphql_context



# Setup distributed tracing (issue #336)
_tracer_provider = setup_tracing()

# Create GraphQL router with query depth limiting and authentication
graphql_app = GraphQLRouter(
    schema,
    context_getter=get_graphql_context,
)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    session_factory = get_async_session_factory()

    try:
        from api.database import _sync_session_factory
        from api.routers.auth import ensure_default_admin

        db = _sync_session_factory()()
        try:
            ensure_default_admin(db)
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        pass

    try:
        from astroml.api.scheduler import build_score_fn, start_scheduler  # noqa: PLC0415

        if os.environ.get("DISABLE_SCHEDULER", "").lower() not in ("1", "true", "yes"):
            start_scheduler(session_factory, score_fn=build_score_fn())
    except Exception:  # noqa: BLE001
        pass

    poll_task = None
    if os.environ.get("DISABLE_WS_POLLER", "").lower() not in ("1", "true", "yes"):
        try:
            poll_task = asyncio.create_task(
                poll_and_broadcast_transactions(),
                name="ws-transaction-poller",
            )
        except Exception:  # noqa: BLE001
            poll_task = None

    # Startup finished — startup/readiness probes may now pass (issue #569).
    readiness_state.mark_started()

    yield

    # Drain traffic before dependencies are torn down.
    readiness_state.set_ready(False, "Application is shutting down.")

    try:
        from astroml.api.scheduler import stop_scheduler  # noqa: PLC0415

        await stop_scheduler()
    except Exception:  # noqa: BLE001
        pass

    if poll_task is not None:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="AstroML API",
    version="1.0.0",
    description="Fraud detection, account management, model monitoring, and loyalty points.",
    lifespan=lifespan,
)

app.add_middleware(VersionMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(ValidationMiddleware)
app.add_middleware(AuditLoggingMiddleware)
app.add_middleware(
    CSPMiddleware,
    report_only=settings.csp_report_only,
    report_uri=settings.csp_report_uri,
    enable_nonce=settings.csp_enable_nonce,
)
app.add_middleware(
    HTTPSRedirectMiddleware,
    enabled=settings.https_enabled,
    allowed_hosts=settings.https_allowed_hosts if settings.https_allowed_hosts else None,
)
app.add_middleware(
    HSTSMiddleware,
    max_age=settings.hsts_max_age,
    include_subdomains=settings.hsts_include_subdomains,
    preload=settings.hsts_preload,
    enabled=settings.hsts_enabled,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


def _route_template(request: Request) -> str:
    """Return the matched route path (not the raw URL) to bound cardinality."""
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else request.url.path


@app.middleware("http")
async def _correlation_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Request-ID")
    set_correlation_id(correlation_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = get_correlation_id() or ""
    clear_correlation_id()
    return response


@app.middleware("http")
async def _latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        record_latency(elapsed * 1000)
        # Prometheus HTTP latency + request count (issue #567).
        observe_http_request(
            request.method, _route_template(request), status_code, elapsed
        )


# Include all routers from both branches
app.include_router(auth_router)
app.include_router(audit_router)
app.include_router(compliance_router)
app.include_router(rate_limit_router)
app.include_router(errors_router)
app.include_router(contact_router)
app.include_router(transactions_router)
app.include_router(fraud_router)
app.include_router(accounts_router)
app.include_router(monitoring_router)
app.include_router(loyalty_router)
app.include_router(models_router)
app.include_router(contributors_router)
app.include_router(discussions_router)
app.include_router(mentorship_router)
app.include_router(notifications_router)
app.include_router(onboarding_router)
app.include_router(faq_router)
app.include_router(feedback_router)
app.include_router(validation_router)
app.include_router(backup_router)
app.include_router(chat_router)
app.include_router(ws_router)
app.include_router(streaming_router)
app.include_router(llm_router)
app.include_router(query_router)
app.include_router(explanations_router)
app.include_router(agents_router)
app.include_router(cost_router)
app.include_router(ws_llm_router)
app.include_router(query_router)
app.include_router(llm_usage_router)
app.include_router(llm_cache_metrics_router)
app.include_router(voice_router)
app.include_router(llm_router)
app.include_router(llm_metrics_router)
app.include_router(search_router)
app.include_router(stream_router)
app.include_router(llm_health_router)
app.include_router(reports_router)
app.include_router(alerts_router)
# HEAD branch routers (health, admin, GraphQL)
app.include_router(health.router)
app.include_router(healthz.router)
app.include_router(admin.router)
app.include_router(graphql_app, prefix="/graphql")
# upstream/main branch routers
app.include_router(query_router)


# Add GraphQL playground endpoint (for development)
if os.environ.get("ENV", "development") == "development":
    @app.get("/graphql/playground")
    async def graphql_playground():
        from strawberry.fastapi import GraphQLPlayground
        return GraphQLPlayground()


@app.get("/health", tags=["ops"])

async def health():
    return {"status": "ok"}


@app.get("/metrics", tags=["ops"])
async def prometheus_metrics():
    """Prometheus exposition endpoint (issue #567)."""
    _refresh_pool_gauges()
    body, content_type = render_latest()
    return Response(body, media_type=content_type)


def _refresh_pool_gauges() -> None:
    """Sample DB pool counters into the gauges just before a scrape (#550)."""
    try:
        from api.database import get_async_engine  # noqa: PLC0415
        from astroml.db.pool_health import collect_pool_stats  # noqa: PLC0415
        from astroml.observability.metrics import (  # noqa: PLC0415
            update_db_pool_metrics,
        )

        update_db_pool_metrics(collect_pool_stats(get_async_engine()))
    except Exception:  # noqa: BLE001 - a scrape must never 500
        pass


@app.get("/api/v1", tags=["ops"])
async def api_root():
    return {"version": settings.api_version, "status": "ok"}