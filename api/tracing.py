"""OpenTelemetry distributed tracing setup (issue #336)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from api.config import settings

_TRACE_PROVIDER: Optional[TracerProvider] = None


def _build_resource() -> Resource:
    """Build the OpenTelemetry resource for the API service."""
    return Resource.create(
        {
            SERVICE_NAME: settings.service_name,
            "service.version": settings.api_version,
            "deployment.environment": os.environ.get("ENV", "development"),
        }
    )


def setup_tracing() -> Optional[TracerProvider]:
    """Initialize OpenTelemetry tracing with the configured exporter."""
    global _TRACE_PROVIDER
    if not settings.tracing_enabled:
        return None

    if _TRACE_PROVIDER is not None:
        return _TRACE_PROVIDER

    provider = TracerProvider(resource=_build_resource())

    if settings.tracing_exporter == "jaeger":
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter

        exporter = JaegerExporter(
            agent_host_name=settings.jaeger_agent_host,
            agent_port=settings.jaeger_agent_port,
        )
    elif settings.tracing_exporter == "zipkin":
        from opentelemetry.exporter.zipkin.proto.http import ZipkinExporter

        exporter = ZipkinExporter(
            endpoint=settings.zipkin_endpoint,
        )
    else:
        exporter = ConsoleSpanExporter()

    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    try:
        trace.set_tracer_provider(provider)
    except Exception:  # noqa: BLE001
        # The global provider may already be set in test or reload scenarios.
        pass

    FastAPIInstrumentor().instrument(
        tracer_provider=provider,
        excluded_urls="/health,/docs,/openapi.json",
    )

    HTTPXClientInstrumentor().instrument(
        tracer_provider=provider,
    )

    try:
        from api.database import _async_engine, _sync_engine

        SQLAlchemyInstrumentor().instrument(
            engine=_sync_engine(),
            tracer_provider=provider,
        )
        SQLAlchemyInstrumentor().instrument(
            engine=_async_engine().sync_engine,
            tracer_provider=provider,
        )
    except Exception:  # noqa: BLE001
        # Engines might not be initialized yet
        pass

    _TRACE_PROVIDER = provider
    return provider


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer for the current module."""
    return trace.get_tracer(name)


@contextmanager
def trace_operation(
    operation_name: str,
    attributes: Optional[dict[str, str]] = None,
):
    """Context manager for tracing an operation."""
    tracer = get_tracer()
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


def add_span_attributes(attributes: dict[str, str]) -> None:
    """Add attributes to the current span."""
    current_span = trace.get_current_span()
    if current_span:
        for key, value in attributes.items():
            current_span.set_attribute(key, value)


def add_span_event(name: str, attributes: Optional[dict[str, str]] = None) -> None:
    """Add an event to the current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.add_event(name, attributes or {})


def record_exception(exception: Exception) -> None:
    """Record an exception in the current span."""
    current_span = trace.get_current_span()
    if current_span:
        current_span.record_exception(exception)
