"""Shared pytest fixtures for LLM tests.

Resolves #458: Provides mock providers, safety guards, observability singletons,
and temporary database fixtures for the entire test suite.
"""
from __future__ import annotations

import pytest
import pytest_asyncio

from tests.llm.mocks import DeterministicMockProvider, ErrorInjectingProvider
from tests.llm.fixtures import (
    SAMPLE_CONVERSATIONS,
    TEST_PROMPTS,
    RAG_TEST_DOCUMENTS,
    MOCK_EMBEDDINGS,
    TOOL_DEFINITIONS,
    MOCK_TOOL_RESPONSES,
)
from astroml.llm.safety.guards import SafetyGuard, StrictnessLevel
from astroml.llm.safety.audit import SafetyAuditLog
from astroml.llm.observability.tracer import LLMTracer
from astroml.llm.observability.metrics import LLMMetrics
from astroml.llm.observability.audit import LLMAuditLog


# ─── Mock providers ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_provider() -> DeterministicMockProvider:
    """A deterministic mock LLM provider with no latency."""
    return DeterministicMockProvider(latency_ms=0.0)


@pytest.fixture
def slow_provider() -> DeterministicMockProvider:
    """A mock provider with 100ms simulated latency."""
    return DeterministicMockProvider(latency_ms=100.0)


@pytest.fixture
def error_provider() -> ErrorInjectingProvider:
    """A provider that always raises on every call."""
    return ErrorInjectingProvider()


@pytest.fixture
def custom_response_provider() -> DeterministicMockProvider:
    """A mock provider with predefined custom responses."""
    return DeterministicMockProvider(
        custom_responses={
            "What is AstroML?": "AstroML is a fraud detection ML platform.",
            "2 + 2": "4",
        }
    )


# ─── Safety fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def safety_audit_log() -> SafetyAuditLog:
    """Fresh in-memory safety audit log."""
    return SafetyAuditLog()


@pytest.fixture
def strict_guard(safety_audit_log: SafetyAuditLog) -> SafetyGuard:
    """Safety guard in STRICT mode."""
    return SafetyGuard(strictness=StrictnessLevel.STRICT, audit_log=safety_audit_log)


@pytest.fixture
def moderate_guard(safety_audit_log: SafetyAuditLog) -> SafetyGuard:
    """Safety guard in MODERATE mode (default)."""
    return SafetyGuard(strictness=StrictnessLevel.MODERATE, audit_log=safety_audit_log)


@pytest.fixture
def permissive_guard(safety_audit_log: SafetyAuditLog) -> SafetyGuard:
    """Safety guard in PERMISSIVE mode."""
    return SafetyGuard(strictness=StrictnessLevel.PERMISSIVE, audit_log=safety_audit_log)


# ─── Observability fixtures ───────────────────────────────────────────────────

@pytest.fixture
def tracer() -> LLMTracer:
    """Fresh LLM tracer instance."""
    return LLMTracer(service_name="test-llm")


@pytest.fixture
def metrics() -> LLMMetrics:
    """Fresh LLM metrics instance."""
    return LLMMetrics()


@pytest.fixture
def audit_log() -> LLMAuditLog:
    """Fresh in-memory LLM audit log."""
    return LLMAuditLog()


# ─── Data fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_conversations() -> list:
    return SAMPLE_CONVERSATIONS


@pytest.fixture
def test_prompts() -> list:
    return TEST_PROMPTS


@pytest.fixture
def rag_documents() -> list:
    return RAG_TEST_DOCUMENTS


@pytest.fixture
def mock_embeddings() -> dict:
    return MOCK_EMBEDDINGS


@pytest.fixture
def tool_definitions() -> list:
    return TOOL_DEFINITIONS


@pytest.fixture
def mock_tool_responses() -> dict:
    return MOCK_TOOL_RESPONSES


# ─── Async test support ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    """Ensure asyncio event loop policy is set for pytest-asyncio."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
