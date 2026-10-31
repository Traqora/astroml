"""Tests for GraphQL query depth limiting (issue #848).

api/graphql/__init__.py's module docstring and api/app.py's GraphQLRouter
comment both claimed "query depth limiting for security", but
strawberry.Schema(...) was never given a QueryDepthLimiter (or any
extensions at all) -- an unauthenticated client could submit an arbitrarily
deeply nested query and the server would start resolving it regardless.

AstroML's actual business schema (Account, Transaction, ...) happens to be
flat today, so it can't demonstrate rejection through its own relations; the
depth limiter is tested against a small recursive schema built the same way
as the real one (same extension, same wiring), plus a check that the real
schema actually carries the extension with the configured max depth.
"""

from __future__ import annotations

import importlib
import sys

import strawberry
from strawberry.extensions import QueryDepthLimiter


def _import_fresh_schema_module():
    """Re-import api.graphql.schema as a standalone module object.

    `api/graphql/__init__.py` does `from api.graphql.schema import schema`,
    which rebinds the `schema` *attribute* on the `api.graphql` package to
    the Schema instance -- so `import api.graphql.schema` after that can
    resolve to that instance instead of the submodule. Popping it from
    sys.modules and re-importing sidesteps that.
    """
    sys.modules.pop("api.graphql.schema", None)
    return importlib.import_module("api.graphql.schema")


@strawberry.type
class _Node:
    name: str

    @strawberry.field
    def child(self) -> "_Node":
        return _Node(name="child")


@strawberry.type
class _Query:
    @strawberry.field
    def root(self) -> _Node:
        return _Node(name="root")


def _schema_with_max_depth(max_depth: int) -> strawberry.Schema:
    return strawberry.Schema(
        query=_Query,
        extensions=[lambda: QueryDepthLimiter(max_depth=max_depth)],
    )


class TestQueryDepthLimiterWiring:
    def test_query_within_the_depth_limit_is_allowed(self):
        schema = _schema_with_max_depth(3)
        result = schema.execute_sync("{ root { child { name } } }")
        assert result.errors is None

    def test_query_past_the_depth_limit_is_rejected(self):
        schema = _schema_with_max_depth(2)
        result = schema.execute_sync("{ root { child { child { name } } } }")
        assert result.errors is not None
        assert "exceeds maximum operation depth" in str(result.errors[0])

    def test_query_exactly_at_the_depth_limit_is_allowed(self):
        schema = _schema_with_max_depth(2)
        result = schema.execute_sync("{ root { child { name } } }")
        assert result.errors is None


class TestRealSchemaHasDepthLimiting:
    def test_schema_carries_a_query_depth_limiter_extension(self):
        schema_module = _import_fresh_schema_module()
        extension_types = {type(ext).__name__ for ext in schema_module.schema.get_extensions()}
        assert "QueryDepthLimiter" in extension_types

    def test_max_depth_is_configurable_via_environment(self, monkeypatch):
        monkeypatch.setenv("GRAPHQL_MAX_QUERY_DEPTH", "7")
        schema_module = _import_fresh_schema_module()

        assert schema_module.GRAPHQL_MAX_QUERY_DEPTH == 7

        monkeypatch.delenv("GRAPHQL_MAX_QUERY_DEPTH", raising=False)
        _import_fresh_schema_module()
