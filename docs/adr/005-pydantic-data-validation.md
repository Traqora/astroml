# ADR-005: Pydantic for Data Validation

## Status

Accepted

## Context

The AstroML API server and data validation pipelines require strong data validation, serialization, and deserialization for HTTP endpoints, internal model schemas, and data exchange interfaces. Input parameters from API clients and ledger payloads must be validated at runtime to prevent invalid state, injection attacks, or runtime errors.

Alternatives considered:
- **Marshmallow**: Popular schema library, but slower runtime performance and less seamless integration with modern FastAPI.
- **Dataclasses with custom validators**: Built-in to Python, but lacks automatic JSON schema generation, OpenAPI documentation generation, and built-in type coercion.
- **Cerberus / jsonschema**: Dict-based validation without native Python class/type-hint integration.

## Decision

We chose **Pydantic** (v2) for runtime data validation, schema definition, and serialization across the FastAPI backend and internal domain validation modules.

Key reasons:
- Native integration with FastAPI for auto-generating OpenAPI documentation and handling HTTP body validation.
- High-performance Rust-backed core (`pydantic-core`) in Pydantic v2.
- Python type hint compatibility (`BaseModel`, `Field`, `field_validator`).
- Automatic type casting, validation error reporting, and JSON serialization.

## Consequences

### Positive
- Strict, declarative runtime validation with clear error messages.
- Automatic interactive API docs (`/docs`, `/redoc`) generated directly from schemas.
- Clean integration with Python static type checkers (`mypy`).

### Negative / Tradeoffs
- Codebase must maintain schema compatibility across Pydantic version updates (e.g. v1 to v2 migration patterns).
- Minimal memory/runtime cost per validated object during heavy serialization loops.
