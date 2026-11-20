# V1 Errors

The supported v1 workflow reports its failures with a small set of public exception types in
`astroml.utils.exceptions`. Catch the base class to handle any v1 failure, or a stage
subclass to handle one stage of the workflow.

## Purpose and boundaries

The v1 profile is PostgreSQL, Redis, the AstroML API, the feature store and ingestion (see
[V1 Persistent Volumes](v1-persistent-volumes.md)). The types map onto it as follows:

| v1 component | Exception family |
|---|---|
| Ingestion | `IngestionError`, `ParseError`, `SourceUnavailableError` |
| PostgreSQL | `DatabaseError` |
| Feature store | `FeatureError` |
| Any component, at startup or reload | `ConfigurationError` |
| AstroML API | does not define failures of its own; it presents the types above (see below) |

- **In scope:** the exception types, their stable machine-readable codes, whether each is
  retryable, and a `to_public_dict()` that is safe to return to an API client.
- **Not part of the types:** logging, HTTP status mapping and retry policy. Those read `code`
  and `retryable` but live with the code that logs, responds or retries.
- **Not covered:** there is no dedicated type for Redis or cache failures. `ModelError` exists
  for model operations, which are outside the v1 profile.
- The types depend on no optional service (a database driver, Horizon, Redis, a model
  library), so they can be imported and caught anywhere in the workflow without any of them.

## Hierarchy

```
AstroMLError
+-- ConfigurationError
+-- IngestionError
|   +-- ParseError
|   +-- SourceUnavailableError
+-- FeatureError
+-- ModelError
+-- DatabaseError
```

| Exception | `code` | Raised when | Retryable | Public message |
|---|---|---|---|---|
| `AstroMLError` | `ASTROML_ERROR` | Base class; use a subclass when one fits | no | An unexpected error occurred. |
| `ConfigurationError` | `CONFIGURATION_ERROR` | Configuration is missing or invalid | no | The service is misconfigured. |
| `IngestionError` | `INGESTION_ERROR` | Ledger data could not be ingested | no | Ledger data could not be ingested. |
| `ParseError` | `PARSE_ERROR` | A ledger, transaction, operation or effect payload could not be parsed | no | Ledger data could not be parsed. |
| `SourceUnavailableError` | `SOURCE_UNAVAILABLE` | The data source (such as Horizon) is unreachable or rate limiting | **yes** | The ledger data source is temporarily unavailable. |
| `FeatureError` | `FEATURE_ERROR` | Features could not be computed | no | Features could not be computed. |
| `ModelError` | `MODEL_ERROR` | The model could not complete the request | no | The model could not complete the request. |
| `DatabaseError` | `DATABASE_ERROR` | A storage error occurred | no | A storage error occurred. |

`code` values are part of the public contract: they will not change, so clients may match on
them. New errors add new codes.

## Using them

```python
from astroml.utils.exceptions import DatabaseError, IngestionError, ParseError

try:
    ledger = parse_ledger(payload)
except KeyError as exc:
    # Keep the original error: it is shown in logs as "The above exception was the direct
    # cause of the following exception".
    raise ParseError(
        "ledger payload is missing a field",
        ledger_sequence=12554,
        transaction_hash="abc123...",
        field="closed_at",
    ) from exc

try:
    run_ingest()
except IngestionError as err:      # also catches ParseError and SourceUnavailableError
    log.error("ingest failed: %s", err, extra=err.context)
    if err.retryable:
        schedule_retry()
```

### Parse errors name the record

`ParseError` takes `ledger_sequence`, `transaction_hash` and `field`. Whichever are given are
added to the message (`bad timestamp [ledger=12554 tx=abc123 field=created_at]`) and to
`context`, and are available as attributes. A ledger sequence of `0` counts as given.

## Internal detail versus what clients see

`str(exc)`, `exc.message` and `exc.context` are **internal**. They may contain hostnames, SQL,
tokens or other details and belong in logs only.

`exc.to_public_dict()` is the only representation meant for API clients:

```json
{"code": "DATABASE_ERROR", "message": "A storage error occurred.", "retryable": false}
```

It never contains the internal message, the context, the `__cause__` or a traceback.
`ParseError` adds the ledger sequence and transaction hash under `details`, because they are
public on-chain identifiers; the field name and any other context stay internal.

```json
{
  "code": "PARSE_ERROR",
  "message": "Ledger data could not be parsed.",
  "retryable": false,
  "details": {"ledger_sequence": 12554, "transaction_hash": "abc123..."}
}
```

## Compatibility

`AstroMLError(message, **context)` and the original `IngestionError`, `FeatureError`,
`ModelError` and `DatabaseError` are unchanged for existing callers; `message`, `code`,
`public_message` and `retryable` are additions.

## Adoption status

This change defines the types. Using them at existing call sites is tracked separately:

- Translating dependency errors with `raise ... from exc` throughout ingestion (#1235).
- Raising `ParseError` with ledger and transaction identifiers from `astroml.ingestion.parsers` (#1237).
- Returning `to_public_dict()` from the API error handlers, and auditing routers for exception
  text in responses (#1238).

Until those land, parsers still raise their original `KeyError`/`ValueError`, and routers that
build a response from `str(exc)` are unchanged.
