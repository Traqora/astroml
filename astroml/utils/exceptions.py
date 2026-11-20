"""Public exception types for the AstroML v1 workflow.

Every failure the v1 workflow can surface to a caller or an API client is an
:class:`AstroMLError`. Catch the base class to handle any of them, or a stage
subclass to handle one stage of the workflow::

    AstroMLError
    +-- ConfigurationError        invalid or missing configuration
    +-- IngestionError            ledger data could not be ingested
    |   +-- ParseError            a payload could not be parsed (ledger / transaction ids)
    |   +-- SourceUnavailableError  the data source is temporarily unreachable (retryable)
    +-- FeatureError              features could not be computed
    +-- ModelError                the model could not complete the request
    +-- DatabaseError             a storage error

Conventions
-----------
* Raise the most specific subclass for the stage that failed.
* When translating an exception from a dependency (an HTTP client, SQLAlchemy,
  a ``KeyError`` from a payload), use ``raise SomeError(...) from exc`` so the
  original stays available on ``__cause__`` for logs and debugging.
* ``str(exc)``, :attr:`AstroMLError.message` and :attr:`AstroMLError.context`
  are for operators: they may contain hostnames, SQL or other internals and
  belong in logs. Only :meth:`AstroMLError.to_public_dict` is meant for API
  clients; it never contains the internal message, the context, the cause or a
  traceback.

See ``docs/v1-errors.md`` for the reference table and examples.
"""

from __future__ import annotations

from typing import Any, ClassVar

__all__ = [
    "AstroMLError",
    "ConfigurationError",
    "IngestionError",
    "ParseError",
    "SourceUnavailableError",
    "FeatureError",
    "ModelError",
    "DatabaseError",
]


class AstroMLError(Exception):
    """Base class for every failure the v1 workflow reports.

    Args:
        message: Internal, human-readable description. Safe for logs, not for API clients.
        **context: Extra keys to help debugging (stored on :attr:`context`, never made public).

    Attributes:
        message: The internal message, same as ``str(exc)``.
        context: The keyword context passed at construction.
        code: Stable machine-readable identifier, safe to expose and to match on.
        public_message: Generic description safe to show to an API client.
        retryable: Whether repeating the same operation later could succeed.
    """

    code: ClassVar[str] = "ASTROML_ERROR"
    public_message: ClassVar[str] = "An unexpected error occurred."
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_public_dict(self) -> dict[str, Any]:
        """Return the representation that is safe to send to an API client.

        Contains only the stable ``code``, a generic ``message`` and ``retryable``.
        It never includes the internal message, :attr:`context`, ``__cause__`` or a
        traceback, so a database error cannot leak SQL or a hostname through it.
        """
        return {
            "code": self.code,
            "message": self.public_message,
            "retryable": self.retryable,
        }


class ConfigurationError(AstroMLError):
    """Configuration is missing or invalid, so the workflow cannot run as requested."""

    code: ClassVar[str] = "CONFIGURATION_ERROR"
    public_message: ClassVar[str] = "The service is misconfigured."


class IngestionError(AstroMLError):
    """Ledger data could not be ingested."""

    code: ClassVar[str] = "INGESTION_ERROR"
    public_message: ClassVar[str] = "Ledger data could not be ingested."


class ParseError(IngestionError):
    """A ledger, transaction, operation or effect payload could not be parsed.

    The ledger sequence and transaction hash identify the offending record, so an
    operator can find it in Horizon and a client can report it. Both are public
    on-chain identifiers, so unlike the internal message they are part of
    :meth:`to_public_dict`.

    Args:
        message: What was wrong with the payload.
        ledger_sequence: Sequence number of the ledger the record belongs to, if known.
        transaction_hash: Hash of the transaction the record belongs to, if known.
        field: Name of the field that failed to parse, if known.
        **context: Extra debugging keys (never made public).
    """

    code: ClassVar[str] = "PARSE_ERROR"
    public_message: ClassVar[str] = "Ledger data could not be parsed."

    def __init__(
        self,
        message: str,
        *,
        ledger_sequence: int | None = None,
        transaction_hash: str | None = None,
        field: str | None = None,
        **context: Any,
    ) -> None:
        located = {
            "ledger": ledger_sequence,
            "tx": transaction_hash,
            "field": field,
        }
        where = " ".join(f"{name}={value}" for name, value in located.items() if value is not None)
        super().__init__(f"{message} [{where}]" if where else message, **context)
        self.ledger_sequence = ledger_sequence
        self.transaction_hash = transaction_hash
        self.field = field
        for name, value in (
            ("ledger_sequence", ledger_sequence),
            ("transaction_hash", transaction_hash),
            ("field", field),
        ):
            if value is not None:
                self.context.setdefault(name, value)

    def to_public_dict(self) -> dict[str, Any]:
        """Return the public representation, plus the ledger and transaction identifiers."""
        public = super().to_public_dict()
        details = {
            name: value
            for name, value in (
                ("ledger_sequence", self.ledger_sequence),
                ("transaction_hash", self.transaction_hash),
            )
            if value is not None
        }
        if details:
            public["details"] = details
        return public


class SourceUnavailableError(IngestionError):
    """The ledger data source (such as Horizon) is temporarily unreachable or rate limiting."""

    code: ClassVar[str] = "SOURCE_UNAVAILABLE"
    public_message: ClassVar[str] = "The ledger data source is temporarily unavailable."
    retryable: ClassVar[bool] = True


class FeatureError(AstroMLError):
    """Features could not be computed."""

    code: ClassVar[str] = "FEATURE_ERROR"
    public_message: ClassVar[str] = "Features could not be computed."


class ModelError(AstroMLError):
    """The model could not complete the request."""

    code: ClassVar[str] = "MODEL_ERROR"
    public_message: ClassVar[str] = "The model could not complete the request."


class DatabaseError(AstroMLError):
    """A storage error occurred."""

    code: ClassVar[str] = "DATABASE_ERROR"
    public_message: ClassVar[str] = "A storage error occurred."
