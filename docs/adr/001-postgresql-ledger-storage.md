# ADR-001: PostgreSQL for Ledger Storage

## Status

Accepted

## Context

AstroML ingests, parses, and stores transaction ledgers from the Stellar network. The data contains relational relationships between accounts, ledgers, transactions, operations, and asset balances. The system requires strong ACID guarantees, structured schema enforcement, efficient composite indexing (e.g. `(account_id, timestamp)` for time-windowed subgraphs), and support for semi-structured metadata.

Alternatives considered:
- **MongoDB / NoSQL**: Flexible schema but weaker multi-table relational join support and transactional guarantees.
- **SQLite**: Great for embedded metadata, but lacks concurrency and scaling capacity for large multi-gigabyte blockchain transaction datasets.
- **DynamoDB / Cloud NoSQL**: High scalability, but vendor lock-in and high cost for ad-hoc analytical subqueries.

## Decision

We chose **PostgreSQL** as the primary storage database for raw Stellar ledger data, graph mirrors (`api_accounts`, `api_transactions`), and operational models.

Key reasons:
- Native relational engine with full ACID compliance.
- Support for `JSONB` for flexible event/metric payload storage.
- High-performance B-Tree and GIN indexing for fast time-windowed graph lookups.
- Ecosystem compatibility with SQLAlchemy, Alembic, and asyncpg/psycopg.

## Consequences

### Positive
- Reliable transaction storage with strict integrity.
- Fast queries on indexed fields like `(account_id, timestamp)`.
- Seamless ORM integration across API and pipeline layers.

### Negative / Tradeoffs
- Requires running a PostgreSQL server or container in development and production environments.
- Database scaling beyond a single instance requires read replicas or partitioned tables.
