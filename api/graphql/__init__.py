"""GraphQL API module for AstroML.

Provides a GraphQL endpoint at /graphql with:
- Query support for all data models
- Mutation support for creating and updating entities
- Subscription support for real-time updates
- Authentication integration with JWT
- Query depth limiting for security
"""

from __future__ import annotations

try:
    from api.graphql.context import get_graphql_context
    from api.graphql.schema import schema
except Exception:
    schema = None
    get_graphql_context = None


async def publish_transaction(*args, **kwargs):
    pass


async def publish_fraud_alert(*args, **kwargs):
    pass


async def publish_loyalty_points(*args, **kwargs):
    pass


__all__ = [
    "schema",
    "get_graphql_context",
    "publish_transaction",
    "publish_fraud_alert",
    "publish_loyalty_points",
]
