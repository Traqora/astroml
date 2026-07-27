"""GraphQL context with authentication."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from api.auth.dependencies import get_current_user_from_token
from api.database import _sync_session_factory


class GraphQLContext:
    """GraphQL context containing database session and authenticated user."""

    def __init__(self, session: Session, user: Optional[dict] = None, request=None):
        self.session = session
        self.user = user
        self.request = request

    def close(self) -> None:
        """Close the database session."""
        self.session.close()


def get_graphql_context(request=None) -> GraphQLContext:
    """Create a GraphQL context with authentication."""
    session = _sync_session_factory()()

    user = None
    try:
        auth_header = request.headers.get("Authorization", "") if request else ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            user = get_current_user_from_token(token, session)
    except Exception:
        pass

    return GraphQLContext(session, user, request)