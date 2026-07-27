"""Permission system for tool access control."""

from typing import Any


class PermissionDenied(Exception):
    """Raised when a user does not have permission to use a tool."""
    pass


class PermissionChecker:
    """Checks whether a user is allowed to execute a given tool."""

    def __init__(self):
        self._acl: dict[str, set[str | None]] = {}

    def allow(self, tool_name: str, user_id: str | None = None) -> None:
        """Grant access to a tool. None means all users."""
        self._acl.setdefault(tool_name, set()).add(user_id)

    def deny(self, tool_name: str, user_id: str | None = None) -> None:
        """Revoke access to a tool."""
        s = self._acl.get(tool_name)
        if s:
            s.discard(user_id)

    def check(self, tool_name: str, user_id: str | None = None) -> None:
        """Raise PermissionDenied if the user is not allowed."""
        s = self._acl.get(tool_name)
        if s is None:
            return
        if None in s:
            return
        if user_id in s:
            return
        raise PermissionDenied(f"User '{user_id}' is not allowed to use tool '{tool_name}'")
