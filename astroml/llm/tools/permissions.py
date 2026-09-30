from typing import Any, Dict, List, Optional, Union, Callable
"""Permission system for tool access control."""


class PermissionDeniedError(Exception):
    """Raised when a user does not have permission to use a tool."""

    pass


# Backward-compatible alias
PermissionDenied = PermissionDeniedError


class PermissionChecker:
    """Checks whether a user is allowed to execute a given tool."""

    def __init__(self) -> Any:
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
        """Raise PermissionDeniedError if the user is not allowed."""
        s = self._acl.get(tool_name)
        if s is None:
            return
        if None in s:
            return
        if user_id in s:
            return
        raise PermissionDeniedError(f"User '{user_id}' is not allowed to use tool '{tool_name}'")
