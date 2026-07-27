"""Tool execution engine with timeout, retry, and output limits."""

import asyncio
import logging
import time
from typing import Any

from .registry import ToolRegistry
from .validators import validate_parameters, validate_output_size, ValidationError
from .permissions import PermissionChecker
from .audit import ToolAuditLog

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_OUTPUT_SIZE = 100_000
MAX_RETRIES = 2


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    pass


class ToolExecutor:
    """Executes tool calls with safety controls."""

    def __init__(
        self,
        registry: ToolRegistry,
        permission_checker: PermissionChecker | None = None,
        audit_log: ToolAuditLog | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
    ):
        self._registry = registry
        self._permission_checker = permission_checker
        self._audit_log = audit_log
        self._timeout = timeout
        self._max_output_size = max_output_size

    async def execute(
        self,
        tool_name: str,
        tool_params: dict[str, Any],
        user_id: str | None = None,
    ) -> Any:
        """Execute a single tool call with all safety checks."""
        start = time.time()
        tool = self._registry.get(tool_name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: '{tool_name}'")

        if self._permission_checker:
            self._permission_checker.check(tool_name, user_id)

        validate_parameters(tool_params, tool.parameters)

        last_error: Exception | None = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    tool.execute(tool_params),
                    timeout=self._timeout,
                )
                validate_output_size(result, self._max_output_size)

                duration = time.time() - start
                if self._audit_log:
                    self._audit_log.record(
                        tool_name=tool_name,
                        params=tool_params,
                        result=result,
                        user_id=user_id,
                        duration=duration,
                        error=None,
                    )
                return result

            except asyncio.TimeoutError:
                last_error = ToolExecutionError(
                    f"Tool '{tool_name}' timed out after {self._timeout}s"
                )
                logger.warning("Tool %s timed out (attempt %d)", tool_name, attempt + 1)
                continue
            except ValidationError:
                raise
            except Exception as e:
                last_error = ToolExecutionError(f"Tool '{tool_name}' failed: {e}")
                logger.warning("Tool %s failed (attempt %d): %s", tool_name, attempt + 1, e)
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(1 * (attempt + 1))
                continue

        duration = time.time() - start
        if self._audit_log:
            self._audit_log.record(
                tool_name=tool_name,
                params=tool_params,
                result=None,
                user_id=user_id,
                duration=duration,
                error=str(last_error) if last_error else "Unknown error",
            )
        raise last_error or ToolExecutionError(f"Tool '{tool_name}' failed")
