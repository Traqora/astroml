"""Tool-use framework for LLM function calling."""

from .audit import ToolAuditLog
from .definitions import BaseTool
from .executor import ToolExecutionError, ToolExecutor
from .permissions import PermissionChecker, PermissionDenied, PermissionDeniedError
from .registry import ToolRegistry, get_global_registry, reset_registry
from .validators import ValidationError, validate_parameters

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "get_global_registry",
    "reset_registry",
    "ToolExecutor",
    "ToolExecutionError",
    "validate_parameters",
    "ValidationError",
    "PermissionChecker",
    "PermissionDenied",
    "PermissionDeniedError",
    "ToolAuditLog",
]
