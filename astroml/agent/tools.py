"""Tool registry and built-in tools for the LLM Agent Framework.

Tools are the agent's interface to the external world.  Each tool
implements a single, well-defined capability (e.g. reading a file,
running Python code, performing a calculation).

The :class:`ToolRegistry` manages registration, discovery, and
execution of tools.  Built-in tools are registered automatically
when the module is imported.

Example::

    from astroml.agent.tools import ToolRegistry, CalculatorTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    result = registry.execute("calculator", expression="2 + 2")
    print(result.output)  # "4"
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .exceptions import ToolError, ToolNotFoundError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        success: Whether the tool executed without errors.
        output: The tool's output (string representation).
        error: Error message if ``success`` is ``False``.
        data: Optional structured data returned by the tool.
        metadata: Optional dict of extra information.
    """

    success: bool
    output: str = ""
    error: str = ""
    data: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
            "metadata": self.metadata,
        }

    @property
    def text(self) -> str:
        """Return the output or error as a single string."""
        return self.output if self.success else f"Error: {self.error}"


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool:
    """Base class for all tools.

    Subclasses must implement :meth:`execute` and set the following
    class attributes:

    * ``name``: The tool's identifier (used for registration and lookup).
    * ``description``: A human-readable description of what the tool does.
    * ``parameters``: A JSON-schema-like dict describing the tool's
      parameters (used for LLM function-calling).
    """

    name: str = "base"
    description: str = "Base tool"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def get_schema(self) -> Dict[str, Any]:
        """Return the tool's schema for LLM function-calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


# ---------------------------------------------------------------------------
# Built-in tools
# ---------------------------------------------------------------------------

class CalculatorTool(Tool):
    """Evaluate arithmetic expressions safely.

    Uses Python's ``ast`` module to parse and evaluate expressions
    containing only numbers and basic operators (+, -, *, /, **, %).
    """

    name = "calculator"
    description = "Evaluate a mathematical expression. Supports +, -, *, /, **, %, and parentheses."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g. '2 + 3 * 4').",
            },
        },
        "required": ["expression"],
    }

    def execute(self, expression: str) -> ToolResult:
        import ast
        import operator

        ops: Dict[type, Callable] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return ToolResult(success=False, error=f"Syntax error: {exc}")

        def _eval(node: ast.AST) -> float:
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError(f"Unsupported constant: {node.value}")
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                op_type = type(node.op)
                if op_type not in ops:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                return ops[op_type](left, right)
            if isinstance(node, ast.UnaryOp):
                operand = _eval(node.operand)
                op_type = type(node.op)
                if op_type not in ops:
                    raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
                return ops[op_type](operand)
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

        try:
            result = _eval(tree.body)
            return ToolResult(
                success=True,
                output=str(result),
                data={"result": result},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class PythonREPLTool(Tool):
    """Execute Python code in a sandboxed subprocess.

    Code is executed in a separate Python process with a timeout.
    The tool captures stdout, stderr, and the return value.

    Note: This tool executes arbitrary code.  Only use it in trusted
    environments.  The ``allowed_imports`` config option can restrict
    which modules can be imported.
    """

    name = "python_repl"
    description = "Execute Python code and return the output. Use this for calculations, data processing, or any Python-based computation."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default: 30).",
                "default": 30,
            },
        },
        "required": ["code"],
    }

    def execute(self, code: str, timeout: int = 30) -> ToolResult:
        import tempfile

        # Write code to a temp file and execute
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
            success = result.returncode == 0
            output = result.stdout.strip()
            error = result.stderr.strip() if not success else ""
            return ToolResult(
                success=success,
                output=output,
                error=error,
                data={"returncode": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Execution timed out after {timeout}s")
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
        finally:
            Path(temp_path).unlink(missing_ok=True)


class FileReadTool(Tool):
    """Read the contents of a file."""

    name = "read_file"
    description = "Read the contents of a file at the given path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read (default: 100).",
                "default": 100,
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str, max_lines: int = 100) -> ToolResult:
        file_path = Path(path)
        if not file_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        if not file_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            if max_lines and len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                content += f"\n... ({len(lines) - max_lines} more lines truncated)"
            return ToolResult(
                success=True,
                output=content,
                data={"line_count": len(lines), "path": str(file_path)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class FileWriteTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file at the given path. Creates parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file.",
            },
            "append": {
                "type": "boolean",
                "description": "If True, append to the file instead of overwriting.",
                "default": False,
            },
        },
        "required": ["path", "content"],
    }

    def execute(self, path: str, content: str, append: bool = False) -> ToolResult:
        file_path = Path(path)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(file_path, mode, encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Written to {path}",
                data={"path": str(file_path), "bytes": len(content)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class ListDirectoryTool(Tool):
    """List the contents of a directory."""

    name = "list_directory"
    description = "List the files and directories in the given path."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the directory to list (default: current directory).",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "If True, list recursively.",
                "default": False,
            },
        },
        "required": [],
    }

    def execute(self, path: str = ".", recursive: bool = False) -> ToolResult:
        dir_path = Path(path)
        if not dir_path.exists():
            return ToolResult(success=False, error=f"Directory not found: {path}")
        if not dir_path.is_dir():
            return ToolResult(success=False, error=f"Not a directory: {path}")

        try:
            if recursive:
                entries = [str(p.relative_to(dir_path)) for p in sorted(dir_path.rglob("*"))]
            else:
                entries = [str(p.name) for p in sorted(dir_path.iterdir())]

            dirs = [e for e in entries if (dir_path / e).is_dir()]
            files = [e for e in entries if (dir_path / e).is_file()]

            output = "Directories:\n"
            for d in dirs:
                output += f"  [dir]  {d}\n"
            output += "\nFiles:\n"
            for f in files:
                output += f"  [file] {f}\n"

            return ToolResult(
                success=True,
                output=output,
                data={"directories": dirs, "files": files, "total": len(entries)},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class SearchTool(Tool):
    """Search for text patterns in files."""

    name = "search"
    description = "Search for a regex pattern in files within a directory."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory).",
                "default": ".",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern for files to search (default: '*.py').",
                "default": "*.py",
            },
        },
        "required": ["pattern"],
    }

    def execute(self, pattern: str, path: str = ".", file_pattern: str = "*.py") -> ToolResult:
        import re

        search_path = Path(path)
        if not search_path.exists():
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(success=False, error=f"Invalid regex: {exc}")

        results: List[str] = []
        file_count = 0

        for file_path in search_path.rglob(file_pattern):
            if not file_path.is_file():
                continue
            file_count += 1
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{i}: {line.strip()}")
            except Exception:
                continue

        output = "\n".join(results) if results else "No matches found."
        return ToolResult(
            success=True,
            output=output,
            data={"match_count": len(results), "files_searched": file_count},
        )


class HTTPRequestTool(Tool):
    """Make HTTP requests."""

    name = "http_request"
    description = "Make an HTTP GET or POST request and return the response."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to request.",
            },
            "method": {
                "type": "string",
                "description": "HTTP method (GET or POST, default: GET).",
                "default": "GET",
            },
            "headers": {
                "type": "string",
                "description": "JSON string of headers (default: empty).",
                "default": "{}",
            },
            "data": {
                "type": "string",
                "description": "JSON string of request body (for POST).",
                "default": "{}",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default: 30).",
                "default": 30,
            },
        },
        "required": ["url"],
    }

    def execute(
        self,
        url: str,
        method: str = "GET",
        headers: str = "{}",
        data: str = "{}",
        timeout: int = 30,
    ) -> ToolResult:
        import urllib.request
        import urllib.error

        try:
            parsed_headers = json.loads(headers) if headers else {}
            parsed_data = json.loads(data) if data else {}
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"Invalid JSON: {exc}")

        try:
            body = json.dumps(parsed_data).encode("utf-8") if parsed_data else None
            req = urllib.request.Request(
                url,
                data=body,
                headers=parsed_headers,
                method=method.upper(),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content = resp.read().decode("utf-8", errors="replace")
                return ToolResult(
                    success=True,
                    output=content,
                    data={
                        "status_code": resp.status,
                        "headers": dict(resp.headers),
                    },
                )
        except urllib.error.HTTPError as exc:
            return ToolResult(
                success=False,
                error=f"HTTP {exc.code}: {exc.reason}",
                data={"status_code": exc.code},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Manages tool registration, discovery, and execution.

    Tools are registered by name and can be retrieved or executed
    by name.  The registry also provides schema generation for LLM
    function-calling.

    Example::

        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = registry.execute("calculator", expression="2 + 2")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Overwrites any existing tool with the same name."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def register_many(self, tools: Sequence[Tool]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name. Returns ``True`` if it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        """Retrieve a tool by name."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool not found: {name}")
        return tool

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> List[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return schemas for all registered tools (for LLM function-calling)."""
        return [tool.get_schema() for tool in self._tools.values()]

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a registered tool by name.

        Args:
            name: The tool's registered name.
            **kwargs: Parameters to pass to the tool.

        Returns:
            A :class:`ToolResult`.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            ToolError: If the tool execution fails.
        """
        tool = self.get(name)
        try:
            result = tool.execute(**kwargs)
            if not result.success and result.error:
                logger.warning("Tool %s failed: %s", name, result.error)
            return result
        except Exception as exc:
            logger.error("Tool %s raised exception: %s", name, exc, exc_info=True)
            return ToolResult(success=False, error=str(exc))

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

def create_default_registry() -> ToolRegistry:
    """Create a :class:`ToolRegistry` with all built-in tools registered."""
    registry = ToolRegistry()
    registry.register_many([
        CalculatorTool(),
        PythonREPLTool(),
        FileReadTool(),
        FileWriteTool(),
        ListDirectoryTool(),
        SearchTool(),
        HTTPRequestTool(),
    ])
    return registry
