from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

from .registry import Tool, ToolRegistry

MAX_EXPRESSION_CHARS = 256
MAX_AST_NODES = 64
MAX_ABSOLUTE_RESULT = 1e100
MAX_TEXT_FILE_BYTES = 256 * 1024
_ALLOWED_TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".py", ".pyi", ".c", ".cc",
    ".cpp", ".h", ".hh", ".hpp", ".java", ".js", ".jsx", ".ts", ".tsx",
    ".json", ".yaml", ".yml", ".toml", ".csv", ".log", ".sql", ".cmake",
    ".sh", ".ps1",
})
_IGNORED_COMPONENTS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "build", "dist",
    "__pycache__", ".pytest_cache",
})
_DENIED_FILENAMES = frozenset({
    ".env", "credentials", "secrets", "id_rsa", "id_ed25519",
})


def _bounded_number(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("calculator result is not a real number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("calculator result must be finite")
    if abs(value) > MAX_ABSOLUTE_RESULT:
        raise ValueError("calculator result exceeds the configured magnitude limit")
    return value


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only integer and decimal numeric literals are allowed")
        return _bounded_number(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        operand = _evaluate_node(node.operand)
        return _bounded_number(operand if isinstance(node.op, ast.UAdd) else -operand)

    if not isinstance(node, ast.BinOp):
        raise ValueError("expression contains an unsupported operation")

    left = _evaluate_node(node.left)
    right = _evaluate_node(node.right)
    operator = type(node.op)

    if operator is ast.Add:
        value = left + right
    elif operator is ast.Sub:
        value = left - right
    elif operator is ast.Mult:
        value = left * right
    elif operator is ast.Div:
        if right == 0:
            raise ValueError("division by zero")
        value = left / right
    elif operator is ast.FloorDiv:
        if right == 0:
            raise ValueError("division by zero")
        value = left // right
    elif operator is ast.Mod:
        if right == 0:
            raise ValueError("division by zero")
        value = left % right
    elif operator is ast.Pow:
        if abs(right) > 12:
            raise ValueError("exponent magnitude cannot exceed 12")
        value = left ** right
    else:
        raise ValueError("expression contains an unsupported operator")
    return _bounded_number(value)


def calculate(arguments: dict[str, Any]) -> dict[str, int | float]:
    expression = arguments["expression"]
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must not be empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"expression exceeds {MAX_EXPRESSION_CHARS} characters")
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError("invalid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(parsed)) > MAX_AST_NODES:
        raise ValueError(f"expression exceeds {MAX_AST_NODES} syntax nodes")
    result = _evaluate_node(parsed.body)
    return {"result": result}


def _make_text_reader(workspace: str | Path, max_bytes: int = MAX_TEXT_FILE_BYTES) -> Tool:
    root = Path(workspace).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {root}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    def read_text_file(arguments: dict[str, Any]) -> dict[str, Any]:
        relative = Path(arguments["path"])
        if relative.is_absolute() or not relative.parts:
            raise ValueError("path must be a non-empty relative path inside the workspace")
        if any(part in {"", ".", ".."} or part.startswith(".") or part in _IGNORED_COMPONENTS
               for part in relative.parts):
            raise PermissionError("hidden, parent, and generated paths are not allowed")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PermissionError("resolved path escapes the selected workspace") from exc
        if not candidate.is_file():
            raise ValueError("path must refer to a regular file")
        if candidate.name.casefold() in _DENIED_FILENAMES or candidate.suffix.lower() not in _ALLOWED_TEXT_SUFFIXES:
            raise PermissionError("file type is not allowed for the read-only tool")
        size = candidate.stat().st_size
        if size > max_bytes:
            raise ValueError(f"file exceeds max_bytes ({size} > {max_bytes})")
        content = candidate.read_text(encoding="utf-8-sig")
        return {
            "path": candidate.relative_to(root).as_posix(),
            "bytes": size,
            "text": content,
        }

    return Tool(
        name="read_text_file",
        description="Read one bounded text/code file inside the explicitly selected workspace.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 1024},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=read_text_file,
        capability="filesystem_read",
    )


def create_builtin_registry(
    *,
    workspace: str | Path | None = None,
    max_text_file_bytes: int = MAX_TEXT_FILE_BYTES,
) -> ToolRegistry:
    """Create the default allow-listed tools; no shell or network tools exist."""
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="Evaluate bounded arithmetic expressions without code execution.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "minLength": 1, "maxLength": MAX_EXPRESSION_CHARS},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculate,
        capability="compute",
    ))
    if workspace is not None:
        registry.register(_make_text_reader(workspace, max_text_file_bytes))
    return registry
