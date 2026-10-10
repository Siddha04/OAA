from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


class ToolValidationError(ValueError):
    """The tool name or its arguments do not satisfy the declared contract."""


class ToolPermissionError(PermissionError):
    """The active policy does not allow the requested tool capability."""


ToolHandler = Callable[[dict[str, Any]], Any]
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler = field(repr=False, compare=False)
    capability: str = "compute"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError("tool name must match [a-z][a-z0-9_]{0,63}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be non-empty")
        if not isinstance(self.capability, str) or not _NAME_PATTERN.fullmatch(self.capability):
            raise ValueError("tool capability must match [a-z][a-z0-9_]{0,63}")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be an object schema")
        if self.parameters.get("type") != "object":
            raise ValueError('tool parameters schema must declare type="object"')
        # Reject unsupported schema keywords at registration, not during execution.
        _validate_schema_definition(dict(self.parameters), "$")


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    """Explicit allow-list. The default policy denies every tool."""

    allowed_tools: frozenset[str] = frozenset()
    allowed_capabilities: frozenset[str] = frozenset()

    def permits(self, tool: Tool) -> bool:
        return tool.name in self.allowed_tools and tool.capability in self.allowed_capabilities


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    name: str
    success: bool
    output: Any = None
    error: str | None = None
    truncated: bool = False


_ALLOWED_SCHEMA_KEYS = {
    "type", "properties", "required", "additionalProperties", "enum",
    "minLength", "maxLength", "minimum", "maximum", "items", "minItems", "maxItems",
}
_SUPPORTED_TYPES = {"object", "string", "integer", "number", "boolean", "array", "null"}


def _validate_schema_definition(schema: dict[str, Any], path: str) -> None:
    unknown = set(schema) - _ALLOWED_SCHEMA_KEYS
    if unknown:
        raise ValueError(f"unsupported schema keyword at {path}: {sorted(unknown)[0]}")
    kind = schema.get("type")
    if kind not in _SUPPORTED_TYPES:
        raise ValueError(f"unsupported or missing schema type at {path}")
    for key in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
        if key in schema and (not isinstance(schema[key], (int, float)) or isinstance(schema[key], bool)):
            raise ValueError(f"{key} at {path} must be numeric")
        if key in schema and isinstance(schema[key], (int, float)) and not math.isfinite(schema[key]):
            raise ValueError(f"{key} at {path} must be finite")
    if "properties" in schema:
        if kind != "object" or not isinstance(schema["properties"], dict):
            raise ValueError(f"properties at {path} is valid only for object schemas")
        for key, value in schema["properties"].items():
            if not isinstance(key, str) or not isinstance(value, dict):
                raise ValueError(f"invalid property schema at {path}")
            _validate_schema_definition(value, f"{path}.{key}")
    if "required" in schema:
        required = schema["required"]
        if kind != "object" or not isinstance(required, list) or not all(isinstance(x, str) for x in required):
            raise ValueError(f"required at {path} must be a string list on objects")
        if len(set(required)) != len(required):
            raise ValueError(f"required at {path} must not contain duplicates")
        unknown_required = set(required) - set(schema.get("properties", {}))
        if unknown_required:
            raise ValueError(f"required at {path} references undefined properties")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise ValueError(f"additionalProperties at {path} must be boolean")
    if "items" in schema:
        if kind != "array" or not isinstance(schema["items"], dict):
            raise ValueError(f"items at {path} is valid only for array schemas")
        _validate_schema_definition(schema["items"], f"{path}[]")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"enum at {path} must be a list")


def validate_arguments(arguments: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Validate arguments against the deliberately small, documented schema subset."""
    if not isinstance(arguments, dict):
        raise ToolValidationError("tool arguments must be a JSON object")
    _validate_schema_definition(dict(schema), "$")
    _validate_value(arguments, dict(schema), "$")
    return arguments


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    kind = schema["type"]
    valid = {
        "object": lambda x: isinstance(x, dict),
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "array": lambda x: isinstance(x, list),
        "null": lambda x: x is None,
    }[kind](value)
    if not valid:
        raise ToolValidationError(f"{path} must be {kind}")

    if "enum" in schema and value not in schema["enum"]:
        raise ToolValidationError(f"{path} must be one of the allowed values")

    if kind == "object":
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ToolValidationError(f"{path} is missing required field: {sorted(missing)[0]}")
        if schema.get("additionalProperties", True) is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ToolValidationError(f"{path} contains unknown field: {sorted(unknown)[0]}")
        for key, item in value.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}")

    if kind == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ToolValidationError(f"{path} is shorter than minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ToolValidationError(f"{path} exceeds maxLength")

    if kind in {"integer", "number"}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolValidationError(f"{path} must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolValidationError(f"{path} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolValidationError(f"{path} exceeds maximum")

    if kind == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolValidationError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ToolValidationError(f"{path} has too many items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_value(item, schema["items"], f"{path}[{index}]")


class ToolRegistry:
    """Process-local registry; tool discovery never executes a handler."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("register() expects a Tool")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def list_tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools[name] for name in sorted(self._tools))

    def execute(
        self,
        name: str,
        arguments: Any,
        *,
        policy: ToolPolicy | None = None,
        max_output_chars: int = 8192,
    ) -> ToolExecutionResult:
        if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool) or max_output_chars <= 0:
            raise ValueError("max_output_chars must be a positive integer")
        tool = self.get(name)
        active_policy = policy or ToolPolicy()
        if not active_policy.permits(tool):
            raise ToolPermissionError(
                f"policy denies tool '{tool.name}' (capability '{tool.capability}')"
            )
        clean_arguments = validate_arguments(arguments, tool.parameters)
        try:
            output = tool.handler(clean_arguments)
            encoded = json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except Exception as exc:
            return ToolExecutionResult(
                name=tool.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        if len(encoded) > max_output_chars:
            preview_limit = max(0, max_output_chars - 64)
            return ToolExecutionResult(
                name=tool.name,
                success=True,
                output={
                    "truncated": True,
                    "preview": encoded[:preview_limit],
                },
                truncated=True,
            )
        return ToolExecutionResult(name=tool.name, success=True, output=output)
