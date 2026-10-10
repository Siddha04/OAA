from __future__ import annotations

import pytest

from oaa.tools import (
    Tool,
    ToolPermissionError,
    ToolPolicy,
    ToolRegistry,
    ToolValidationError,
)


BASIC_SCHEMA = {
    "type": "object",
    "properties": {
        "value": {"type": "integer", "minimum": 1, "maximum": 9},
        "label": {"type": "string", "minLength": 1, "maxLength": 16},
    },
    "required": ["value"],
    "additionalProperties": False,
}


def make_tool(handler=lambda args: {"value": args["value"]}, *, capability="compute") -> Tool:
    return Tool(
        name="sample_tool",
        description="A test tool",
        parameters=BASIC_SCHEMA,
        handler=handler,
        capability=capability,
    )


def test_registry_lists_tools_deterministically_without_running_them() -> None:
    calls = []
    registry = ToolRegistry()
    registry.register(make_tool(lambda args: calls.append(args) or "ok"))
    registry.register(Tool(
        name="another_tool",
        description="Second tool",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: "second",
    ))
    assert [tool.name for tool in registry.list_tools()] == ["another_tool", "sample_tool"]
    assert calls == []


def test_registry_requires_explicit_tool_and_capability_allowlist() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    with pytest.raises(ToolPermissionError):
        registry.execute("sample_tool", {"value": 2})
    denied = ToolPolicy(
        allowed_tools=frozenset({"sample_tool"}),
        allowed_capabilities=frozenset({"filesystem_read"}),
    )
    with pytest.raises(ToolPermissionError):
        registry.execute("sample_tool", {"value": 2}, policy=denied)
    allowed = ToolPolicy(
        allowed_tools=frozenset({"sample_tool"}),
        allowed_capabilities=frozenset({"compute"}),
    )
    assert registry.execute("sample_tool", {"value": 2}, policy=allowed).output == {"value": 2}


@pytest.mark.parametrize(
    "arguments",
    [
        {"value": True},
        {"value": 0},
        {"value": 10},
        {"value": 2, "unexpected": "not allowed"},
        {},
        {"value": 2, "label": ""},
    ],
)
def test_registry_validates_argument_contracts(arguments) -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    policy = ToolPolicy(
        allowed_tools=frozenset({"sample_tool"}),
        allowed_capabilities=frozenset({"compute"}),
    )
    with pytest.raises(ToolValidationError):
        registry.execute("sample_tool", arguments, policy=policy)


def test_registry_rejects_duplicates_and_bad_schemas() -> None:
    registry = ToolRegistry()
    registry.register(make_tool())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_tool())
    with pytest.raises(ValueError, match="unsupported schema keyword"):
        Tool("bad_schema", "bad", {"type": "object", "execute": True}, lambda args: None)
    with pytest.raises(ValueError, match="tool name"):
        Tool("Not Valid!", "bad", {"type": "object"}, lambda args: None)


def test_tool_handler_failures_are_returned_as_results() -> None:
    registry = ToolRegistry()
    registry.register(make_tool(lambda args: (_ for _ in ()).throw(RuntimeError("controlled failure"))))
    policy = ToolPolicy(
        allowed_tools=frozenset({"sample_tool"}),
        allowed_capabilities=frozenset({"compute"}),
    )
    result = registry.execute("sample_tool", {"value": 2}, policy=policy)
    assert not result.success
    assert "RuntimeError" in result.error
    assert "controlled failure" in result.error


def test_registry_bounds_large_outputs_and_rejects_invalid_output() -> None:
    registry = ToolRegistry()
    policy = ToolPolicy(
        allowed_tools=frozenset({"sample_tool"}),
        allowed_capabilities=frozenset({"compute"}),
    )
    registry.register(make_tool(lambda args: "x" * 1000))
    result = registry.execute("sample_tool", {"value": 2}, policy=policy, max_output_chars=100)
    assert result.success
    assert result.truncated
    assert len(result.output["preview"]) <= 100

    other = ToolRegistry()
    other.register(make_tool(lambda args: float("nan")))
    result = other.execute("sample_tool", {"value": 2}, policy=policy)
    assert not result.success
    assert "ValueError" in result.error
