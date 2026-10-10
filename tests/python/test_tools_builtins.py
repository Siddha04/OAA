from __future__ import annotations

from pathlib import Path

import pytest

from oaa.tools import ToolPolicy
from oaa.tools.builtins import create_builtin_registry


def _policy(registry, name: str) -> ToolPolicy:
    tool = registry.get(name)
    return ToolPolicy(
        allowed_tools=frozenset({name}),
        allowed_capabilities=frozenset({tool.capability}),
    )


def test_calculator_handles_arithmetic_without_dynamic_execution() -> None:
    registry = create_builtin_registry()
    result = registry.execute(
        "calculator",
        {"expression": "(2 + 3) * 4 - 1"},
        policy=_policy(registry, "calculator"),
    )
    assert result.success
    assert result.output == {"result": 19}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo unsafe')",
        "open('secrets.txt').read()",
        "2 ** 99",
        "1 / 0",
        "[1, 2, 3]",
        "x + 1",
    ],
)
def test_calculator_rejects_unsafe_or_unbounded_expressions(expression: str) -> None:
    registry = create_builtin_registry()
    result = registry.execute(
        "calculator",
        {"expression": expression},
        policy=_policy(registry, "calculator"),
    )
    assert not result.success
    assert result.error


def test_text_reader_is_scoped_and_read_only(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manual.md").write_text("Local OAA manual", encoding="utf-8")
    (workspace / "private.txt").write_text("outside", encoding="utf-8")
    registry = create_builtin_registry(workspace=workspace)
    policy = _policy(registry, "read_text_file")

    result = registry.execute(
        "read_text_file", {"path": "manual.md"}, policy=policy
    )
    assert result.success
    assert result.output["text"] == "Local OAA manual"
    assert result.output["path"] == "manual.md"

    for path in ("../private.txt", ".env", ".git/config", str(workspace / "manual.md")):
        denied = registry.execute("read_text_file", {"path": path}, policy=policy)
        assert not denied.success


def test_text_reader_blocks_symlink_escape_and_oversized_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    try:
        (workspace / "linked.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable in this environment")
    registry = create_builtin_registry(workspace=workspace)
    result = registry.execute(
        "read_text_file", {"path": "linked.md"},
        policy=_policy(registry, "read_text_file"),
    )
    assert not result.success
    assert "escapes" in result.error

    (workspace / "large.md").write_text("123456", encoding="utf-8")
    small_limit = create_builtin_registry(workspace=workspace, max_text_file_bytes=3)
    large_result = small_limit.execute(
        "read_text_file", {"path": "large.md"},
        policy=_policy(small_limit, "read_text_file"),
    )
    assert not large_result.success
    assert "max_bytes" in large_result.error


def test_workspace_tool_is_not_registered_without_workspace() -> None:
    registry = create_builtin_registry()
    assert [tool.name for tool in registry.list_tools()] == ["calculator"]
    with pytest.raises(KeyError, match="unknown tool"):
        registry.get("read_text_file")
