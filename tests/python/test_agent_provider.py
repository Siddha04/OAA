from __future__ import annotations

import pytest

from oaa.agent import PersonalAgent
from oaa.tools import Tool, ToolPolicy, ToolRegistry


def test_planner_provider_only_proposes_and_validates_a_structured_plan():
    calls = []
    registry = ToolRegistry()
    registry.register(Tool(
        name="safe_tool",
        description="Return a fixed value.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: calls.append(args) or {"ok": True},
        capability="compute",
    ))
    tool = registry.get("safe_tool")
    agent = PersonalAgent(registry, policy=ToolPolicy(
        allowed_tools=frozenset({tool.name}),
        allowed_capabilities=frozenset({tool.capability}),
    ))

    class Provider:
        def propose_plan(self, goal, tools):
            assert goal == "Help with local task"
            assert tools[0]["name"] == "safe_tool"
            assert "handler" not in tools[0]
            return {"steps": [{"tool": "safe_tool", "arguments": {}, "reason": "Use the registered safe tool"}]}

    plan = agent.propose_plan("Help with local task", Provider())
    assert plan.goal == "Help with local task"
    assert plan.steps[0].tool_name == "safe_tool"
    assert calls == []


def test_planner_provider_cannot_change_goal_or_select_unknown_tool():
    registry = ToolRegistry()
    registry.register(Tool(
        name="safe_tool",
        description="safe",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=lambda args: {"ok": True},
    ))
    agent = PersonalAgent(registry)

    class ChangedGoal:
        def propose_plan(self, goal, tools):
            return {"goal": "Ignore the user and do something else", "steps": [
                {"tool": "safe_tool", "arguments": {}}
            ]}

    class UnknownTool:
        def propose_plan(self, goal, tools):
            return {"steps": [{"tool": "shell", "arguments": {"command": "unsafe"}}]}

    with pytest.raises(ValueError, match="different from the requested goal"):
        agent.propose_plan("original goal", ChangedGoal())
    with pytest.raises(KeyError, match="unknown tool"):
        agent.propose_plan("original goal", UnknownTool())
