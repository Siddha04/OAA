from __future__ import annotations

import pytest

from oaa.agent import AgentPlan, PersonalAgent
from oaa.tools import Tool, ToolPermissionError, ToolPolicy, ToolRegistry


def make_registry(calls: list[dict]) -> ToolRegistry:
    registry = ToolRegistry()

    def calculate(arguments):
        calls.append(dict(arguments))
        return {"result": arguments["value"] * 2}

    registry.register(Tool(
        name="double_value",
        description="Double a small integer for tests.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer", "minimum": 0, "maximum": 100}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=calculate,
        capability="compute",
    ))
    return registry


def payload(*steps):
    return {
        "goal": "Calculate a value for the user",
        "steps": list(steps),
    }


def call(value: int, reason: str = "Compute the requested value") -> dict:
    return {
        "tool": "double_value",
        "arguments": {"value": value},
        "reason": reason,
    }


def allow(registry: ToolRegistry) -> ToolPolicy:
    tool = registry.get("double_value")
    return ToolPolicy(
        allowed_tools=frozenset({tool.name}),
        allowed_capabilities=frozenset({tool.capability}),
    )


def test_plan_preview_validates_without_executing_any_handler():
    calls = []
    registry = make_registry(calls)
    agent = PersonalAgent(registry, policy=allow(registry))
    plan = agent.prepare_plan(payload(call(7)))

    assert plan.goal == "Calculate a value for the user"
    assert plan.to_dict()["steps"][0]["arguments"] == {"value": 7}
    assert calls == []

    results = agent.run_plan(plan)
    assert [step.status for step in results] == ["approval_required"]
    assert calls == []


def test_execution_requires_policy_and_per_step_approval():
    calls = []
    registry = make_registry(calls)
    agent = PersonalAgent(registry, policy=allow(registry))
    plan = agent.prepare_plan(payload(call(7)))

    results = agent.run_plan(plan, approve=lambda proposal: proposal.tool_name == "double_value")
    assert [step.status for step in results] == ["executed"]
    assert results[0].execution.output == {"result": 14}
    assert calls == [{"value": 7}]


def test_default_policy_denies_entire_plan_before_any_side_effect():
    calls = []
    registry = make_registry(calls)
    agent = PersonalAgent(registry)
    plan = agent.prepare_plan(payload(call(7)))

    with pytest.raises(ToolPermissionError, match="policy denies"):
        agent.run_plan(plan, approve=lambda _: True)
    assert calls == []


def test_declined_approval_stops_remaining_steps():
    calls = []
    registry = make_registry(calls)
    agent = PersonalAgent(registry, policy=allow(registry))
    plan = agent.prepare_plan(payload(call(1), call(2)))

    results = agent.run_plan(plan, approve=lambda _: False)
    assert len(results) == 1
    assert results[0].status == "rejected"
    assert calls == []


def test_approval_is_requested_for_each_step_and_execution_stops_on_failure():
    calls = []
    registry = make_registry(calls)
    approvals = []
    agent = PersonalAgent(registry, policy=allow(registry))
    plan = agent.prepare_plan(payload(call(1), call(2)))

    def approve(proposal):
        approvals.append(proposal.arguments["value"])
        return proposal.arguments["value"] == 1

    results = agent.run_plan(plan, approve=approve)
    assert approvals == [1, 2]
    assert [step.status for step in results] == ["executed", "rejected"]
    assert calls == [{"value": 1}]


def test_plan_rejects_unknown_fields_invalid_arguments_and_unknown_tools():
    registry = make_registry([])
    agent = PersonalAgent(registry, policy=allow(registry))

    with pytest.raises(ValueError, match="unknown field"):
        agent.prepare_plan({"goal": "test", "steps": [call(1)], "auto_execute": True})
    with pytest.raises(ValueError, match="missing field"):
        agent.prepare_plan({"goal": "test", "steps": [{"tool": "double_value"}]})
    with pytest.raises(ValueError, match="unknown field"):
        agent.prepare_plan(payload({**call(1), "shell": "echo unsafe"}))
    with pytest.raises(ValueError, match="below minimum"):
        agent.prepare_plan(payload(call(-1)))
    with pytest.raises(KeyError, match="unknown tool"):
        agent.prepare_plan(payload({"tool": "shell", "arguments": {"command": "id"}}))


def test_plan_bounds_and_json_payloads():
    registry = make_registry([])
    agent = PersonalAgent(registry, policy=allow(registry), max_steps=2)
    with pytest.raises(ValueError, match="maximum of 2 steps"):
        agent.prepare_plan(payload(call(1), call(2), call(3)))
    with pytest.raises(ValueError, match="positive integer"):
        PersonalAgent(registry, max_output_chars=0)
    with pytest.raises(ValueError, match="non-empty"):
        AgentPlan.from_mapping({"goal": "", "steps": [call(1)]})
    with pytest.raises(ValueError, match="finite JSON"):
        from oaa.agent import AgentToolCall
        AgentToolCall("double_value", {"value": float("nan")})


def test_tool_arguments_are_detached_from_caller_mutations():
    registry = make_registry([])
    agent = PersonalAgent(registry, policy=allow(registry))
    raw = payload(call(5))
    plan = agent.prepare_plan(raw)
    raw["steps"][0]["arguments"]["value"] = 99
    assert plan.steps[0].arguments["value"] == 5
