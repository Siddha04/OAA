"""Bounded, approval-gated orchestration for explicitly proposed local tool calls."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .tools.registry import (
    ToolExecutionResult,
    ToolPermissionError,
    ToolPolicy,
    ToolRegistry,
    validate_arguments,
)

MAX_PLAN_GOAL_CHARS = 2048
MAX_TOOL_ARGUMENTS_CHARS = 16 * 1024
MAX_TOOL_RATIONALE_CHARS = 512
MAX_AGENT_STEPS = 16


@dataclass(frozen=True, slots=True)
class AgentToolCall:
    """A proposal, not an execution request."""

    tool_name: str
    arguments: dict[str, Any]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name or len(self.tool_name) > 64:
            raise ValueError("tool_name must be a non-empty string of at most 64 characters")
        if not isinstance(self.arguments, dict):
            raise ValueError("tool arguments must be a JSON object")
        if not isinstance(self.rationale, str) or len(self.rationale) > MAX_TOOL_RATIONALE_CHARS:
            raise ValueError(f"rationale must be a string of at most {MAX_TOOL_RATIONALE_CHARS} characters")
        try:
            encoded = json.dumps(
                self.arguments, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("tool arguments must contain only finite JSON values") from exc
        if len(encoded) > MAX_TOOL_ARGUMENTS_CHARS:
            raise ValueError(
                f"tool arguments exceed {MAX_TOOL_ARGUMENTS_CHARS} characters"
            )
        # Detach from caller-owned mutable structures and normalize to JSON data.
        object.__setattr__(self, "arguments", json.loads(encoded))


@dataclass(frozen=True, slots=True)
class AgentPlan:
    """A finite list of tool proposals supplied by a caller or planner."""

    goal: str
    steps: tuple[AgentToolCall, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValueError("plan goal must be a non-empty string")
        if len(self.goal) > MAX_PLAN_GOAL_CHARS:
            raise ValueError(f"plan goal exceeds {MAX_PLAN_GOAL_CHARS} characters")
        if not isinstance(self.steps, (tuple, list)) or not self.steps:
            raise ValueError("plan must contain at least one tool step")
        if not all(isinstance(step, AgentToolCall) for step in self.steps):
            raise TypeError("plan steps must be AgentToolCall instances")
        object.__setattr__(self, "steps", tuple(self.steps))

    @classmethod
    def from_mapping(cls, payload: Any) -> "AgentPlan":
        """Parse the strict JSON-compatible plan shape without running any tool."""
        if not isinstance(payload, Mapping):
            raise ValueError("plan must be a JSON object")
        unknown = set(payload) - {"goal", "steps"}
        missing = {"goal", "steps"} - set(payload)
        if missing:
            raise ValueError(f"plan is missing field: {sorted(missing)[0]}")
        if unknown:
            raise ValueError(f"plan contains unknown field: {sorted(unknown)[0]}")
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, list):
            raise ValueError("plan steps must be a JSON array")
        calls: list[AgentToolCall] = []
        for index, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, Mapping):
                raise ValueError(f"plan step {index} must be a JSON object")
            unknown_step = set(raw) - {"tool", "arguments", "reason"}
            missing_step = {"tool", "arguments"} - set(raw)
            if missing_step:
                raise ValueError(
                    f"plan step {index} is missing field: {sorted(missing_step)[0]}"
                )
            if unknown_step:
                raise ValueError(
                    f"plan step {index} contains unknown field: {sorted(unknown_step)[0]}"
                )
            calls.append(
                AgentToolCall(
                    tool_name=raw["tool"],
                    arguments=raw["arguments"],
                    rationale=raw.get("reason", ""),
                )
            )
        return cls(goal=payload["goal"], steps=tuple(calls))

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "tool": call.tool_name,
                    "arguments": call.arguments,
                    "reason": call.rationale,
                }
                for call in self.steps
            ],
        }


@dataclass(frozen=True, slots=True)
class AgentStepResult:
    """Outcome for one proposal; a missing execution means no handler ran."""

    step: int
    tool_name: str
    status: str
    rationale: str
    execution: ToolExecutionResult | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "step": self.step,
            "tool": self.tool_name,
            "status": self.status,
            "reason": self.rationale,
        }
        if self.execution is not None:
            data["success"] = self.execution.success
            data["output"] = self.execution.output
            data["error"] = self.execution.error
            data["truncated"] = self.execution.truncated
        if self.message is not None:
            data["message"] = self.message
        return data


ApprovalCallback = Callable[[AgentToolCall], bool]


class PersonalAgent:
    """Run a fixed, bounded plan only after policy checks and per-step approval.

    The default policy denies all tools. A plan is never inferred from free-form
    model text here: callers provide structured proposals, and every proposal is
    validated by the same ToolRegistry contract used by the explicit tool CLI.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: ToolPolicy | None = None,
        max_steps: int = 4,
        max_output_chars: int = 8192,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or not 1 <= max_steps <= MAX_AGENT_STEPS
        ):
            raise ValueError(f"max_steps must be between 1 and {MAX_AGENT_STEPS}")
        if (
            not isinstance(max_output_chars, int)
            or isinstance(max_output_chars, bool)
            or max_output_chars <= 0
        ):
            raise ValueError("max_output_chars must be a positive integer")
        if policy is not None and not isinstance(policy, ToolPolicy):
            raise TypeError("policy must be a ToolPolicy")
        self._registry = registry
        self._policy = policy or ToolPolicy()
        self._max_steps = max_steps
        self._max_output_chars = max_output_chars

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def prepare_plan(self, payload: Any) -> AgentPlan:
        """Parse and validate every proposed tool and its arguments; do not execute."""
        plan = AgentPlan.from_mapping(payload)
        if len(plan.steps) > self._max_steps:
            raise ValueError(f"plan exceeds configured maximum of {self._max_steps} steps")
        for step in plan.steps:
            tool = self._registry.get(step.tool_name)
            validate_arguments(step.arguments, tool.parameters)
        return plan

    def run_plan(
        self,
        plan: AgentPlan,
        *,
        approve: ApprovalCallback | None = None,
    ) -> tuple[AgentStepResult, ...]:
        """Execute in order only with an allow-list and positive approval for each step.

        When approve is omitted, the plan is a preview only and every returned
        step has status 'approval_required'. A declined approval or failed tool
        stops the remainder of the plan. No model text or tool output is executed
        as code, and tool results are not silently promoted into new actions.
        """
        if not isinstance(plan, AgentPlan):
            raise TypeError("plan must be an AgentPlan")
        validated = self.prepare_plan(plan.to_dict())

        # Preflight the whole plan before allowing its first side effect.
        for call in validated.steps:
            tool = self._registry.get(call.tool_name)
            if not self._policy.permits(tool):
                raise ToolPermissionError(
                    f"policy denies tool '{tool.name}' (capability '{tool.capability}')"
                )

        results: list[AgentStepResult] = []
        for index, call in enumerate(validated.steps, start=1):
            if approve is None:
                results.append(AgentStepResult(
                    step=index,
                    tool_name=call.tool_name,
                    status="approval_required",
                    rationale=call.rationale,
                    message="No approval callback was supplied; no tool was executed.",
                ))
                continue
            if not approve(call):
                results.append(AgentStepResult(
                    step=index,
                    tool_name=call.tool_name,
                    status="rejected",
                    rationale=call.rationale,
                    message="Approval was not granted; remaining steps were not run.",
                ))
                break

            execution = self._registry.execute(
                call.tool_name,
                call.arguments,
                policy=self._policy,
                max_output_chars=self._max_output_chars,
            )
            succeeded = execution.success
            results.append(AgentStepResult(
                step=index,
                tool_name=call.tool_name,
                status="executed" if succeeded else "failed",
                rationale=call.rationale,
                execution=execution,
                message=None if succeeded else "Tool failed; remaining steps were not run.",
            ))
            if not succeeded:
                break
        return tuple(results)
