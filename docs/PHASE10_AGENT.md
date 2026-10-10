# OAA Phase 10 - Approval-Gated Personal Agent

## Scope

Phase 10 introduces a reusable Python agent that validates and executes a bounded, structured list of tool proposals through the existing Phase 9 registry. A plan contains a goal and ordered steps. Each step names a registered tool, provides JSON-object arguments, and may include a brief reason.

## Safety and execution contract

- Plan parsing and previewing never call tool handlers.
- The plan has a non-empty goal, at least one step, bounded text/argument sizes, and a configurable step limit (maximum 16).
- All tools and argument schemas are validated before any handler can run.
- The plan must pass the exact tool-name and capability allow-list before the first execution.
- Each step separately requires an approval callback; omitting the callback is preview-only.
- A rejected step or failed tool stops the rest of the plan.
- Outputs remain bounded by the existing registry and configured output character limit.
- The agent has no shell or network tool and never evaluates a plan as Python code.
- Tool output is returned to the caller; it is not silently converted into another tool action.

## CLI

Preview a JSON plan file:

    genai agent plan --plan-file ./agent-plan.json

Execute with an explicit tool allow-list and per-step terminal approval:

    genai agent run --plan-file ./agent-plan.json --allow-tool calculator

The JSON shape is:

    {
      "goal": "Calculate a half of 100",
      "steps": [
        {
          "tool": "calculator",
          "arguments": {"expression": "100 / 2"},
          "reason": "Compute the requested value"
        }
      ]
    }

For multi-step plans, repeat --allow-tool for every tool name that should be permitted. Tool calls outside that allow-list are denied before any step executes. The read-only read_text_file tool is available only when the CLI receives --workspace ./some-directory.

## Current limitation

Plans are structured caller/provider inputs; Phase 10 does not claim that the seeded tiny transformer can author reliable tool plans from free-form language. The project has not yet added a pretrained LLM, autonomous tool-call decoding, shell access, network access, or automatic chaining of tool outputs.
