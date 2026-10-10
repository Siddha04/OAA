# OAA Phase 9 - Controlled Tool System

## Step 9A: registry, contracts, and permissions

- Tools are registered explicitly with a name, description, JSON-compatible object schema, handler, and capability.
- Tool names and capability names follow a restricted identifier format.
- The supported schema subset validates objects, required fields, additional-property policy, scalar types, enums, strings, numeric bounds, arrays, and nested array items.
- Registry discovery is deterministic and does not execute tools.
- Execution defaults to deny. A ToolPolicy must allow both the exact tool name and its capability.
- Handler failures become structured results. Non-JSON/non-finite results are rejected and serialized output size is bounded.
- No arbitrary shell, dynamic Python evaluation, or network execution is provided by the registry.

## Step 9B: built-in local tools

Built-ins include a bounded arithmetic calculator and a read-only text-file tool limited to an explicitly supplied workspace root. File paths must remain inside that root after resolution; hidden/generated folders and oversized files are rejected.

Example:

    genai tools list
    genai tools run calculator --args '{"expression":"(2 + 3) * 4"}'
    genai tools run read_text_file --workspace ./knowledge --args '{"path":"manual.md"}'

The CLI only invokes the named tool on explicit user command. Tool registry policies are allow-listed per invocation; the framework does not automatically execute tools based on model-generated text.

## Future extensions

Phase 10 may let a personal agent propose tool calls, but those calls should still be parsed, validated, and checked against the same explicit permission policy. Destructive or network tools should require additional consent and should not be registered by default.


## Step 9C: explicit tool CLI

- `genai tools list` lists registered tools without invoking their handlers. The read-file tool appears only when `--workspace` points to an existing local directory.
- `genai tools run <tool> --args '<JSON object>'` invokes one explicitly named tool after parsing its JSON arguments.
- The CLI creates a policy scoped to that named tool and its declared capability for this explicit invocation.
- Invalid JSON/schema, denied paths, unsafe math, handler failures, and oversized outputs are reported without falling back to shell execution.
