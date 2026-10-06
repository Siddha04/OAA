# OAA Phase 6 - Personal Assistant

Phase 6 adds the personal assistant orchestration and local CLI on top of the Phase 5 inference runtime.

6A:
- typed system/user/assistant messages
- bounded in-memory session history
- system prompt control
- structured prompt rendering
- stateful chat and token streaming
- session reset and fork

6B:
- genai ask for one-shot local requests
- genai chat for an interactive local session
- local slash commands for reset, system prompt, stats, and exit
- generation controls exposed as CLI flags

CLI examples:

genai ask --model tiny.manifest --max-tokens 32 "Hello"

genai chat --model tiny.manifest --temperature 0.0

The CLI is local-only and uses the existing Python facade and C++ inference engine. It does not add a network service.

Limits:
- sessions are in memory only
- history is message-count based
- the Phase 5 tiny transformer is a runtime validation model, not a pretrained conversational LLM
- persistent memory and retrieval remain future phases
