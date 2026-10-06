# OAA Phase 6 - Personal Assistant

Phase 6 adds personal assistant orchestration and a local CLI on top of the Phase 5 inference runtime.

6A:
- typed system/user/assistant messages
- bounded in-memory session history
- system prompt control
- structured prompt rendering
- stateful chat and token streaming
- session reset and fork

6B:
- genai ask for one-shot local inference
- genai chat for an interactive local session
- reset, system, stats, and exit commands
- generation controls exposed as CLI flags
- max history configuration

The CLI is local-only and uses the existing Python facade and C++ inference engine. No network service is introduced.

Limits:
- sessions are in memory only
- history is message-count based
- Phase 5 tiny transformer is a runtime validation model, not a pretrained conversational LLM
- persistent memory and retrieval remain future phases
