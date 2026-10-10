# OAA Phase 8 - Long-Term Memory

## Step 8A: persistent memory store

- Uses Python's standard-library SQLite; no external database service is required.
- Stores an explicit memory content string, category, source, and created/updated timestamps.
- Supports add, get, list, filtered list, lexical search, update, delete, and clear.
- Uses parameterized SQL queries. Limits and identifiers are validated.
- The database path is local, defaults to `.oaa/memory.sqlite3`, and the repository ignores `.oaa/` and SQLite database files.
- Memory records persist when a new MemoryStore instance opens the same file.

## Privacy rules

- The store never receives chat transcripts automatically.
- Call `MemoryStore.add()` only after a user explicitly asks OAA to remember something.
- The SQLite file is not encrypted by this implementation. Use OS disk/account protections and do not save passwords, authentication tokens, financial account secrets, or other highly sensitive data.
- Deleting a row removes it from the logical store; SQLite journaling and filesystem backups may retain data until the relevant files/backups are removed.

## Scope and limitations

This step builds the storage layer only. Automatic recall/injection and explicit CLI memory-management commands are implemented in the following Phase 8 verification gates. Search is local lexical matching, not semantic vector retrieval.

## Step 8B: explicit memory recall

- PersonalAssistant can optionally receive a MemoryStore and recall relevant saved notes for normal and streaming chat.
- Saved memories enter a separate, bounded `<memory_context>` block and are identified as reference notes, not instructions.
- Stored markup is escaped before prompt insertion.
- Recall is opt-in and lexical; ordinary chat without a MemoryStore keeps the same prompt behavior.
- User and assistant turns are not added to persistent memory. Saving remains a separate explicit operation.
- Common query stopwords are ignored during lexical memory search to reduce matches on generic wording.

## Step 8C: explicit memory-management CLI

- `genai memory add/list/search/update/delete/clear` manages a local SQLite database.
- The default path is `.oaa/memory.sqlite3`; commands also accept `--db`.
- Clearing all memories requires the explicit `--yes` flag. A request without it is rejected and preserves all records.
- `genai chat --memory-db <path>` opts into recall of saved records. It does not automatically save the user's prompt or generated response.
