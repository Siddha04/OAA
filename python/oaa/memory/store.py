from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: int
    content: str
    category: str
    source: str
    created_at: str
    updated_at: str


class MemoryStore:
    """SQLite-backed, local persistent memory with explicit CRUD operations.

    No chat messages are saved automatically. Call add() only when the user
    explicitly asks to remember something. Use a local path such as
    .oaa/memory.sqlite3; the database is not encrypted by this class.
    """

    def __init__(self, path: str | Path = ".oaa/memory.sqlite3") -> None:
        if str(path) == ":memory:":
            raise ValueError("use a local file path for persistent MemoryStore")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds")

    @staticmethod
    def _record(row: sqlite3.Row | None) -> MemoryRecord | None:
        if row is None:
            return None
        return MemoryRecord(
            id=int(row["id"]),
            content=str(row["content"]),
            category=str(row["category"]),
            source=str(row["source"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _validate_text(value: str, field: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{field} must not be empty")
        return cleaned

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS oaa_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT 'user',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_oaa_memories_updated "
                "ON oaa_memories(updated_at DESC, id DESC)"
            )

    def add(
        self,
        content: str,
        *,
        category: str = "fact",
        source: str = "user",
    ) -> MemoryRecord:
        content = self._validate_text(content, "content")
        category = self._validate_text(category, "category")
        source = self._validate_text(source, "source")
        now = self._now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO oaa_memories (content, category, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (content, category, source, now, now),
            )
            row = connection.execute(
                "SELECT * FROM oaa_memories WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        record = self._record(row)
        assert record is not None
        return record

    def get(self, memory_id: int) -> MemoryRecord | None:
        self._validate_id(memory_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM oaa_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._record(row)

    def list_memories(
        self,
        *,
        category: str | None = None,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        self._validate_limit(limit)
        if category is not None:
            category = self._validate_text(category, "category")
        with self._connect() as connection:
            if category is None:
                rows = connection.execute(
                    "SELECT * FROM oaa_memories ORDER BY id ASC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM oaa_memories WHERE category = ? ORDER BY id ASC LIMIT ?",
                    (category, limit),
                ).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    def search(self, query: str, *, limit: int = 10) -> list[MemoryRecord]:
        query = self._validate_text(query, "query")
        self._validate_limit(limit)
        tokens = list(dict.fromkeys(re.findall(r"[\w-]+", query.casefold())))
        if not tokens:
            return []

        predicates = []
        parameters: list[object] = []
        for token in tokens:
            predicates.append(
                "(instr(lower(content), ?) > 0 OR "
                "instr(lower(category), ?) > 0 OR instr(lower(source), ?) > 0)"
            )
            parameters.extend((token, token, token))

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM oaa_memories WHERE "
                + " OR ".join(predicates)
                + " ORDER BY updated_at DESC, id DESC",
                parameters,
            ).fetchall()

        records = [record for row in rows if (record := self._record(row)) is not None]
        records.sort(
            key=lambda item: (
                -sum(token in (item.content + " " + item.category + " " + item.source).casefold()
                     for token in tokens),
                item.updated_at,
                item.id,
            ),
            reverse=True,
        )
        return records[:limit]

    def update(
        self,
        memory_id: int,
        content: str,
        *,
        category: str | None = None,
        source: str | None = None,
    ) -> MemoryRecord | None:
        self._validate_id(memory_id)
        content = self._validate_text(content, "content")
        current = self.get(memory_id)
        if current is None:
            return None
        if category is not None:
            category = self._validate_text(category, "category")
        if source is not None:
            source = self._validate_text(source, "source")
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE oaa_memories
                SET content = ?, category = ?, source = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    category if category is not None else current.category,
                    source if source is not None else current.source,
                    now,
                    memory_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM oaa_memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._record(row)

    def delete(self, memory_id: int) -> bool:
        self._validate_id(memory_id)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM oaa_memories WHERE id = ?", (memory_id,)
            )
        return cursor.rowcount > 0

    def clear(self) -> int:
        """Delete all saved memories; callers should require explicit confirmation."""
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM oaa_memories")
        return cursor.rowcount

    @staticmethod
    def _validate_id(memory_id: int) -> None:
        if not isinstance(memory_id, int) or isinstance(memory_id, bool) or memory_id <= 0:
            raise ValueError("memory_id must be a positive integer")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
