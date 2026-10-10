from __future__ import annotations

from pathlib import Path

import pytest

from oaa.memory import MemoryStore


def test_memory_persists_across_store_instances(tmp_path: Path) -> None:
    path = tmp_path / "private" / "memory.sqlite3"
    first = MemoryStore(path)
    saved = first.add("Prefers concise technical summaries", category="preference")
    second = MemoryStore(path)
    loaded = second.get(saved.id)
    assert loaded is not None
    assert loaded.content == saved.content
    assert loaded.category == "preference"
    assert loaded.source == "user"
    assert loaded.created_at == saved.created_at


def test_memory_search_prioritizes_more_matching_tokens(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add("Prefers Python", category="preference")
    store.add("Python examples include more examples", category="preference")
    results = store.search("Python examples")
    assert len(results) == 2
    assert results[0].content == "Python examples include more examples"


def test_memory_update_and_delete_lifecycle(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    saved = store.add("Old preference", category="preference")
    updated = store.update(saved.id, "Updated preference", source="user-approved")
    assert updated is not None
    assert updated.content == "Updated preference"
    assert updated.category == "preference"
    assert updated.source == "user-approved"
    assert updated.created_at == saved.created_at
    assert updated.updated_at >= saved.updated_at
    assert store.delete(saved.id) is True
    assert store.delete(saved.id) is False
    assert store.get(saved.id) is None


def test_list_filter_limit_and_clear(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add("Likes concise output", category="preference")
    store.add("OAA project uses C++", category="project")
    store.add("Python package uses pybind11", category="project")
    assert len(store.list_memories(limit=2)) == 2
    assert len(store.list_memories(category="project")) == 2
    assert store.clear() == 3
    assert store.list_memories() == []


def test_memory_inputs_are_validated(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    with pytest.raises(ValueError, match="content"):
        store.add("   ")
    with pytest.raises(ValueError, match="category"):
        store.add("fact", category=" ")
    with pytest.raises(ValueError, match="limit"):
        store.list_memories(limit=0)
    with pytest.raises(ValueError, match="memory_id"):
        store.delete(0)
    with pytest.raises(ValueError, match="persistent"):
        MemoryStore(":memory:")


def test_memory_search_does_not_treat_sql_as_code(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add("The safe memory contains SQL text like OR 1=1")
    assert store.search("' OR 1=1 --")
    assert len(store.list_memories()) == 1
