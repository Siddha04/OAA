from __future__ import annotations

from pathlib import Path

from oaa.assistant import PersonalAssistant
from oaa.memory import MemoryStore


class FakeEngine:
    def __init__(self) -> None:
        self.prompt = ""
        self.calls = 0

    def generate(self, prompt: str, config=None) -> str:
        self.prompt = prompt
        self.calls += 1
        return "local response"

    def generate_stream(self, prompt: str, config=None):
        self.prompt = prompt
        self.calls += 1
        yield "local "
        yield "response"

    def get_stats(self):
        return {"calls": self.calls}


def test_assistant_recalls_only_matching_persisted_memories_without_autosaving(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    saved = store.add("Prefers concise Python examples", category="preference")
    store.add("Favorite pasta sauce uses basil", category="food")
    engine = FakeEngine()
    assistant = PersonalAssistant(engine, memory_store=store)

    assert assistant.chat("Please show Python examples") == "local response"
    assert "<memory_context>" in engine.prompt
    assert "Prefers concise Python examples" in engine.prompt
    assert "Favorite pasta sauce" not in engine.prompt
    assert "<memory_policy>" in engine.prompt
    assert len(store.list_memories()) == 2
    assert len(assistant.history()) == 2
    assert store.get(saved.id) is not None


def test_assistant_streaming_can_recall_saved_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add("Prefers concise technical answers", category="preference")
    engine = FakeEngine()
    assistant = PersonalAssistant(engine, memory_store=store)
    assert "".join(assistant.chat_stream("Give concise technical answers")) == "local response"
    assert "<memory_context>" in engine.prompt
    assert len(store.list_memories()) == 1


def test_assistant_memory_context_is_opt_in() -> None:
    engine = FakeEngine()
    assistant = PersonalAssistant(engine)
    assistant.chat("Hello there")
    assert "<memory_context>" not in engine.prompt
    assert "<memory_policy>" not in engine.prompt
