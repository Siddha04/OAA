from pathlib import Path

import pytest

from oaa import ChatSession, Engine, GenerationConfig, PersonalAssistant


def write_manifest(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "architecture=tiny_transformer_v1",
                "vocab_size=95",
                "hidden_size=16",
                "num_layers=1",
                "num_heads=4",
                "intermediate_size=32",
                "context_length=64",
                "seed=12",
            ]
        ),
        encoding="utf-8",
    )


def test_session_builds_structured_prompt() -> None:
    session = ChatSession("You are a test assistant.", max_history_messages=4)
    session.add("user", "Hello")
    session.add("assistant", "Hi")
    prompt = session.build_prompt("How are you?")
    assert "<system>You are a test assistant.</system>" in prompt
    assert "<user>Hello</user>" in prompt
    assert "<assistant>Hi</assistant>" in prompt
    assert prompt.endswith("<assistant>")


def test_session_history_is_bounded() -> None:
    session = ChatSession(max_history_messages=2)
    session.add("user", "one")
    session.add("assistant", "two")
    session.add("user", "three")
    assert [m.content for m in session.messages] == ["two", "three"]


def test_session_validation() -> None:
    with pytest.raises(ValueError):
        ChatSession(max_history_messages=0)
    session = ChatSession()
    with pytest.raises(ValueError):
        session.add("user", "")


def test_assistant_chat_updates_history(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    engine = Engine()
    engine.load_model(str(manifest))
    assistant = PersonalAssistant(engine)
    response = assistant.chat(
        "Hello",
        GenerationConfig(max_tokens=4, temperature=0.0, seed=1),
    )
    assert len(response) == 4
    assert len(assistant.history()) == 2
    assert assistant.history()[0].role == "user"
    assert assistant.history()[0].content == "Hello"
    assert assistant.history()[1].role == "assistant"
    assert assistant.history()[1].content == response


def test_assistant_stream_updates_history(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    engine = Engine()
    engine.load_model(str(manifest))
    assistant = PersonalAssistant(engine)
    chunks = list(
        assistant.chat_stream(
            "Hello",
            GenerationConfig(max_tokens=4, temperature=0.0, seed=2),
        )
    )
    assert len(chunks) == 4
    assert len(assistant.history()) == 2
    assert assistant.history()[1].content == "".join(chunks)


def test_assistant_reset(tmp_path: Path) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    engine = Engine()
    engine.load_model(str(manifest))
    assistant = PersonalAssistant(engine)
    assistant.chat("Hello", GenerationConfig(max_tokens=2, temperature=0.0))
    assistant.reset()
    assert assistant.history() == ()
