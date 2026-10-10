from pathlib import Path
from types import SimpleNamespace

from oaa.cli import build_parser, main


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
                "context_length=256",
                "seed=3",
            ]
        ),
        encoding="utf-8",
    )


def test_cli_parser() -> None:
    args = build_parser().parse_args(
        [
            "ask",
            "--model",
            "tiny.manifest",
            "--max-tokens",
            "4",
            "--temperature",
            "0.0",
            "Hello",
        ]
    )
    assert args.command == "ask"
    assert args.max_tokens == 4
    assert args.prompt == "Hello"


def test_cli_ask(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    exit_code = main(
        [
            "ask", "--model", str(manifest), "--max-tokens", "4",
            "--temperature", "0.0", "--seed", "4", "Hello",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert len(captured.out.strip()) == 4
    assert captured.err == ""


def test_cli_chat_control_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)

    commands = iter(["/stats", "/system Be precise", "/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))

    exit_code = main(
        [
            "chat", "--model", str(manifest),
            "--max-history-messages", "4",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OAA chat." in captured.out
    assert "session reset" in captured.out


def test_cli_search_local_documents(tmp_path: Path, capsys) -> None:
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "manual.md").write_text(
        "OAA loads model weights from a local manifest and stores project notes.",
        encoding="utf-8",
    )
    (docs / "other.txt").write_text(
        "Cooking pasta requires boiling water.", encoding="utf-8"
    )

    exit_code = main(
        [
            "search", "--docs", str(docs), "--top-k", "3",
            "where does OAA load model weights",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "manual.md" in captured.out
    assert "model weights" in captured.out
    assert "other.txt" not in captured.out


def test_cli_chat_can_attach_local_rag(tmp_path: Path, monkeypatch, capsys) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "manual.txt").write_text(
        "OAA keeps local notes in its retrieval index.", encoding="utf-8"
    )

    class FakeAssistant:
        def __init__(
            self,
            engine,
            *,
            rag_pipeline=None,
            rag_top_k=3,
            rag_max_context_chars=512,
        ) -> None:
            assert rag_pipeline is not None
            assert rag_pipeline.store.count == 1
            self.session = SimpleNamespace(max_history_messages=32)
            self.rag_pipeline = rag_pipeline

        def set_system_prompt(self, prompt):
            self.system_prompt = prompt

        def chat_stream(self, message, config):
            yield "RAG"
            yield " works"

        def reset(self):
            pass

        def stats(self):
            return {"ok": True}

    monkeypatch.setattr("oaa.cli.PersonalAssistant", FakeAssistant)
    commands = iter(["What do my local notes say?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))

    exit_code = main(
        ["chat", "--model", str(manifest), "--docs", str(docs)]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Indexed 1 chunks" in captured.out
    assert "RAG works" in captured.out


def test_cli_search_empty_directory(tmp_path: Path, capsys) -> None:
    docs = tmp_path / "empty"
    docs.mkdir()
    exit_code = main(["search", "--docs", str(docs), "anything"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No extractable document chunks" in captured.out



def test_cli_memory_add_search_list_update_delete(tmp_path: Path, capsys) -> None:
    db = tmp_path / "memory.sqlite3"
    assert main([
        "memory", "add", "--db", str(db), "--category", "preference",
        "Prefers concise Python examples",
    ]) == 0
    added = capsys.readouterr()
    assert "Saved memory 1" in added.out

    assert main(["memory", "search", "--db", str(db), "Python examples"]) == 0
    searched = capsys.readouterr()
    assert "Prefers concise Python examples" in searched.out

    assert main(["memory", "list", "--db", str(db), "--category", "preference"]) == 0
    listed = capsys.readouterr()
    assert "[1] [preference]" in listed.out

    assert main([
        "memory", "update", "--db", str(db), "1", "Prefers technical summaries",
    ]) == 0
    updated = capsys.readouterr()
    assert "Updated memory 1" in updated.out

    assert main(["memory", "delete", "--db", str(db), "1"]) == 0
    deleted = capsys.readouterr()
    assert "Deleted memory 1" in deleted.out


def test_cli_memory_clear_requires_confirmation(tmp_path: Path, capsys) -> None:
    db = tmp_path / "memory.sqlite3"
    assert main(["memory", "add", "--db", str(db), "Keep me"]) == 0
    capsys.readouterr()

    assert main(["memory", "clear", "--db", str(db)]) == 2
    refused = capsys.readouterr()
    assert "without --yes" in refused.err

    assert main(["memory", "list", "--db", str(db)]) == 0
    assert "Keep me" in capsys.readouterr().out

    assert main(["memory", "clear", "--db", str(db), "--yes"]) == 0
    assert "Deleted 1 memories" in capsys.readouterr().out


def test_cli_chat_can_enable_persistent_memory(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    db = tmp_path / "memory.sqlite3"

    class FakeAssistant:
        def __init__(self, engine, *, memory_store=None, **kwargs) -> None:
            assert memory_store is not None
            assert len(memory_store.list_memories()) == 1
            self.session = SimpleNamespace(max_history_messages=32)

        def set_system_prompt(self, prompt):
            self.system_prompt = prompt

        def chat_stream(self, message, config):
            yield "memory"
            yield " enabled"

        def reset(self):
            pass

        def stats(self):
            return {"ok": True}

    assert main([
        "memory", "add", "--db", str(db), "Prefers short examples",
        "--category", "preference",
    ]) == 0
    capsys.readouterr()
    monkeypatch.setattr("oaa.cli.PersonalAssistant", FakeAssistant)
    commands = iter(["show short examples", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))

    exit_code = main([
        "chat", "--model", str(manifest), "--memory-db", str(db),
    ])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Persistent memory enabled" in captured.out
    assert "memory enabled" in captured.out
