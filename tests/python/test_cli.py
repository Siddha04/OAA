from pathlib import Path

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
            "ask",
            "--model",
            str(manifest),
            "--max-tokens",
            "4",
            "--temperature",
            "0.0",
            "--seed",
            "4",
            "Hello",
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
            "chat",
            "--model",
            str(manifest),
            "--max-history-messages",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "OAA chat." in captured.out
    assert "session reset" in captured.out
