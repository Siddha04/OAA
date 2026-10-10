from __future__ import annotations

import json
from pathlib import Path

from oaa.cli import main


def test_tools_list_and_calculator_cli(capsys) -> None:
    assert main(["tools", "list"]) == 0
    listed = capsys.readouterr()
    assert "calculator" in listed.out
    assert "read_text_file" not in listed.out

    assert main([
        "tools", "run", "calculator", "--args",
        json.dumps({"expression": "(8 / 2) + 3"}),
    ]) == 0
    output = capsys.readouterr()
    assert '"result":7.0' in output.out.replace(" ", "")


def test_tools_cli_file_reader_needs_explicit_workspace(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "knowledge"
    workspace.mkdir()
    (workspace / "guide.md").write_text("OAA tool guide", encoding="utf-8")

    assert main(["tools", "list", "--workspace", str(workspace)]) == 0
    assert "read_text_file" in capsys.readouterr().out

    assert main([
        "tools", "run", "read_text_file",
        "--workspace", str(workspace),
        "--args", json.dumps({"path": "guide.md"}),
    ]) == 0
    output = capsys.readouterr()
    assert "OAA tool guide" in output.out


def test_tools_cli_rejects_bad_json_and_unsafe_math(capsys) -> None:
    assert main(["tools", "run", "calculator", "--args", "{bad-json"]) == 2
    assert "error:" in capsys.readouterr().err

    assert main([
        "tools", "run", "calculator", "--args",
        json.dumps({"expression": "__import__('os').system('unsafe')"}),
    ]) == 1
    assert "Tool 'calculator' failed" in capsys.readouterr().err
