from __future__ import annotations

import json
from pathlib import Path

from oaa.cli import build_parser, main


def write_manifest(path: Path) -> None:
    path.write_text("\n".join([
        "architecture=tiny_transformer_v1",
        "vocab_size=95",
        "hidden_size=16",
        "num_layers=1",
        "num_heads=4",
        "intermediate_size=32",
        "context_length=64",
        "seed=3",
    ]) + "\n", encoding="utf-8")


def test_runtime_info_cli_reports_current_limits(tmp_path: Path, capsys):
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    args = build_parser().parse_args(["runtime-info", "--model", str(manifest)])
    assert args.command == "runtime-info"
    assert main(["runtime-info", "--model", str(manifest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["engine_loaded"] is True
    assert result["backend"]["pretrained_weights_loaded"] is False
    assert result["backend"]["true_streaming"] is False
    assert result["backend"]["structured_tool_calls"] is False
    assert result["known_limitations"]
