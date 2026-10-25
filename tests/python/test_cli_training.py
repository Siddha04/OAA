from __future__ import annotations

import json
from pathlib import Path

from oaa.cli import build_parser, main


def _dataset(path: Path):
    rows = [
        {"instruction": f"Task {i}", "input": "", "output": f"Answer {i}"}
        for i in range(20)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_train_data_parser_and_validate(tmp_path: Path, capsys):
    path = tmp_path / "sft.jsonl"
    _dataset(path)
    args = build_parser().parse_args(["train-data", "validate", "--path", str(path)])
    assert args.command == "train-data"
    assert args.train_data_action == "validate"

    assert main(["train-data", "validate", "--path", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid_records"] == 20
    assert report["schema"] == "oaa_sft_jsonl_v1"


def test_train_data_split_creates_local_splits_and_refuses_overwrite(tmp_path: Path, capsys):
    path = tmp_path / "sft.jsonl"
    _dataset(path)
    output = tmp_path / "splits"
    assert main([
        "train-data", "split", "--path", str(path), "--output-dir", str(output), "--seed", "12",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["split_counts"] == {"train": 16, "validation": 2, "test": 2}
    assert all(Path(p).is_file() for p in report["output_paths"].values())

    assert main([
        "train-data", "split", "--path", str(path), "--output-dir", str(output),
    ]) == 2
    assert "non-empty output directory" in capsys.readouterr().err


def test_train_data_split_rejects_invalid_jsonl(tmp_path: Path, capsys):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"instruction":"no output"}\n', encoding="utf-8")
    assert main(["train-data", "validate", "--path", str(path)]) == 2
    assert "missing required field" in capsys.readouterr().err
