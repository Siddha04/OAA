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


def test_evaluate_cli_runs_local_model_and_writes_report(tmp_path: Path, capsys):
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Secret prompt", "expected": "reference text"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    args = build_parser().parse_args([
        "evaluate", "--model", str(manifest), "--dataset", str(dataset),
        "--report", str(output), "--max-tokens", "4", "--temperature", "0",
    ])
    assert args.command == "evaluate"
    assert main([
        "evaluate", "--model", str(manifest), "--dataset", str(dataset),
        "--report", str(output), "--max-tokens", "4", "--temperature", "0",
    ]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert report["schema"] == "oaa_evaluation_report_v1"
    assert report["case_count"] == 1
    assert "Secret prompt" not in output.read_text(encoding="utf-8")


def test_evaluate_refuses_overwrite_and_dataset_clobber(tmp_path: Path, capsys):
    manifest = tmp_path / "tiny.manifest"
    write_manifest(manifest)
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"prompt": "p", "expected": "r"}) + "\n", encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    base = ["evaluate", "--model", str(manifest), "--dataset", str(dataset), "--report", str(output)]
    assert main(base) == 2
    assert "pass --overwrite" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "keep"
    assert main([
        "evaluate", "--model", str(manifest), "--dataset", str(dataset), "--report", str(dataset),
    ]) == 2
    assert "must not overwrite" in capsys.readouterr().err


def test_evaluate_rejects_bad_dataset_before_model_load(tmp_path: Path, capsys):
    invalid = tmp_path / "eval.jsonl"
    invalid.write_text("{}\n", encoding="utf-8")
    code = main([
        "evaluate", "--model", str(tmp_path / "missing.manifest"),
        "--dataset", str(invalid), "--report", str(tmp_path / "report.json"),
    ])
    assert code == 2
    assert "missing required field" in capsys.readouterr().err
