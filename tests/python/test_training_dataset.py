from __future__ import annotations

import json
from pathlib import Path

import pytest

from oaa.training import (
    DatasetValidationError,
    SFTExample,
    load_sft_jsonl,
    split_sft_examples,
    write_sft_splits,
)


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row(instruction: str, output: str, input_text: str = "") -> dict[str, str]:
    return {"instruction": instruction, "input": input_text, "output": output}


def test_load_sft_jsonl_validates_schema_and_reports_duplicates(tmp_path: Path):
    path = tmp_path / "training.jsonl"
    write_jsonl(path, [
        row("Summarize", "Short summary", "Long report"),
        row("Translate", "Vanakkam", "Hello"),
        row("Summarize", "Short summary", "Long report"),
    ])
    examples, report = load_sft_jsonl(path)
    assert len(examples) == 2
    assert report.records_read == 3
    assert report.valid_records == 2
    assert report.duplicate_records == 1
    assert len(report.sha256) == 64
    assert report.output_characters == len("Short summary") + len("Vanakkam")


@pytest.mark.parametrize("record, message", [
    ('not json', "invalid UTF-8 JSONL"),
    (json.dumps({"instruction": "question"}), "missing required field 'output'"),
    (json.dumps({"instruction": "question", "output": "answer", "code": "x"}), "unsupported field 'code'"),
    (json.dumps({"instruction": " ", "output": "answer"}), "instruction must not be empty"),
    (json.dumps({"instruction": "question", "output": ""}), "output must not be empty"),
])
def test_load_sft_jsonl_rejects_invalid_records(tmp_path: Path, record: str, message: str):
    path = tmp_path / "bad.jsonl"
    path.write_text(record + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match=message):
        load_sft_jsonl(path)


def test_dataset_limits_and_empty_files_are_rejected(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [row("A", "B")])
    with pytest.raises(DatasetValidationError, match="max_dataset_bytes"):
        load_sft_jsonl(path, max_dataset_bytes=1)
    with pytest.raises(ValueError, match="max_examples"):
        load_sft_jsonl(path, max_examples=0)  # invalid limit configuration
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="empty"):
        load_sft_jsonl(path)


def test_split_is_deterministic_and_uses_separate_examples(tmp_path: Path):
    examples = tuple(SFTExample(instruction=f"Task {i}", output=f"Result {i}") for i in range(20))
    first = split_sft_examples(examples, seed=17)
    second = split_sft_examples(examples, seed=17)
    assert {k: [v.instruction for v in values] for k, values in first.items()} == {
        k: [v.instruction for v in values] for k, values in second.items()
    }
    assert sum(len(value) for value in first.values()) == len(examples)
    assert first["train"]
    assert first["validation"]
    assert first["test"]
    all_items = [item.fingerprint for values in first.values() for item in values]
    assert len(all_items) == len(set(all_items))


def test_write_splits_is_local_and_refuses_overwrite(tmp_path: Path):
    examples = tuple(SFTExample(instruction=f"Task {i}", output=f"Result {i}") for i in range(9))
    splits = split_sft_examples(examples)
    output = tmp_path / "prepared"
    paths = write_sft_splits(splits, output)
    assert set(paths) == {"train", "validation", "test"}
    assert all(Path(value).is_file() for value in paths.values())
    assert len(load_sft_jsonl(paths["train"])[0]) == len(splits["train"])
    with pytest.raises(FileExistsError, match="non-empty"):
        write_sft_splits(splits, output)


def test_split_rejects_invalid_fractions_and_types():
    with pytest.raises(ValueError, match="less than 1"):
        split_sft_examples([SFTExample("A", "B")], validation_fraction=0.5, test_fraction=0.5)
    with pytest.raises(ValueError, match="seed"):
        split_sft_examples([SFTExample("A", "B")], seed=-1)
    with pytest.raises(TypeError, match="SFTExample"):
        split_sft_examples([{"instruction": "A"}])
