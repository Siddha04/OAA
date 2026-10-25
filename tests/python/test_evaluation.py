from __future__ import annotations

import json
from pathlib import Path

import pytest

from oaa.evaluation import (
    EvaluationCase,
    EvaluationDataError,
    evaluate_cases,
    evaluation_dataset_sha256,
    load_evaluation_jsonl,
)


def write_cases(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_evaluation_dataset_schema_and_fingerprint(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    write_cases(path, [
        {"prompt": "Say hello", "expected": "Hello", "category": "format"},
        {"prompt": "Return two", "expected": "2"},
    ])
    cases = load_evaluation_jsonl(path)
    assert len(cases) == 2
    assert cases[0].category == "format"
    assert len(evaluation_dataset_sha256(path)) == 64


@pytest.mark.parametrize("row, message", [
    ({"expected": "answer"}, "missing required field 'prompt'"),
    ({"prompt": "hello", "expected": "yes", "unexpected": 1}, "unsupported field 'unexpected'"),
    ({"prompt": " ", "expected": "answer"}, "prompt must not be empty"),
    ({"prompt": "hello", "expected": ""}, "expected must not be empty"),
])
def test_evaluation_dataset_rejects_bad_records(tmp_path: Path, row, message):
    path = tmp_path / "bad.jsonl"
    write_cases(path, [row])
    with pytest.raises(EvaluationDataError, match=message):
        load_evaluation_jsonl(path)


def test_evaluation_dataset_enforces_file_and_case_limits(tmp_path: Path):
    path = tmp_path / "eval.jsonl"
    write_cases(path, [{"prompt": "one", "expected": "1"}] * 2)
    with pytest.raises(EvaluationDataError, match="max_cases"):
        load_evaluation_jsonl(path, max_cases=1)
    with pytest.raises(EvaluationDataError, match="max_dataset_bytes"):
        load_evaluation_jsonl(path, max_dataset_bytes=1)


def test_metrics_report_exact_match_f1_and_no_raw_prompts_or_outputs():
    cases = (
        EvaluationCase("case one secret prompt", "Hello world", "greeting"),
        EvaluationCase("case two secret prompt", "red blue", "colors"),
    )
    outputs = iter([" hello   WORLD ", "red green"])
    report = evaluate_cases(cases, lambda _: next(outputs), dataset_sha256="abc123")
    data = report.to_dict()
    assert data["case_count"] == 2
    assert data["exact_match_rate"] == 0.5
    assert data["expected_substring_rate"] == 0.5
    assert data["mean_token_f1"] > 0
    assert data["dataset_sha256"] == "abc123"
    encoded = json.dumps(data)
    assert "secret prompt" not in encoded
    assert "hello   WORLD" not in encoded
    assert data["p95_latency_seconds"] >= data["p50_latency_seconds"]


def test_metrics_validate_generator_and_case_types():
    with pytest.raises(ValueError, match="not be empty"):
        evaluate_cases([], lambda _: "yes")
    with pytest.raises(TypeError, match="return a string"):
        evaluate_cases((EvaluationCase("p", "r"),), lambda _: 123)
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_cases((EvaluationCase("p", "r"),), lambda _: "x" * 100_001)
