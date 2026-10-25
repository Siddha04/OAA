"""Local, reproducible reference-based evaluation utilities.

Scores are task-specific string-matching diagnostics, not a general measure of
truthfulness, reasoning, safety, or model quality.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

DEFAULT_MAX_DATASET_BYTES = 25 * 1024 * 1024
DEFAULT_MAX_CASES = 2_000
DEFAULT_MAX_LINE_BYTES = 256 * 1024
MAX_TEXT_CHARS = 100_000


class EvaluationDataError(ValueError):
    """An evaluation case file is malformed or exceeds configured limits."""


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    prompt: str
    expected: str
    category: str = "general"

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt", self.prompt),
            ("expected", self.expected),
            ("category", self.category),
        ):
            if not isinstance(value, str):
                raise EvaluationDataError(f"{name} must be a string")
            if len(value) > MAX_TEXT_CHARS:
                raise EvaluationDataError(f"{name} exceeds {MAX_TEXT_CHARS} characters")
        if not self.prompt.strip():
            raise EvaluationDataError("prompt must not be empty")
        if not self.expected.strip():
            raise EvaluationDataError("expected must not be empty")
        if not self.category.strip():
            raise EvaluationDataError("category must not be empty")


def load_evaluation_jsonl(
    path: str | Path,
    *,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
    max_cases: int = DEFAULT_MAX_CASES,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
) -> tuple[EvaluationCase, ...]:
    """Load bounded JSONL cases with prompt and expected reference text."""
    for name, value in (
        ("max_dataset_bytes", max_dataset_bytes),
        ("max_cases", max_cases),
        ("max_line_bytes", max_line_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"evaluation dataset does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise EvaluationDataError("evaluation dataset is empty")
    if size > max_dataset_bytes:
        raise EvaluationDataError(
            f"evaluation dataset exceeds max_dataset_bytes ({size} > {max_dataset_bytes})"
        )

    cases: list[EvaluationCase] = []
    with source.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            if len(raw) > max_line_bytes:
                raise EvaluationDataError(
                    f"line {line_number}: record exceeds max_line_bytes ({max_line_bytes})"
                )
            if len(cases) >= max_cases:
                raise EvaluationDataError(f"evaluation dataset exceeds max_cases ({max_cases})")
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EvaluationDataError(
                    f"line {line_number}: invalid UTF-8 JSONL record"
                ) from exc
            if not isinstance(value, dict):
                raise EvaluationDataError(f"line {line_number}: record must be an object")
            unknown = set(value) - {"prompt", "expected", "category"}
            missing = {"prompt", "expected"} - set(value)
            if missing:
                raise EvaluationDataError(
                    f"line {line_number}: missing required field '{sorted(missing)[0]}'"
                )
            if unknown:
                raise EvaluationDataError(
                    f"line {line_number}: unsupported field '{sorted(unknown)[0]}'"
                )
            try:
                cases.append(EvaluationCase(
                    prompt=value["prompt"],
                    expected=value["expected"],
                    category=value.get("category", "general"),
                ))
            except EvaluationDataError as exc:
                raise EvaluationDataError(f"line {line_number}: {exc}") from exc
    if not cases:
        raise EvaluationDataError("evaluation dataset contains no cases")
    return tuple(cases)


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _token_scores(expected: str, output: str) -> tuple[float, float, float]:
    expected_counts = Counter(_tokens(expected))
    output_counts = Counter(_tokens(output))
    expected_total = sum(expected_counts.values())
    output_total = sum(output_counts.values())
    overlap = sum((expected_counts & output_counts).values())
    precision = overlap / output_total if output_total else 0.0
    recall = overlap / expected_total if expected_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@dataclass(frozen=True, slots=True)
class CaseMetrics:
    case_index: int
    category: str
    exact_match: bool
    expected_substring: bool
    token_precision: float
    token_recall: float
    token_f1: float
    latency_seconds: float
    output_characters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "category": self.category,
            "exact_match": self.exact_match,
            "expected_substring": self.expected_substring,
            "token_precision": round(self.token_precision, 6),
            "token_recall": round(self.token_recall, 6),
            "token_f1": round(self.token_f1, 6),
            "latency_seconds": round(self.latency_seconds, 6),
            "output_characters": self.output_characters,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    case_count: int
    exact_match_rate: float
    expected_substring_rate: float
    mean_token_precision: float
    mean_token_recall: float
    mean_token_f1: float
    mean_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    output_characters: int
    dataset_sha256: str
    cases: tuple[CaseMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "oaa_evaluation_report_v1",
            "metric_scope": "reference string matching and local generation timing; not overall model accuracy",
            "case_count": self.case_count,
            "exact_match_rate": round(self.exact_match_rate, 6),
            "expected_substring_rate": round(self.expected_substring_rate, 6),
            "mean_token_precision": round(self.mean_token_precision, 6),
            "mean_token_recall": round(self.mean_token_recall, 6),
            "mean_token_f1": round(self.mean_token_f1, 6),
            "mean_latency_seconds": round(self.mean_latency_seconds, 6),
            "p50_latency_seconds": round(self.p50_latency_seconds, 6),
            "p95_latency_seconds": round(self.p95_latency_seconds, 6),
            "output_characters": self.output_characters,
            "dataset_sha256": self.dataset_sha256,
            "cases": [case.to_dict() for case in self.cases],
        }


def evaluate_cases(
    cases: tuple[EvaluationCase, ...] | list[EvaluationCase],
    generate: Callable[[str], str],
    *,
    dataset_sha256: str = "",
) -> EvaluationReport:
    """Evaluate a caller-supplied local generator without writing prompts/outputs."""
    if not cases:
        raise ValueError("cases must not be empty")
    if not callable(generate):
        raise TypeError("generate must be callable")
    if not isinstance(dataset_sha256, str):
        raise TypeError("dataset_sha256 must be a string")

    metrics: list[CaseMetrics] = []
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, EvaluationCase):
            raise TypeError("cases must contain EvaluationCase values")
        started = time.perf_counter()
        output = generate(case.prompt)
        latency = time.perf_counter() - started
        if not isinstance(output, str):
            raise TypeError("generator must return a string")
        if len(output) > MAX_TEXT_CHARS:
            raise ValueError(f"case {index}: generated output exceeds {MAX_TEXT_CHARS} characters")
        normalized_expected = _normalize(case.expected)
        normalized_output = _normalize(output)
        precision, recall, f1 = _token_scores(case.expected, output)
        metrics.append(CaseMetrics(
            case_index=index,
            category=case.category,
            exact_match=normalized_output == normalized_expected,
            expected_substring=bool(normalized_expected and normalized_expected in normalized_output),
            token_precision=precision,
            token_recall=recall,
            token_f1=f1,
            latency_seconds=max(0.0, latency),
            output_characters=len(output),
        ))

    latencies = sorted(item.latency_seconds for item in metrics)
    def percentile(q: float) -> float:
        return latencies[min(len(latencies) - 1, max(0, math.ceil(q * len(latencies)) - 1))]

    count = len(metrics)
    return EvaluationReport(
        case_count=count,
        exact_match_rate=sum(item.exact_match for item in metrics) / count,
        expected_substring_rate=sum(item.expected_substring for item in metrics) / count,
        mean_token_precision=statistics.fmean(item.token_precision for item in metrics),
        mean_token_recall=statistics.fmean(item.token_recall for item in metrics),
        mean_token_f1=statistics.fmean(item.token_f1 for item in metrics),
        mean_latency_seconds=statistics.fmean(latencies),
        p50_latency_seconds=percentile(0.50),
        p95_latency_seconds=percentile(0.95),
        output_characters=sum(item.output_characters for item in metrics),
        dataset_sha256=dataset_sha256,
        cases=tuple(metrics),
    )


def evaluation_dataset_sha256(path: str | Path) -> str:
    """Hash the exact local evaluation file without returning its content."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
