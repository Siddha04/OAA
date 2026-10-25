"""Safe local JSONL preparation for supervised fine-tuning (SFT).

This module validates and splits data only. It does not perform gradient updates,
modify model weights, download datasets, or upload user data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_MAX_DATASET_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_EXAMPLES = 200_000
DEFAULT_MAX_LINE_BYTES = 1024 * 1024
MAX_FIELD_CHARS = 100_000
_ALLOWED_FIELDS = frozenset({"instruction", "input", "output"})


class DatasetValidationError(ValueError):
    """A local SFT dataset did not meet the supported schema or safety limits."""


@dataclass(frozen=True, slots=True)
class SFTExample:
    instruction: str
    output: str
    input: str = ""

    def __post_init__(self) -> None:
        for name, value in (("instruction", self.instruction), ("input", self.input), ("output", self.output)):
            if not isinstance(value, str):
                raise DatasetValidationError(f"{name} must be a string")
            if len(value) > MAX_FIELD_CHARS:
                raise DatasetValidationError(f"{name} exceeds {MAX_FIELD_CHARS} characters")
        if not self.instruction.strip():
            raise DatasetValidationError("instruction must not be empty")
        if not self.output.strip():
            raise DatasetValidationError("output must not be empty")
        if "\x00" in self.instruction or "\x00" in self.input or "\x00" in self.output:
            raise DatasetValidationError("fields must not contain NUL characters")

    @classmethod
    def from_mapping(cls, value: Any, *, line_number: int = 1) -> "SFTExample":
        if not isinstance(value, dict):
            raise DatasetValidationError(f"line {line_number}: each record must be a JSON object")
        extra = set(value) - _ALLOWED_FIELDS
        if extra:
            raise DatasetValidationError(
                f"line {line_number}: unsupported field '{sorted(extra)[0]}'"
            )
        missing = {"instruction", "output"} - set(value)
        if missing:
            raise DatasetValidationError(
                f"line {line_number}: missing required field '{sorted(missing)[0]}'"
            )
        try:
            return cls(
                instruction=value["instruction"],
                input=value.get("input", ""),
                output=value["output"],
            )
        except DatasetValidationError as exc:
            raise DatasetValidationError(f"line {line_number}: {exc}") from exc

    def to_dict(self) -> dict[str, str]:
        result = {"instruction": self.instruction, "output": self.output}
        if self.input:
            result["input"] = self.input
        return result

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetReport:
    path: str
    bytes_read: int
    records_read: int
    valid_records: int
    duplicate_records: int
    instruction_characters: int
    input_characters: int
    output_characters: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes_read": self.bytes_read,
            "records_read": self.records_read,
            "valid_records": self.valid_records,
            "duplicate_records": self.duplicate_records,
            "instruction_characters": self.instruction_characters,
            "input_characters": self.input_characters,
            "output_characters": self.output_characters,
            "sha256": self.sha256,
            "schema": "oaa_sft_jsonl_v1",
        }


def load_sft_jsonl(
    path: str | Path,
    *,
    max_dataset_bytes: int = DEFAULT_MAX_DATASET_BYTES,
    max_examples: int = DEFAULT_MAX_EXAMPLES,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    deduplicate: bool = True,
) -> tuple[tuple[SFTExample, ...], DatasetReport]:
    """Read, validate and optionally deduplicate local SFT examples."""
    for name, value in (
        ("max_dataset_bytes", max_dataset_bytes),
        ("max_examples", max_examples),
        ("max_line_bytes", max_line_bytes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(deduplicate, bool):
        raise TypeError("deduplicate must be a boolean")

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"dataset file does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise DatasetValidationError("dataset file is empty")
    if size > max_dataset_bytes:
        raise DatasetValidationError(
            f"dataset exceeds max_dataset_bytes ({size} > {max_dataset_bytes})"
        )

    digest = hashlib.sha256()
    examples: list[SFTExample] = []
    seen: set[str] = set()
    records_read = 0
    duplicates = 0
    instruction_chars = input_chars = output_chars = 0

    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            if len(raw_line) > max_line_bytes:
                raise DatasetValidationError(
                    f"line {line_number}: record exceeds max_line_bytes ({max_line_bytes})"
                )
            records_read += 1
            if records_read > max_examples:
                raise DatasetValidationError(
                    f"dataset exceeds max_examples ({max_examples})"
                )
            try:
                decoded = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DatasetValidationError(
                    f"line {line_number}: invalid UTF-8 JSONL record"
                ) from exc
            example = SFTExample.from_mapping(decoded, line_number=line_number)
            key = example.fingerprint
            if deduplicate and key in seen:
                duplicates += 1
                continue
            seen.add(key)
            examples.append(example)
            instruction_chars += len(example.instruction)
            input_chars += len(example.input)
            output_chars += len(example.output)

    if not examples:
        raise DatasetValidationError("dataset contains no valid records")
    report = DatasetReport(
        path=str(source.resolve()),
        bytes_read=size,
        records_read=records_read,
        valid_records=len(examples),
        duplicate_records=duplicates,
        instruction_characters=instruction_chars,
        input_characters=input_chars,
        output_characters=output_chars,
        sha256=digest.hexdigest(),
    )
    return tuple(examples), report


def split_sft_examples(
    examples: Iterable[SFTExample],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, tuple[SFTExample, ...]]:
    """Create deterministic train/validation/test splits for a small local dataset."""
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or validation_fraction < 0
    ):
        raise ValueError("validation_fraction must be finite and non-negative")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not math.isfinite(test_fraction)
        or test_fraction < 0
    ):
        raise ValueError("test_fraction must be finite and non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    data = list(examples)
    if not data:
        raise DatasetValidationError("cannot split an empty dataset")
    if not all(isinstance(item, SFTExample) for item in data):
        raise TypeError("examples must contain SFTExample values")
    rng = random.Random(seed)
    rng.shuffle(data)
    count = len(data)
    n_test = int(count * test_fraction)
    n_validation = int(count * validation_fraction)
    # For small sets, keep a train example and make requested holdouts explicit.
    if count >= 3 and test_fraction > 0:
        n_test = max(1, n_test)
    if count >= 3 and validation_fraction > 0:
        n_validation = max(1, n_validation)
    while n_test + n_validation >= count:
        if n_validation >= n_test and n_validation > 0:
            n_validation -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            raise DatasetValidationError("split fractions leave no training records")
    return {
        "train": tuple(data[:count - n_test - n_validation]),
        "validation": tuple(data[count - n_test - n_validation:count - n_test]),
        "test": tuple(data[count - n_test:]),
    }


def write_sft_splits(
    splits: dict[str, tuple[SFTExample, ...]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write three JSONL files into an empty/new directory without overwriting data."""
    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise ValueError("splits must contain exactly train, validation, and test")
    target = Path(output_dir).expanduser()
    if target.exists():
        if not target.is_dir():
            raise ValueError("output_dir exists and is not a directory")
        if any(target.iterdir()):
            raise FileExistsError(f"refusing to write into non-empty output directory: {target}")
    else:
        target.mkdir(parents=True, exist_ok=False)

    staged: list[tuple[Path, Path]] = []
    try:
        for name in ("train", "validation", "test"):
            destination = target / f"{name}.jsonl"
            fd, temporary_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=target)
            temporary = Path(temporary_name)
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for item in splits[name]:
                    if not isinstance(item, SFTExample):
                        raise TypeError(f"{name} split contains a non-SFTExample value")
                    stream.write(json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":")))
                    stream.write("\n")
            staged.append((temporary, destination))
        # The empty destination directory is checked before files are staged.
        for temporary, destination in staged:
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing split: {destination}")
        for temporary, destination in staged:
            temporary.replace(destination)
        return {name: str((target / f"{name}.jsonl").resolve()) for name in ("train", "validation", "test")}
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
