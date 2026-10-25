# OAA Phase 12 - Fine-Tuning Data Preparation

## Status

Implemented and CI-tested: schema validation, bounded local JSONL ingestion, exact duplicate detection, dataset SHA-256 fingerprinting, deterministic splitting, and overwrite protection.

Not implemented: optimizer/backpropagation, loss computation, checkpoint loading/saving, LoRA/QLoRA, or a real fine-tuning run. The existing tiny transformer is initialized from a seed and has no training path.

## SFT schema

One UTF-8 JSON object per line:

    {"instruction":"Summarize the note","input":"A local note","output":"A short summary."}

Required fields are non-empty string instruction and output. Input is optional. Unexpected fields, invalid JSON/UTF-8, empty required fields, NULs, oversize files/records, and empty datasets are rejected. The loader is bounded by default to 100 MiB and 200,000 records.

## Validate and split

    genai train-data validate --path D:/oaa-data/sft.jsonl
    genai train-data split --path D:/oaa-data/sft.jsonl --output-dir D:/oaa-data/splits --seed 42

Split defaults are 80/10/10. Seed and fractions are configurable. Split output is generated locally, and output creation refuses to overwrite a non-empty directory.

Keep the original data and generated splits outside the Git checkout. Validation hashes the original file and reports record counts/duplicates; it does not upload content, fetch data, or log example text.
