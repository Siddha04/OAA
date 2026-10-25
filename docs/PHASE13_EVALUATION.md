# OAA Phase 13 - Evaluation and Benchmarking

## Scope

The evaluation utilities load bounded local JSONL cases containing prompt, expected reference text, and an optional category. The runner accepts an injected generator, so it can be tested independently of any model backend.

The report contains:
- case count and local evaluation dataset SHA-256
- normalized exact-match and expected-substring rates
- mean token-level precision, recall and F1 based on multiset overlap
- mean, p50 and p95 generation latency
- per-case aggregate metrics without raw prompt or generated text

## CLI

    genai evaluate --model ./tiny.manifest --dataset D:/oaa-data/eval.jsonl --report D:/oaa-results/eval-report.json --max-tokens 16

JSONL records:

    {"prompt":"Return the word hello","expected":"hello","category":"format"}

Required fields are prompt and expected. The evaluation file and report are local. Existing report files require --overwrite. Evaluation data is not downloaded or uploaded.

## Interpretation

This is a reproducible reference-matching and latency harness, not a complete benchmark suite. It does not establish general language-model accuracy, factuality, safety, agent success, robustness, or clinical/legal reliability. Use task-specific held-out data, human review, safety/red-team suites, multilingual tests, and controlled hardware when drawing quality conclusions.
