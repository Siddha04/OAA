# OAA — GenAI Readiness and Production-Grade Roadmap

**Assessment date:** 25 October 2026  
**Repository:** https://github.com/Siddha04/OAA  
**Branch:** main  
**Audited commit at report time:** ec4115a9a44923d3cdf6cd53e175a9b0e52ea6d6  
**Package version:** 0.12.0  
**Final CI run:** https://github.com/Siddha04/OAA/actions/runs/40178646291

## 1. Executive verdict

OAA has completed the planned 15-stage engineering scaffold: a C++ runtime with a Python/pybind11 interface, local assistant and retrieval components, persistent memory, controlled tools, an approval-gated agent framework, media ingestion contracts, supervised-fine-tuning data preparation, reference-based evaluation, a standalone C++ CLI, and backend capability reporting.

The latest GitHub Actions run passed all **6/6 C++ tests and 134 Python tests**, and built the Python wheel **oaa-0.12.0**. See the [verified final run](https://github.com/Siddha04/OAA/actions/runs/40178646291).

**Important product verdict: OAA is a tested GenAI runtime prototype and orchestration scaffold, not yet a best-grade or production-capable conversational LLM.** The core transformer initializes weights from a manifest seed. It does not load pretrained model checkpoints, use a production tokenizer, perform gradient training, or deliver dependable natural-language reasoning. The media module prepares images and PCM audio but does not include vision, OCR, ASR, or TTS models. Most important next step: integrate a real, appropriately licensed pretrained model and then measure quality, safety, latency, and memory use against explicit acceptance criteria.

“Best-grade GenAI” is not a test-count milestone. It requires a capable pretrained model, valid model weights and tokenizer, relevant real-world evaluations, controlled tool execution, secure data handling, stable packaging, and reproducible hardware benchmarks.

## 2. What is implemented

| Stage | Implemented foundation | Critical limit still present |
|---|---|---|
| 1. Foundation | CMake/C++20 project, Python package, pybind11 boundary, CI | Platform matrix is narrow; current CI is Ubuntu/Python 3.12 |
| 2. Runtime core | Tensor operations, memory primitives, tokenizer interface, manifest loader | Manifest is metadata, not a real checkpoint |
| 3. Python/C++ integration | Stable engine/config facade and streaming-shaped API | Current generation computes a response before yielding chunks; no true incremental decoding |
| 4. CUDA | CUDA kernel integration plus CPU fallback | Real transformer inference on a supported GPU is not verified; the development setup previously had no NVIDIA CUDA toolchain |
| 5. Transformer inference | Tiny decoder-only transformer, KV-cache structures, sampling controls | Seed-initialized weights; printable-ASCII tokenizer; no pretrained checkpoint or production tokenizer |
| 6. Personal assistant | CLI ask/chat, bounded session history, prompt controls | Random tiny-model output is not a useful general assistant |
| 7. RAG | Local file ingestion, chunking, lexical hash embeddings, retrieval/context injection | Hash embeddings are not semantic model embeddings; index is in-memory and rebuilt each run |
| 8. Long-term memory | Explicit SQLite memory CRUD and optional chat recall | Database is local but not encrypted; chat transcripts are not automatically persisted |
| 9. Tools | Deny-by-default registry, schema validation, bounded calculator, workspace-scoped read-only file reader | No general production tool sandbox or end-to-end prompt-injection defense |
| 10. Personal agent | Structured plans, policy preflight, per-step approval, stop-on-rejection/failure | The current model does not independently produce reliable tool plans |
| 11. Vision/voice foundation | Bounded local image decoding to RGB; WAV/PCM loading; ASR/TTS adapter contracts | No bundled image-understanding, OCR, ASR, or TTS model |
| 12. Fine-tuning preparation | Bounded SFT JSONL validation, duplicate detection, dataset hash, deterministic splits | No loss function, backpropagation, optimizer, LoRA/QLoRA, checkpoint save/load, or actual fine-tuning |
| 13. Evaluation | Reference text matching, token overlap metrics, latency percentiles, report hash | Metrics are diagnostic string comparisons, not general accuracy, factuality, safety, or reasoning scores |
| 14. Native CLI | Standalone C++ executable, command parsing, manifest-load/generation smoke test, CMake install target | Only the tiny seed-initialized model family is supported |
| 15. Backend contracts | Capability reporting, generation budgets/telemetry, pluggable structured plan-provider contract | These are extension points, not a production model backend or an autonomous LLM planner |

## 3. Verification record

The most relevant completed gates for the final roadmap stages are listed below. Each successful run builds the project, runs CTest, builds the Python wheel, and runs the Python tests.

| Stage/gate | Commit | GitHub Actions | Result |
|---|---|---|---|
| Phase 11 media CLI and docs | 21ffdd684b8749e824110943261ad2cdbf4a8012 | [Run #30](https://github.com/Siddha04/OAA/actions/runs/38058998925) | 4/4 C++, 104 Python |
| Phase 12 SFT data core test fix | ae3dac3ed0813ba0768cd8580798971625deb13d | [Run #33](https://github.com/Siddha04/OAA/actions/runs/40177563282) | 4/4 C++, 114 Python |
| Phase 12 data CLI | a8e6dca81ea5ad0eb748b7d2a8777542048d0d7a | [Run #34](https://github.com/Siddha04/OAA/actions/runs/40177711800) | 4/4 C++, 117 Python |
| Phase 13 evaluation harness | 7d0baaf5b3eadd8372e99b94109b314220dfdec7 | [Run #35](https://github.com/Siddha04/OAA/actions/runs/40177844344) | 4/4 C++, 125 Python |
| Phase 13 evaluation CLI | b32cca2782c627cdfa3461e19ba834b76af5029a | [Run #36](https://github.com/Siddha04/OAA/actions/runs/40177972567) | 4/4 C++, 128 Python |
| Phase 14 native C++ CLI | c1eec86dfca59c06aab22b9978a3e1de6aa95517 | [Run #37](https://github.com/Siddha04/OAA/actions/runs/40178216105) | 6/6 C++, 128 Python |
| Phase 15 backend/agent contracts | ec4115a9a44923d3cdf6cd53e175a9b0e52ea6d6 | [Run #38](https://github.com/Siddha04/OAA/actions/runs/40178646291) | **6/6 C++, 134 Python; wheel built** |

Earlier gates for phases 1–10 are also represented in repository history and CI. These checks establish software integration and regression behavior. They do **not** demonstrate meaningful LLM quality, GPU acceleration, fine-tuning quality, semantic retrieval quality, or multimodal understanding.

## 4. Highest-priority gaps

### P0 — Load and run a genuine pretrained language model

This is the blocker that dominates all others. The current seed-initialized transformer is useful for validating the runtime pipeline, but it is not a trained conversational model. More CLI commands, tests, or agent wrappers will not turn it into one.

Choose one of two architectural paths:

1. **Practical product path (recommended):** keep OAA’s native runtime as a learning/research core, and add an optional production inference backend using a mature local inference engine. A GGUF-capable engine such as [llama.cpp](https://github.com/ggml-org/llama.cpp) already documents quantization, CPU/GPU backend choices, and local model execution. Wrap it with OAA’s backend capability contract and keep model loading, quantization, and tokenizer compatibility in that adapter.
2. **Build-the-runtime-from-scratch path:** implement a verified weight-file format and loader, the correct tokenizer and chat template for the chosen model, architecture-specific tensor layouts/operators, rotary position encoding where required, KV-cache lifecycle, sampler compatibility, checkpoint validation, and memory accounting. Then test outputs against a reference implementation of the same checkpoint.

Do not silently claim both routes are complete. Select a single reference model and architecture first, check model-weight terms and redistribution rights, record the exact model revision and file hashes, and use a known-good reference runner as the quality baseline.

**Acceptance gate:** load a real pretrained model locally, verify tokenizer and checkpoint metadata, produce coherent responses on a held-out test set, reproduce known outputs or logits within stated tolerance where feasible, and report load time, peak RAM/VRAM, time-to-first-token, time-per-output-token, and tokens per second.

### P0 — Make inference genuinely efficient and bounded

The current text API presents a streaming interface, but the current generation path forms the generated token sequence before yielding it. The backend service checks prompt length before generation and output size after generation; it cannot cancel a slow backend or prevent the backend from allocating an oversized result internally.

Required work:
- Implement true token-by-token streaming and cancellation.
- Enforce context/token limits before the expensive prefill stage.
- Add model-load and generation timeouts where supported, memory budgets, and reliable error recovery.
- Measure prefill separately from decode; report TTFT, TPOT, tokens/sec, peak RAM/VRAM, model-load duration, and concurrency behavior.
- Add actual inference quantization, memory mapping, optimized CPU kernels, and supported GPU kernels only after validating quality against the non-quantized baseline.
- Track KV-cache memory, cache limits, reset semantics, and multi-request isolation.
- Benchmark on specified CPU/GPU hardware. CTest on GitHub’s general runner is not evidence of GPU performance.

Use task-appropriate, repeatable benchmark definitions. [MLPerf Inference](https://docs.mlcommons.org/inference/benchmarks/) provides an example of separating model/task quality references from system latency constraints; OAA should publish its own smaller reproducible suite before any performance claims.

### P0 — Implement actual fine-tuning and checkpoint management

Phase 12 currently prepares data only. It does not update model weights.

The most practical route is to add an optional Python training backend rather than hand-build autodiff in the existing C++ runtime first:
- Load a supported pretrained checkpoint and matching tokenizer.
- Use a mature SFT trainer such as Hugging Face [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer), with training/evaluation splits and a defined completion-only or assistant-only loss policy.
- Add parameter-efficient adaptation using [PEFT/LoRA](https://huggingface.co/docs/transformers/peft); consider QLoRA only after validating model/quantization support and hardware memory.
- Save adapter weights, training configuration, tokenizer/template references, dataset fingerprints, random seeds, dependency versions, and evaluation reports.
- Resume interrupted runs; compare the base model and fine-tuned adapter on the exact same held-out evaluation suite.
- Make model export/loading compatibility explicit. Do not treat prepared JSONL files or a success exit code as a successful fine-tune.

**Acceptance gate:** a real training run completes; training loss is finite and logged; a held-out score changes for a defensible reason; checkpoint load/save round-trips; the base model remains available; and no test/evaluation examples leak into training.

### P1 — Replace lexical-only RAG with evaluated semantic retrieval

Keep the existing loader and citation plumbing, but add a local embedding backend with model/version metadata, a persistent index, document/chunk IDs, provenance, deterministic re-indexing, and explicit deletion/rebuild behavior. Add a retrieval test set with answerable/unanswerable questions and measure recall@k, MRR/nDCG where appropriate, citation correctness, grounded answer quality, and latency.

Retrieved documents must be treated as untrusted data, not instructions. Test malicious instructions embedded in documents, direct/indirect prompt injection, irrelevant context, stale documents, and source conflicts. A future semantic index must preserve source traceability and should not be assumed to eliminate injection risk.

### P1 — Make voice and vision real model capabilities

Phase 11 is image/audio preprocessing, not multimodal inference. Select explicit local vision and audio backends, then support model-appropriate preprocessing, input limits, model lifecycle, device selection, output schemas, and evaluation:
- Vision: image question-answering or captioning on a licensed compatible vision-language model, not just RGB conversion.
- Audio input: local ASR/transcription with supported sample rates and quality tests.
- Audio output: local TTS with defined voices/licenses and audio quality tests.
- Realtime voice is a separate project: streaming capture, chunking, latency, echo/noise handling, interruption/barge-in, and privacy controls.

Avoid advertising a camera/microphone voice assistant until end-to-end model tests pass on supported hardware.

### P1 — Strengthen tool and agent security

The existing explicit allow-list, schema checks, bounded tools, and per-step approval are a good foundation. Maintain those constraints when a real model planner is connected.

Priorities:
- Constrained structured output or grammar/schema decoding for plans; never parse free-form assistant prose as an executable command.
- Treat user input, retrieved content, model output, and tool output as untrusted.
- Preserve the exact user goal and tool schemas; validate all arguments before any side effect.
- Require explicit consent for writes, external calls, destructive operations, and changes to user data.
- Add process/filesystem sandboxing, time/resource budgets, audit logs that avoid secrets, and defense-in-depth against symlink/path races.
- No shell, arbitrary Python, or outbound network tool should be added by default.
- Add red-team tests for prompt injection, data exposure, unsafe file access, excessive agency, malformed model responses, and resource exhaustion.

These priorities align with risks highlighted in the [OWASP LLM/GenAI Top 10 (2025)](https://genai.owasp.org/llm-top-10/), including prompt injection, sensitive-information disclosure, supply-chain issues, training/model poisoning, improper output handling, and excessive agency.

### P1 — Establish a real quality benchmark

The current evaluation metrics are normalized string match, expected-substring rate, token overlap, and latency. They are helpful diagnostics but can fail to detect hallucinations, reasoning failures, unsafe actions, or incorrect but linguistically similar responses.

Build an evaluation suite around the intended use cases:
- Held-out domain questions with reference answers and source documents.
- Exact-match tasks where exact match is meaningful; semantic/human-reviewed scoring where it is not.
- Retrieval and citation correctness.
- Tool-call schema validity, policy refusal, approval compliance, and successful task completion.
- Prompt injection, privacy leakage, jailbreak attempts, out-of-scope questions, and abuse/resource-exhaustion cases.
- Unicode, multilingual inputs, long context, malformed data, and regression tests.
- Human review rubric with blinded comparisons for subjective quality.
- Repeated benchmark runs on documented hardware, with confidence intervals or variability reports where practical.

Publish per-task metrics and known failure modes instead of a single “accuracy” percentage. There is no universal accuracy target that makes a general-purpose GenAI best-grade.

### P1 — Production engineering and delivery

Before distribution:
- Add CI for Windows, Linux, and macOS as supported; test the supported Python versions rather than one version only. Add sanitizer builds, static analysis, fuzzing for manifest/model/data parsers, and regression tests for security boundaries.
- Align project/package/native version numbers and automate release version checks.
- Produce native release artifacts for declared platforms, install/uninstall tests, and a documented clean-machine smoke test.
- Pin or constrain dependencies, scan vulnerabilities, maintain a software bill of materials, verify upstream model and package provenance, and record model licenses.
- Add installation, troubleshooting, configuration, supported hardware/model table, privacy policy, threat model, model card, data-processing notes, and release notes.
- Establish opt-in diagnostics; do not include prompts, local paths, output text, or dataset records in telemetry by default.
- Consider permission-restricted or encrypted storage for sensitive long-term memory. The current SQLite store is not encrypted.
- Define backup/deletion behavior and test that generated datasets, reports, model weights, and private memory do not enter Git.

Use [NIST AI 600-1, the Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) as a risk-management checklist across design, measurement, and deployment, not merely as documentation after the fact.

## 5. Recommended 90-day build order

The sequence below prioritizes useful model quality before investing further in optimization or new features.

| Time window | Priority deliverables | Completion evidence |
|---|---|---|
| Days 1–15 | Choose a licensed instruct model and production backend; define supported OS/hardware; add real model loading and tokenizer path; preserve tiny model as a CI fixture | Local end-to-end coherent chat, documented model hashes/license, baseline hardware/memory metrics |
| Days 16–30 | Build a real domain evaluation set; add model and RAG baselines; improve semantic retrieval and source citations | Reproducible report on held-out tasks, retrieval metrics, citation audit, no dataset committed |
| Days 31–45 | Add an SFT backend using a mature trainer; connect prepared data and deterministic splits; add LoRA adapter save/load | Successful real fine-tune and checkpoint round-trip; base-vs-adapter comparison |
| Days 46–60 | Add constrained model-planned tool calls, preserving approval and policy preflight; red-team security behavior | Tests show no unauthorized tool execution in adversarial/malformed plan cases |
| Days 61–75 | Integrate one local vision model and one local ASR/TTS backend if required; otherwise defer multimodality until text quality is strong | End-to-end modality tests, explicit model/device/license support, latency and quality metrics |
| Days 76–90 | Optimize the chosen deployment: true streaming, quantization, GPU/CPU benchmarks, resource caps, cross-platform packaging and supply-chain checks | Clean release artifacts and documented benchmarks; no unsupported capability claims |

The exact durations depend on target hardware, selected model, available datasets, and whether a mature inference backend is used. Do not optimize the current seed-initialized tiny transformer as if it were the final model.

## 6. Definition of “best-grade” acceptance

OAA should not be called production-grade until all applicable gates below pass:

1. **Real model:** a licensed pretrained model and matching tokenizer load and generate coherent output without fake weights.
2. **Quality evidence:** held-out domain benchmark, retrieval and citation tests, human review, and documented known failure modes.
3. **Training reproducibility:** actual fine-tune or adapter training, logged loss/configuration, deterministic data fingerprinting, checkpoint round-trip, and post-training evaluation.
4. **Reliable inference:** true incremental streaming, controlled memory/context, cancellation/error handling, and repeatable performance metrics.
5. **Safe agents:** structured tool-call decoding, least-privilege policies, per-action consent, sandboxing, and adversarial tests.
6. **Real multimodality if advertised:** compatible trained models plus successful end-to-end image/audio tests.
7. **Privacy and supply chain:** model/dependency provenance, dataset licensing, local storage protections, no accidental data commits, and release integrity.
8. **Release quality:** cross-platform CI, install-from-scratch tests, clear supported configurations, version alignment, security scans and release documentation.

## 7. Current strengths to preserve

- Local-first design and explicit local file paths.
- Clear package boundaries between C++ runtime and Python orchestration.
- Reusable test fixtures rather than committing model weights or datasets.
- Deny-by-default tool policies and approval-gated execution.
- Local dataset preparation and hashed evaluation input tracking.
- Reports that omit prompts and generated outputs by default.
- C++ and Python regression tests running in CI.
- Honest runtime capability reporting; keep this truthful as real backends are added.

## 8. External references

These references informed the recommendations and are useful implementation starting points:

1. Hugging Face TRL, SFT Trainer: https://huggingface.co/docs/trl/sft_trainer — supervised fine-tuning, dataset formats, loss masking and training/evaluation datasets.
2. Hugging Face Transformers, PEFT: https://huggingface.co/docs/transformers/peft — parameter-efficient adapters and training/checkpoint handling.
3. llama.cpp project documentation: https://github.com/ggml-org/llama.cpp — local model inference, GGUF and a range of CPU/GPU backends and quantization formats.
4. MLCommons, MLPerf Inference benchmark definitions: https://docs.mlcommons.org/inference/benchmarks/ — task-specific quality references and system latency constraints.
5. NIST AI 600-1, Generative AI Profile: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence — lifecycle risk management for generative AI.
6. OWASP GenAI Security Project, LLM/GenAI Top 10 (2025): https://genai.owasp.org/llm-top-10/ — prompt injection, data disclosure, supply-chain and model/data poisoning, improper output handling, excessive agency, and other application risks.

## Final conclusion

The planned engineering phases have reached a tested checkpoint. The main thing that prevents OAA from being a best-grade GenAI is **not another orchestration feature**: it is the absence of real pretrained model weights and tokenizer support, real training, high-quality semantic retrieval, true streaming/optimization, end-to-end multimodal models, and comprehensive quality/safety evaluation.

The best next engineering investment is therefore: **select a real local model and inference backend, establish a rigorous held-out quality benchmark, then implement reproducible fine-tuning and secure model-driven tool use**. Preserve the existing OAA core, tests, local-first boundaries, and explicit permissions as the foundation around that model.
