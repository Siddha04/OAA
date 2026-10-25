# OAA

**OAA — Personal Local-First GenAI Runtime**

OAA is a personal GenAI system built around a Python orchestration layer and a C++ runtime. The intended design keeps document ingestion, retrieval, sessions, and orchestration local; performance-critical inference runs through the native runtime.

## Language responsibilities

### Python

- AI orchestration and assistant sessions
- Local document ingestion and retrieval-augmented prompts
- Agents, tools, and memory
- Training, fine-tuning, and evaluation workflows

### C++

- Model runtime and tensor operations
- Token generation and KV cache
- Memory management
- CPU execution and optional CUDA kernels

## Install and use

Install the package from a local checkout in an environment where the C++ build tools are installed:

    python -m pip install .

PDF extraction is optional:

    python -m pip install "oaa[rag]"

Search your local documents without loading a language model:

    genai search --docs ./knowledge "where are model settings stored"

Start a retrieval-assisted chat using a model manifest and a local knowledge directory:

    genai chat --model ./tiny.manifest --docs ./knowledge

Use a normal chat session without document retrieval by omitting the docs option:

    genai chat --model ./tiny.manifest

The search and chat commands read documents from the selected local directory. Documents and the vector index are not uploaded or committed. The current vector index is in memory, so it is rebuilt on each command invocation.

## Current implementation status

- Phases 1-3: foundation, C++ runtime core, and Python/C++ boundary implemented and CI verified.
- Phase 4: CUDA support has a CPU fallback and software integration tests. Real NVIDIA GPU execution is not verified on the project's AMD-only development machine.
- Phase 5: a small decoder-only transformer validates inference and sampling. It initializes test weights from a manifest seed; it is not a pretrained conversational LLM or a GGUF/Safetensors loader.
- Phase 6: local assistant sessions and CLI chat/ask implemented.
- Phase 7A: local document loaders, size limits, and overlapping chunking implemented.
- Phase 7B: deterministic lexical hash embeddings and an in-memory cosine-search index implemented.
- Phase 7C: retrieved context can be inserted into normal and streaming assistant prompts with source attribution.
- Phase 7D: local document search and optional RAG-enabled chat CLI implemented.
- Phase 8: explicit local SQLite long-term memory with list/search/update/delete and optional chat recall.
- Phase 9: a deny-by-default tool registry, bounded calculator, workspace-scoped read-only text reader, and explicit tool CLI.
- Phase 10: structured Personal Agent plans with bounded steps, schema checks, an exact tool allow-list, and per-step user approval.
- Phase 11: bounded local image/WAV ingestion, normalized PCM access, and explicit speech-model adapter contracts; no pretrained vision or speech model is bundled.
- Phase 12: local SFT JSONL validation, duplicate detection, dataset fingerprints, and deterministic train/validation/test splitting. Weight optimization is not implemented yet.
- Phase 13: local reference-based evaluation, string-match/token-overlap metrics, latency percentiles, and privacy-minimizing JSON reports.
- Phase 14: standalone C++ executable with local manifest loading, generation controls, help/version output, install target, and CTest smoke coverage.

## Phase 7 limitations

- Hash embeddings are a lexical baseline, not pretrained semantic embeddings. Questions phrased with little word overlap can miss relevant passages.
- The retrieval index is memory-only and is rebuilt each run; persistent indexing is future work.
- PDFs need the optional pypdf dependency and must contain selectable text. Scanned PDFs require OCR, which is not included.
- This project does not yet load pretrained model weights in GGUF, Safetensors, or Hugging Face formats.

## Python / C++ boundary

OAA uses pybind11 for the in-process bridge. The Python-facing runtime API includes model loading, generation, streaming, runtime statistics, and unload operations.


## Long-term memory (Phase 8)

Save a memory explicitly; OAA does not automatically store chat transcripts:

    genai memory add --db ./.oaa/memory.sqlite3 --category preference "Prefers concise technical examples"

List or search saved memories:

    genai memory list --db ./.oaa/memory.sqlite3
    genai memory search --db ./.oaa/memory.sqlite3 "Python examples"

Update or delete one memory by its displayed ID:

    genai memory update --db ./.oaa/memory.sqlite3 1 "Prefers concise code examples"
    genai memory delete --db ./.oaa/memory.sqlite3 1

Clear all records only with explicit confirmation:

    genai memory clear --db ./.oaa/memory.sqlite3 --yes

Enable recall during local chat:

    genai chat --model ./tiny.manifest --memory-db ./.oaa/memory.sqlite3

The database is local and ignored by Git. This implementation does not encrypt it, so do not store passwords, tokens, or other highly sensitive information. SQLite deletion removes logical records but is not a guarantee of secure physical erasure.


## Controlled local tools (Phase 9)

List available local tools:

    genai tools list
    genai tools list --workspace ./knowledge

Run the bounded arithmetic calculator:

    genai tools run calculator --args '{"expression":"(2 + 3) * 4"}'

Read a text file only from an explicitly selected workspace:

    genai tools run read_text_file --workspace ./knowledge --args '{"path":"manual.md"}'

The calculator accepts arithmetic only and does not execute Python code. The file reader is read-only, limits file size, rejects hidden/generated paths, and refuses paths that resolve outside the selected workspace. A tool must pass the tool-name and capability allow-list. The CLI invokes only the tool explicitly named by the user; no shell or network tool is registered by default.


## Approval-gated Personal Agent (Phase 10)

Preview a structured plan without running any tool:

    genai agent plan --plan-file ./agent-plan.json

A plan is ordinary JSON with a goal and an ordered list of tool calls:

    {
      "goal": "Calculate a half of 100",
      "steps": [
        {
          "tool": "calculator",
          "arguments": {"expression": "100 / 2"},
          "reason": "Compute the requested value"
        }
      ]
    }

Run a plan only after naming the tool or tools allowed for this invocation:

    genai agent run --plan-file ./agent-plan.json --allow-tool calculator

The CLI displays each proposed call and asks for approval before each step. Without a positive approval, that step and the remaining plan do not execute. All tools and arguments are validated before the first action, and the complete plan must fit the configured step limit. To use the read-only file tool, pass an explicit workspace and include the tool in the allow-list.

Phase 10 accepts structured, caller-supplied plans; it does not claim that the seeded tiny transformer can reliably invent tool calls from natural-language requests. The agent does not add shell or network tools, does not evaluate plan text as code, and does not automatically turn tool output into further actions. OAA still does not yet load a pretrained conversational LLM.


## Local multimodal foundation (Phase 11)

Install the optional image decoder:

    python -m pip install "oaa[vision]"

Inspect a local image (JPEG, PNG, or WebP); the CLI decodes it to bounded, orientation-corrected RGB pixels:

    genai media image-info --path ./photo.png

Inspect a local uncompressed PCM WAV file:

    genai media audio-info --path ./speech.wav

The Python API is available from the oaa.media module or via the package exports load_image, load_wav, transcribe_audio, and synthesize_speech. WAV input supports mono/stereo PCM with 8–192 kHz sample rates and bounded duration/file size. Images and audio are read from local paths; no camera, microphone, network or cloud API is accessed.

This phase prepares media and defines explicit SpeechToTextBackend / TextToSpeechBackend adapter contracts, but does not bundle a vision encoder, OCR engine, ASR model, TTS model, or model-driven image/audio understanding. Calling speech transcription or synthesis without supplying a backend fails clearly instead of silently connecting to a service.


## Fine-tuning data preparation (Phase 12)

SFT input is a local UTF-8 JSONL file with one JSON object per line. Each record needs "instruction" and "output" string fields; "input" is optional. Example record:

    {"instruction":"Summarize the note","input":"A local note","output":"A short summary."}

Validate the data before using it:

    genai train-data validate --path D:/oaa-data/sft.jsonl

Create deterministic splits outside the checkout:

    genai train-data split --path D:/oaa-data/sft.jsonl --output-dir D:/oaa-data/splits --seed 42

The split command refuses to write into a non-empty directory. Exact duplicate records are removed by default. Validation includes bounded file/record sizes and a SHA-256 fingerprint. Keep private datasets and generated split files outside the Git working tree and never commit them.

This is data preparation, not model fine-tuning. The current C++ transformer has no gradient/backpropagation, optimizer, checkpoint loading, or weight-saving support. No accuracy gain is claimed from this phase.


## Local evaluation and benchmarking (Phase 13)

Use a local JSONL file with one {"prompt":"...", "expected":"..."} record per line (optional "category"):

    {"prompt":"Return the word hello","expected":"hello","category":"format"}

Run evaluation against the current local model manifest:

    genai evaluate --model ./tiny.manifest --dataset D:/oaa-data/eval.jsonl --report D:/oaa-results/eval-report.json --max-tokens 16

The report records dataset SHA-256, exact/substring reference matching, token-overlap precision/recall/F1, and latency percentiles. It intentionally does not include raw prompts or generated outputs. An existing report is not replaced unless --overwrite is passed.

These are task-specific reference string diagnostics and timing—not a general model accuracy, safety, reasoning, or truthfulness score. The tiny seeded transformer is not a pretrained conversational LLM.


## Standalone native C++ CLI (Phase 14)

Configure and build without Python bindings:

    cmake -S . -B build-native -DOAA_BUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON
    cmake --build build-native --config Release
    ctest --test-dir build-native --output-on-failure -C Release

Run native inference on a local manifest:

    ./build-native/oaa_native_cli --model ./tiny.manifest --prompt "Hello OAA" --max-tokens 24 --temperature 0

Show help/version with --help or --version. The executable depends only on the C++ runtime, not Python/pybind11. It currently supports the deterministic tiny_transformer_v1 manifest; it does not yet load production checkpoint formats such as GGUF or Safetensors.
