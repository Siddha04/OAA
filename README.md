# OAA

**OAA — Personal Local-First GenAI Runtime**

OAA is a personal GenAI system built around a Python orchestration layer and a C++ runtime. The intended design keeps document ingestion, retrieval, sessions, and orchestration local; performance-critical inference runs through the native runtime.

## Architecture

    Python
    ├── RAG
    ├── Agents
    ├── Memory
    ├── Tools
    ├── Training
    ├── Evaluation
    └── Orchestration
            |
          pybind11
            |
            v
       C++ Runtime
       ├── Inference
       ├── Tensor operations
       ├── KV cache
       ├── Quantization
       ├── Memory management
       └── CPU / optional CUDA execution

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
