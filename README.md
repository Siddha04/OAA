# OAA

**OAA — Personal Local-First GenAI Runtime**

OAA is a personal GenAI system built around a Python orchestration layer and a high-performance C++ runtime. The initial architecture is local-first: Python handles orchestration while C++ handles performance-critical runtime and inference work.

## Architecture

```text
Python
├── RAG
├── Agents
├── Memory
├── Tools
├── Training
├── Evaluation
└── Orchestration
        │
      pybind11
        │
        ▼
C++ Runtime
├── Inference
├── Tensor operations
├── KV cache
├── Quantization
├── Memory management
└── CUDA acceleration
        │
       GPU
        │
       LLM
```

## Language responsibilities

### Python
- AI orchestration
- RAG and retrieval
- Memory
- Agents and tools
- Training and fine-tuning workflows
- Evaluation
- APIs and high-level application logic

### C++
- Model runtime
- Tensor operations
- Token generation
- KV cache
- Memory management
- Quantization
- CPU/GPU execution
- CUDA acceleration

## Python ↔ C++ boundary

OAA uses **pybind11** as the primary in-process bridge. The Python-facing runtime API is intentionally small and stable:

- `load_model()`
- `generate()`
- `generate_stream()`
- `embed()`
- `get_stats()`
- `unload()`

Python orchestrates; C++ executes performance-critical runtime work.

## Roadmap

1. Foundation: C++20, CMake, Python package, tests, CI
2. C++ model/runtime core
3. Python ↔ C++ integration with pybind11
4. CUDA acceleration
5. Local LLM inference
6. Personal chat assistant
7. Personal RAG
8. Long-term memory
9. Tool system
10. Personal agent
11. Vision and voice
12. Fine-tuning
13. Evaluation and optimization
14. Native CLI personal GenAI platform
15. Advanced inference and agent research

## Phase 1 acceptance criteria

- C++ builds with CMake
- Python package imports successfully
- Python can instantiate the C++ engine through pybind11
- A deterministic Python → C++ → Python smoke test passes
- No model-specific implementation is required in Phase 1

## Status

Phase 1 foundation is being established.
