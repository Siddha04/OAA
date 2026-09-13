# OAA Phase 1 Architecture

## Goal

Phase 1 establishes a stable Python/C++ boundary before model-specific inference is introduced.

## Layers

```text
Python application layer
    │
    │  small stable API
    ▼
pybind11
    │
    ▼
C++ runtime core
    │
    ├── engine state
    ├── model lifecycle
    ├── generation interface
    └── runtime statistics
```

## Responsibility split

Python owns orchestration and user-facing application logic. C++ owns runtime operations that will later become performance-critical, including inference, tensor operations, KV cache, quantization, memory management and CUDA execution.

## Phase 1 runtime behavior

The current `generate()` implementation is intentionally deterministic and returns an `OAA_PHASE1_ECHO` response. It is a boundary test, not an LLM implementation.

`load_model()` also establishes the lifecycle contract without parsing or executing a real model yet.

## Public boundary

The first Python-facing surface is deliberately small:

- `load_model(path)`
- `generate(prompt)`
- `unload()`
- `get_stats()`
- `status()`
- `loaded`

Future APIs such as streaming generation and embeddings should be added only when their underlying C++ runtime implementations exist.

## Build model

- C++20 with CMake
- C++ core is a static library
- pybind11 exposes the C++ engine as `oaa_cpp`
- `python/oaa` provides the stable Python facade
- CTest covers the C++ smoke test
- pytest covers the Python-to-C++ smoke test
