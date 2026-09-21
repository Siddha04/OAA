# OAA Phase 2 — C++ Runtime Core

## Scope

Phase 2 establishes model-independent C++ runtime primitives that later LLM inference can build on.

## Implemented

- Tensor: shape-aware float32 storage, indexing, 2D matrix multiplication, and numerically stable last-dimension softmax.
- MemoryBuffer / MemoryManager: explicit byte-buffer ownership for runtime memory.
- Tokenizer interface plus a lossless ByteTokenizer baseline implementation.
- ModelLoader interface plus a small manifest loader for validating model metadata before a real model format is introduced.
- CpuInferenceEngine: linear projection and softmax primitives for CPU execution.

## Deliberate boundary

Phase 2 does not claim transformer inference, attention, KV cache, sampling, or a specific model format. Those belong to later phases.

The manifest loader is metadata-only. It does not interpret model weights.

The byte tokenizer is a correctness baseline, not the tokenizer of a production LLM.

## Acceptance criteria

1. C++20 core compiles with the new runtime sources.
2. Tensor operations produce deterministic expected results.
3. Memory allocation smoke tests pass.
4. Tokenization round-trips byte strings losslessly.
5. Model metadata manifests validate into a typed configuration.
6. CPU linear and softmax operations execute on real tensors.
7. The existing Python/pybind11 package remains buildable.

## Next phase

Phase 3 hardens and expands the Python to C++ boundary around these runtime primitives without duplicating runtime logic in Python.
