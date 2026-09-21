# OAA Phase 3 — Python ↔ C++ Integration

## Scope

Phase 3 hardens the native boundary so Python can configure and consume the C++ runtime without duplicating runtime behavior.

## Implemented

- Typed Python GenerationConfig with validation and conversion to the native configuration.
- Native Engine.generate_stream() exposed through pybind11 and surfaced as a Python iterator.
- C++ validation for invalid generation configuration values.
- Explicit Python input validation for model paths and prompts.
- Expanded tests for configuration, streaming, and cross-language error behavior.

## Streaming boundary

The current stream is a deterministic chunked transport boundary. It is not token-level streaming from a transformer model yet. Real token streaming belongs to the model inference phase.

## Acceptance criteria

1. Native C++ extension builds successfully.
2. Python configuration reaches C++ as typed values.
3. Invalid configuration is rejected deterministically.
4. Stream chunks can be consumed from Python and reconstruct the native result exactly.
5. Existing Phase 1 and Phase 2 tests continue to pass.
6. No runtime implementation is duplicated in Python.

## Next phase

Phase 4 introduces an optional CUDA execution backend while retaining the CPU runtime as the portability baseline.
