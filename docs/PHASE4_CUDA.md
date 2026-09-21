# OAA Phase 4 — Optional CUDA Acceleration

## Scope

Phase 4 adds a real CUDA execution backend without making CUDA a mandatory dependency.

The CPU runtime remains the portability baseline. A normal build continues to use the C++ stub backend when CUDA is disabled.

## Implemented

- CMake option: `OAA_BUILD_CUDA`, default `OFF`.
- CUDA device detection through the native runtime.
- Exception-safe cleanup for CUDA allocations on both success and failure paths.
- First real GPU kernel: element-wise float32 vector addition.
- C++ and Python tests that execute the GPU kernel when a CUDA device is available.
- Python status functions for build-time CUDA support and runtime device availability.
- Existing CPU and Python workflows remain unchanged when CUDA is disabled.

## Build modes

CPU-only build:

```text
cmake -S . -B build -DOAA_BUILD_CUDA=OFF
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release
```

CUDA build:

```text
cmake -S . -B build-cuda -DOAA_BUILD_CUDA=ON
cmake --build build-cuda --config Release
ctest --test-dir build-cuda --output-on-failure -C Release
```

For a Python package build with CUDA enabled, pass the same CMake option through the scikit-build-core configuration, for example with the environment variable used by your shell:

```text
CMAKE_ARGS=-DOAA_BUILD_CUDA=ON python -m pip install .
```

On Windows PowerShell, use:

```powershell
$env:CMAKE_ARGS='-DOAA_BUILD_CUDA=ON'
py -m pip install .
```

## Verification boundary

Generic GitHub-hosted CPU runners are used for the normal CI workflow. They verify that CUDA-disabled builds remain healthy.

Actual GPU execution must be verified on a machine with a compatible NVIDIA driver, CUDA toolkit, and visible CUDA device. The CUDA test intentionally skips the kernel assertion when no device is available rather than pretending that a CPU runner is a GPU validation environment.

## Next phase

Phase 5 replaces the current deterministic echo generator with real transformer inference: tokenizer → embeddings → transformer blocks → attention/KV cache → logits → sampling → streaming.
