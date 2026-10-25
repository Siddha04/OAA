# OAA Phase 14 - Standalone Native CLI

## Build

The native executable is built from the C++ runtime and has no dependency on Python or pybind11:

    cmake -S . -B build-native -DOAA_BUILD_PYTHON_BINDINGS=OFF -DBUILD_TESTING=ON
    cmake --build build-native --config Release
    ctest --test-dir build-native --output-on-failure -C Release

CMake installs the executable under bin/ when the project is installed.

## Usage

    oaa_native_cli --help
    oaa_native_cli --version
    oaa_native_cli --model ./tiny.manifest --prompt "Hello OAA" --max-tokens 24 --temperature 0
    oaa_native_cli --model ./tiny.manifest --prompt "A short prompt" --top-k 20 --top-p 0.95 --seed 42

The command checks required options and numeric arguments, catches manifest/runtime errors, prints generated text to stdout, and sends errors to stderr with a non-zero status.

## Verification

CTest includes native CLI help and one generation smoke test using a tiny test manifest. These validate build, command parsing, model-manifest loading and generation integration without requiring Python.

## Limitations

- Only the current tiny_transformer_v1 manifest with seed-initialized in-memory weights is supported.
- This is native executable delivery, not a trained production LLM. GGUF/Safetensors checkpoints, real tokenizers, memory mapping, CUDA model inference and training checkpoints remain unimplemented.
