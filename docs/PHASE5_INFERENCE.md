# OAA Phase 5 — Real Transformer Inference

## Scope

Phase 5 replaces the Phase 1 deterministic echo path with an actual decoder-only transformer inference pipeline on the CPU baseline.

The first supported model format is a small textual manifest describing a deterministic tiny transformer. The manifest contains architecture parameters and a seed. Transformer weights are initialized in memory from that seed, so binary model weights are not committed to the repository.

This is a runtime and inference validation model, not a pretrained production LLM.

## Pipeline

Prompt
-> printable ASCII tokenizer
-> token IDs
-> token and position embeddings
-> transformer blocks
   -> RMS normalization
   -> Q/K/V projections
   -> causal self-attention
   -> residual connection
   -> RMS normalization
   -> GELU MLP
   -> residual connection
-> final RMS normalization
-> vocabulary logits
-> sampler
   -> greedy decoding / temperature
   -> top-k
   -> top-p
   -> repetition penalty
-> generated tokens
-> streaming text

## KV cache

Each transformer layer retains key/value vectors in KvCache. New tokens attend to the retained sequence instead of rebuilding previous key/value projections.

The current cache lives in CPU memory and is reset at the beginning of each generation call.

## Model manifest

Example:

architecture=tiny_transformer_v1
vocab_size=95
hidden_size=16
num_layers=2
num_heads=4
intermediate_size=32
context_length=64
seed=42

The tokenizer has 95 printable ASCII symbols. Non-printable input bytes are mapped to the space token.

## Python usage

from oaa import Engine, GenerationConfig

engine = Engine()
engine.load_model("tiny.manifest")

text = engine.generate(
    "Hello",
    GenerationConfig(
        max_tokens=32,
        temperature=0.8,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.1,
        seed=123,
    ),
)

print(text)

for chunk in engine.generate_stream("Hello"):
    print(chunk, end="")

## Verification

$env:CMAKE_GENERATOR="Ninja"

cmake -S . -B build -G Ninja -DOAA_BUILD_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build --config Release
ctest --test-dir build --output-on-failure -C Release

py -m pip install . --no-cache-dir
py -m pytest tests/python -v

## Limits

- This phase does not load pretrained GGUF, Safetensors, or Hugging Face checkpoints.
- The deterministic tiny transformer is not intended to be a useful conversational model.
- GPU execution remains dependent on an NVIDIA CUDA environment; the current development machine uses AMD graphics.
- Quantization, large-model memory mapping, optimized kernels, production tokenizer formats, and external pretrained weights remain future work.

## Next phase

Phase 6 builds the personal assistant layer: sessions, prompt templates, system instructions, conversation history, and user-facing CLI behavior.
