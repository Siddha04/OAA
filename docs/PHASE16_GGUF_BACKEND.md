# OAA Phase 16 - Real Local GGUF Inference Backend

## Status

Phase 16 adds an optional production-checkpoint adapter using llama-cpp-python. Unlike OAA's tiny_transformer_v1 fixture, this adapter loads a real local GGUF file, uses the tokenizer/model metadata packaged with that checkpoint, sends chat messages through llama.cpp's chat-completion API, and supports incremental stream chunks.

The repository does not contain model files. A user must obtain a compatible GGUF checkpoint from a trusted source, verify its license and provenance, and store it locally.

## Install

    python -m pip install "oaa[gguf]"

The llama-cpp-python package builds native llama.cpp bindings. A working C/C++ toolchain is required for source builds. Hardware-specific builds may use the documented CPU, CUDA, Metal, HIP/ROCm, or Vulkan options. Start with CPU inference to verify the model and chat template before enabling GPU offload.

## Commands

Inspect a local checkpoint:

    genai gguf-info --model ./models/instruct-model.gguf --n-ctx 4096

One-shot request:

    genai gguf-ask --model ./models/instruct-model.gguf --system "You are OAA." --max-tokens 128 "Explain local inference."

Interactive streaming chat:

    genai gguf-chat --model ./models/instruct-model.gguf --n-ctx 4096 --n-gpu-layers 0 --max-tokens 256

Use --chat-format only if the model's metadata does not provide the correct template or the model documentation explicitly requires an override. GPU layer configuration is an execution preference, not proof that a GPU was actually used.

## Python API

    from oaa import GenerationConfig, LlamaCppBackend

    backend = LlamaCppBackend(
        "./models/instruct-model.gguf",
        n_ctx=4096,
        n_gpu_layers=0,
    )
    answer = backend.chat(
        [
            {"role": "system", "content": "You are OAA."},
            {"role": "user", "content": "Explain local inference."},
        ],
        GenerationConfig(max_tokens=128, temperature=0.2),
    )
    for piece in backend.stream("Continue with a concise summary."):
        print(piece, end="")

The adapter exposes backend capabilities and GGUF metadata. It reports that a checkpoint file was loaded, but it does not certify that the checkpoint is pretrained, safe, correctly licensed, or high quality. It does not claim structured tool calling, multimodal support, or model-specific function calling.

## Safety and operational boundaries

- The backend requires an existing local .gguf file; it does not call a model hub or network download.
- Only explicit local model paths are accepted. Model files and adapters should stay outside Git.
- Message roles are limited to system, user, and assistant; tool-role messages and arbitrary extra fields are rejected.
- Context length, GPU-layer count, thread count, messages, and generation settings are validated before the model call.
- A model's output is untrusted text. This phase does not connect model output to tools or shell commands.
- A GGUF file can contain untrusted or malicious artifacts; acquire it from a trusted source, verify checksums, license, and origin, and isolate model loading appropriately.

## Verification scope and remaining gate

CI uses a fake llama_cpp API object to verify the adapter contract, message mapping, stream chunk handling, and CLI behavior without downloading or committing model weights. A real end-to-end inference acceptance test still requires a separately supplied GGUF file and the target machine's supported native backend.

Before describing OAA as a useful conversational assistant, run a real model on the intended hardware and evaluate held-out tasks, quality, context handling, latency, peak RAM/VRAM, and GPU utilization. This phase closes the checkpoint-loading architecture gap; it does not close the training, quality, privacy, or production-readiness gaps.
