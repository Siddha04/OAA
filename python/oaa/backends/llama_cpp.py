"""Optional local GGUF inference through llama-cpp-python.

This module requires the optional 'gguf' extra and an explicit local GGUF file.
It does not download model files or select a remote endpoint.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..backend import BackendCapabilities
from ..config import GenerationConfig

MAX_MESSAGES = 128
MAX_MESSAGE_CHARS = 32_768


class LlamaCppBackend:
    """Load a local GGUF checkpoint and expose chat, streaming and capabilities."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        chat_format: str | None = None,
        verbose: bool = False,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model file does not exist: {path}")
        if path.suffix.lower() != ".gguf":
            raise ValueError("model_path must point to a local .gguf file")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer")
        if (
            not isinstance(n_gpu_layers, int)
            or isinstance(n_gpu_layers, bool)
            or n_gpu_layers < -1
        ):
            raise ValueError("n_gpu_layers must be -1 or a non-negative integer")
        if n_threads is not None and (
            not isinstance(n_threads, int)
            or isinstance(n_threads, bool)
            or n_threads <= 0
        ):
            raise ValueError("n_threads must be a positive integer or None")
        if chat_format is not None and (
            not isinstance(chat_format, str) or not chat_format.strip()
        ):
            raise ValueError("chat_format must be a non-empty string or None")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                'GGUF inference requires the optional dependency; install with '
                '"python -m pip install oaa[gguf]"'
            ) from exc

        kwargs: dict[str, Any] = {
            "model_path": str(path.resolve()),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "verbose": verbose,
        }
        if n_threads is not None:
            kwargs["n_threads"] = n_threads
        if chat_format is not None:
            kwargs["chat_format"] = chat_format

        self._model = Llama(**kwargs)
        metadata = getattr(self._model, "metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        architecture = metadata.get("general.architecture", "gguf_unknown")
        file_type = metadata.get("general.file_type", "unknown")
        self._model_path = path.resolve()
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._architecture = str(architecture)
        self._file_type = str(file_type)
        self._capabilities = BackendCapabilities(
            backend_name="llama_cpp_python",
            model_family=self._architecture,
            modalities=("text",),
            pretrained_weights_loaded=False,
            checkpoint_loaded=True,
            true_streaming=True,
            structured_tool_calls=False,
            quantization=f"gguf_file_type:{self._file_type}",
            max_context_tokens=n_ctx,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @property
    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": str(self._model_path),
            "model_format": "GGUF",
            "architecture": self._architecture,
            "file_type": self._file_type,
            "context_size": self._n_ctx,
            "gpu_layers_requested": self._n_gpu_layers,
            "capabilities": self._capabilities.to_dict(),
        }

    @staticmethod
    def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
        if not isinstance(messages, (list, tuple)) or not messages:
            raise ValueError("messages must be a non-empty list")
        if len(messages) > MAX_MESSAGES:
            raise ValueError(f"messages exceed the {MAX_MESSAGES}-message limit")
        validated: list[dict[str, str]] = []
        for index, message in enumerate(messages, start=1):
            if not isinstance(message, Mapping):
                raise TypeError(f"message {index} must be an object")
            if set(message) != {"role", "content"}:
                raise ValueError(f"message {index} must contain only role and content")
            role = message["role"]
            content = message["content"]
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"message {index} has unsupported role")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"message {index} content must be a non-empty string")
            if len(content) > MAX_MESSAGE_CHARS:
                raise ValueError(f"message {index} exceeds {MAX_MESSAGE_CHARS} characters")
            validated.append({"role": role, "content": content})
        if validated[0]["role"] not in {"system", "user"}:
            raise ValueError("first message must be a system or user message")
        return validated

    @staticmethod
    def _generation_kwargs(config: GenerationConfig) -> dict[str, Any]:
        if not isinstance(config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig")
        return {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repeat_penalty": config.repetition_penalty,
            "seed": config.seed,
        }

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        config: GenerationConfig | None = None,
    ) -> str:
        validated = self._validate_messages(messages)
        result = self._model.create_chat_completion(
            messages=validated,
            stream=False,
            **self._generation_kwargs(config or GenerationConfig()),
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("GGUF backend returned an invalid chat-completion response") from exc
        if not isinstance(content, str):
            raise RuntimeError("GGUF backend returned a non-text assistant response")
        return content

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return self.chat(
            [{"role": "user", "content": prompt}],
            config=config,
        )

    def chat_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        validated = self._validate_messages(messages)
        chunks = self._model.create_chat_completion(
            messages=validated,
            stream=True,
            **self._generation_kwargs(config or GenerationConfig()),
        )
        for chunk in chunks:
            try:
                content = chunk["choices"][0]["delta"].get("content")
            except (KeyError, IndexError, TypeError, AttributeError) as exc:
                raise RuntimeError("GGUF backend returned an invalid stream chunk") from exc
            if content is None:
                continue
            if not isinstance(content, str):
                raise RuntimeError("GGUF backend returned a non-text stream delta")
            if content:
                yield content

    def stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        yield from self.chat_stream(
            [{"role": "user", "content": prompt}],
            config=config,
        )
