"""Capability-aware generation service and adapter for the current native runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .config import GenerationConfig
from .engine import Engine

_SUPPORTED_MODALITIES = frozenset({"text", "image", "audio"})


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    backend_name: str
    model_family: str
    modalities: tuple[str, ...] = ("text",)
    pretrained_weights_loaded: bool = False
    true_streaming: bool = False
    structured_tool_calls: bool = False
    quantization: str = "unknown"
    max_context_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("backend_name", self.backend_name),
            ("model_family", self.model_family),
        ):
            if not isinstance(value, str) or not value.strip() or len(value) > 128:
                raise ValueError(f"{field_name} must be a non-empty string of at most 128 characters")
        if not isinstance(self.modalities, (tuple, list)) or not self.modalities:
            raise ValueError("modalities must be a non-empty sequence")
        modalities = tuple(self.modalities)
        if any(item not in _SUPPORTED_MODALITIES for item in modalities):
            raise ValueError("modalities may contain only text, image, and audio")
        if len(set(modalities)) != len(modalities):
            raise ValueError("modalities must not contain duplicates")
        object.__setattr__(self, "modalities", modalities)
        for name in ("pretrained_weights_loaded", "true_streaming", "structured_tool_calls"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.quantization, str) or not self.quantization.strip():
            raise ValueError("quantization must be a non-empty string")
        if self.max_context_tokens is not None and (
            not isinstance(self.max_context_tokens, int)
            or isinstance(self.max_context_tokens, bool)
            or self.max_context_tokens <= 0
        ):
            raise ValueError("max_context_tokens must be a positive integer or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "model_family": self.model_family,
            "modalities": list(self.modalities),
            "pretrained_weights_loaded": self.pretrained_weights_loaded,
            "true_streaming": self.true_streaming,
            "structured_tool_calls": self.structured_tool_calls,
            "quantization": self.quantization,
            "max_context_tokens": self.max_context_tokens,
        }


@runtime_checkable
class TextGenerationBackend(Protocol):
    @property
    def capabilities(self) -> BackendCapabilities: ...

    def generate(self, prompt: str, config: GenerationConfig) -> str: ...


class RuntimeEngineBackend:
    """Honest adapter for OAA's currently supported seed-initialized C++ model."""

    def __init__(self, engine: Engine, *, max_context_tokens: int | None = None) -> None:
        if not isinstance(engine, Engine):
            raise TypeError("engine must be an oaa.Engine instance")
        if not engine.loaded:
            raise ValueError("engine must have a loaded model")
        self._engine = engine
        self._capabilities = BackendCapabilities(
            backend_name="oaa_cpp_tiny_transformer",
            model_family="tiny_transformer_v1",
            modalities=("text",),
            pretrained_weights_loaded=False,
            true_streaming=False,
            structured_tool_calls=False,
            quantization="float32_unquantized_seeded",
            max_context_tokens=max_context_tokens,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        return self._engine.generate(prompt, config)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    backend_name: str
    elapsed_seconds: float
    output_characters: int
    requested_max_tokens: int
    characters_per_second: float

    def metrics(self) -> dict[str, Any]:
        """Return telemetry without including prompt or generated text."""
        return {
            "backend_name": self.backend_name,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "output_characters": self.output_characters,
            "requested_max_tokens": self.requested_max_tokens,
            "characters_per_second": round(self.characters_per_second, 3),
        }


class GenerationService:
    """Validate generation budgets and normalize telemetry around any backend."""

    def __init__(
        self,
        backend: TextGenerationBackend,
        *,
        max_prompt_chars: int = 32_768,
        max_output_tokens: int = 2_048,
        max_output_chars: int = 100_000,
    ) -> None:
        capabilities = getattr(backend, "capabilities", None)
        generate = getattr(backend, "generate", None)
        if not isinstance(capabilities, BackendCapabilities) or not callable(generate):
            raise TypeError("backend must expose BackendCapabilities and generate(prompt, config)")
        for name, value in (
            ("max_prompt_chars", max_prompt_chars),
            ("max_output_tokens", max_output_tokens),
            ("max_output_chars", max_output_chars),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._backend = backend
        self._max_prompt_chars = max_prompt_chars
        self._max_output_tokens = max_output_tokens
        self._max_output_chars = max_output_chars

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._backend.capabilities

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > self._max_prompt_chars:
            raise ValueError(f"prompt exceeds max_prompt_chars ({len(prompt)} > {self._max_prompt_chars})")
        if "text" not in self.capabilities.modalities:
            raise ValueError("selected backend does not support text generation")
        generation_config = config or GenerationConfig()
        if not isinstance(generation_config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig")
        if generation_config.max_tokens > self._max_output_tokens:
            raise ValueError(
                f"max_tokens exceeds service limit ({generation_config.max_tokens} > {self._max_output_tokens})"
            )
        started = time.perf_counter()
        output = self._backend.generate(prompt, generation_config)
        elapsed = max(0.0, time.perf_counter() - started)
        if not isinstance(output, str):
            raise TypeError("backend generate() must return a string")
        if len(output) > self._max_output_chars:
            raise ValueError(
                f"generated output exceeds max_output_chars ({len(output)} > {self._max_output_chars})"
            )
        return GenerationResult(
            text=output,
            backend_name=self.capabilities.backend_name,
            elapsed_seconds=elapsed,
            output_characters=len(output),
            requested_max_tokens=generation_config.max_tokens,
            characters_per_second=(len(output) / elapsed) if elapsed > 0 else 0.0,
        )
