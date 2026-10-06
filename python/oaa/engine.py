from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import oaa_cpp

from .config import GenerationConfig


class Engine:
    """Stable Python facade over the native OAA runtime."""

    def __init__(self) -> None:
        self._runtime = oaa_cpp.Engine()

    def load_model(self, path: str) -> bool:
        if not isinstance(path, str) or not path.strip():
            raise ValueError("model manifest path must be a non-empty string")
        return bool(self._runtime.load_model(path))

    def unload(self) -> None:
        self._runtime.unload()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        cfg = config or GenerationConfig()
        return str(self._runtime.generate(prompt, cfg.to_cpp()))

    def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        cfg = config or GenerationConfig()
        for chunk in self._runtime.generate_stream(prompt, cfg.to_cpp()):
            yield str(chunk)

    def status(self) -> str:
        return str(self._runtime.status())

    def get_stats(self) -> dict[str, Any]:
        stats = self._runtime.get_stats()
        tokens = int(stats.generated_tokens)
        seconds = float(stats.last_generation_seconds)
        return {
            "model_loaded": bool(stats.model_loaded),
            "generation_calls": int(stats.generation_calls),
            "prompt_tokens": int(stats.prompt_tokens),
            "generated_tokens": tokens,
            "last_generation_seconds": seconds,
            "tokens_per_second": (tokens / seconds) if seconds > 0 else 0.0,
            "model_path": str(stats.model_path),
        }

    @property
    def loaded(self) -> bool:
        return bool(self._runtime.loaded())
