from __future__ import annotations

from dataclasses import dataclass

import oaa_cpp


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_cpp(self) -> oaa_cpp.GenerationConfig:
        config = oaa_cpp.GenerationConfig()
        config.max_tokens = self.max_tokens
        config.temperature = self.temperature
        config.top_k = self.top_k
        config.top_p = self.top_p
        config.repetition_penalty = self.repetition_penalty
        config.seed = self.seed
        return config
