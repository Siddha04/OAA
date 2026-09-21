from __future__ import annotations

from dataclasses import dataclass

import oaa_cpp


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.7

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")

    def to_cpp(self) -> oaa_cpp.GenerationConfig:
        config = oaa_cpp.GenerationConfig()
        config.max_tokens = self.max_tokens
        config.temperature = self.temperature
        return config
