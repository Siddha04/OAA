from __future__ import annotations

from typing import Any

import oaa_cpp


class Engine:
    """Small Python facade over the C++ runtime."""

    def __init__(self) -> None:
        self._runtime = oaa_cpp.Engine()

    def load_model(self, path: str) -> bool:
        return bool(self._runtime.load_model(path))

    def unload(self) -> None:
        self._runtime.unload()

    def generate(self, prompt: str) -> str:
        return str(self._runtime.generate(prompt))

    def status(self) -> str:
        return str(self._runtime.status())

    def get_stats(self) -> dict[str, Any]:
        stats = self._runtime.get_stats()
        return {
            "model_loaded": bool(stats.model_loaded),
            "generation_calls": int(stats.generation_calls),
            "model_path": str(stats.model_path),
        }

    @property
    def loaded(self) -> bool:
        return bool(self._runtime.loaded())
