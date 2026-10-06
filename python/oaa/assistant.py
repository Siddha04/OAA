from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .config import GenerationConfig
from .engine import Engine
from .session import ChatMessage, ChatSession


class PersonalAssistant:
    def __init__(
        self,
        engine: Engine,
        session: ChatSession | None = None,
    ) -> None:
        self._engine = engine
        self._session = session or ChatSession()

    @property
    def session(self) -> ChatSession:
        return self._session

    def set_system_prompt(self, system_prompt: str) -> None:
        self._session.system_prompt = system_prompt

    def reset(self) -> None:
        self._session.clear()

    def history(self) -> tuple[ChatMessage, ...]:
        return self._session.messages

    def chat(
        self,
        user_message: str,
        config: GenerationConfig | None = None,
    ) -> str:
        prompt = self._session.build_prompt(user_message)
        response = self._engine.generate(prompt, config)
        self._session.add("user", user_message)
        self._session.add("assistant", response)
        return response

    def chat_stream(
        self,
        user_message: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        prompt = self._session.build_prompt(user_message)
        chunks: list[str] = []
        try:
            for chunk in self._engine.generate_stream(prompt, config):
                chunks.append(chunk)
                yield chunk
        finally:
            if chunks:
                self._session.add("user", user_message)
                self._session.add("assistant", "".join(chunks))

    def stats(self) -> dict[str, Any]:
        return self._engine.get_stats()
