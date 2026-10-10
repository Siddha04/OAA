from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .config import GenerationConfig
from .engine import Engine
from .rag.pipeline import RAGPipeline
from .session import ChatMessage, ChatSession


class PersonalAssistant:
    def __init__(
        self,
        engine: Engine,
        session: ChatSession | None = None,
        *,
        rag_pipeline: RAGPipeline | None = None,
        rag_top_k: int = 3,
        rag_max_context_chars: int = 512,
    ) -> None:
        if rag_top_k <= 0:
            raise ValueError("rag_top_k must be greater than zero")
        if rag_max_context_chars <= 0:
            raise ValueError("rag_max_context_chars must be greater than zero")
        self._engine = engine
        self._session = session or ChatSession()
        self._rag_pipeline = rag_pipeline
        self._rag_top_k = rag_top_k
        self._rag_max_context_chars = rag_max_context_chars

    @property
    def session(self) -> ChatSession:
        return self._session

    def set_system_prompt(self, system_prompt: str) -> None:
        self._session.system_prompt = system_prompt

    def reset(self) -> None:
        self._session.clear()

    def history(self) -> tuple[ChatMessage, ...]:
        return self._session.messages

    def _build_prompt(self, user_message: str) -> str:
        context = None
        if self._rag_pipeline is not None:
            context = self._rag_pipeline.build_context(
                user_message,
                top_k=self._rag_top_k,
                max_chars=self._rag_max_context_chars,
            )
        return self._session.build_prompt(user_message, retrieved_context=context)

    def chat(
        self,
        user_message: str,
        config: GenerationConfig | None = None,
    ) -> str:
        prompt = self._build_prompt(user_message)
        response = self._engine.generate(prompt, config)
        self._session.add("user", user_message)
        self._session.add("assistant", response)
        return response

    def chat_stream(
        self,
        user_message: str,
        config: GenerationConfig | None = None,
    ) -> Iterator[str]:
        prompt = self._build_prompt(user_message)
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
