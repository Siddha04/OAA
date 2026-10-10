from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .documents import DocumentChunk
from .embeddings import EmbeddingModel, HashEmbeddingModel


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: DocumentChunk
    score: float

    @property
    def source(self) -> str:
        return self.chunk.source


class InMemoryVectorStore:
    """Small local vector index with deterministic cosine-similarity search."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or HashEmbeddingModel()
        self._entries: dict[str, tuple[DocumentChunk, tuple[float, ...]]] = {}
        self._dimensions: int | None = None

    @property
    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._dimensions = None

    @staticmethod
    def _normalize(vector: Iterable[float]) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if not values:
            raise ValueError("embedding vectors must not be empty")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("embedding vectors must contain only finite values")
        norm = math.sqrt(sum(value * value for value in values))
        return tuple(value / norm for value in values) if norm else values

    def add(self, chunks: Iterable[DocumentChunk]) -> int:
        prepared = list(chunks)
        encoded: list[tuple[DocumentChunk, tuple[float, ...]]] = []
        expected_dimensions = self._dimensions
        for chunk in prepared:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError("add() expects DocumentChunk items")
            vector = self._normalize(self.embedding_model.embed(chunk.text))
            if expected_dimensions is None:
                expected_dimensions = len(vector)
            if len(vector) != expected_dimensions:
                raise ValueError("embedding dimension mismatch")
            encoded.append((chunk, vector))

        for chunk, vector in encoded:
            self._entries[chunk.chunk_id] = (chunk, vector)
        self._dimensions = expected_dimensions
        return len(encoded)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if not math.isfinite(min_score) or min_score < -1.0 or min_score > 1.0:
            raise ValueError("min_score must be finite and between -1 and 1")
        if not self._entries:
            return []

        query_vector = self._normalize(self.embedding_model.embed(query))
        if self._dimensions is not None and len(query_vector) != self._dimensions:
            raise ValueError("query embedding dimension mismatch")
        if not any(query_vector):
            return []

        results: list[SearchResult] = []
        for chunk, vector in self._entries.values():
            score = sum(left * right for left, right in zip(query_vector, vector))
            if score >= min_score:
                results.append(SearchResult(chunk=chunk, score=max(-1.0, min(1.0, score))))
        results.sort(key=lambda item: (
            -item.score, item.chunk.source.casefold(), item.chunk.chunk_index, item.chunk.chunk_id
        ))
        return results[:top_k]
