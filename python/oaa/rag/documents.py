from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class Document:
    """A source document and its small, serializable metadata."""

    source: str
    text: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("document source must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("document text must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("document metadata must be a mapping")


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A text slice with source and character-offset provenance."""

    source: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("chunk source must be non-empty")
        if not self.text.strip():
            raise ValueError("chunk text must be non-empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.start_char < 0 or self.end_char < self.start_char:
            raise ValueError("invalid chunk character offsets")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("chunk metadata must be a mapping")

    @property
    def chunk_id(self) -> str:
        return f"{self.source}::chunk-{self.chunk_index}"
