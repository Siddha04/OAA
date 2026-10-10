from __future__ import annotations

from collections.abc import Mapping

from .documents import Document, DocumentChunk


def chunk_text(
    text: str,
    *,
    source: str = "<memory>",
    chunk_size: int = 1000,
    overlap: int = 150,
    metadata: Mapping[str, str] | None = None,
) -> list[DocumentChunk]:
    """Split text into overlapping, source-attributed character chunks.

    Boundaries prefer paragraph or line breaks and then sentence/word boundaries.
    Character offsets refer to the original text, not byte offsets.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not source:
        raise ValueError("source must be non-empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    if not text.strip():
        return []

    chunks: list[DocumentChunk] = []
    start = 0
    meta = dict(metadata or {})
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            floor = start + max(1, int(chunk_size * 0.65))
            window = text[start:end]
            candidates: list[int] = []
            for separator in ("\n\n", "\n", ". ", " "):
                index = window.rfind(separator)
                if index >= 0 and start + index + len(separator) >= floor:
                    candidates.append(start + index + len(separator))
            if candidates:
                end = max(candidates)

        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        content = raw.strip()
        if content:
            chunks.append(
                DocumentChunk(
                    source=source,
                    text=content,
                    chunk_index=len(chunks),
                    start_char=start + left_trim,
                    end_char=start + right_trim,
                    metadata=meta,
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)

    return chunks


def chunk_document(
    document: Document,
    *,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[DocumentChunk]:
    return chunk_text(
        document.text,
        source=document.source,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata=document.metadata,
    )
