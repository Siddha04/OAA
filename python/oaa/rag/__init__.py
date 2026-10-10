"""Local-first retrieval-augmented generation primitives."""

from .chunking import chunk_document, chunk_text
from .documents import Document, DocumentChunk
from .embeddings import EmbeddingModel, HashEmbeddingModel
from .loaders import SUPPORTED_EXTENSIONS, load_directory, load_document
from .store import InMemoryVectorStore, SearchResult

__all__ = [
    "Document", "DocumentChunk", "EmbeddingModel", "HashEmbeddingModel",
    "InMemoryVectorStore", "SUPPORTED_EXTENSIONS", "SearchResult",
    "chunk_document", "chunk_text", "load_directory", "load_document",
]
