from __future__ import annotations

import math

import pytest

from oaa.rag import Document, chunk_document
from oaa.rag.embeddings import HashEmbeddingModel
from oaa.rag.store import InMemoryVectorStore


def _chunks(source: str, text: str):
    return chunk_document(Document(source, text), chunk_size=500, overlap=20)


def test_hash_embeddings_are_deterministic_and_normalized() -> None:
    model_a = HashEmbeddingModel(dimensions=256)
    model_b = HashEmbeddingModel(dimensions=256)
    vector_a = model_a.embed("Local retrieval uses document chunks.")
    vector_b = model_b.embed("Local retrieval uses document chunks.")
    assert vector_a == vector_b
    assert len(vector_a) == 256
    assert math.isclose(math.sqrt(sum(value * value for value in vector_a)), 1.0)
    assert model_a.embed("!!!") == (0.0,) * 256


def test_vector_store_ranks_matching_lexical_content_and_keeps_provenance() -> None:
    store = InMemoryVectorStore(HashEmbeddingModel(dimensions=512))
    store.add(_chunks("training.md", "Model training uses GPUs and tensor batches."))
    store.add(_chunks("database.md", "SQL database query planning relies on indexes."))
    store.add(_chunks("support.md", "Customer support tickets track product feedback."))

    results = store.search("SQL database indexes", top_k=3)
    assert len(results) == 1
    assert results[0].source == "database.md"
    assert results[0].score > 0.0
    assert results[0].chunk.text.startswith("SQL database")
    assert 0.0 <= results[0].score <= 1.0


def test_vector_store_top_k_and_empty_index() -> None:
    store = InMemoryVectorStore()
    assert store.search("anything") == []
    store.add(_chunks("a.txt", "alpha beta gamma"))
    assert len(store.search("alpha", top_k=1)) == 1
    with pytest.raises(ValueError, match="top_k"):
        store.search("alpha", top_k=0)
    with pytest.raises(ValueError, match="query"):
        store.search("   ")


def test_vector_store_replaces_same_chunk_id_and_clear_resets() -> None:
    store = InMemoryVectorStore()
    store.add(_chunks("same.md", "first version"))
    store.add(_chunks("same.md", "replacement version"))
    assert store.count == 1
    assert store.search("replacement")[0].chunk.text == "replacement version"
    store.clear()
    assert store.count == 0
    assert store.search("replacement") == []


def test_embedding_input_validation() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        HashEmbeddingModel(dimensions=0)
    with pytest.raises(TypeError, match="text"):
        HashEmbeddingModel().embed(42)  # type: ignore[arg-type]
