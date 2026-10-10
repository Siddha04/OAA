from __future__ import annotations

from pathlib import Path

import pytest

from oaa.rag import Document, chunk_document, chunk_text, load_directory, load_document


def test_load_document_preserves_text_and_type_metadata(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Local notes\nNo network required.", encoding="utf-8")
    document = load_document(path)
    assert document.source.endswith("notes.md")
    assert document.text == "# Local notes\nNo network required."
    assert document.metadata["file_type"] == "md"


def test_load_directory_is_sorted_and_skips_build_artifacts(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("zed", encoding="utf-8")
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "leak.txt").write_text("ignored", encoding="utf-8")
    build = tmp_path / "build"
    build.mkdir()
    (build / "generated.txt").write_text("ignored", encoding="utf-8")
    docs = load_directory(tmp_path)
    assert [doc.source for doc in docs] == ["a.md", "z.txt"]


def test_load_document_rejects_unsupported_and_oversized_files(tmp_path: Path) -> None:
    unsupported = tmp_path / "data.bin"
    unsupported.write_bytes(b"abc")
    with pytest.raises(ValueError, match="unsupported"):
        load_document(unsupported)
    text = tmp_path / "large.txt"
    text.write_text("123456", encoding="utf-8")
    with pytest.raises(ValueError, match="max_bytes"):
        load_document(text, max_bytes=3)


def test_chunking_keeps_offsets_overlap_and_source_metadata() -> None:
    text = "Alpha section. " + ("beta gamma delta " * 9) + "\nFinal section."
    document = Document("notes.md", text, {"file_type": "md"})
    chunks = chunk_document(document, chunk_size=45, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.source == "notes.md" for chunk in chunks)
    assert all(chunk.metadata["file_type"] == "md" for chunk in chunks)
    assert all(text[chunk.start_char:chunk.end_char].strip() == chunk.text for chunk in chunks)
    assert any(a.end_char > b.start_char for a, b in zip(chunks, chunks[1:]))


def test_chunking_empty_and_validation() -> None:
    assert chunk_text("   ") == []
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=5, overlap=5)
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("text", chunk_size=0)


def test_chunking_handles_one_short_document() -> None:
    chunks = chunk_text("hello world", source="x.txt", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 11
