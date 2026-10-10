# OAA Phase 7 - Personal RAG

## Step 7A: ingestion and chunking

- Reads supported local text and source files, including Markdown, Python/C++ source, JSON, YAML, TOML, CSV, logs, and SQL.
- Recursively scans directories in deterministic path order and skips hidden/generated directories.
- Enforces a per-file size limit (5 MiB by default).
- Splits text into overlapping chunks with source names and original character offsets.
- PDF extraction is optional: install with pip install "oaa[rag]". Scanned PDFs require OCR, which is not included here.

## Step 7B: embedding interface and retrieval index

- EmbeddingModel is a replaceable protocol for local embedding implementations.
- HashEmbeddingModel creates deterministic normalized lexical feature vectors from unigrams and adjacent-token bigrams.
- InMemoryVectorStore stores vectors in process memory and ranks by cosine similarity.
- Results preserve document source, metadata, chunk index, and a bounded similarity score.
- Search tie-breaking is deterministic. Duplicate source/chunk IDs replace earlier entries.

Example usage:

    from oaa.rag import HashEmbeddingModel, InMemoryVectorStore, chunk_document, load_directory

    index = InMemoryVectorStore(HashEmbeddingModel())
    for document in load_directory("./knowledge"):
        index.add(chunk_document(document))
    for result in index.search("how does model loading work", top_k=4):
        print(result.source, result.score, result.chunk.text)

All processing is local. Source documents and the in-memory index are not committed to the repository.

## Limitations

- HashEmbeddingModel is a lexical retrieval baseline, not a semantic pretrained embedding model. Paraphrases with little word overlap can be missed.
- The index is in memory and is lost when the process exits. Persistent indexing is a later step.
- PDF support extracts selectable text only; scanned-document OCR is not included.
