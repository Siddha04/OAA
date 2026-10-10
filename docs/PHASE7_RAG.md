# OAA Phase 7 - Personal RAG

Phase 7 starts with local document ingestion and deterministic chunking.

## Step 7A: ingestion and chunking

- Reads supported local text and source files as UTF-8, including Markdown, Python/C++ source, JSON, YAML, TOML, CSV, logs, and SQL.
- Recursively scans directories in deterministic path order.
- Skips hidden folders and common generated directories such as .git, .venv, build, dist, and node_modules.
- Enforces a per-file size limit (5 MiB by default).
- Splits source text into overlapping chunks with source names and original character offsets.
- PDF extraction is optional and requires pip install "oaa[rag]"; extraction quality depends on the PDF containing selectable text. Scanned PDFs need OCR, which is not included in this step.

Example usage:

    from oaa.rag import chunk_document, load_directory

    documents = load_directory("./knowledge")
    chunks = [chunk for doc in documents for chunk in chunk_document(doc)]
    print(f"Loaded {len(documents)} documents and {len(chunks)} chunks")

All document processing is local; source documents are not copied into the OAA repository. This step does not yet generate embeddings or answer questions. Those are separate verification gates in Phase 7B and 7C.

## Limits

- No OCR for scanned PDFs.
- No semantic pretrained embedding model yet.
- Chunk offsets are character offsets into decoded text, not byte offsets.
