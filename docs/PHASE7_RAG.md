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
- Search tie-breaking is deterministic and the default score threshold excludes zero-similarity results.

## Step 7C: RAG context integration

- RAGPipeline can ingest one file or a directory, chunk the extracted text, index the chunks, search, and produce a size-bounded context.
- PersonalAssistant accepts an optional RAGPipeline. When supplied, it retrieves relevant excerpts and injects them into both normal and streaming prompts.
- Source paths, chunk indices, and similarity values are retained.
- Retrieved document markup is escaped before prompt insertion. The prompt explicitly says retrieved content is untrusted reference data, not instructions.
- Chat behavior remains unchanged when no RAG pipeline is configured.

Example usage:

    from oaa import Engine, PersonalAssistant
    from oaa.rag import RAGPipeline

    rag = RAGPipeline()
    rag.ingest_directory("./knowledge")
    assistant = PersonalAssistant(Engine("tiny.manifest"), rag_pipeline=rag)
    answer = assistant.chat("What do my local notes say about model loading?")
    print(answer)

All processing is local. The source documents and in-memory index are not committed to the repository.

## Limitations

- HashEmbeddingModel is a lexical retrieval baseline, not a semantic pretrained embedding model. Paraphrases with little word overlap can be missed.
- The index is in memory and is lost when the process exits. Persistent indexing is a later step.
- PDF support extracts selectable text only; scanned-document OCR is not included.
