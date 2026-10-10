from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .documents import Document

SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".py", ".pyi", ".c", ".cc", ".cpp",
    ".h", ".hh", ".hpp", ".java", ".js", ".jsx", ".ts", ".tsx", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv", ".log", ".sql",
    ".sh", ".ps1", ".cmake", ".pdf",
})
_IGNORED_DIRECTORY_NAMES = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "build", "dist", ".mypy_cache", ".pytest_cache",
})
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


def _read_pdf(path: Path) -> tuple[str, dict[str, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            'PDF ingestion requires the optional dependency; install with pip install "oaa[rag]".'
        ) from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(page for page in pages if page)
    return text, {"file_type": "pdf", "page_count": str(len(reader.pages))}


def load_document(
    path: str | Path,
    *,
    source: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Document:
    """Load one supported local text document (PDF support is optional)."""
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"document file does not exist: {file_path}")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    size = file_path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"document exceeds max_bytes ({size} > {max_bytes}): {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported document type: {suffix or '<no extension>'}")
    if suffix == ".pdf":
        text, metadata = _read_pdf(file_path)
    else:
        text = file_path.read_text(encoding="utf-8-sig")
        metadata = {"file_type": suffix.lstrip(".")}

    return Document(source=source or file_path.as_posix(), text=text, metadata=metadata)


def load_directory(
    directory: str | Path,
    *,
    extensions: Iterable[str] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Document]:
    """Load supported documents from a directory in deterministic path order."""
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"document directory does not exist: {root}")
    allowed = {
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (extensions if extensions is not None else SUPPORTED_EXTENSIONS)
    }
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(root)
        if any(part.startswith(".") or part in _IGNORED_DIRECTORY_NAMES for part in relative.parts[:-1]):
            continue
        paths.append((relative.as_posix(), path))

    documents: list[Document] = []
    for relative, path in sorted(paths, key=lambda item: item[0].casefold()):
        documents.append(load_document(path, source=relative, max_bytes=max_bytes))
    return documents
