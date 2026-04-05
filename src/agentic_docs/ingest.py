"""End-to-end ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from agentic_docs.chunking import chunk_document
from agentic_docs.git_sync import current_commit_hash
from agentic_docs.parser import discover_markdown_files, parse_markdown_document
from agentic_docs.storage import SQLiteStore
from agentic_docs.tokenizers import get_tokenizer


def ingest_source(
    source: Path,
    db_path: Path,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
) -> dict[str, int | str]:
    """Ingest a markdown corpus into SQLite and return summary counts."""

    tokenizer = get_tokenizer(tokenizer_name)
    repo_commit_hash = current_commit_hash(source)
    files = discover_markdown_files(source)
    store = SQLiteStore(db_path)
    store.initialize()
    store.reindex()

    document_count = 0
    section_count = 0
    chunk_count = 0
    for path in files:
        document = parse_markdown_document(path=path, root=source, repo_commit_hash=repo_commit_hash)
        chunks = chunk_document(
            document=document,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        store.store_document(document, chunks)
        document_count += 1
        section_count += len(document.sections)
        chunk_count += len(chunks)

    return {
        "documents": document_count,
        "sections": section_count,
        "chunks": chunk_count,
        "tokenizer": tokenizer.name(),
        "repo_commit_hash": repo_commit_hash or "",
    }

