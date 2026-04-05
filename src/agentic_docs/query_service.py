"""Retrieval-facing query helpers."""

from __future__ import annotations

from pathlib import Path

from agentic_docs.models import QueryResult
from agentic_docs.storage import SQLiteStore


def query_chunks(db_path: Path, query_text: str, top_k: int) -> list[QueryResult]:
    """Query indexed chunks from the SQLite store."""

    store = SQLiteStore(db_path)
    store.initialize()
    return store.query(query_text=query_text, top_k=top_k)

