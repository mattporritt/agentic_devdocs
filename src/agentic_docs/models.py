"""Canonical data models used across ingestion, chunking, storage, and query."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata captured for an ingested source document."""

    source_path: str
    repo_commit_hash: str | None = None
    last_modified_time: datetime | None = None
    file_hash: str


class SectionModel(BaseModel):
    """Canonical representation of a logical markdown section."""

    id: str
    document_id: str
    section_order: int
    section_title: str | None = None
    heading_level: int = 0
    heading_path: list[str] = Field(default_factory=list)
    content: str


class DocumentModel(BaseModel):
    """Canonical representation of a parsed source document."""

    id: str
    title: str
    metadata: DocumentMetadata
    sections: list[SectionModel]


class ChunkMetadata(BaseModel):
    """Trace metadata for a chunk."""

    document_id: str
    document_title: str
    source_path: str
    repo_commit_hash: str | None = None
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)


class ChunkModel(BaseModel):
    """A token-aware retrieval chunk."""

    id: str
    section_id: str
    chunk_order: int
    content: str
    token_count: int
    start_offset: int | None = None
    end_offset: int | None = None
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: ChunkMetadata


class QueryResult(BaseModel):
    """A retrieval result returned by the query service and CLI."""

    chunk_id: str
    score: float
    content: str
    source_file_path: str
    document_id: str
    document_title: str
    section_id: str
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    repo_commit_hash: str | None = None
    chunk_order: int
    metadata_json: dict[str, Any] | None = None

