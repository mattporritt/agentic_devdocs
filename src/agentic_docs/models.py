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
    normalized_query: str | None = None
    snippet: str | None = None
    metadata_json: dict[str, Any] | None = None
    rerank_score: float | None = None
    rerank_breakdown: dict[str, float | int | str | list[str]] | None = None


class ContextBundleChunk(BaseModel):
    """A single chunk included in an agent-facing context bundle."""

    chunk_id: str
    role: str
    content: str
    token_count: int


class ContextBundle(BaseModel):
    """A compact retrieval bundle intended for agent prompt construction."""

    rank: int
    score: float
    bundle_token_count: int
    source_file_path: str
    document_title: str
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    repo_commit_hash: str | None = None
    snippet: str | None = None
    chunks: list[ContextBundleChunk]
    selection_strategy: str = "match_only"


class EvalCase(BaseModel):
    """A single retrieval evaluation case."""

    id: str
    query: str
    description: str | None = None
    preferred_document_paths: list[str] = Field(default_factory=list)
    acceptable_document_paths: list[str] = Field(default_factory=list)
    preferred_heading_substrings: list[str] = Field(default_factory=list)
    acceptable_heading_substrings: list[str] = Field(default_factory=list)
    disallowed_document_paths: list[str] = Field(default_factory=list)
    top_k: int = 5
    notes: str | None = None


class EvalMatch(BaseModel):
    """The highest-ranked matching retrieval result for an eval case."""

    rank: int
    chunk_id: str
    source_file_path: str
    document_title: str
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    score: float
    snippet: str | None = None
    matched_on: list[str] = Field(default_factory=list)
    grade: str
    matched_rule_type: str


class EvalWindowStats(BaseModel):
    """Aggregate counts for a specific top-k cutoff."""

    strong_passes: int
    weak_passes: int
    misses: int
    strong_pass_rate: float
    weak_pass_rate: float


class EvalOutcome(BaseModel):
    """Evaluation outcome for a single query."""

    case_id: str
    query: str
    top_k: int
    grade: str
    strong_pass_top_1: bool
    strong_pass_top_3: bool
    strong_pass_top_5: bool
    weak_pass_top_1: bool
    weak_pass_top_3: bool
    weak_pass_top_5: bool
    matched_result_rank: int | None = None
    matched_result_path: str | None = None
    matched_result_heading: str | None = None
    matched_rule_type: str | None = None
    matched_result: EvalMatch | None = None
    failure_summary: str | None = None
    preferred_result_rank: int | None = None
    preferred_result_path: str | None = None
    preferred_result_heading: str | None = None
    ranking_diagnostic: str | None = None


class EvalReport(BaseModel):
    """Aggregate retrieval evaluation report."""

    total_queries: int
    strong_passes: int
    weak_passes: int
    misses: int
    top_1: EvalWindowStats
    top_3: EvalWindowStats
    top_5: EvalWindowStats
    outcomes: list[EvalOutcome]
