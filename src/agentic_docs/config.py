"""Configuration models for ingestion and query."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    """Configuration for repository ingestion and chunking."""

    source: Path
    db_path: Path
    tokenizer: str = "openai"
    max_tokens: int = Field(default=400, ge=50)
    overlap_tokens: int = Field(default=60, ge=0)


class QueryConfig(BaseModel):
    """Configuration for lexical retrieval."""

    db_path: Path
    top_k: int = Field(default=5, ge=1, le=100)

