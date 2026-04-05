"""Retrieval-facing query helpers."""

from __future__ import annotations

import re
from pathlib import Path

from agentic_docs.models import ContextBundle, ContextBundleChunk, QueryResult
from agentic_docs.storage import SQLiteStore


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "work",
    "works",
    "write",
    "add",
    "define",
    "create",
}
TOKEN_ALIASES = {
    "language": ["lang"],
    "strings": ["string"],
    "services": ["service", "external"],
    "tasks": ["task"],
    "tests": ["test"],
    "forms": ["form"],
    "plugins": ["plugin"],
}


def canonical_path_key(path: str) -> str:
    """Return a canonicalized path key with versioned docs folded into their base path."""

    normalized = path.strip().lstrip("./")
    match = re.match(r"versioned_docs/version-[^/]+/(.+)", normalized)
    if match:
        return match.group(1)
    return normalized


def is_canonical_path(path: str) -> bool:
    """Return whether a path is from the non-versioned canonical corpus."""

    normalized = path.strip().lstrip("./")
    return not normalized.startswith("versioned_docs/")


def normalize_query_text(query_text: str) -> tuple[str, list[str]]:
    """Normalize a developer query into FTS-friendly tokens."""

    raw_tokens = [token.lower() for token in TOKEN_PATTERN.findall(query_text)]
    filtered_tokens = [token for token in raw_tokens if token not in STOPWORDS]
    source_tokens = filtered_tokens or raw_tokens
    tokens: list[str] = []
    for token in source_tokens:
        if token not in tokens:
            tokens.append(token)
    normalized = " ".join(tokens) if tokens else query_text.strip()
    return normalized, tokens


def _expanded_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        if token not in expanded:
            expanded.append(token)
        if token.endswith("s") and len(token) > 3:
            singular = token[:-1]
            if singular not in expanded:
                expanded.append(singular)
        for alias in TOKEN_ALIASES.get(token, []):
            if alias not in expanded:
                expanded.append(alias)
    return expanded


def _build_fts_queries(query_text: str) -> list[str]:
    normalized, tokens = normalize_query_text(query_text)
    if not tokens:
        return [f'"{query_text.strip()}"'] if query_text.strip() else []

    and_query = " ".join(f'"{token}"' for token in tokens)
    if len(tokens) == 1:
        return [and_query]
    expanded = _expanded_tokens(tokens)
    or_query = " OR ".join(f'"{token}"' for token in expanded)
    phrase_query = f"\"{normalized}\""
    return [and_query, phrase_query, or_query]


def _context_hits(result: QueryResult, tokens: list[str]) -> int:
    fields = [
        result.source_file_path.lower(),
        result.document_title.lower(),
        (result.section_title or "").lower(),
        " > ".join(result.heading_path).lower(),
    ]
    return sum(1 for token in tokens if any(token in field for field in fields))


def _rerank_results(results: list[QueryResult], tokens: list[str], top_k: int, normalized_query: str) -> list[QueryResult]:
    expanded_tokens = _expanded_tokens(tokens)
    ranked = sorted(
        results,
        key=lambda result: (
            not is_canonical_path(result.source_file_path),
            -_context_hits(result, expanded_tokens),
            result.score,
            result.token_count,
            result.chunk_id,
        ),
    )
    deduped: list[QueryResult] = []
    seen_signatures: set[tuple[str, tuple[str, ...], str]] = set()
    for result in ranked:
        signature = (
            canonical_path_key(result.source_file_path),
            tuple(result.heading_path),
            (result.section_title or "").lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(result)
    for result in ranked:
        result.normalized_query = normalized_query
    for result in deduped:
        result.normalized_query = normalized_query
    return deduped[:top_k]


def query_chunks(db_path: Path, query_text: str, top_k: int) -> list[QueryResult]:
    """Query indexed chunks from the SQLite store with light query normalization."""

    store = SQLiteStore(db_path)
    store.initialize()
    normalized_query, tokens = normalize_query_text(query_text)
    candidates: list[QueryResult] = []
    seen_chunk_ids: set[str] = set()
    for fts_query in _build_fts_queries(query_text):
        if not fts_query:
            continue
        try:
            current_results = store.query(query_text=fts_query, top_k=max(top_k * 5, top_k))
        except Exception:
            current_results = []
        for result in current_results:
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)
            candidates.append(result)
    return _rerank_results(candidates, tokens, top_k, normalized_query)


def build_context_bundles(
    db_path: Path,
    results: list[QueryResult],
    include_previous: bool = False,
    include_next: bool = False,
) -> list[ContextBundle]:
    """Build compact, traceable context bundles from query results."""

    store = SQLiteStore(db_path)
    store.initialize()

    bundles: list[ContextBundle] = []
    for rank, result in enumerate(results, start=1):
        adjacent = store.get_adjacent_chunks(
            chunk_id=result.chunk_id,
            include_previous=include_previous,
            include_next=include_next,
        )
        ordered_chunks: list[ContextBundleChunk] = []
        if include_previous:
            for chunk in adjacent:
                if chunk.chunk_order < result.chunk_order:
                    ordered_chunks.append(
                        ContextBundleChunk(
                            chunk_id=chunk.chunk_id,
                            role="previous",
                            content=chunk.content,
                            token_count=chunk.token_count,
                        )
                    )
        ordered_chunks.append(
            ContextBundleChunk(
                chunk_id=result.chunk_id,
                role="match",
                content=result.content,
                token_count=result.token_count,
            )
        )
        if include_next:
            for chunk in adjacent:
                if chunk.chunk_order > result.chunk_order:
                    ordered_chunks.append(
                        ContextBundleChunk(
                            chunk_id=chunk.chunk_id,
                            role="next",
                            content=chunk.content,
                            token_count=chunk.token_count,
                        )
                    )
        bundles.append(
            ContextBundle(
                rank=rank,
                score=result.score,
                bundle_token_count=sum(chunk.token_count for chunk in ordered_chunks),
                source_file_path=result.source_file_path,
                document_title=result.document_title,
                section_title=result.section_title,
                heading_path=result.heading_path,
                repo_commit_hash=result.repo_commit_hash,
                snippet=result.snippet,
                chunks=ordered_chunks,
            )
        )
    return bundles
