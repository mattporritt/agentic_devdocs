# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

"""SQLite persistence, inspection, and FTS5 retrieval support.

The storage layer is intentionally explicit: documents, sections, chunks, and the FTS
index all have first-class tables so artifacts stay easy to inspect and migrate.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentic_docs.models import ChunkModel, DocumentModel, QueryResult


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'repo_markdown',
    source_name TEXT,
    source_url TEXT,
    canonical_url TEXT,
    repo_commit_hash TEXT,
    last_modified_time TEXT,
    scrape_timestamp TEXT,
    file_hash TEXT NOT NULL,
    content_hash TEXT
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading_path TEXT NOT NULL,
    heading_level INTEGER NOT NULL,
    section_title TEXT,
    section_order INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    chunk_order INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    prev_chunk_id TEXT,
    next_chunk_id TEXT,
    metadata_json TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    source_path,
    document_title,
    heading_path
);
"""


class SQLiteStore:
    """Simple explicit SQLite store for documents, sections, and chunks."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row access by column name."""

        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._ensure_document_columns(connection)
            connection.commit()

    def _ensure_document_columns(self, connection: sqlite3.Connection) -> None:
        """Keep local SQLite stores forward-compatible as metadata evolves."""

        existing_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(documents)").fetchall()
        }
        required_columns = {
            "source_type": "TEXT NOT NULL DEFAULT 'repo_markdown'",
            "source_name": "TEXT",
            "source_url": "TEXT",
            "canonical_url": "TEXT",
            "scrape_timestamp": "TEXT",
            "content_hash": "TEXT",
        }
        for column, ddl in required_columns.items():
            if column in existing_columns:
                continue
            connection.execute(f"ALTER TABLE documents ADD COLUMN {column} {ddl}")

    def reindex(self) -> None:
        """Clear all persisted corpus state before rebuilding a database snapshot."""

        with self.connect() as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM sections")
            connection.execute("DELETE FROM documents")
            connection.commit()

    def store_document(self, document: DocumentModel, chunks: list[ChunkModel]) -> None:
        """Persist one canonical document and its chunks into the shared schema."""

        with self.connect() as connection:
            connection.execute("DELETE FROM chunks_fts WHERE source_path = ?", (document.metadata.source_path,))
            connection.execute("DELETE FROM documents WHERE id = ?", (document.id,))
            connection.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, source_path, title, source_type, source_name, source_url, canonical_url,
                    repo_commit_hash, last_modified_time, scrape_timestamp, file_hash, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.metadata.source_path,
                    document.title,
                    document.metadata.source_type,
                    document.metadata.source_name,
                    document.metadata.source_url,
                    document.metadata.canonical_url,
                    document.metadata.repo_commit_hash,
                    document.metadata.last_modified_time.isoformat() if document.metadata.last_modified_time else None,
                    document.metadata.scrape_timestamp.isoformat() if document.metadata.scrape_timestamp else None,
                    document.metadata.file_hash,
                    document.metadata.content_hash or document.metadata.file_hash,
                ),
            )
            for section in document.sections:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO sections (
                        id, document_id, heading_path, heading_level, section_title, section_order
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        section.id,
                        section.document_id,
                        " > ".join(section.heading_path),
                        section.heading_level,
                        section.section_title,
                        section.section_order,
                    ),
                )
            for chunk in chunks:
                metadata_json = chunk.metadata.model_dump_json()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        id, section_id, chunk_order, content, token_count, start_offset,
                        end_offset, prev_chunk_id, next_chunk_id, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.section_id,
                        chunk.chunk_order,
                        chunk.content,
                        chunk.token_count,
                        chunk.start_offset,
                        chunk.end_offset,
                        chunk.prev_chunk_id,
                        chunk.next_chunk_id,
                        metadata_json,
                    ),
                )
                metadata = chunk.metadata
                connection.execute(
                    """
                    INSERT INTO chunks_fts (
                        chunk_id, content, source_path, document_title, heading_path
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.content,
                        metadata.source_path,
                        metadata.document_title,
                        " > ".join(metadata.heading_path),
                    ),
                )
            connection.commit()

    def stats(self) -> dict[str, int]:
        """Return top-level corpus counts for documents, sections, and chunks."""

        with self.connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            sections = connection.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": documents, "sections": sections, "chunks": chunks}

    def inspect_chunk(self, chunk_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    c.id,
                    c.section_id,
                    c.chunk_order,
                    c.content,
                    c.token_count,
                    c.start_offset,
                    c.end_offset,
                    c.prev_chunk_id,
                    c.next_chunk_id,
                    c.metadata_json
                FROM chunks c
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["metadata_json"] = json.loads(data["metadata_json"]) if data["metadata_json"] else None
        return data

    def inspect_document(self, document_id: str) -> dict[str, object] | None:
        with self.connect() as connection:
            document = connection.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            ).fetchone()
            if document is None:
                return None
            sections = connection.execute(
                """
                SELECT id, heading_path, heading_level, section_title, section_order
                FROM sections
                WHERE document_id = ?
                ORDER BY section_order ASC
                """,
                (document_id,),
            ).fetchall()
            chunk_stats = connection.execute(
                """
                SELECT COUNT(*) AS chunk_count, COALESCE(SUM(token_count), 0) AS total_tokens
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                WHERE s.document_id = ?
                """,
                (document_id,),
            ).fetchone()
        return {
            "document": dict(document),
            "sections": [dict(row) for row in sections],
            "chunk_stats": dict(chunk_stats) if chunk_stats is not None else {},
        }

    def query(self, query_text: str, top_k: int) -> list[QueryResult]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    bm25(chunks_fts) AS score,
                    c.content,
                    d.source_path AS source_file_path,
                    d.id AS document_id,
                    d.title AS document_title,
                    s.id AS section_id,
                    s.section_title AS section_title,
                    s.heading_path AS heading_path,
                    c.token_count AS token_count,
                    d.repo_commit_hash AS repo_commit_hash,
                    c.chunk_order AS chunk_order,
                    snippet(chunks_fts, 1, '[', ']', '...', 20) AS snippet,
                    c.metadata_json AS metadata_json
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (query_text, top_k),
            ).fetchall()

        return [
            QueryResult(
                chunk_id=row["chunk_id"],
                score=row["score"],
                content=row["content"],
                source_file_path=row["source_file_path"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                section_id=row["section_id"],
                section_title=row["section_title"],
                heading_path=row["heading_path"].split(" > ") if row["heading_path"] else [],
                token_count=row["token_count"],
                repo_commit_hash=row["repo_commit_hash"],
                chunk_order=row["chunk_order"],
                snippet=row["snippet"],
                metadata_json=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
            )
            for row in rows
        ]

    def get_chunk_by_id(self, chunk_id: str) -> QueryResult | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    0.0 AS score,
                    c.content AS content,
                    d.source_path AS source_file_path,
                    d.id AS document_id,
                    d.title AS document_title,
                    s.id AS section_id,
                    s.section_title AS section_title,
                    s.heading_path AS heading_path,
                    c.token_count AS token_count,
                    d.repo_commit_hash AS repo_commit_hash,
                    c.chunk_order AS chunk_order,
                    NULL AS snippet,
                    c.metadata_json AS metadata_json
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return QueryResult(
            chunk_id=row["chunk_id"],
            score=row["score"],
            content=row["content"],
            source_file_path=row["source_file_path"],
            document_id=row["document_id"],
            document_title=row["document_title"],
            section_id=row["section_id"],
            section_title=row["section_title"],
            heading_path=row["heading_path"].split(" > ") if row["heading_path"] else [],
            token_count=row["token_count"],
            repo_commit_hash=row["repo_commit_hash"],
            chunk_order=row["chunk_order"],
            snippet=row["snippet"],
            metadata_json=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
        )

    def get_adjacent_chunks(self, chunk_id: str, include_previous: bool, include_next: bool) -> list[QueryResult]:
        with self.connect() as connection:
            pivot = connection.execute(
                "SELECT prev_chunk_id, next_chunk_id FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        if pivot is None:
            return []

        adjacent_ids: list[str] = []
        if include_previous and pivot["prev_chunk_id"]:
            adjacent_ids.append(pivot["prev_chunk_id"])
        if include_next and pivot["next_chunk_id"]:
            adjacent_ids.append(pivot["next_chunk_id"])

        results: list[QueryResult] = []
        for adjacent_id in adjacent_ids:
            chunk = self.get_chunk_by_id(adjacent_id)
            if chunk is not None:
                results.append(chunk)
        return results

    def get_section_chunks(self, section_id: str) -> list[QueryResult]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    0.0 AS score,
                    c.content,
                    d.source_path AS source_file_path,
                    d.id AS document_id,
                    d.title AS document_title,
                    s.id AS section_id,
                    s.section_title AS section_title,
                    s.heading_path AS heading_path,
                    c.token_count AS token_count,
                    d.repo_commit_hash AS repo_commit_hash,
                    c.chunk_order AS chunk_order,
                    NULL AS snippet,
                    c.metadata_json AS metadata_json
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                WHERE s.id = ?
                ORDER BY c.chunk_order ASC
                """,
                (section_id,),
            ).fetchall()
        return [
            QueryResult(
                chunk_id=row["chunk_id"],
                score=row["score"],
                content=row["content"],
                source_file_path=row["source_file_path"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                section_id=row["section_id"],
                section_title=row["section_title"],
                heading_path=row["heading_path"].split(" > ") if row["heading_path"] else [],
                token_count=row["token_count"],
                repo_commit_hash=row["repo_commit_hash"],
                chunk_order=row["chunk_order"],
                snippet=row["snippet"],
                metadata_json=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
            )
            for row in rows
        ]

    def get_document_chunks(self, document_id: str) -> list[QueryResult]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    c.id AS chunk_id,
                    0.0 AS score,
                    c.content,
                    d.source_path AS source_file_path,
                    d.id AS document_id,
                    d.title AS document_title,
                    s.id AS section_id,
                    s.section_title AS section_title,
                    s.heading_path AS heading_path,
                    c.token_count AS token_count,
                    d.repo_commit_hash AS repo_commit_hash,
                    c.chunk_order AS chunk_order,
                    NULL AS snippet,
                    c.metadata_json AS metadata_json
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                WHERE d.id = ?
                ORDER BY s.section_order ASC, c.chunk_order ASC
                """,
                (document_id,),
            ).fetchall()
        return [
            QueryResult(
                chunk_id=row["chunk_id"],
                score=row["score"],
                content=row["content"],
                source_file_path=row["source_file_path"],
                document_id=row["document_id"],
                document_title=row["document_title"],
                section_id=row["section_id"],
                section_title=row["section_title"],
                heading_path=row["heading_path"].split(" > ") if row["heading_path"] else [],
                token_count=row["token_count"],
                repo_commit_hash=row["repo_commit_hash"],
                chunk_order=row["chunk_order"],
                snippet=row["snippet"],
                metadata_json=json.loads(row["metadata_json"]) if row["metadata_json"] else None,
            )
            for row in rows
        ]

    def detailed_stats(self, limit: int = 10) -> dict[str, object]:
        with self.connect() as connection:
            overview = {
                "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "sections": connection.execute("SELECT COUNT(*) FROM sections").fetchone()[0],
                "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "avg_chunk_tokens": connection.execute(
                    "SELECT COALESCE(AVG(token_count), 0) FROM chunks"
                ).fetchone()[0],
            }
            chunks_per_document = connection.execute(
                """
                SELECT d.source_path, d.title, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN sections s ON s.document_id = d.id
                LEFT JOIN chunks c ON c.section_id = s.id
                GROUP BY d.id
                ORDER BY chunk_count DESC, d.source_path ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            largest_chunks = connection.execute(
                """
                SELECT c.id AS chunk_id, d.source_path, s.heading_path, c.token_count
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                ORDER BY c.token_count DESC, c.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            small_chunks = connection.execute(
                """
                SELECT c.id AS chunk_id, d.source_path, s.heading_path, c.token_count
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                WHERE c.token_count <= 30
                ORDER BY c.token_count ASC, c.id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            duplicate_candidates = connection.execute(
                """
                SELECT COUNT(*) AS duplicate_count, MIN(c.id) AS sample_chunk_id, MIN(d.source_path) AS sample_source_path
                FROM chunks c
                JOIN sections s ON s.id = c.section_id
                JOIN documents d ON d.id = s.document_id
                GROUP BY c.content
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC, sample_chunk_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            source_counts = connection.execute(
                """
                SELECT
                    d.source_type,
                    COALESCE(d.source_name, '') AS source_name,
                    COUNT(DISTINCT d.id) AS document_count,
                    COUNT(DISTINCT s.id) AS section_count,
                    COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN sections s ON s.document_id = d.id
                LEFT JOIN chunks c ON c.section_id = s.id
                GROUP BY d.source_type, COALESCE(d.source_name, '')
                ORDER BY chunk_count DESC, d.source_type ASC, source_name ASC
                """
            ).fetchall()
        return {
            "overview": overview,
            "sources": [dict(row) for row in source_counts],
            "chunks_per_document": [dict(row) for row in chunks_per_document],
            "largest_chunks": [dict(row) for row in largest_chunks],
            "small_chunks": [dict(row) for row in small_chunks],
            "duplicate_chunk_candidates": [dict(row) for row in duplicate_candidates],
        }
