# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

"""Helpers for keeping source provenance consistent across ingestion and retrieval."""

from __future__ import annotations

from agentic_docs.models import QueryResult


def infer_source_name(source_file_path: str, source_name: str | None, source_type: str | None) -> str | None:
    """Return a stable source name across devdocs, design-system, user-docs, and legacy rows."""

    normalized_path = source_file_path.replace("\\", "/").lower()
    if normalized_path.startswith("design_system/") or "/design_system/" in normalized_path:
        return "design_system"
    if normalized_path.startswith("user_docs/") or "/user_docs/" in normalized_path:
        return "user_docs"
    if source_name:
        return source_name
    if source_type == "repo_markdown" or source_type is None:
        return "devdocs_repo"
    return None


def infer_source_type(source_file_path: str, source_name: str | None, source_type: str | None) -> str | None:
    """Return a stable source type for both explicit and legacy source rows."""

    normalized_path = source_file_path.replace("\\", "/").lower()
    if normalized_path.startswith("design_system/") or "/design_system/" in normalized_path:
        return source_type or "scraped_web"
    if normalized_path.startswith("user_docs/") or "/user_docs/" in normalized_path:
        return source_type or "scraped_web"
    normalized_name = infer_source_name(source_file_path, source_name, source_type)
    if normalized_name == "devdocs_repo":
        return source_type or "repo_markdown"
    return source_type


def source_fields_from_metadata(
    metadata_json: dict[str, object] | None,
    *,
    source_file_path: str = "",
) -> tuple[str | None, str | None, str | None, str | None]:
    """Extract normalized source fields from stored chunk metadata."""

    if not metadata_json:
        return infer_source_name(source_file_path, None, None), None, None, None
    source_name = metadata_json.get("source_name")
    source_type = metadata_json.get("source_type")
    source_url = metadata_json.get("source_url")
    canonical_url = metadata_json.get("canonical_url")
    normalized_name = infer_source_name(
        source_file_path,
        str(source_name) if source_name is not None else None,
        str(source_type) if source_type is not None else None,
    )
    normalized_type = infer_source_type(
        source_file_path,
        str(source_name) if source_name is not None else None,
        str(source_type) if source_type is not None else None,
    )
    return (
        normalized_name,
        normalized_type,
        str(source_url) if source_url is not None else None,
        str(canonical_url) if canonical_url is not None else None,
    )


def result_source_metadata(result: QueryResult) -> dict[str, str | None]:
    """Normalize source provenance for a query result returned from storage."""

    source_name, source_type, source_url, canonical_url = source_fields_from_metadata(
        result.metadata_json,
        source_file_path=result.source_file_path,
    )
    return {
        "source_name": source_name,
        "source_type": source_type,
        "source_url": source_url,
        "canonical_url": canonical_url,
    }
