"""Stable runtime-facing contract builders for `query --json-contract`."""

from __future__ import annotations

import re
from hashlib import sha1

from agentic_docs.models import (
    RuntimeContractContent,
    RuntimeContractDiagnostics,
    RuntimeContractEnvelope,
    RuntimeContractIntent,
    RuntimeContractResult,
    RuntimeContractSection,
    RuntimeContractSource,
)
from agentic_docs.provenance import infer_source_name
from agentic_docs.query_service import build_query_profile


FILE_ANCHOR_PATTERN = re.compile(r"`?([A-Za-z0-9_./-]+\.(?:php|mdx|md|js|mustache|yml|css|scss))`?")


def _extract_contract_file_anchors(texts: list[str]) -> list[str]:
    anchors: list[str] = []
    for text in texts:
        for match in FILE_ANCHOR_PATTERN.findall(text):
            if match not in anchors:
                anchors.append(match)
    return anchors


def _content_without_heading_prefix(content: str) -> str:
    lines = [line.strip() for line in content.splitlines()]
    filtered = [line for line in lines if line and not line.startswith("Heading: ")]
    return " ".join(filtered).strip()


def _sentence_like_points(text: str) -> list[str]:
    points: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        cleaned = part.strip().strip("- ").strip()
        if len(cleaned) < 20:
            continue
        if cleaned not in points:
            points.append(cleaned)
        if len(points) >= 4:
            break
    return points


def _bundle_summary(bundle) -> str:
    heading = " > ".join(bundle.heading_path) if bundle.heading_path else bundle.document_title
    lead = _content_without_heading_prefix(bundle.chunks[0].content) if bundle.chunks else ""
    if lead:
        sentence = _sentence_like_points(lead)
        if sentence:
            return f"{bundle.document_title} — {heading}: {sentence[0]}"
    return f"{bundle.document_title} — {heading}"


def _bundle_key_points(bundle) -> list[str]:
    points: list[str] = []
    for chunk in bundle.chunks:
        text = _content_without_heading_prefix(chunk.content)
        for point in _sentence_like_points(text):
            if point not in points:
                points.append(point)
            if len(points) >= 4:
                return points
    return points


def _runtime_confidence(bundle) -> str:
    if bundle.rank == 1:
        return "high"
    if bundle.rank <= 3:
        return "medium"
    return "low"


def _stable_contract_id(*parts: str) -> str:
    normalized = "||".join(part.strip() for part in parts)
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _runtime_ranking_explanation(bundle) -> str:
    source_name = infer_source_name(bundle.source_file_path, bundle.source_name, bundle.source_type)
    heading = " > ".join(bundle.heading_path) if bundle.heading_path else (bundle.section_title or bundle.document_title)
    explanation = f"Rank {bundle.rank} {source_name or 'unknown-source'} bundle from {heading}."
    support_reason = bundle.diagnostics.get("support_reason") if bundle.diagnostics else None
    if support_reason:
        explanation += f" Added support for {support_reason}."
    return explanation


def build_runtime_contract(query_text: str, bundles: list, top_k: int) -> RuntimeContractEnvelope:
    """Map context bundles into the stable runtime-facing JSON contract."""

    profile = build_query_profile(query_text)
    results: list[RuntimeContractResult] = []
    for bundle in bundles[:top_k]:
        sections = [
            RuntimeContractSection(
                id=chunk.chunk_id,
                role=chunk.role,
                document_title=bundle.document_title,
                source_path=chunk.source_file_path,
                source_url=chunk.source_url,
                canonical_url=chunk.canonical_url,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
                token_count=chunk.token_count,
                content=chunk.content,
            )
            for chunk in bundle.chunks
        ]
        file_anchors = _extract_contract_file_anchors([chunk.content for chunk in bundle.chunks])
        source_name = infer_source_name(bundle.source_file_path, bundle.source_name, bundle.source_type)
        results.append(
            RuntimeContractResult(
                id=_stable_contract_id(
                    bundle.source_file_path,
                    bundle.section_title or "",
                    " > ".join(bundle.heading_path),
                ),
                rank=bundle.rank,
                confidence=_runtime_confidence(bundle),
                source=RuntimeContractSource(
                    name=source_name,
                    type=bundle.source_type,
                    url=bundle.source_url,
                    canonical_url=bundle.canonical_url,
                    path=bundle.source_file_path,
                    document_title=bundle.document_title,
                    section_title=bundle.section_title,
                    heading_path=bundle.heading_path,
                ),
                content=RuntimeContractContent(
                    summary=_bundle_summary(bundle),
                    sections=sections,
                    file_anchors=file_anchors,
                    key_points=_bundle_key_points(bundle),
                ),
                diagnostics=RuntimeContractDiagnostics(
                    ranking_explanation=_runtime_ranking_explanation(bundle),
                    support_reason=(
                        str(bundle.diagnostics.get("support_reason"))
                        if bundle.diagnostics and bundle.diagnostics.get("support_reason") is not None
                        else None
                    ),
                    token_count=bundle.bundle_token_count,
                    selection_strategy=bundle.selection_strategy,
                ),
            )
        )
    return RuntimeContractEnvelope(
        query=query_text,
        normalized_query=profile.normalized_query,
        intent=RuntimeContractIntent(
            query_intent=profile.intent,
            task_intent=profile.task_intent,
            concept_families=profile.concept_families,
        ),
        results=results,
    )
