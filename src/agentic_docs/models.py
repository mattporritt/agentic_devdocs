"""Canonical data models used across ingestion, chunking, storage, and query."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata captured for an ingested source document."""

    source_path: str
    source_type: str = "repo_markdown"
    source_name: str | None = None
    source_url: str | None = None
    canonical_url: str | None = None
    repo_commit_hash: str | None = None
    last_modified_time: datetime | None = None
    scrape_timestamp: datetime | None = None
    file_hash: str
    content_hash: str | None = None


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
    source_type: str = "repo_markdown"
    source_name: str | None = None
    source_url: str | None = None
    canonical_url: str | None = None
    repo_commit_hash: str | None = None
    scrape_timestamp: datetime | None = None
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
    source_file_path: str
    source_name: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    canonical_url: str | None = None
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)


class ContextBundle(BaseModel):
    """A compact retrieval bundle intended for agent prompt construction."""

    rank: int
    score: float
    bundle_token_count: int
    source_file_path: str
    source_name: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    canonical_url: str | None = None
    document_title: str
    section_title: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    repo_commit_hash: str | None = None
    snippet: str | None = None
    chunks: list[ContextBundleChunk]
    selection_strategy: str = "match_only"
    diagnostics: dict[str, Any] | None = None


class RuntimeContractIntent(BaseModel):
    """Stable runtime-facing query intent summary."""

    model_config = ConfigDict(extra="forbid")

    query_intent: str
    task_intent: str
    concept_families: list[str]


class RuntimeContractSource(BaseModel):
    """Runtime-facing provenance for a returned knowledge bundle."""

    model_config = ConfigDict(extra="forbid")

    name: str | None
    type: str | None
    url: str | None
    canonical_url: str | None
    path: str
    document_title: str
    section_title: str | None
    heading_path: list[str]


class RuntimeContractSection(BaseModel):
    """A compact structured section included in a runtime knowledge bundle."""

    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    document_title: str
    source_path: str
    source_url: str | None
    canonical_url: str | None
    section_title: str | None
    heading_path: list[str]
    token_count: int
    content: str


class RuntimeContractContent(BaseModel):
    """Normalized content block for the stable runtime contract."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    sections: list[RuntimeContractSection]
    file_anchors: list[str]
    key_points: list[str]


class RuntimeContractDiagnostics(BaseModel):
    """Minimal runtime diagnostics for debugging orchestration issues."""

    model_config = ConfigDict(extra="forbid")

    ranking_explanation: str | None
    support_reason: str | None
    token_count: int
    selection_strategy: str


class RuntimeContractResult(BaseModel):
    """A single runtime-facing knowledge bundle result."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["knowledge_bundle"]
    rank: int
    confidence: Literal["high", "medium", "low"]
    source: RuntimeContractSource
    content: RuntimeContractContent
    diagnostics: RuntimeContractDiagnostics


class RuntimeContractEnvelope(BaseModel):
    """Stable JSON contract for external runtime consumers."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["agentic_docs"]
    version: Literal["v1"]
    query: str
    normalized_query: str
    intent: RuntimeContractIntent
    results: list[RuntimeContractResult]


class SharedRuntimeContractContent(BaseModel):
    """Tool-specific runtime payload carried inside the shared outer envelope.

    The shared cross-tool contract intentionally does not constrain the inner
    shape beyond "must be an object". `agentic_docs` layers its richer content
    model on top of this shared shell.
    """

    model_config = ConfigDict(extra="allow")


class SharedRuntimeContractResult(BaseModel):
    """Shared cross-tool runtime result wrapper.

    This preserves the common provenance and diagnostics shape while allowing
    each tool to define its own `content` payload.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str
    rank: int
    confidence: Literal["high", "medium", "low"]
    source: RuntimeContractSource
    content: SharedRuntimeContractContent
    diagnostics: RuntimeContractDiagnostics


class SharedRuntimeContractEnvelope(BaseModel):
    """Canonical shared outer envelope for the Moodle agentic tool family."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    version: str
    query: str
    normalized_query: str
    intent: RuntimeContractIntent
    results: list[SharedRuntimeContractResult]


class EvalCase(BaseModel):
    """A single retrieval evaluation case."""

    id: str
    query: str
    bucket: str = "uncategorized"
    query_style: str | None = None
    concept_id: str | None = None
    description: str | None = None
    preferred_source_names: list[str] = Field(default_factory=list)
    acceptable_source_names: list[str] = Field(default_factory=list)
    preferred_document_paths: list[str] = Field(default_factory=list)
    acceptable_document_paths: list[str] = Field(default_factory=list)
    preferred_heading_substrings: list[str] = Field(default_factory=list)
    acceptable_heading_substrings: list[str] = Field(default_factory=list)
    disallowed_document_paths: list[str] = Field(default_factory=list)
    preferred_bundle_source_names: list[str] = Field(default_factory=list)
    acceptable_bundle_source_names: list[str] = Field(default_factory=list)
    allow_mixed_bundle_sources: bool = False
    preferred_bundle_paths: list[str] = Field(default_factory=list)
    preferred_heading_substrings_for_bundle: list[str] = Field(default_factory=list)
    required_heading_substrings_for_bundle: list[str] = Field(default_factory=list)
    max_reasonable_bundle_tokens: int | None = None
    top_k: int = 5
    notes: str | None = None


class EvalMatch(BaseModel):
    """The highest-ranked matching retrieval result for an eval case."""

    rank: int
    chunk_id: str
    source_file_path: str
    source_name: str | None = None
    source_type: str | None = None
    source_url: str | None = None
    canonical_url: str | None = None
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
    bucket: str
    query_style: str | None = None
    concept_id: str | None = None
    expected_source_name: str | None = None
    acceptable_source_names: list[str] = Field(default_factory=list)
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
    matched_result_source_name: str | None = None
    matched_result_source_type: str | None = None
    matched_rule_type: str | None = None
    matched_result: EvalMatch | None = None
    failure_summary: str | None = None
    preferred_result_rank: int | None = None
    preferred_result_path: str | None = None
    preferred_result_heading: str | None = None
    preferred_source_rank: int | None = None
    ranking_diagnostic: str | None = None
    bundle_grade: str | None = None
    bundle_path: str | None = None
    bundle_source_name: str | None = None
    bundle_source_type: str | None = None
    bundle_source_names: list[str] = Field(default_factory=list)
    bundle_source_coherent: bool | None = None
    bundle_token_count: int | None = None
    bundle_chunk_count: int | None = None
    bundle_selection_strategy: str | None = None
    bundle_within_budget: bool | None = None
    bundle_matched_path: bool | None = None
    bundle_required_headings_present: list[str] = Field(default_factory=list)
    bundle_required_headings_missing: list[str] = Field(default_factory=list)
    bundle_diagnostic: str | None = None


class EvalReport(BaseModel):
    """Aggregate retrieval evaluation report."""

    total_queries: int
    strong_passes: int
    weak_passes: int
    misses: int
    top_1: EvalWindowStats
    top_3: EvalWindowStats
    top_5: EvalWindowStats
    buckets: dict[str, "EvalGroupReport"] = Field(default_factory=dict)
    expected_sources: dict[str, "EvalGroupReport"] = Field(default_factory=dict)
    query_styles: dict[str, "EvalGroupReport"] = Field(default_factory=dict)
    concepts: dict[str, "EvalGroupReport"] = Field(default_factory=dict)
    bundle_overall: "BundleGradeStats | None" = None
    bundle_buckets: dict[str, "BundleGradeStats"] = Field(default_factory=dict)
    bundle_expected_sources: dict[str, "BundleGradeStats"] = Field(default_factory=dict)
    source_confusions: list["SourceConfusionCase"] = Field(default_factory=list)
    baseline_comparison: "BaselineComparison | None" = None
    outcomes: list[EvalOutcome]


class EvalGroupReport(BaseModel):
    """Aggregate retrieval report for a bucket or concept group."""

    label: str
    total_queries: int
    strong_passes: int
    weak_passes: int
    misses: int
    top_1: EvalWindowStats
    top_3: EvalWindowStats
    top_5: EvalWindowStats
    case_ids: list[str] = Field(default_factory=list)


class BundleGradeStats(BaseModel):
    """Aggregate usefulness report for context bundles."""

    total_evaluated: int
    complete: int
    partial: int
    insufficient: int
    complete_rate: float
    partial_rate: float
    insufficient_rate: float


class SourceConfusionCase(BaseModel):
    """A case where source expectations and actual retrieved/bundled source diverged."""

    case_id: str
    query: str
    expected_source_name: str | None = None
    acceptable_source_names: list[str] = Field(default_factory=list)
    matched_result_source_name: str | None = None
    matched_result_path: str | None = None
    bundle_source_names: list[str] = Field(default_factory=list)
    grade: str
    bundle_grade: str | None = None


class BaselineMetricDelta(BaseModel):
    """Delta for a single compared metric."""

    current: float | int
    baseline: float | int
    delta: float


class BaselineBucketChange(BaseModel):
    """A changed retrieval or bundle bucket relative to baseline."""

    metric_family: str
    label: str
    status: str
    current: dict[str, float | int]
    baseline: dict[str, float | int]


class BaselineCaseChange(BaseModel):
    """A retrieval and/or bundle outcome change for a single case."""

    case_id: str
    query: str
    retrieval_from: str
    retrieval_to: str
    bundle_from: str | None = None
    bundle_to: str | None = None


class BaselineComparison(BaseModel):
    """Comparison between the current eval report and a supplied baseline."""

    status: str
    baseline_provided: bool
    baseline_path: str | None = None
    retrieval_status: str | None = None
    bundle_status: str | None = None
    retrieval_deltas: dict[str, BaselineMetricDelta] = Field(default_factory=dict)
    bundle_deltas: dict[str, BaselineMetricDelta] = Field(default_factory=dict)
    changed_retrieval_buckets: list[BaselineBucketChange] = Field(default_factory=list)
    changed_bundle_buckets: list[BaselineBucketChange] = Field(default_factory=list)
    changed_cases: list[BaselineCaseChange] = Field(default_factory=list)
