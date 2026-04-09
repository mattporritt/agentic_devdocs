import pytest
from pydantic import ValidationError

from agentic_docs.models import (
    RuntimeContractContent,
    RuntimeContractDiagnostics,
    RuntimeContractEnvelope,
    RuntimeContractIntent,
    RuntimeContractResult,
    RuntimeContractSection,
    RuntimeContractSource,
)


def _contract_result() -> RuntimeContractResult:
    return RuntimeContractResult(
        id="bundle-1",
        rank=1,
        confidence="high",
        source=RuntimeContractSource(
            name="devdocs_repo",
            type="repo_markdown",
            url=None,
            canonical_url=None,
            path="docs/forms.md",
            document_title="Forms API",
            section_title="Validation",
            heading_path=["Validation"],
        ),
        content=RuntimeContractContent(
            summary="Forms API — Validation: Use addRule.",
            sections=[
                RuntimeContractSection(
                    id="chunk-1",
                    role="match",
                    document_title="Forms API",
                    source_path="docs/forms.md",
                    source_url=None,
                    canonical_url=None,
                    section_title="Validation",
                    heading_path=["Validation"],
                    token_count=24,
                    content="Heading: Validation\n\nUse addRule().",
                )
            ],
            file_anchors=[],
            key_points=[],
        ),
        diagnostics=RuntimeContractDiagnostics(
            ranking_explanation="Rank 1 devdocs_repo bundle from Validation.",
            support_reason=None,
            token_count=24,
            selection_strategy="match_only",
        ),
    )


def test_runtime_contract_models_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RuntimeContractSource(
            name="devdocs_repo",
            type="repo_markdown",
            url=None,
            canonical_url=None,
            path="docs/forms.md",
            document_title="Forms API",
            section_title="Validation",
            heading_path=[],
            unexpected_field="nope",
        )


def test_runtime_contract_models_keep_explicit_empty_lists() -> None:
    payload = RuntimeContractEnvelope(
        query="forms",
        normalized_query="forms",
        intent=RuntimeContractIntent(query_intent="keyword", task_intent="general", concept_families=[]),
        results=[_contract_result()],
    )

    dumped = payload.model_dump()

    assert dumped["intent"]["concept_families"] == []
    assert dumped["results"][0]["content"]["sections"]
    assert dumped["results"][0]["content"]["file_anchors"] == []
    assert dumped["results"][0]["content"]["key_points"] == []
    assert dumped["results"][0]["source"]["heading_path"] == ["Validation"]
    assert dumped["results"][0]["diagnostics"]["support_reason"] is None
