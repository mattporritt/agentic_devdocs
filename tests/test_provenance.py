from agentic_docs.models import QueryResult
from agentic_docs.provenance import infer_source_name, result_source_metadata, source_fields_from_metadata


def test_infer_source_name_prefers_design_system_path() -> None:
    assert infer_source_name("design_system/styles/colours.site", "devdocs_repo", "repo_markdown") == "design_system"


def test_source_fields_from_metadata_backfills_legacy_devdocs_name() -> None:
    source_name, source_type, source_url, canonical_url = source_fields_from_metadata(
        {"source_type": "repo_markdown"},
        source_file_path="docs/apis/subsystems/admin/index.md",
    )

    assert source_name == "devdocs_repo"
    assert source_type == "repo_markdown"
    assert source_url is None
    assert canonical_url is None


def test_result_source_metadata_preserves_scraped_source_fields() -> None:
    result = QueryResult(
        chunk_id="chunk-1",
        score=-3.0,
        content="Design content",
        source_file_path="design_system/styles/colours.site",
        document_id="doc-1",
        document_title="Colours",
        section_id="section-1",
        section_title="Semantic colour tokens",
        heading_path=["Colours", "Semantic colour tokens"],
        token_count=20,
        chunk_order=0,
        metadata_json={
            "source_name": "design_system",
            "source_type": "scraped_web",
            "source_url": "https://design.moodle.com/example",
            "canonical_url": "https://design.moodle.com/example",
        },
    )

    payload = result_source_metadata(result)

    assert payload == {
        "source_name": "design_system",
        "source_type": "scraped_web",
        "source_url": "https://design.moodle.com/example",
        "canonical_url": "https://design.moodle.com/example",
    }
