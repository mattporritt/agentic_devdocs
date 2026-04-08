from datetime import datetime, timezone
from pathlib import Path

from agentic_docs.models import ChunkMetadata, ChunkModel, DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.storage import SQLiteStore


def test_sqlite_store_indexes_and_queries(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "index.db")
    store.initialize()
    document = DocumentModel(
        id="doc-1",
        title="Moodle Forms",
        metadata=DocumentMetadata(
            source_path="forms.md",
            repo_commit_hash="abc123",
            last_modified_time=datetime.now(tz=timezone.utc),
            file_hash="filehash",
        ),
        sections=[
            SectionModel(
                id="sec-1",
                document_id="doc-1",
                section_order=0,
                section_title="Form API",
                heading_level=2,
                heading_path=["Moodle Forms", "Form API"],
                content="Use Moodle forms to build forms.",
            )
        ],
    )
    chunk = ChunkModel(
        id="chunk-1",
        section_id="sec-1",
        chunk_order=0,
        content="Heading: Moodle Forms > Form API\n\nUse Moodle forms to build forms.",
        token_count=12,
        metadata=ChunkMetadata(
            document_id="doc-1",
            document_title="Moodle Forms",
            source_path="forms.md",
            repo_commit_hash="abc123",
            section_title="Form API",
            heading_path=["Moodle Forms", "Form API"],
        ),
    )

    store.store_document(document, [chunk])
    results = store.query("forms", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].document_title == "Moodle Forms"
    assert store.stats()["chunks"] == 1
    assert store.detailed_stats(limit=5)["overview"]["chunks"] == 1


def test_sqlite_store_reports_source_metadata_in_stats_and_inspect(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "index.db")
    store.initialize()
    document = DocumentModel(
        id="doc-site",
        title="For Developers",
        metadata=DocumentMetadata(
            source_path="design_system/get-started/for-developers-98e3cb.site",
            source_type="scraped_web",
            source_name="design_system",
            source_url="https://design.moodle.com/98292f05f/p/98e3cb",
            canonical_url="https://design.moodle.com/98292f05f/p/98e3cb",
            scrape_timestamp=datetime.now(tz=timezone.utc),
            file_hash="payloadhash",
            content_hash="payloadhash",
        ),
        sections=[
            SectionModel(
                id="sec-site",
                document_id="doc-site",
                section_order=0,
                section_title="Token Consumption",
                heading_level=2,
                heading_path=["For Developers", "Token Consumption"],
                content="Use CSS tokens.",
            )
        ],
    )
    chunk = ChunkModel(
        id="chunk-site",
        section_id="sec-site",
        chunk_order=0,
        content="Heading: For Developers > Token Consumption\n\nUse CSS tokens.",
        token_count=10,
        metadata=ChunkMetadata(
            document_id="doc-site",
            document_title="For Developers",
            source_path=document.metadata.source_path,
            source_type=document.metadata.source_type,
            source_name=document.metadata.source_name,
            source_url=document.metadata.source_url,
            canonical_url=document.metadata.canonical_url,
            scrape_timestamp=document.metadata.scrape_timestamp,
            section_title="Token Consumption",
            heading_path=["For Developers", "Token Consumption"],
        ),
    )

    store.store_document(document, [chunk])

    details = store.detailed_stats(limit=5)
    inspect = store.inspect_document("doc-site")

    assert details["sources"] == [
        {
            "source_type": "scraped_web",
            "source_name": "design_system",
            "document_count": 1,
            "section_count": 1,
            "chunk_count": 1,
        }
    ]
    assert inspect is not None
    assert inspect["document"]["source_type"] == "scraped_web"
    assert inspect["document"]["source_name"] == "design_system"
