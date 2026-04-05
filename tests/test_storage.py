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

