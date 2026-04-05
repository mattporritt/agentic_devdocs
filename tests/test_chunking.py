from datetime import datetime, timezone

from agentic_docs.chunking import chunk_document
from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.tokenizers import OpenAITokenizer


def test_chunk_document_preserves_heading_context_and_overlap() -> None:
    document = DocumentModel(
        id="doc-1",
        title="Guide",
        metadata=DocumentMetadata(
            source_path="guide.md",
            repo_commit_hash="abc123",
            last_modified_time=datetime.now(tz=timezone.utc),
            file_hash="hash",
        ),
        sections=[
            SectionModel(
                id="sec-1",
                document_id="doc-1",
                section_order=0,
                section_title="Install",
                heading_level=2,
                heading_path=["Guide", "Install"],
                content="\n\n".join(
                    [
                        "Paragraph one about installation and setup.",
                        "Paragraph two adds more installation detail and repeated words setup install configuration.",
                        "Paragraph three closes out the section with deployment advice.",
                    ]
                ),
            )
        ],
    )

    chunks = chunk_document(document, OpenAITokenizer(), max_tokens=30, overlap_tokens=5)

    assert len(chunks) >= 2
    assert chunks[0].content.startswith("Heading: Guide > Install")
    assert chunks[0].next_chunk_id is not None
    assert chunks[1].prev_chunk_id == chunks[0].id
    assert all(chunk.token_count <= 30 for chunk in chunks)

