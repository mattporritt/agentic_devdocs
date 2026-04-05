from pathlib import Path

from agentic_docs.ingest import ingest_source
from agentic_docs.query_service import build_context_bundles, canonical_path_key, normalize_query_text, query_chunks


def test_normalize_query_text_strips_punctuation() -> None:
    normalized, tokens = normalize_query_text("Forms API validation!!! db/tasks.php?")

    assert normalized == "forms api validation db tasks php"
    assert tokens == ["forms", "api", "validation", "db", "tasks", "php"]


def test_query_chunks_and_context_bundle(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "forms.md").write_text(
        "\n".join(
            [
                "---",
                "title: Forms API",
                "---",
                "",
                "Validation rules live in the Forms API.",
                "",
                "## Validation",
                "",
                "Use addRule() for field validation.",
                "",
                "## Repeated elements",
                "",
                "Repeated elements allow multiple sections.",
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=40, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Forms API validation!!!", top_k=2)

    assert results
    assert results[0].normalized_query == "forms api validation"

    bundles = build_context_bundles(db_path=db_path, results=results[:1], include_previous=False, include_next=True)

    assert len(bundles) == 1
    assert bundles[0].chunks[0].role == "match"
    assert bundles[0].bundle_token_count >= bundles[0].chunks[0].token_count


def test_query_prefers_canonical_doc_over_versioned_duplicate(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "forms.md").write_text(
        "---\n"
        "title: Forms API\n"
        "---\n\n"
        "Use addRule for validation in Moodle forms.\n",
        encoding="utf-8",
    )
    versioned_dir = tmp_path / "versioned_docs" / "version-5.1"
    versioned_dir.mkdir(parents=True)
    (versioned_dir / "forms.md").write_text(
        "---\n"
        "title: Forms API\n"
        "---\n\n"
        "Use addRule for validation in Moodle forms.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=tmp_path, db_path=db_path, tokenizer_name="openai", max_tokens=60, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="forms validation", top_k=2)

    assert results
    assert canonical_path_key(results[0].source_file_path) == "docs/forms.md"
    assert results[0].source_file_path == "docs/forms.md"
