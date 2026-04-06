from pathlib import Path

from agentic_docs.ingest import ingest_source
from agentic_docs.query_service import (
    build_context_bundles,
    build_query_profile,
    canonical_path_key,
    normalize_query_text,
    query_chunks,
)


def test_normalize_query_text_strips_punctuation() -> None:
    normalized, tokens = normalize_query_text("Forms API validation!!! db/tasks.php?")

    assert normalized == "forms api validation db tasks php"
    assert tokens == ["forms", "api", "validation", "db", "tasks", "php"]


def test_normalize_query_text_keeps_intentful_action_verbs() -> None:
    normalized, tokens = normalize_query_text("How do I define web services for a plugin?")

    assert normalized == "define web services plugin"
    assert tokens == ["define", "web", "services", "plugin"]


def test_build_query_profile_marks_conceptual_queries_and_families() -> None:
    profile = build_query_profile("How do Mustache templates fit into Moodle output?")

    assert profile.intent == "conceptual"
    assert "output_templates" in profile.concept_families


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
    assert results[0].rerank_score is not None
    assert results[0].rerank_breakdown is not None


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


def test_query_reranks_heading_match_above_generic_content_match(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "overview.md").write_text(
        "---\n"
        "title: APIs overview\n"
        "---\n\n"
        "Moodle includes forms, services, tasks, events, and admin settings APIs.\n",
        encoding="utf-8",
    )
    (docs_dir / "services.md").write_text(
        "# External services\n\n## Writing a new service\n\nDefine plugin web services in db/services.php.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="define web services plugin", top_k=2)

    assert len(results) == 2
    assert results[0].source_file_path == "services.md"
    assert results[0].rerank_breakdown is not None
    assert int(results[0].rerank_breakdown["heading_overlap"]) >= int(results[1].rerank_breakdown["heading_overlap"])


def test_query_penalizes_example_sections_when_canonical_subsystem_doc_exists(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "plugintypes" / "local").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add admin settings for plugins.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "plugintypes" / "local" / "index.mdx").write_text(
        "# Local plugins\n\n## Examples\n\n### Adding Site Wide Settings For Your Local Plugin\n\nExample code for settings.php in a local plugin.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I add admin settings for a plugin?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/admin/index.md"
    assert results[0].rerank_breakdown is not None
    assert float(results[0].rerank_breakdown["subsystem_bonus"]) > 0


def test_query_prefers_upgrade_explainer_over_plugin_type_page_for_conceptual_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides" / "upgrade").mkdir(parents=True)
    (docs_dir / "apis" / "plugintypes" / "mod").mkdir(parents=True)
    (docs_dir / "guides" / "upgrade" / "index.md").write_text(
        "---\n"
        "title: Plugin Upgrades\n"
        "---\n\n"
        "## Upgrade helpers\n\nUse db/upgrade.php and savepoints for plugin upgrade steps.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "plugintypes" / "mod" / "index.mdx").write_text(
        "# Activity modules\n\n## Standard Files and their Functions\n\n### `upgrade.php` - Upgrade steps\n\nExample upgrade steps for activity modules.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do plugin upgrade steps work in Moodle?", top_k=2)

    assert results[0].source_file_path == "guides/upgrade/index.md"
    assert results[0].rerank_breakdown is not None
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


def test_query_prefers_language_string_explainer_over_plugin_type_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "_files").mkdir(parents=True)
    (docs_dir / "apis" / "plugintypes" / "mod").mkdir(parents=True)
    (docs_dir / "apis" / "_files" / "lang.md").write_text(
        "---\n"
        "title: lang\n"
        "---\n\n"
        "Language strings for plugins live in lang/en and are declared in lang files.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "plugintypes" / "mod" / "index.mdx").write_text(
        "# Activity modules\n\n## Standard Files and their Functions\n\n### `/lang/en/[modname].php` - Language string definitions\n\nExample language string definitions for activity modules.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do language strings work in plugins?", top_k=2)

    assert results[0].source_file_path == "apis/_files/lang.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


def test_query_prefers_templates_guide_over_plugin_specific_format_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides" / "templates").mkdir(parents=True)
    (docs_dir / "apis" / "plugintypes" / "format").mkdir(parents=True)
    (docs_dir / "guides" / "templates" / "index.md").write_text(
        "---\n"
        "title: Templates\n"
        "---\n\n"
        "Moodle uses Mustache templates to render HTML output.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "plugintypes" / "format" / "index.md").write_text(
        "# Course format\n\n## Override mustache blocks\n\nFormat plugins can override course format mustache blocks.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do Mustache templates fit into Moodle output?", top_k=2)

    assert results[0].source_file_path == "guides/templates/index.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


def test_query_prefers_templates_guide_over_output_api_for_mustache_concept_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides" / "templates").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "output").mkdir(parents=True)
    (docs_dir / "guides" / "templates" / "index.md").write_text(
        "---\n"
        "title: Templates\n"
        "---\n\n"
        "## Mustache templates\n\nMoodle uses Mustache templates to render HTML output.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "output" / "index.md").write_text(
        "---\n"
        "title: Output API\n"
        "---\n\n"
        "## Renderable\n\nRenderable objects can be rendered and may use templates in the output pipeline.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do Mustache templates fit into Moodle output?", top_k=2)

    assert results[0].source_file_path == "guides/templates/index.md"
    assert results[0].rerank_breakdown["query_intent"] == "conceptual"
    assert "output_templates" in results[0].rerank_breakdown["concept_families"]


def test_query_prefers_events_concept_page_over_incidental_plugin_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "_files").mkdir(parents=True)
    (docs_dir / "apis" / "plugintypes" / "assign").mkdir(parents=True)
    (docs_dir / "apis" / "_files" / "db-events-php.mdx").write_text(
        "---\n"
        "title: db-events-php\n"
        "---\n\n"
        "Register event observers in db/events.php for your plugin.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "plugintypes" / "assign" / "feedback.md").write_text(
        "# Assign feedback plugins\n\n## Other features\n\n### Add calendar events\n\nAssign feedback plugins can add calendar events.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do events work in plugins?", top_k=2)

    assert results[0].source_file_path == "apis/_files/db-events-php.mdx"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


def test_query_prefers_db_events_explainer_over_general_api_index_for_plugin_events_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "_files").mkdir(parents=True)
    (docs_dir / "apis").mkdir(exist_ok=True)
    (docs_dir / "apis" / "_files" / "db-events-php.mdx").write_text(
        "---\n"
        "title: db-events-php\n"
        "---\n\n"
        "Register plugin event observers in db/events.php.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis.md").write_text(
        "# API Guides\n\n## Other General API\n\n### Events API (event)\n\nEvents API allows plugins to fire and observe events.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do events work in plugins?", top_k=2)

    assert results[0].source_file_path == "apis/_files/db-events-php.mdx"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > float(
        results[1].rerank_breakdown["family_specific_bonus"]
    )


def test_context_bundle_respects_token_budget_and_compacts_heading_prefix(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "forms.md").write_text(
        "\n".join(
            [
                "# Forms API",
                "",
                "## Validation",
                "",
                "Use addRule() to validate required fields in Moodle forms. " * 6,
                "",
                "Validation callbacks can inspect cross-field dependencies. " * 6,
                "",
                "Use repeated elements when you need multiple grouped entries. " * 6,
            ]
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=50, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="forms validation callbacks", top_k=1)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results,
        include_previous=True,
        include_next=True,
        bundle_max_tokens=90,
    )

    assert len(bundles) == 1
    assert bundles[0].selection_strategy in {"match_only", "section_window", "truncated_match"}
    assert bundles[0].bundle_token_count <= 90
    if len(bundles[0].chunks) > 1:
        for chunk in bundles[0].chunks[1:]:
            assert not chunk.content.startswith("Heading:")
