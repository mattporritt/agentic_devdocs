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


def test_build_query_profile_treats_where_docs_queries_as_conceptual() -> None:
    profile = build_query_profile("Where do I find Moodle Behat test docs?")

    assert profile.intent == "conceptual"
    assert "testing" in profile.concept_families


def test_build_query_profile_detects_plugin_string_concept_without_language_token() -> None:
    profile = build_query_profile("How do Moodle plugin strings get defined?")

    assert profile.intent == "conceptual"
    assert "language_strings" in profile.concept_families


def test_build_query_profile_detects_task_oriented_intents() -> None:
    location_profile = build_query_profile("Where do I put plugin lang strings?")
    implementation_profile = build_query_profile("How do I define web services for a plugin?")

    assert location_profile.task_intent == "file_location"
    assert implementation_profile.task_intent == "implementation_guide"
    assert "web_services" in implementation_profile.concept_families


def test_build_query_profile_detects_flow_and_requirement_intents() -> None:
    flow_profile = build_query_profile("Where does template rendering fit in the output flow?")
    requirement_profile = build_query_profile("What do I need to implement for plugin privacy metadata?")

    assert flow_profile.task_intent == "flow_explainer"
    assert "output_templates" in flow_profile.concept_families
    assert requirement_profile.task_intent == "implementation_guide"
    assert "privacy" in requirement_profile.concept_families


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


def test_query_prefers_writing_a_service_over_external_services_overview_for_web_services_howto(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "external").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "external" / "index.md").write_text(
        "---\n"
        "title: External Services\n"
        "---\n\n"
        "Moodle has a full-featured web service framework for external systems.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "external" / "writing-a-service.md").write_text(
        "---\n"
        "title: Writing a new service\n"
        "---\n\n"
        "## Declare the web service function\n\n"
        "Declare plugin external functions in `db/services.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I define web services for a plugin?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/external/writing-a-service.md"
    assert results[0].rerank_breakdown is not None
    assert "web_services" in results[0].rerank_breakdown["concept_families"]
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > float(
        results[1].rerank_breakdown["family_specific_bonus"]
    )


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


def test_query_prefers_language_string_explainer_over_generic_string_pages(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "_files").mkdir(parents=True)
    (docs_dir / "guides" / "javascript").mkdir(parents=True)
    (docs_dir / "community" / "plugincontribution").mkdir(parents=True)
    (docs_dir / "apis" / "_files" / "lang.md").write_text(
        "---\n"
        "title: lang\n"
        "---\n\n"
        "## Language files\n\n"
        "Define plugin strings in lang/en/<plugintype>_<pluginname>.php and retrieve them with get_string().\n",
        encoding="utf-8",
    )
    (docs_dir / "guides" / "javascript" / "index.md").write_text(
        "# JavaScript guide\n\n## Working with Strings\n\nClient-side code can manipulate strings in JavaScript.\n",
        encoding="utf-8",
    )
    (docs_dir / "community" / "plugincontribution" / "checklist.md").write_text(
        "# Contribution checklist\n\n## Coding\n\n### Strings\n\nRemember to review plugin strings before publishing.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do Moodle plugin strings get defined?", top_k=3)

    assert results[0].source_file_path == "apis/_files/lang.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0
    assert float(results[0].rerank_breakdown["section_focus_bonus"]) > 0


def test_query_prefers_language_string_file_doc_over_app_language_files_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "_files").mkdir(parents=True)
    (docs_dir / "general" / "app" / "development").mkdir(parents=True)
    (docs_dir / "apis" / "_files" / "lang.md").write_text(
        "---\n"
        "title: lang\n"
        "---\n\n"
        "## Language files\n\n"
        "Plugin strings are defined in `lang/en/<plugintype>_<pluginname>.php`.\n",
        encoding="utf-8",
    )
    (docs_dir / "general" / "app" / "development" / "development-guide.md").write_text(
        "# Development guide\n\n## Folder structure\n\n### Language files\n\nApp language files live in the app development tree.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="What file contains a plugin's language strings?", top_k=2)

    assert results[0].source_file_path == "apis/_files/lang.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > float(
        results[1].rerank_breakdown["family_specific_bonus"]
    )


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


def test_query_prefers_output_flow_explainer_over_templates_index_for_flow_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides" / "templates").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "output").mkdir(parents=True)
    (docs_dir / "guides" / "templates" / "index.md").write_text(
        "---\n"
        "title: Templates\n"
        "---\n\n"
        "## Templates\n\nTemplates can be overridden in plugins.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "output" / "index.md").write_text(
        "---\n"
        "title: Output API\n"
        "---\n\n"
        "## Page output journey\n\n"
        "This explains how renderables, renderers, and templates fit into the output flow.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Where does template rendering fit in the output flow?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/output/index.md"
    assert results[0].rerank_breakdown["query_intent"] == "conceptual"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > float(
        results[1].rerank_breakdown["family_specific_bonus"]
    )


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


def test_query_prefers_output_api_renderer_section_over_templates_renderer_section_for_renderer_docs_query(
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides" / "templates").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "output").mkdir(parents=True)
    (docs_dir / "guides" / "templates" / "index.md").write_text(
        "---\n"
        "title: Templates\n"
        "---\n\n"
        "## Rendering in PHP\n\n### Renderers\n\nTemplates can be rendered in PHP through renderers.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "output" / "index.md").write_text(
        "---\n"
        "title: Output API\n"
        "---\n\n"
        "## Page Output Journey\n\n### Accessing renderers with dependency injection\n\nRenderers are documented in the Output API.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Where are Moodle renderers documented?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/output/index.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


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


def test_context_bundle_adds_file_anchor_support_from_same_document(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Individual settings\n\n"
        "Choose a suitable admin setting and configure defaults for your plugin.\n\n"
        "## Adding settings in settings.php\n\n"
        "Plugin admin settings live in settings.php and are registered there.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=40, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Where do Moodle plugin admin settings go?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="Where do Moodle plugin admin settings go?",
        bundle_max_tokens=120,
    )

    assert len(bundles) == 1
    assert any("settings.php" in chunk.content for chunk in bundles[0].chunks)
    assert bundles[0].diagnostics["task_intent"] == "file_location"
    if bundles[0].selection_strategy in {"task_support", "task_support_truncated"}:
        assert any(chunk.role == "support" for chunk in bundles[0].chunks)
    assert any("settings.php" in chunk.content for chunk in bundles[0].chunks)


def test_context_bundle_trims_primary_chunk_to_fit_file_anchor_support(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Individual settings\n\n"
        + ("Choose a suitable admin setting and configure defaults for your plugin. " * 30)
        + "\n\n## Where to find the code\n\n"
        "Plugin admin settings live in `settings.php` and are managed from `admin/settings.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    results = query_chunks(db_path=db_path, query_text="Where do Moodle plugin admin settings go?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="Where do Moodle plugin admin settings go?",
        bundle_max_tokens=180,
    )

    assert len(bundles) == 1
    assert bundles[0].selection_strategy in {"task_support", "task_support_truncated"}
    assert bundles[0].bundle_token_count <= 180
    assert any("settings.php" in chunk.content for chunk in bundles[0].chunks)
    assert bundles[0].diagnostics["support_added"] is True


def test_context_bundle_adds_file_anchor_support_from_other_result(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apis.md").write_text(
        "# API Guides\n\n## Web services API\n\nThe Web services API lets plugins expose external functions.\n",
        encoding="utf-8",
    )
    (docs_dir / "writing-a-service.md").write_text(
        "# Writing a new service\n\n## Declare the web service function\n\nDeclare the function in `db/services.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I define web services for a plugin?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="How do I define web services for a plugin?",
        bundle_max_tokens=160,
    )

    assert len(bundles) == 1
    assert bundles[0].source_file_path == "writing-a-service.md"
    assert any("db/services.php" in chunk.content for chunk in bundles[0].chunks)


def test_context_bundle_prefers_anchor_rich_support_chunk_from_other_document(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apis.md").write_text(
        "# API Guides\n\n"
        "## Web services API\n\n"
        "The Web services API lets plugins expose external functions.\n\n"
        "## Admin settings API\n\n"
        "The Admin settings API lets plugins expose configuration.\n",
        encoding="utf-8",
    )
    (docs_dir / "writing-a-service.md").write_text(
        "# Writing a new service\n\n"
        "## Bump the plugin version\n\n"
        "Increase version.php after adding your service.\n\n"
        "## Write the external function descriptions\n\n"
        "External functions are declared through `db/services.php` and related classes.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I define web services for a plugin?", top_k=5)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="How do I define web services for a plugin?",
        bundle_max_tokens=220,
    )

    assert len(bundles) == 1
    assert any(chunk.role == "support" for chunk in bundles[0].chunks)
    assert any(chunk.source_file_path == "writing-a-service.md" for chunk in bundles[0].chunks)
    assert any("db/services.php" in chunk.content for chunk in bundles[0].chunks)
    assert all("Admin settings API" not in chunk.content for chunk in bundles[0].chunks if chunk.role == "support")


def test_context_bundle_adds_web_service_support_when_primary_already_has_db_services_anchor(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "writing-a-service.md").write_text(
        "# Writing a new service\n\n"
        "## Declare the web service function\n\n"
        "Declare the function in `db/services.php`.\n\n"
        "## Write the external function descriptions\n\n"
        "External functions are described in dedicated classes.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I define web services for a plugin?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="How do I define web services for a plugin?",
        bundle_max_tokens=220,
    )

    assert len(bundles) == 1
    assert bundles[0].selection_strategy in {"task_support", "task_support_truncated"}
    assert any(chunk.role == "support" for chunk in bundles[0].chunks)
    assert any("External functions" in chunk.content or "Write the external function descriptions" in " > ".join(chunk.heading_path) for chunk in bundles[0].chunks)


def test_context_bundle_prefers_web_service_definition_support_over_version_note_for_wire_up_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "writing-a-service.md").write_text(
        "# Writing a new service\n\n"
        "## Bump the plugin version\n\n"
        "Increase version.php after adding your service.\n\n"
        "## Write the external function descriptions\n\n"
        "Register the function description in `db/services.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I wire up an external service in a plugin?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="How do I wire up an external service in a plugin?",
        bundle_max_tokens=220,
    )

    assert bundles[0].diagnostics["task_intent"] == "implementation_guide"
    assert any("db/services.php" in chunk.content for chunk in bundles[0].chunks)
    assert all("version.php" not in chunk.content for chunk in bundles[0].chunks if chunk.role == "support")


def test_context_bundle_adds_behat_writing_support_for_location_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "general" / "development" / "tools" / "behat").mkdir(parents=True)
    (docs_dir / "general" / "development" / "tools" / "behat" / "index.md").write_text(
        "# Behat\n\n## Moodle integration\n\n### Admin tool \"Acceptance testing\"\n\nUse the admin tool to inspect available steps.\n",
        encoding="utf-8",
    )
    (docs_dir / "general" / "development" / "tools" / "behat" / "writing.md").write_text(
        "# Writing acceptance tests\n\n## Writing acceptance tests\n\nThis guide explains how to write Behat tests.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Where do I find Moodle Behat test docs?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="Where do I find Moodle Behat test docs?",
        bundle_max_tokens=220,
    )

    assert len(bundles) == 1
    assert any(chunk.source_file_path.endswith("writing.md") for chunk in bundles[0].chunks)
    assert any("Writing acceptance tests" in " > ".join(chunk.heading_path) or "Writing acceptance tests" in chunk.content for chunk in bundles[0].chunks)


def test_context_bundle_adds_privacy_metadata_support_for_requirement_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "privacy").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "privacy" / "index.md").write_text(
        "---\n"
        "title: Privacy API\n"
        "---\n\n"
        "## Background\n\n"
        "### Implementing a provider\n\n"
        "Plugins declare privacy providers for stored data.\n\n"
        "## Plugins which do not store personal data\n\n"
        "### Implementation requirements\n\n"
        "Use the metadata provider or `null_provider` when the plugin does not store personal data.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="What do I need to implement for plugin privacy metadata?", top_k=3)
    bundles = build_context_bundles(
        db_path=db_path,
        results=results[:1],
        support_results=results,
        query_text="What do I need to implement for plugin privacy metadata?",
        bundle_max_tokens=220,
    )

    assert bundles[0].diagnostics["task_intent"] == "implementation_guide"
    assert any("metadata provider" in chunk.content.lower() or "null_provider" in chunk.content.lower() for chunk in bundles[0].chunks)


def test_query_prefers_behat_guide_over_mdk_workflow_for_docs_location_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "general" / "development" / "tools" / "behat").mkdir(parents=True)
    (docs_dir / "general" / "development" / "tools").mkdir(parents=True, exist_ok=True)
    (docs_dir / "general" / "development" / "tools" / "behat" / "writing.md").write_text(
        "# Writing acceptance tests\n\n## Behat\n\nUse this guide to write Behat tests in Moodle.\n",
        encoding="utf-8",
    )
    (docs_dir / "general" / "development" / "tools" / "mdk.md").write_text(
        "# MDK\n\n## Typical workflows using MDK\n\n### Executing behat tests\n\nMDK can run Behat tests from the command line.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="Where do I find Moodle Behat test docs?", top_k=2)

    assert results[0].source_file_path == "general/development/tools/behat/writing.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0
    assert float(results[1].rerank_breakdown["family_specific_bonus"]) < float(
        results[0].rerank_breakdown["family_specific_bonus"]
    )


def test_query_prefers_privacy_provider_doc_over_unrelated_provider_page(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "privacy").mkdir(parents=True)
    (docs_dir / "integrations" / "providers").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "privacy" / "index.md").write_text(
        "---\n"
        "title: Privacy API\n"
        "---\n\n"
        "## Privacy provider\n\nPlugins implement privacy providers to describe stored data.\n",
        encoding="utf-8",
    )
    (docs_dir / "integrations" / "providers" / "index.md").write_text(
        "# Integration providers\n\n## Provider registry\n\nProviders can register integration services.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do privacy providers work for plugins?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/privacy/index.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0


def test_query_prefers_form_validation_section_over_field_reference(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "form" / "fields").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "form").mkdir(parents=True, exist_ok=True)
    (docs_dir / "apis" / "subsystems" / "form" / "index.md").write_text(
        "---\n"
        "title: Forms API\n"
        "---\n\n"
        "## Commonly used functions\n\n### addRule()\n\nUse addRule() to validate Moodle form fields.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "form" / "fields" / "choicedropdown.md").write_text(
        "# Choice dropdown\n\nChoice dropdown fields are used for form field selection.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    results = query_chunks(db_path=db_path, query_text="How do I validate a Moodle form field?", top_k=2)

    assert results[0].source_file_path == "apis/subsystems/form/index.md"
    assert float(results[0].rerank_breakdown["family_specific_bonus"]) > 0
    assert float(results[1].rerank_breakdown["incidental_penalty"]) < 0
