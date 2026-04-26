# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

from datetime import datetime, timezone

from agentic_docs.wiki_ingest import (
    WikiScrapeContext,
    clean_wikitext,
    is_redirect,
    page_source_path,
    sections_from_wikitext,
    wiki_api_url,
    wiki_page_to_document,
    wiki_page_url,
)


def _ctx() -> WikiScrapeContext:
    return WikiScrapeContext(
        base_url="https://docs.moodle.org/502/en",
        api_url="https://docs.moodle.org/502/en/api.php",
        scrape_timestamp=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )


def test_wiki_api_url_appends_api_php() -> None:
    assert wiki_api_url("https://docs.moodle.org/502/en") == "https://docs.moodle.org/502/en/api.php"
    assert wiki_api_url("https://docs.moodle.org/502/en/") == "https://docs.moodle.org/502/en/api.php"


def test_wiki_page_url_encodes_spaces() -> None:
    url = wiki_page_url("https://docs.moodle.org/502/en", "Using Forum")
    assert url == "https://docs.moodle.org/502/en/Using_Forum"


def test_page_source_path_slugifies_title() -> None:
    assert page_source_path("Using Forum") == "user_docs/using-forum.wiki"
    assert page_source_path("Quiz activity") == "user_docs/quiz-activity.wiki"
    assert page_source_path("Gradebook FAQ") == "user_docs/gradebook-faq.wiki"


def test_is_redirect_detects_redirect_pages() -> None:
    assert is_redirect("#REDIRECT [[Forum]]") is True
    assert is_redirect("#redirect [[Forum]]") is True
    assert is_redirect("This is a normal page about forums.") is False
    assert is_redirect("== Introduction ==\nContent here.") is False


def test_clean_wikitext_strips_templates() -> None:
    wikitext = "Before {{SomeTemplate|arg1|arg2}} after."
    result = clean_wikitext(wikitext)
    assert "{{" not in result
    assert "}}" not in result
    assert "Before" in result
    assert "after" in result


def test_clean_wikitext_strips_nested_templates() -> None:
    wikitext = "Text {{outer|{{inner|value}}}} end."
    result = clean_wikitext(wikitext)
    assert "{{" not in result
    assert "Text" in result
    assert "end" in result


def test_clean_wikitext_simplifies_wikilinks() -> None:
    wikitext = "See the [[Forum|Forum activity]] and [[Quiz]] pages."
    result = clean_wikitext(wikitext)
    assert "Forum activity" in result
    assert "Quiz" in result
    assert "[[" not in result


def test_clean_wikitext_removes_file_links() -> None:
    wikitext = "Here is an image [[File:Forum.png|thumb|A forum screenshot]]."
    result = clean_wikitext(wikitext)
    assert "[[File:" not in result


def test_clean_wikitext_simplifies_external_links() -> None:
    wikitext = "See [https://moodle.org/forums Moodle Forums] for help."
    result = clean_wikitext(wikitext)
    assert "Moodle Forums" in result
    assert "https://" not in result


def test_clean_wikitext_strips_bold_italic_markers() -> None:
    wikitext = "This is '''bold''' and ''italic'' text."
    result = clean_wikitext(wikitext)
    assert "'''" not in result
    assert "''" not in result
    assert "bold" in result
    assert "italic" in result


def test_clean_wikitext_strips_html_tags() -> None:
    wikitext = "Before <ref>Citation text</ref> after."
    result = clean_wikitext(wikitext)
    assert "<ref>" not in result
    assert "Citation text" not in result
    assert "Before" in result
    assert "after" in result


def test_sections_from_wikitext_creates_root_section_for_intro() -> None:
    wikitext = "This is the introduction text before any heading."
    sections = sections_from_wikitext(wikitext, page_title="Forum", document_id="doc-1")
    assert len(sections) == 1
    assert sections[0].section_title == "Forum"
    assert sections[0].heading_path == ["Forum"]
    assert sections[0].heading_level == 0
    assert "introduction" in sections[0].content


def test_sections_from_wikitext_splits_on_headings() -> None:
    wikitext = """\
Introduction text.

== Overview ==
This section covers the overview.

=== Details ===
More detailed information here.

== Settings ==
Configuration options.
"""
    sections = sections_from_wikitext(wikitext, page_title="Forum", document_id="doc-1")
    titles = [s.section_title for s in sections]
    assert "Forum" in titles
    assert "Overview" in titles
    assert "Details" in titles
    assert "Settings" in titles


def test_sections_from_wikitext_builds_heading_path() -> None:
    wikitext = """\
== Overview ==
Some content.

=== Details ===
More content.
"""
    sections = sections_from_wikitext(wikitext, page_title="Quiz", document_id="doc-2")
    detail_section = next(s for s in sections if s.section_title == "Details")
    assert detail_section.heading_path == ["Quiz", "Overview", "Details"]


def test_sections_from_wikitext_skips_empty_sections() -> None:
    wikitext = "== Empty ==\n\n== Non-empty ==\nActual content here."
    sections = sections_from_wikitext(wikitext, page_title="Page", document_id="doc-3")
    titles = [s.section_title for s in sections]
    assert "Empty" not in titles
    assert "Non-empty" in titles


def test_wiki_page_to_document_sets_provenance() -> None:
    wikitext = "== Overview ==\nThe forum lets users post discussions."
    doc = wiki_page_to_document(page_title="Forum", wikitext=wikitext, ctx=_ctx())

    assert doc.title == "Forum"
    assert doc.metadata.source_type == "scraped_web"
    assert doc.metadata.source_name == "user_docs"
    assert doc.metadata.canonical_url == "https://docs.moodle.org/502/en/Forum"
    assert doc.metadata.source_path == "user_docs/forum.wiki"
    assert doc.metadata.scrape_timestamp == datetime(2026, 4, 27, tzinfo=timezone.utc)
    assert len(doc.sections) == 1
    assert doc.sections[0].section_title == "Overview"


def test_wiki_page_to_document_stable_ids() -> None:
    wikitext = "Content for the page."
    doc1 = wiki_page_to_document(page_title="Forum", wikitext=wikitext, ctx=_ctx())
    doc2 = wiki_page_to_document(page_title="Forum", wikitext=wikitext, ctx=_ctx())
    assert doc1.id == doc2.id
    assert doc1.sections[0].id == doc2.sections[0].id
