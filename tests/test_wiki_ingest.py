# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from agentic_docs.wiki_ingest import (
    WikiScrapeContext,
    WikiSession,
    _load_dotenv,
    _request_json,
    clean_wikitext,
    is_redirect,
    page_source_path,
    sections_from_wikitext,
    wiki_api_url,
    wiki_page_to_document,
    wiki_page_url,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx() -> WikiScrapeContext:
    return WikiScrapeContext(
        base_url="https://docs.moodle.org/502/en",
        api_url="https://docs.moodle.org/502/en/api.php",
        scrape_timestamp=datetime(2026, 4, 27, tzinfo=timezone.utc),
    )


def _session() -> WikiSession:
    return WikiSession(
        cookies={"cf_clearance": "abc123", "__cf_bm": "xyz"},
        user_agent="Mozilla/5.0 Firefox/138.0",
        base_url="https://docs.moodle.org/502/en",
    )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def test_load_dotenv_sets_missing_vars() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("MOODLE_DOCS_CF_CLEARANCE=test-cookie\n")
        f.write("MOODLE_DOCS_USER_AGENT=Mozilla/5.0 Firefox\n")
        env_path = Path(f.name)
    try:
        os.environ.pop("MOODLE_DOCS_CF_CLEARANCE", None)
        os.environ.pop("MOODLE_DOCS_USER_AGENT", None)
        _load_dotenv(env_path)
        assert os.environ["MOODLE_DOCS_CF_CLEARANCE"] == "test-cookie"
        assert os.environ["MOODLE_DOCS_USER_AGENT"] == "Mozilla/5.0 Firefox"
    finally:
        os.environ.pop("MOODLE_DOCS_CF_CLEARANCE", None)
        os.environ.pop("MOODLE_DOCS_USER_AGENT", None)
        env_path.unlink()


def test_load_dotenv_does_not_overwrite_existing_vars() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("MOODLE_DOCS_CF_CLEARANCE=from-file\n")
        env_path = Path(f.name)
    try:
        os.environ["MOODLE_DOCS_CF_CLEARANCE"] = "from-shell"
        _load_dotenv(env_path)
        assert os.environ["MOODLE_DOCS_CF_CLEARANCE"] == "from-shell"
    finally:
        os.environ.pop("MOODLE_DOCS_CF_CLEARANCE", None)
        env_path.unlink()


def test_load_dotenv_ignores_comments_and_blanks() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("# this is a comment\n\nMOODLE_DOCS_CF_CLEARANCE=value\n")
        env_path = Path(f.name)
    try:
        os.environ.pop("MOODLE_DOCS_CF_CLEARANCE", None)
        _load_dotenv(env_path)
        assert os.environ["MOODLE_DOCS_CF_CLEARANCE"] == "value"
    finally:
        os.environ.pop("MOODLE_DOCS_CF_CLEARANCE", None)
        env_path.unlink()


def test_load_dotenv_is_noop_for_missing_file() -> None:
    _load_dotenv(Path("/nonexistent/.env"))  # must not raise


# ---------------------------------------------------------------------------
# WikiSession
# ---------------------------------------------------------------------------

def test_wiki_session_cookie_header_formats_correctly() -> None:
    session = _session()
    header = session.cookie_header
    assert "cf_clearance=abc123" in header
    assert "__cf_bm=xyz" in header


def test_wiki_session_update_replaces_credentials() -> None:
    session = _session()
    session.update("new-cookie", "Mozilla/5.0 Firefox/150.0")
    assert session.cookies == {"cf_clearance": "new-cookie"}
    assert session.user_agent == "Mozilla/5.0 Firefox/150.0"


def test_wiki_session_from_cookie_builds_session() -> None:
    session = WikiSession.from_cookie("test-clearance-value", "https://docs.moodle.org/502/en")
    assert session.cookies == {"cf_clearance": "test-clearance-value"}
    assert session.base_url == "https://docs.moodle.org/502/en"
    assert "Mozilla" in session.user_agent


def test_wiki_session_from_cookie_accepts_custom_ua() -> None:
    ua = "Mozilla/5.0 Firefox/150.0"
    session = WikiSession.from_cookie("tok", "https://docs.moodle.org/502/en", user_agent=ua)
    assert session.user_agent == ua


# ---------------------------------------------------------------------------
# _request_json — session integration and 403 re-prompt
# ---------------------------------------------------------------------------

def _mock_urlopen_response(payload: dict) -> MagicMock:
    body = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = body
    return mock_resp


def test_request_json_injects_session_headers() -> None:
    session = _session()
    captured: list[dict] = []

    def fake_urlopen(req, timeout, context):
        captured.append(dict(req.headers))
        return _mock_urlopen_response({"ok": True})

    with patch("agentic_docs.wiki_ingest.urlopen", side_effect=fake_urlopen):
        _request_json("https://example.com/api", session=session)

    assert captured[0].get("User-agent") == session.user_agent
    assert "cf_clearance=abc123" in captured[0].get("Cookie", "")


def test_request_json_reprompts_on_403() -> None:
    session = _session()
    call_count = 0

    def fake_urlopen(req, timeout, context):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise HTTPError(url="u", code=403, msg="Forbidden", hdrs={}, fp=None)
        return _mock_urlopen_response({"ok": True})

    with patch("agentic_docs.wiki_ingest.urlopen", side_effect=fake_urlopen), \
         patch("builtins.input", side_effect=["new-cookie", ""]):
        result = _request_json("https://example.com/api", session=session)

    assert result == {"ok": True}
    assert call_count == 2
    assert session.cookies == {"cf_clearance": "new-cookie"}


def test_request_json_reraises_non_403_http_errors() -> None:
    session = _session()

    def fake_urlopen(req, timeout, context):
        raise HTTPError(url="u", code=500, msg="Server Error", hdrs={}, fp=None)

    with patch("agentic_docs.wiki_ingest.urlopen", side_effect=fake_urlopen):
        try:
            _request_json("https://example.com/api", session=session)
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 500


# ---------------------------------------------------------------------------
# Redirect detection
# ---------------------------------------------------------------------------

def test_is_redirect_detects_redirect_pages() -> None:
    assert is_redirect("#REDIRECT [[Forum]]") is True
    assert is_redirect("#redirect [[Forum]]") is True
    assert is_redirect("This is a normal page about forums.") is False
    assert is_redirect("== Introduction ==\nContent here.") is False


# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

def test_clean_wikitext_strips_templates() -> None:
    result = clean_wikitext("Before {{SomeTemplate|arg1|arg2}} after.")
    assert "{{" not in result
    assert "Before" in result and "after" in result


def test_clean_wikitext_strips_nested_templates() -> None:
    result = clean_wikitext("Text {{outer|{{inner|value}}}} end.")
    assert "{{" not in result
    assert "Text" in result and "end" in result


def test_clean_wikitext_simplifies_wikilinks() -> None:
    result = clean_wikitext("See the [[Forum|Forum activity]] and [[Quiz]] pages.")
    assert "Forum activity" in result and "Quiz" in result
    assert "[[" not in result


def test_clean_wikitext_removes_file_links() -> None:
    result = clean_wikitext("Here is an image [[File:Forum.png|thumb|A forum screenshot]].")
    assert "[[File:" not in result


def test_clean_wikitext_simplifies_external_links() -> None:
    result = clean_wikitext("See [https://moodle.org/forums Moodle Forums] for help.")
    assert "Moodle Forums" in result and "https://" not in result


def test_clean_wikitext_strips_bold_italic_markers() -> None:
    result = clean_wikitext("This is '''bold''' and ''italic'' text.")
    assert "'''" not in result and "''" not in result
    assert "bold" in result and "italic" in result


def test_clean_wikitext_strips_ref_blocks() -> None:
    result = clean_wikitext("Before <ref>Citation text</ref> after.")
    assert "<ref>" not in result and "Citation text" not in result
    assert "Before" in result and "after" in result


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

def test_sections_from_wikitext_creates_root_section_for_intro() -> None:
    sections = sections_from_wikitext(
        "This is the introduction text before any heading.",
        page_title="Forum", document_id="doc-1",
    )
    assert len(sections) == 1
    assert sections[0].section_title == "Forum"
    assert sections[0].heading_path == ["Forum"]
    assert sections[0].heading_level == 0


def test_sections_from_wikitext_splits_on_headings() -> None:
    wikitext = "Intro.\n\n== Overview ==\nOverview text.\n\n=== Details ===\nDetails.\n\n== Settings ==\nSettings text."
    sections = sections_from_wikitext(wikitext, page_title="Forum", document_id="doc-1")
    titles = [s.section_title for s in sections]
    assert "Forum" in titles
    assert "Overview" in titles
    assert "Details" in titles
    assert "Settings" in titles


def test_sections_from_wikitext_builds_heading_path() -> None:
    wikitext = "== Overview ==\nSome content.\n\n=== Details ===\nMore content."
    sections = sections_from_wikitext(wikitext, page_title="Quiz", document_id="doc-2")
    detail = next(s for s in sections if s.section_title == "Details")
    assert detail.heading_path == ["Quiz", "Overview", "Details"]


def test_sections_from_wikitext_skips_empty_sections() -> None:
    sections = sections_from_wikitext(
        "== Empty ==\n\n== Non-empty ==\nActual content here.",
        page_title="Page", document_id="doc-3",
    )
    titles = [s.section_title for s in sections]
    assert "Empty" not in titles
    assert "Non-empty" in titles


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def test_wiki_page_to_document_sets_provenance() -> None:
    doc = wiki_page_to_document(
        page_title="Forum",
        wikitext="== Overview ==\nThe forum lets users post discussions.",
        ctx=_ctx(),
    )
    assert doc.title == "Forum"
    assert doc.metadata.source_type == "scraped_web"
    assert doc.metadata.source_name == "user_docs"
    assert doc.metadata.canonical_url == "https://docs.moodle.org/502/en/Forum"
    assert doc.metadata.source_path == "user_docs/forum.wiki"
    assert doc.metadata.scrape_timestamp == datetime(2026, 4, 27, tzinfo=timezone.utc)
    assert len(doc.sections) == 1


def test_wiki_page_to_document_stable_ids() -> None:
    wikitext = "Content for the page."
    doc1 = wiki_page_to_document(page_title="Forum", wikitext=wikitext, ctx=_ctx())
    doc2 = wiki_page_to_document(page_title="Forum", wikitext=wikitext, ctx=_ctx())
    assert doc1.id == doc2.id
    assert doc1.sections[0].id == doc2.sections[0].id
