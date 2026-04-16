# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

from datetime import datetime, timezone
import json

from agentic_docs.site_ingest import (
    DesignSiteBootstrap,
    build_page_index,
    design_page_to_document,
    extract_meta_content,
    extract_window_json,
    page_source_path,
    should_include_design_page,
)


def _bootstrap() -> DesignSiteBootstrap:
    return DesignSiteBootstrap(
        base_url="https://design.moodle.com",
        styleguide_id=131542,
        share_id="98292f05f",
        viewer_token="viewer-token",
        csrf_token="csrf-token",
        styleguide_details={},
        user_info={},
        scrape_timestamp=datetime(2026, 4, 8, tzinfo=timezone.utc),
    )


def test_extract_window_json_and_meta_content() -> None:
    html = """
    <html>
      <head>
        <meta name="csrf-token" content="csrf-token-value" />
      </head>
      <body>
        <script>
          window.USER_INFO = {"token":"viewer-token","styleguideId":131542};
        </script>
      </body>
    </html>
    """

    payload = extract_window_json(html, "USER_INFO")

    assert payload["token"] == "viewer-token"
    assert extract_meta_content(html, "csrf-token") == "csrf-token-value"


def test_build_page_index_and_filter_cover_pages() -> None:
    styleguide_details = {
        "categories": [
            {
                "name": "Foundation",
                "navigation_id": 1,
                "pages": [
                    {"id": 10, "uid": "abc123", "name": "Colours", "visibility": "visible"},
                    {"id": 11, "uid": "def456", "name": "___cover", "visibility": None},
                ],
            }
        ]
    }

    page_index = build_page_index(styleguide_details)

    assert page_index[10]["category_name"] == "Foundation"
    assert should_include_design_page({"id": 10, "name": "Colours", "content_node": "{}", "introduction_node": None}, page_index[10]) is True
    assert should_include_design_page({"id": 11, "name": "___cover", "content_node": "{}", "introduction_node": None}, page_index[11]) is False


def test_design_page_to_document_extracts_intro_tabs_and_provenance() -> None:
    page = {
        "id": 8079523,
        "name": "For Developers",
        "introduction_node": json.dumps(
            {
                "type": "doc",
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Developer quickstart for the design system."}],
                    }
                ],
            }
        ),
        "content_node": json.dumps(
            {
                "tabs": {
                    "overview": {
                        "uid": "overview",
                        "name": "Overview",
                        "order": 1,
                        "contentNode": {
                            "type": "doc",
                            "content": [
                                {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": "Token Consumption"}]},
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "Use CSS tokens and the npm package."}],
                                },
                            ],
                        },
                    },
                    "npm": {
                        "uid": "npm",
                        "name": "NPM Package",
                        "order": 2,
                        "contentNode": {
                            "type": "doc",
                            "content": [
                                {"type": "heading", "attrs": {"level": 2}, "content": [{"type": "text", "text": "Install"}]},
                                {
                                    "type": "paragraph",
                                    "content": [
                                        {"type": "text", "text": "Install from "},
                                        {
                                            "type": "text",
                                            "text": "@moodle/design-tokens",
                                            "marks": [{"type": "code", "attrs": {}}],
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                }
            }
        ),
    }
    page_info = {"uid": "98e3cb", "page_name": "For Developers", "category_name": "Get started"}

    document = design_page_to_document(page=page, page_info=page_info, bootstrap=_bootstrap())

    assert document.title == "For Developers"
    assert document.metadata.source_type == "scraped_web"
    assert document.metadata.source_name == "design_system"
    assert document.metadata.canonical_url == "https://design.moodle.com/98292f05f/p/98e3cb"
    assert document.metadata.source_path == page_source_path("Get started", "For Developers", "98e3cb")
    assert [section.section_title for section in document.sections] == [
        "For Developers",
        "Token Consumption",
        "Install",
    ]
    assert document.sections[1].heading_path == ["For Developers", "Token Consumption"]
    assert document.sections[2].heading_path == ["For Developers", "NPM Package", "Install"]
    assert "`@moodle/design-tokens`" in document.sections[2].content
