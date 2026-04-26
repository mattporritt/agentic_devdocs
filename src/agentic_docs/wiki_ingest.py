# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

"""Bounded MediaWiki ingestion for the Moodle user documentation wiki."""

from __future__ import annotations

import json
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from agentic_docs.chunking import chunk_document
from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.storage import SQLiteStore
from agentic_docs.tokenizers import get_tokenizer
from agentic_docs.utils import sha1_text, stable_id

DEFAULT_BASE_URL = "https://docs.moodle.org/502/en"
SOURCE_NAME = "user_docs"
SOURCE_TYPE = "scraped_web"

_HEADING_PATTERN = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_REF_BLOCK = re.compile(r"<ref\b[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_HTML_TAGS = re.compile(r"<[^>]+>")
_EXTERNAL_LINK_WITH_TEXT = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_EXTERNAL_LINK_BARE = re.compile(r"\[https?://\S+\]")
_FORMATTING = re.compile(r"'{2,3}")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


@dataclass(slots=True)
class WikiScrapeContext:
    """Lightweight context captured once at the start of a wiki ingestion run."""

    base_url: str
    api_url: str
    scrape_timestamp: datetime


def wiki_api_url(base_url: str) -> str:
    return base_url.rstrip("/") + "/api.php"


def wiki_page_url(base_url: str, page_title: str) -> str:
    return base_url.rstrip("/") + "/" + quote(page_title.replace(" ", "_"), safe="/:@!$&'()*+,;=-._~")


def page_source_path(page_title: str) -> str:
    slug = _SLUG_STRIP.sub("-", page_title.strip().lower()).strip("-") or "page"
    return f"user_docs/{slug}.wiki"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _request_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    full_url = url
    if params:
        full_url = f"{url}?{urlencode(params)}"
    req = Request(full_url, headers={"User-Agent": "agentic-docs/0.1"})
    with urlopen(req, timeout=30, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_all_page_titles(api_url: str, max_pages: int | None = None) -> list[str]:
    """Enumerate main-namespace page titles via the MediaWiki allpages API."""

    titles: list[str] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "allpages",
        "aplimit": 500,
        "apnamespace": 0,
        "apfilterredir": "nonredirects",
        "format": "json",
    }
    while True:
        data = _request_json(api_url, params=params)
        for page in data.get("query", {}).get("allpages", []):
            titles.append(page["title"])
            if max_pages is not None and len(titles) >= max_pages:
                return titles
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
    return titles


def fetch_page_wikitext(api_url: str, page_title: str) -> str | None:
    """Fetch the raw wikitext for a single page. Returns None if the page is missing."""

    data = _request_json(api_url, params={
        "action": "parse",
        "page": page_title,
        "prop": "wikitext",
        "formatversion": "2",
        "format": "json",
    })
    error = data.get("error")
    if error:
        return None
    return data.get("parse", {}).get("wikitext")


def is_redirect(wikitext: str) -> bool:
    return bool(re.match(r"#REDIRECT", wikitext.strip(), re.IGNORECASE))


def _strip_templates(text: str) -> str:
    """Remove {{...}} template calls, handling nesting."""
    result: list[str] = []
    depth = 0
    i = 0
    while i < len(text):
        if text[i : i + 2] == "{{":
            depth += 1
            i += 2
        elif text[i : i + 2] == "}}" and depth > 0:
            depth -= 1
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    return "".join(result)


def _strip_tables(text: str) -> str:
    """Remove wiki table markup, preserving cell text."""
    lines: list[str] = []
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{|"):
            in_table = True
            continue
        if in_table:
            if stripped == "|}":
                in_table = False
            elif stripped.startswith("|") or stripped.startswith("!"):
                cell = re.sub(r"^[|!][^|]*\|", "", stripped).strip()
                if cell:
                    lines.append(cell)
            continue
        lines.append(line)
    return "\n".join(lines)


def _simplify_wikilinks(text: str) -> str:
    text = re.sub(r"\[\[(?:File|Image|Media|Category):[^\]]+\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    return text


def clean_wikitext(wikitext: str) -> str:
    """Convert raw wikitext to plain readable text for indexing."""

    text = _strip_templates(wikitext)
    text = _strip_tables(text)
    text = _REF_BLOCK.sub("", text)
    text = _HTML_TAGS.sub("", text)
    text = _simplify_wikilinks(text)
    text = _EXTERNAL_LINK_WITH_TEXT.sub(r"\1", text)
    text = _EXTERNAL_LINK_BARE.sub("", text)
    text = _FORMATTING.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def sections_from_wikitext(
    wikitext: str,
    *,
    page_title: str,
    document_id: str,
) -> list[SectionModel]:
    """Split cleaned wikitext into canonical sections on heading boundaries."""

    cleaned = clean_wikitext(wikitext)
    sections: list[SectionModel] = []
    heading_stack: list[tuple[int, str]] = []
    current_title: str | None = page_title
    current_level = 0
    current_path: list[str] = [page_title]
    order = 0

    def _flush(content: str) -> None:
        nonlocal order
        content = content.strip()
        if not content:
            return
        section_id = stable_id(
            "section",
            document_id,
            str(order),
            "/".join(current_path) if current_path else "__root__",
        )
        sections.append(
            SectionModel(
                id=section_id,
                document_id=document_id,
                section_order=order,
                section_title=current_title,
                heading_level=current_level,
                heading_path=list(current_path),
                content=content,
            )
        )
        order += 1

    last_end = 0
    for match in _HEADING_PATTERN.finditer(cleaned):
        pre = cleaned[last_end : match.start()]
        _flush(pre)
        heading_level = len(match.group(1))
        heading_text = match.group(2).strip()
        while heading_stack and heading_stack[-1][0] >= heading_level:
            heading_stack.pop()
        heading_stack.append((heading_level, heading_text))
        current_title = heading_text
        current_level = heading_level
        current_path = [page_title] + [text for _, text in heading_stack]
        last_end = match.end()

    _flush(cleaned[last_end:])
    return sections


def wiki_page_to_document(
    *,
    page_title: str,
    wikitext: str,
    ctx: WikiScrapeContext,
) -> DocumentModel:
    """Transform a fetched wiki page into the canonical document model."""

    source_path = page_source_path(page_title)
    canonical_url = wiki_page_url(ctx.base_url, page_title)
    document_id = stable_id("document", source_path)
    sections = sections_from_wikitext(wikitext, page_title=page_title, document_id=document_id)
    metadata = DocumentMetadata(
        source_path=source_path,
        source_type=SOURCE_TYPE,
        source_name=SOURCE_NAME,
        source_url=canonical_url,
        canonical_url=canonical_url,
        scrape_timestamp=ctx.scrape_timestamp,
        file_hash=sha1_text(wikitext),
        content_hash=sha1_text(wikitext),
    )
    return DocumentModel(id=document_id, title=page_title, metadata=metadata, sections=sections)


def fetch_wiki_documents(
    base_url: str,
    max_pages: int | None = None,
) -> tuple[WikiScrapeContext, list[DocumentModel]]:
    """Enumerate and fetch wiki pages, returning canonical document models."""

    api_url = wiki_api_url(base_url)
    ctx = WikiScrapeContext(
        base_url=base_url.rstrip("/"),
        api_url=api_url,
        scrape_timestamp=datetime.now(tz=timezone.utc),
    )
    titles = fetch_all_page_titles(api_url, max_pages=max_pages)
    documents: list[DocumentModel] = []
    for title in titles:
        wikitext = fetch_page_wikitext(api_url, title)
        if not wikitext or is_redirect(wikitext):
            continue
        doc = wiki_page_to_document(page_title=title, wikitext=wikitext, ctx=ctx)
        if doc.sections:
            documents.append(doc)
    return ctx, documents


def ingest_wiki_source(
    *,
    base_url: str,
    db_path,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
    max_pages: int | None = None,
) -> dict[str, int | str]:
    """Ingest the Moodle user documentation wiki into the shared SQLite schema."""

    tokenizer = get_tokenizer(tokenizer_name)
    ctx, documents = fetch_wiki_documents(base_url=base_url, max_pages=max_pages)
    store = SQLiteStore(db_path)
    store.initialize()
    store.reindex()

    document_count = 0
    section_count = 0
    chunk_count = 0
    for document in documents:
        chunks = chunk_document(
            document=document,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        store.store_document(document, chunks)
        document_count += 1
        section_count += len(document.sections)
        chunk_count += len(chunks)

    return {
        "documents": document_count,
        "sections": section_count,
        "chunks": chunk_count,
        "tokenizer": tokenizer.name(),
        "source_type": SOURCE_TYPE,
        "source_name": SOURCE_NAME,
        "base_url": ctx.base_url,
        "scrape_timestamp": ctx.scrape_timestamp.isoformat(),
    }
