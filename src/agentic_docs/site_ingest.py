"""Bounded live-site ingestion for the Moodle Design System docs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import ssl
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from agentic_docs.chunking import chunk_document
from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.storage import SQLiteStore
from agentic_docs.tokenizers import get_tokenizer
from agentic_docs.utils import sha1_text, stable_id

WINDOW_JSON_PATTERN = r"window\.{name}\s*=\s*(\{{.*?\}});"
SLUG_PART_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class DesignSiteBootstrap:
    """Runtime bootstrap needed to fetch shared design-system pages."""

    base_url: str
    styleguide_id: int
    share_id: str
    viewer_token: str
    csrf_token: str
    styleguide_details: dict[str, Any]
    user_info: dict[str, Any]
    scrape_timestamp: datetime


def extract_window_json(html: str, name: str) -> dict[str, Any]:
    """Extract a `window.NAME = {...};` JSON payload from a shell page."""

    match = re.search(WINDOW_JSON_PATTERN.format(name=re.escape(name)), html, re.S)
    if match is None:
        raise ValueError(f"Could not find window.{name} payload in site HTML")
    return json.loads(match.group(1))


def extract_meta_content(html: str, meta_name: str) -> str:
    """Extract a meta tag content value by name."""

    match = re.search(
        rf'<meta\s+name="{re.escape(meta_name)}"\s+content="([^"]+)"',
        html,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not find meta tag {meta_name}")
    return match.group(1)


def slugify(text: str) -> str:
    """Create a stable readable slug."""

    lowered = text.strip().lower()
    collapsed = SLUG_PART_PATTERN.sub("-", lowered).strip("-")
    return collapsed or "page"


def normalize_base_url(base_url: str) -> str:
    """Normalize the base URL used for the live design-system site."""

    return base_url.rstrip("/")


def page_source_path(category_name: str | None, page_name: str, page_uid: str) -> str:
    """Build a deterministic source path for a scraped site page."""

    category_slug = slugify(category_name or "site")
    page_slug = slugify(page_name)
    return f"design_system/{category_slug}/{page_slug}-{page_uid}.site"


def page_url(base_url: str, share_id: str, page_uid: str) -> str:
    """Build the public page URL for a shared design-system page."""

    return f"{normalize_base_url(base_url)}/{share_id}/p/{page_uid}"


def _request_text(url: str, *, headers: dict[str, str] | None = None, form_data: dict[str, Any] | None = None) -> str:
    """Fetch a URL and return decoded UTF-8 text."""

    data = None
    request_headers = {"User-Agent": "agentic-docs/0.1"}
    if headers:
        request_headers.update(headers)
    if form_data is not None:
        encoded = urlencode(form_data).encode("utf-8")
        data = encoded
        request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
    request = Request(url, data=data, headers=request_headers)
    context = _ssl_context()
    with urlopen(request, timeout=30, context=context) as response:
        return response.read().decode("utf-8")


def _ssl_context() -> ssl.SSLContext:
    """Build an HTTPS context that works in local dev and CI environments."""

    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_design_site_bootstrap(base_url: str) -> DesignSiteBootstrap:
    """Fetch the live design-system shell page and extract shared-view bootstrap data."""

    normalized_base = normalize_base_url(base_url)
    html = _request_text(normalized_base)
    styleguide_details = extract_window_json(html, "styleguideDetails")
    user_info = extract_window_json(html, "USER_INFO")
    csrf_token = extract_meta_content(html, "csrf-token")
    return DesignSiteBootstrap(
        base_url=normalized_base,
        styleguide_id=int(styleguide_details["id"]),
        share_id=str(styleguide_details["share_id"]),
        viewer_token=str(user_info["token"]),
        csrf_token=csrf_token,
        styleguide_details=styleguide_details,
        user_info=user_info,
        scrape_timestamp=datetime.now(tz=timezone.utc),
    )


def _shared_headers(bootstrap: DesignSiteBootstrap) -> dict[str, str]:
    return {
        "Authorization": f'Token token="{bootstrap.viewer_token}"',
        "X-CSRF-Token": bootstrap.csrf_token,
        "X-Requested-With": "XMLHttpRequest",
    }


def fetch_design_site_pages(bootstrap: DesignSiteBootstrap) -> list[dict[str, Any]]:
    """Fetch page bodies for the shared design-system site."""

    payload_text = _request_text(
        f"{bootstrap.base_url}/api/styleguide/load_pages",
        headers=_shared_headers(bootstrap),
        form_data={"id": bootstrap.styleguide_id, "share_link": "true"},
    )
    payload = json.loads(payload_text)
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Design site load_pages payload did not contain a page list")
    return pages


def build_page_index(styleguide_details: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Index page metadata from the shell navigation tree."""

    page_index: dict[int, dict[str, Any]] = {}
    for category in styleguide_details.get("categories") or []:
        category_name = category.get("name")
        navigation_id = category.get("navigation_id")
        for page in category.get("pages") or []:
            page_index[int(page["id"])] = {
                "uid": page.get("uid"),
                "page_name": page.get("name"),
                "category_name": category_name,
                "navigation_id": navigation_id,
                "page_visibility": page.get("visibility"),
                "category_overview_page": page.get("category_overview_page"),
            }
    return page_index


def should_include_design_page(page: dict[str, Any], page_info: dict[str, Any] | None) -> bool:
    """Filter out clearly non-content or hidden pages from the live site payload."""

    page_name = str(page.get("name") or "")
    if not page_name or page_name.startswith("___"):
        return False
    page_visibility = page.get("page_visibility") or (page_info or {}).get("page_visibility")
    if page_visibility == "hidden":
        return False
    return bool(page.get("content_node") or page.get("introduction_node"))


def _parse_json_node(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return json.loads(text)
    raise TypeError(f"Unsupported node payload type: {type(raw)!r}")


def _render_inline_text(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "text":
        text = node.get("text", "")
        for mark in node.get("marks") or []:
            mark_type = mark.get("type")
            attrs = mark.get("attrs") or {}
            if mark_type == "code":
                text = f"`{text}`"
            elif mark_type == "link":
                href = attrs.get("href")
                if href and href != text:
                    text = f"{text} ({href})"
        return text
    if node_type == "hardBreak":
        return "\n"
    return "".join(_render_inline_text(child) for child in node.get("content") or [])


def _render_table(node: dict[str, Any]) -> str:
    rows: list[str] = []
    for row in node.get("content") or []:
        cells: list[str] = []
        for cell in row.get("content") or []:
            cell_parts = [_render_block(child) for child in cell.get("content") or []]
            cell_text = " ".join(part.strip() for part in cell_parts if part.strip())
            cells.append(cell_text)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(row for row in rows if row.strip())


def _render_list(node: dict[str, Any], ordered: bool) -> str:
    items: list[str] = []
    for index, item in enumerate(node.get("content") or [], start=1):
        item_parts = [_render_block(child) for child in item.get("content") or []]
        item_text = " ".join(part.strip() for part in item_parts if part.strip())
        if not item_text:
            continue
        prefix = f"{index}. " if ordered else "- "
        items.append(f"{prefix}{item_text}")
    return "\n".join(items)


def _render_block(node: dict[str, Any]) -> str:
    node_type = node.get("type")
    if node_type == "paragraph":
        return _render_inline_text(node).strip()
    if node_type == "bulletList":
        return _render_list(node, ordered=False)
    if node_type == "orderedList":
        return _render_list(node, ordered=True)
    if node_type == "blockquote":
        inner = "\n".join(
            part for part in (_render_block(child) for child in node.get("content") or []) if part.strip()
        )
        return "\n".join(f"> {line}" for line in inner.splitlines())
    if node_type == "table":
        return _render_table(node)
    if node_type == "image":
        attrs = node.get("attrs") or {}
        descriptor = attrs.get("altText") or attrs.get("caption") or ""
        return f"Image: {descriptor}".strip() if descriptor else ""
    if node_type == "codeBlock":
        code_text = "".join(_render_inline_text(child) for child in node.get("content") or []).strip()
        return f"```\n{code_text}\n```" if code_text else ""
    if node_type == "tokensManagement":
        tokens = [token.get("path") for token in (node.get("attrs") or {}).get("tokens") or [] if token.get("path")]
        return f"Tokens: {', '.join(tokens)}" if tokens else ""
    if node_type == "shortcut-tiles":
        lines: list[str] = []
        for tile in (node.get("attrs") or {}).get("shortcutTiles") or []:
            title = tile.get("title")
            description = tile.get("description")
            link = tile.get("link")
            parts = [part for part in [title, description, link] if part]
            if parts:
                lines.append(" - ".join(parts))
        return "\n".join(lines)
    if node_type in {"horizontalRule", "rule"}:
        return ""
    if node_type == "heading":
        return _render_inline_text(node).strip()
    child_parts = [_render_block(child) for child in node.get("content") or []]
    return "\n\n".join(part for part in child_parts if part.strip())


def _sections_from_prosemirror_doc(
    *,
    document_id: str,
    root_node: dict[str, Any] | None,
    order_start: int,
    base_heading_path: list[str],
    default_section_title: str | None,
) -> tuple[list[SectionModel], int]:
    """Convert a ProseMirror doc node into canonical sections."""

    if not root_node or root_node.get("type") != "doc":
        return [], order_start

    sections: list[SectionModel] = []
    heading_stack: list[tuple[int, str]] = []
    current_title = default_section_title
    current_level = 0
    current_parts: list[str] = []
    current_path = list(base_heading_path)
    current_order = order_start

    def flush_current() -> None:
        nonlocal current_parts, current_order
        content = "\n\n".join(part for part in current_parts if part.strip()).strip()
        if not content:
            current_parts = []
            return
        section_path = list(current_path)
        section_id = stable_id(
            "section",
            document_id,
            str(current_order),
            "/".join(section_path) if section_path else "__root__",
        )
        sections.append(
            SectionModel(
                id=section_id,
                document_id=document_id,
                section_order=current_order,
                section_title=current_title,
                heading_level=current_level,
                heading_path=section_path,
                content=content,
            )
        )
        current_order += 1
        current_parts = []

    for child in root_node.get("content") or []:
        if child.get("type") == "heading":
            flush_current()
            heading_text = _render_inline_text(child).strip()
            heading_level = int((child.get("attrs") or {}).get("level") or 0)
            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()
            heading_stack.append((heading_level, heading_text))
            current_title = heading_text
            current_level = heading_level
            current_path = list(base_heading_path) + [text for _, text in heading_stack]
            continue
        rendered = _render_block(child).strip()
        if rendered:
            current_parts.append(rendered)

    flush_current()
    return sections, current_order


def design_page_to_document(
    *,
    page: dict[str, Any],
    page_info: dict[str, Any],
    bootstrap: DesignSiteBootstrap,
) -> DocumentModel:
    """Transform a fetched design-system page into the canonical document schema."""

    page_name = str(page.get("name") or page_info.get("page_name") or "Untitled")
    page_uid = str(page_info["uid"])
    category_name = page_info.get("category_name")
    source_path = page_source_path(category_name, page_name, page_uid)
    public_url = page_url(bootstrap.base_url, bootstrap.share_id, page_uid)
    intro_node = _parse_json_node(page.get("introduction_node"))
    content_node = _parse_json_node(page.get("content_node"))
    raw_payload = json.dumps(
        {
            "page_id": page.get("id"),
            "page_name": page_name,
            "introduction_node": intro_node,
            "content_node": content_node,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    document_id = stable_id("document", source_path)
    sections: list[SectionModel] = []
    order = 0

    intro_sections, order = _sections_from_prosemirror_doc(
        document_id=document_id,
        root_node=intro_node,
        order_start=order,
        base_heading_path=[page_name],
        default_section_title=page_name,
    )
    sections.extend(intro_sections)

    if content_node:
        tabs = content_node.get("tabs") or {}
        if tabs:
            ordered_tabs = sorted(
                tabs.values(),
                key=lambda tab: (tab.get("order") is None, tab.get("order", 0), tab.get("name", "")),
            )
            for tab in ordered_tabs:
                tab_name = str(tab.get("name") or "Tab")
                include_tab_in_path = len(ordered_tabs) > 1 and tab_name.lower() != "overview"
                base_path = [page_name]
                if include_tab_in_path:
                    base_path.append(tab_name)
                tab_sections, order = _sections_from_prosemirror_doc(
                    document_id=document_id,
                    root_node=tab.get("contentNode"),
                    order_start=order,
                    base_heading_path=base_path,
                    default_section_title=tab_name if include_tab_in_path else page_name,
                )
                sections.extend(tab_sections)
        else:
            body_sections, order = _sections_from_prosemirror_doc(
                document_id=document_id,
                root_node=content_node,
                order_start=order,
                base_heading_path=[page_name],
                default_section_title=page_name,
            )
            sections.extend(body_sections)

    metadata = DocumentMetadata(
        source_path=source_path,
        source_type="scraped_web",
        source_name="design_system",
        source_url=public_url,
        canonical_url=public_url,
        scrape_timestamp=bootstrap.scrape_timestamp,
        file_hash=sha1_text(raw_payload),
        content_hash=sha1_text(raw_payload),
    )
    return DocumentModel(id=document_id, title=page_name, metadata=metadata, sections=sections)


def fetch_design_site_documents(base_url: str, max_pages: int | None = None) -> tuple[DesignSiteBootstrap, list[DocumentModel]]:
    """Fetch and canonicalize the live design-system site into documents."""

    bootstrap = fetch_design_site_bootstrap(base_url)
    pages = fetch_design_site_pages(bootstrap)
    page_index = build_page_index(bootstrap.styleguide_details)
    documents: list[DocumentModel] = []
    for page in pages:
        page_id = int(page["id"])
        info = page_index.get(page_id)
        if info is None or not should_include_design_page(page, info):
            continue
        document = design_page_to_document(page=page, page_info=info, bootstrap=bootstrap)
        if not document.sections:
            continue
        documents.append(document)
        if max_pages is not None and len(documents) >= max_pages:
            break
    return bootstrap, documents


def ingest_site_source(
    *,
    base_url: str,
    db_path: Path,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
    max_pages: int | None = None,
) -> dict[str, int | str]:
    """Ingest the live design-system site into the shared SQLite schema."""

    tokenizer = get_tokenizer(tokenizer_name)
    bootstrap, documents = fetch_design_site_documents(base_url=base_url, max_pages=max_pages)
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
        "source_type": "scraped_web",
        "source_name": "design_system",
        "base_url": bootstrap.base_url,
        "scrape_timestamp": bootstrap.scrape_timestamp.isoformat(),
        "styleguide_id": str(bootstrap.styleguide_id),
        "share_id": bootstrap.share_id,
    }
