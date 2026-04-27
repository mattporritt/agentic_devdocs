# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

"""Bounded MediaWiki ingestion for the Moodle user documentation wiki.

docs.moodle.org is behind Cloudflare's Managed Challenge, which requires a
real browser session cookie (cf_clearance) to access. There are two ways to
supply it:

1. Manual (recommended for developers):
   Open docs.moodle.org/502/en in your regular browser, then copy the
   cf_clearance cookie value from DevTools → Application → Cookies.
   Pass it via --cf-clearance or set MOODLE_DOCS_CF_CLEARANCE in the env.

2. Automatic (Playwright + bundled Firefox):
   pip install playwright && playwright install firefox
   The browser opens visibly, attempts to auto-solve the challenge, then
   prompts for human interaction if needed (or accepts a pasted cookie).

All MediaWiki API calls use plain urllib with the cookie injected — no
browser overhead during the actual scrape.
"""

from __future__ import annotations

import json
import random
import re
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError
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

# Refresh the Cloudflare session 5 minutes before cf_clearance's 30-min expiry.
_CF_REFRESH_AFTER_SECONDS = 1500

_HEADING_PATTERN = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$", re.MULTILINE)
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_REF_BLOCK = re.compile(r"<ref\b[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_HTML_TAGS = re.compile(r"<[^>]+>")
_EXTERNAL_LINK_WITH_TEXT = re.compile(r"\[https?://\S+\s+([^\]]+)\]")
_EXTERNAL_LINK_BARE = re.compile(r"\[https?://\S+\]")
_FORMATTING = re.compile(r"'{2,3}")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# Firefox UA used as the fallback when no Playwright browser is launched.
# Must match the browser family the cf_clearance cookie was issued for —
# Cloudflare binds clearance to User-Agent. We use Firefox everywhere
# (automated Playwright flow and manual cookie path) to keep this consistent.
_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:138.0) "
    "Gecko/20100101 Firefox/138.0"
)


# ---------------------------------------------------------------------------
# Cloudflare clearance
# ---------------------------------------------------------------------------

def _find_firefox() -> str | None:
    """Always returns None — Playwright's Firefox requires its own patched build.

    Unlike Chromium (which uses the standard CDP protocol and works with any
    system Chrome binary), Playwright's Firefox uses a custom juggler-pipe
    protocol that is only present in Playwright's bundled Firefox build.
    Passing the system Firefox binary causes an immediate launch failure.
    """
    return None


def _jitter_mouse(page: Any) -> None:
    """Move the mouse in a few randomised arcs to mimic human hand movement."""
    for _ in range(random.randint(3, 6)):
        page.mouse.move(
            random.randint(150, 1100),
            random.randint(100, 600),
        )
        page.wait_for_timeout(random.randint(40, 120))


def _try_click_turnstile(page: Any) -> None:
    """Click the Cloudflare Turnstile 'I am human' checkbox if it is visible.

    The checkbox lives inside a cross-origin iframe served from
    challenges.cloudflare.com. Playwright can interact with cross-origin
    frames directly; bounding_box() returns viewport-relative coordinates so
    page.mouse is used for the jittered click rather than element.click().
    """
    try:
        cf_frame = next(
            (f for f in page.frames if "challenges.cloudflare.com" in (f.url or "")),
            None,
        )
        if cf_frame is None:
            return
        checkbox = cf_frame.locator("input[type='checkbox']").first
        if not checkbox.is_visible(timeout=300):
            return
        box = checkbox.bounding_box()
        if not box:
            return
        cx = box["x"] + box["width"] / 2
        cy = box["y"] + box["height"] / 2
        # Approach from a nearby random position, pause, then click with a tiny offset.
        page.mouse.move(cx + random.uniform(-30, 30), cy + random.uniform(-30, 30))
        page.wait_for_timeout(random.randint(250, 500))
        page.mouse.move(cx + random.uniform(-3, 3), cy + random.uniform(-3, 3))
        page.wait_for_timeout(random.randint(80, 180))
        page.mouse.click(cx + random.uniform(-1, 1), cy + random.uniform(-1, 1))
    except Exception:
        pass  # Non-fatal — challenge may auto-solve or be a different variant


_CF_AUTO_SOLVE_SECONDS = 30


def _acquire_cf_clearance(url: str) -> tuple[dict[str, str], str]:
    """Open a visible browser, solve the Cloudflare challenge, return (cookies, user_agent).

    Phase 1 (auto): polls for up to _CF_AUTO_SOLVE_SECONDS, trying to click the
    Turnstile checkbox each tick.  The Cloudflare JS/fingerprint challenge often
    auto-resolves without any interaction; the Turnstile 'I am human' click covers
    the interactive variant.

    Phase 2 (human-in-the-loop): if still blocked after the auto phase, the browser
    window stays open and the user is prompted to solve the challenge manually, then
    press Enter.  This reliably handles the Cloudflare Managed Challenge that detects
    automation fingerprints.

    The browser closes as soon as cf_clearance is obtained; all subsequent HTTP
    requests use plain urllib with the captured cookie.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is required to acquire Cloudflare clearance.\n"
            "Install it with:\n"
            "  pip install playwright\n"
            "  playwright install firefox"
        )

    print("Acquiring Cloudflare clearance (Playwright bundled Firefox)...", flush=True)

    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=False,
            slow_mo=random.randint(20, 50),
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = context.new_page()

        # Jitter before navigation so the browser doesn't look like it jumped
        # straight to the target URL from a cold start.
        _jitter_mouse(page)

        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        # --- Phase 1: auto-solve ---
        for tick in range(_CF_AUTO_SOLVE_SECONDS):
            cookies = {c["name"]: c["value"] for c in context.cookies()}
            if "cf_clearance" in cookies:
                break
            if tick % 5 == 0:
                print(f"  auto-solving… ({tick}s, title={page.title()!r})", flush=True)
            _try_click_turnstile(page)
            page.wait_for_timeout(random.randint(900, 1100))

        cookies = {c["name"]: c["value"] for c in context.cookies()}

        # --- Phase 2: human-in-the-loop ---
        manual_cookie: str | None = None
        manual_ua: str = _DEFAULT_USER_AGENT
        if "cf_clearance" not in cookies:
            print(
                "\n[ACTION REQUIRED]\n"
                "The Cloudflare challenge requires human interaction.\n"
                "Choose an option:\n"
                "\n"
                "  1. Solve in the open browser window:\n"
                "     a. Click the 'I am human' / 'Verify you are human' checkbox.\n"
                "     b. Wait for the Moodle docs page to load.\n"
                "     c. Press Enter here to continue.\n"
                "\n"
                "  2. Provide the cf_clearance cookie from your regular browser:\n"
                "     a. Open the docs URL in Firefox (or any browser).\n"
                "     b. Open DevTools → Storage (Firefox) or Application (Chrome).\n"
                "     c. Select Cookies → https://docs.moodle.org.\n"
                "     d. Copy the VALUE of the 'cf_clearance' cookie.\n"
                "     e. Paste it here and press Enter  (the automated browser will close).\n"
                "\n"
                "Enter cookie value, or press Enter to use the open browser: ",
                end="",
                flush=True,
            )
            response = input().strip()
            if response:
                manual_cookie = response
                # cf_clearance is bound to the User-Agent it was issued for.
                # Ask for the browser's UA so urllib requests won't be rejected.
                print(
                    "Paste your browser's User-Agent and press Enter\n"
                    "(Firefox DevTools console: copy the output of  navigator.userAgent)\n"
                    "or press Enter to use the built-in Firefox UA: ",
                    end="",
                    flush=True,
                )
                manual_ua = input().strip() or _DEFAULT_USER_AGENT
            else:
                cookies = {c["name"]: c["value"] for c in context.cookies()}

        user_agent: str = page.evaluate("() => navigator.userAgent")
        browser.close()

    if manual_cookie:
        return {"cf_clearance": manual_cookie}, manual_ua

    if "cf_clearance" not in cookies:
        raise RuntimeError(
            "Cloudflare challenge was not resolved.\n\n"
            "Re-run and either solve the challenge in the browser window, or paste\n"
            "the cf_clearance cookie value when prompted.\n"
            "\n"
            "To extract cf_clearance from your regular browser:\n"
            "  Chrome/Edge : DevTools (F12) → Application → Cookies → docs.moodle.org\n"
            "  Firefox     : DevTools (F12) → Storage → Cookies → docs.moodle.org\n"
            "  Safari      : Develop → Show Web Inspector → Storage → Cookies\n"
            "Copy the VALUE of the 'cf_clearance' entry and pass it with:\n"
            "  --cf-clearance <value>   or   MOODLE_DOCS_CF_CLEARANCE=<value>"
        )

    print("Clearance acquired.", flush=True)
    return cookies, user_agent


@dataclass
class WikiSession:
    """Cloudflare clearance credentials for a single wiki scraping run.

    Holds the cf_clearance cookie and the matching User-Agent string (Cloudflare
    ties clearance to the UA it was issued for). Tracks acquisition time so
    urllib callers can refresh proactively before expiry.
    """

    cookies: dict[str, str]
    user_agent: str
    base_url: str
    acquired_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    @property
    def cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def needs_refresh(self) -> bool:
        age = (datetime.now(tz=timezone.utc) - self.acquired_at).total_seconds()
        return age > _CF_REFRESH_AFTER_SECONDS

    def refresh(self) -> None:
        print("Refreshing Cloudflare session...", flush=True)
        self.cookies, self.user_agent = _acquire_cf_clearance(self.base_url)
        self.acquired_at = datetime.now(tz=timezone.utc)

    @classmethod
    def acquire(cls, base_url: str) -> WikiSession:
        """Acquire a session via Playwright (for auto-solvable JS challenges)."""
        cookies, user_agent = _acquire_cf_clearance(base_url)
        return cls(cookies=cookies, user_agent=user_agent, base_url=base_url)

    @classmethod
    def from_cookie(
        cls,
        cf_clearance: str,
        base_url: str,
        user_agent: str | None = None,
    ) -> WikiSession:
        """Build a session from a cf_clearance cookie copied from a real browser.

        Pass user_agent matching the browser that issued the cookie — Cloudflare
        binds cf_clearance to the UA it was issued for.
        """
        return cls(
            cookies={"cf_clearance": cf_clearance},
            user_agent=user_agent or _DEFAULT_USER_AGENT,
            base_url=base_url,
        )


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    session: WikiSession | None = None,
) -> Any:
    """Fetch a JSON endpoint, using Cloudflare credentials when a session is provided.

    On a 403 response the session is refreshed once and the request retried.
    A proactive refresh is also triggered when the session is within 5 minutes
    of its expiry window.
    """
    full_url = url
    if params:
        full_url = f"{url}?{urlencode(params)}"

    def _build_headers() -> dict[str, str]:
        if session is None:
            return {"User-Agent": "agentic-docs/0.1"}
        headers: dict[str, str] = {"User-Agent": session.user_agent}
        if session.cookies:
            headers["Cookie"] = session.cookie_header
        return headers

    def _do_request() -> Any:
        req = Request(full_url, headers=_build_headers())
        with urlopen(req, timeout=30, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))

    if session is None:
        return _do_request()

    if session.needs_refresh():
        session.refresh()

    try:
        return _do_request()
    except HTTPError as exc:
        if exc.code == 403:
            session.refresh()
            return _do_request()
        raise


# ---------------------------------------------------------------------------
# MediaWiki API
# ---------------------------------------------------------------------------

def fetch_all_page_titles(
    api_url: str,
    max_pages: int | None = None,
    *,
    session: WikiSession | None = None,
) -> list[str]:
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
        data = _request_json(api_url, params=params, session=session)
        for page in data.get("query", {}).get("allpages", []):
            titles.append(page["title"])
            if max_pages is not None and len(titles) >= max_pages:
                return titles
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
    return titles


def fetch_page_wikitext(
    api_url: str,
    page_title: str,
    *,
    session: WikiSession | None = None,
) -> str | None:
    """Fetch the raw wikitext for a single page. Returns None if missing."""

    data = _request_json(
        api_url,
        params={
            "action": "parse",
            "page": page_title,
            "prop": "wikitext",
            "formatversion": "2",
            "format": "json",
        },
        session=session,
    )
    if data.get("error"):
        return None
    return data.get("parse", {}).get("wikitext")


# ---------------------------------------------------------------------------
# Wikitext cleaning and section parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

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
    cf_clearance: str | None = None,
    user_agent: str | None = None,
) -> tuple[WikiScrapeContext, list[DocumentModel]]:
    """Acquire a Cloudflare session, enumerate pages, and fetch each as a document model."""

    api_url = wiki_api_url(base_url)
    if cf_clearance:
        session = WikiSession.from_cookie(cf_clearance, base_url, user_agent=user_agent)
    else:
        session = WikiSession.acquire(base_url)
    ctx = WikiScrapeContext(
        base_url=base_url.rstrip("/"),
        api_url=api_url,
        scrape_timestamp=datetime.now(tz=timezone.utc),
    )

    titles = fetch_all_page_titles(api_url, max_pages=max_pages, session=session)
    total = len(titles)
    print(f"Fetching {total} pages from {base_url}...", flush=True)

    documents: list[DocumentModel] = []
    for i, title in enumerate(titles, 1):
        if i % 100 == 0 or i == total:
            print(f"  {i}/{total} pages processed", flush=True)
        wikitext = fetch_page_wikitext(api_url, title, session=session)
        if not wikitext or is_redirect(wikitext):
            continue
        doc = wiki_page_to_document(page_title=title, wikitext=wikitext, ctx=ctx)
        if doc.sections:
            documents.append(doc)

    return ctx, documents


# ---------------------------------------------------------------------------
# Ingestion entry point
# ---------------------------------------------------------------------------

def ingest_wiki_source(
    *,
    base_url: str,
    db_path: Any,
    tokenizer_name: str,
    max_tokens: int,
    overlap_tokens: int,
    max_pages: int | None = None,
    append: bool = False,
    cf_clearance: str | None = None,
    user_agent: str | None = None,
) -> dict[str, int | str]:
    """Ingest the Moodle user documentation wiki into the shared SQLite schema."""

    tokenizer = get_tokenizer(tokenizer_name)
    ctx, documents = fetch_wiki_documents(
        base_url=base_url, max_pages=max_pages, cf_clearance=cf_clearance, user_agent=user_agent
    )
    store = SQLiteStore(db_path)
    store.initialize()
    if not append:
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
