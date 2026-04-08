"""Markdown discovery and canonical parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from markdown_it import MarkdownIt
import yaml

from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.utils import read_text, sha1_text, stable_id

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
COMMENT_LINE_PATTERN = re.compile(r"^\s*<!--.*-->\s*$")
IMPORT_EXPORT_PATTERN = re.compile(r"^\s*(import|export)\s+")
JSX_WRAPPER_PATTERN = re.compile(r"^\s*</?[A-Z][A-Za-z0-9_.]*(?:\s[^>]*)?/?>\s*$")
DETAILS_SUMMARY_PATTERN = re.compile(r"^\s*</?(details|summary)[^>]*>\s*$", re.IGNORECASE)
NOISE_ONLY_PATTERN = re.compile(r"^[\s`~:#*_\-<>{}\[\]()/.,|=+\"']+$")


def discover_markdown_files(source: Path) -> list[Path]:
    """Discover markdown files beneath a source directory."""

    paths: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts[:-1]):
            continue
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        paths.append(path)
    return sorted(paths)


def _extract_front_matter(text: str) -> tuple[dict[str, object], str]:
    """Extract YAML front matter when present and return metadata plus body."""

    if not text.startswith("---\n"):
        return {}, text

    lines = text.splitlines()
    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break
    if closing_index is None:
        return {}, text

    front_matter_text = "\n".join(lines[1:closing_index])
    body = "\n".join(lines[closing_index + 1 :])
    try:
        loaded = yaml.safe_load(front_matter_text) or {}
    except yaml.YAMLError:
        loaded = {}
    metadata = loaded if isinstance(loaded, dict) else {}
    return metadata, body


def _clean_section_content(text: str) -> str:
    """Remove low-value MDX and editorial wrapper lines while preserving prose."""

    cleaned_lines: list[str] = []
    in_code_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)
            continue
        if in_code_block:
            cleaned_lines.append(line)
            continue
        if not stripped:
            cleaned_lines.append("")
            continue
        if COMMENT_LINE_PATTERN.match(stripped):
            continue
        if IMPORT_EXPORT_PATTERN.match(stripped):
            continue
        if JSX_WRAPPER_PATTERN.match(stripped):
            continue
        if DETAILS_SUMMARY_PATTERN.match(stripped):
            continue
        if NOISE_ONLY_PATTERN.match(stripped):
            continue
        cleaned_lines.append(line)

    normalized: list[str] = []
    previous_blank = False
    for line in cleaned_lines:
        if not line.strip():
            if previous_blank:
                continue
            previous_blank = True
            normalized.append("")
            continue
        previous_blank = False
        normalized.append(line.rstrip())
    return "\n".join(normalized).strip()


def parse_markdown_document(path: Path, root: Path, repo_commit_hash: str | None = None) -> DocumentModel:
    """Parse markdown into a canonical document with heading-aware sections."""

    raw_text = read_text(path)
    front_matter, text = _extract_front_matter(raw_text)
    md = MarkdownIt()
    tokens = md.parse(text)
    lines = text.splitlines()
    relative_path = path.relative_to(root).as_posix()
    file_hash = sha1_text(raw_text)
    document_id = stable_id("document", relative_path)

    heading_entries: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        if token.map is None:
            continue
        inline_token = tokens[index + 1]
        start_line = token.map[0]
        heading_level = int(token.tag.removeprefix("h"))
        heading_entries.append((start_line, heading_level, inline_token.content.strip()))

    sections: list[SectionModel] = []
    line_ranges: list[tuple[int, int, int, str | None, list[str]]] = []
    if not heading_entries:
        default_title = str(front_matter.get("title")) if front_matter.get("title") else None
        default_path = [default_title] if default_title else []
        line_ranges.append((0, len(lines), 0, default_title, default_path))
    else:
        first_heading_start = heading_entries[0][0]
        if first_heading_start > 0:
            preamble_title = str(front_matter.get("title")) if front_matter.get("title") else None
            preamble_path = [preamble_title] if preamble_title else []
            line_ranges.append((0, first_heading_start, 0, preamble_title, preamble_path))

        heading_stack: list[tuple[int, str]] = []
        for idx, (start_line, heading_level, heading_text) in enumerate(heading_entries):
            end_line = heading_entries[idx + 1][0] if idx + 1 < len(heading_entries) else len(lines)
            while heading_stack and heading_stack[-1][0] >= heading_level:
                heading_stack.pop()
            heading_stack.append((heading_level, heading_text))
            heading_path = [title for _, title in heading_stack]
            body_start = min(start_line + 1, end_line)
            line_ranges.append((body_start, end_line, heading_level, heading_text, heading_path))

    for order, (start_line, end_line, heading_level, section_title, heading_path) in enumerate(line_ranges):
        content = _clean_section_content("\n".join(lines[start_line:end_line]))
        if not content:
            continue
        section_id = stable_id(
            "section",
            document_id,
            str(order),
            "/".join(heading_path) if heading_path else "__root__",
        )
        sections.append(
            SectionModel(
                id=section_id,
                document_id=document_id,
                section_order=order,
                section_title=section_title,
                heading_level=heading_level,
                heading_path=heading_path,
                content=content,
            )
        )

    front_matter_title = str(front_matter.get("title")) if front_matter.get("title") else None
    first_heading_title = heading_entries[0][2] if heading_entries else None
    title = front_matter_title or first_heading_title or next(
        (section.section_title for section in sections if section.section_title),
        path.stem,
    )
    metadata = DocumentMetadata(
        source_path=relative_path,
        source_type="repo_markdown",
        source_name="devdocs_repo",
        repo_commit_hash=repo_commit_hash,
        last_modified_time=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        file_hash=file_hash,
        content_hash=file_hash,
    )
    return DocumentModel(id=document_id, title=title, metadata=metadata, sections=sections)
