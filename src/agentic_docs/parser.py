"""Markdown discovery and canonical parsing."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from markdown_it import MarkdownIt

from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
from agentic_docs.utils import read_text, sha1_text, stable_id

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}


def discover_markdown_files(source: Path) -> list[Path]:
    """Discover markdown files beneath a source directory."""

    paths: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() not in MARKDOWN_SUFFIXES:
            continue
        paths.append(path)
    return sorted(paths)


def parse_markdown_document(path: Path, root: Path, repo_commit_hash: str | None = None) -> DocumentModel:
    """Parse markdown into a canonical document with heading-aware sections."""

    text = read_text(path)
    md = MarkdownIt()
    tokens = md.parse(text)
    lines = text.splitlines()
    relative_path = path.relative_to(root).as_posix()
    file_hash = sha1_text(text)
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
        line_ranges.append((0, len(lines), 0, None, []))
    else:
        first_heading_start = heading_entries[0][0]
        if first_heading_start > 0:
            line_ranges.append((0, first_heading_start, 0, None, []))

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
        content = "\n".join(lines[start_line:end_line]).strip()
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

    title = next((section.section_title for section in sections if section.section_title), path.stem)
    metadata = DocumentMetadata(
        source_path=relative_path,
        repo_commit_hash=repo_commit_hash,
        last_modified_time=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        file_hash=file_hash,
    )
    return DocumentModel(id=document_id, title=title, metadata=metadata, sections=sections)

