# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

"""Token-aware chunking for canonical sections."""

from __future__ import annotations

from dataclasses import dataclass
import re

from agentic_docs.models import ChunkMetadata, ChunkModel, DocumentModel, SectionModel
from agentic_docs.tokenizers import Tokenizer
from agentic_docs.utils import stable_id

LOW_SIGNAL_LINE_PATTERN = re.compile(r"^\s*</?[A-Z][A-Za-z0-9_.]*(?:\s[^>]*)?/?>\s*$")
WORD_PATTERN = re.compile(r"[A-Za-z]{3,}")


@dataclass(slots=True)
class Paragraph:
    text: str
    start_offset: int
    end_offset: int


def _split_paragraphs(text: str) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    start = 0
    cursor = 0
    for block in text.split("\n\n"):
        stripped = block.strip()
        segment_start = text.find(block, cursor)
        segment_end = segment_start + len(block)
        cursor = segment_end + 2
        if not stripped:
            continue
        paragraphs.append(Paragraph(text=stripped, start_offset=segment_start, end_offset=segment_end))
        start = segment_end
    if not paragraphs and text.strip():
        paragraphs.append(Paragraph(text=text.strip(), start_offset=0, end_offset=len(text)))
    return paragraphs


def _heading_prefix(section: SectionModel) -> str:
    if not section.heading_path:
        return ""
    breadcrumb = " > ".join(section.heading_path)
    return f"Heading: {breadcrumb}\n\n"


def _truncate_to_fit(text: str, available_tokens: int, tokenizer: Tokenizer) -> str:
    encoded = tokenizer.encode(text)
    if len(encoded) <= available_tokens:
        return text
    return tokenizer.decode(encoded[:available_tokens]).strip()


def chunk_document(
    document: DocumentModel,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkModel]:
    """Chunk a canonical document into retrieval-oriented token-aware chunks."""

    chunks: list[ChunkModel] = []
    for section in document.sections:
        chunks.extend(chunk_section(document, section, tokenizer, max_tokens, overlap_tokens))

    for index, chunk in enumerate(chunks):
        previous = chunks[index - 1].id if index > 0 else None
        following = chunks[index + 1].id if index + 1 < len(chunks) else None
        chunk.prev_chunk_id = previous
        chunk.next_chunk_id = following

    return chunks


def chunk_section(
    document: DocumentModel,
    section: SectionModel,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkModel]:
    """Chunk a single section while preserving heading context."""

    prefix = _heading_prefix(section)
    prefix_tokens = tokenizer.count_tokens(prefix)
    if prefix_tokens >= max_tokens:
        prefix = ""
        prefix_tokens = 0

    paragraphs = _split_paragraphs(section.content)
    if not paragraphs:
        return []

    results: list[ChunkModel] = []
    chunk_order = 0
    current_texts: list[str] = []
    current_start: int | None = None
    current_end: int | None = None
    overlap_text = ""

    for paragraph in paragraphs:
        paragraph_text = paragraph.text
        tentative_parts = [part for part in [overlap_text, *current_texts, paragraph_text] if part]
        tentative_body = "\n\n".join(tentative_parts)
        tentative_tokens = tokenizer.count_tokens(prefix + tentative_body)

        if tentative_tokens <= max_tokens:
            if current_start is None:
                current_start = paragraph.start_offset
            current_end = paragraph.end_offset
            current_texts.append(paragraph_text)
            continue

        if current_texts:
            chunk = _build_chunk(
                document=document,
                section=section,
                tokenizer=tokenizer,
                prefix=prefix,
                body="\n\n".join([part for part in [overlap_text, *current_texts] if part]),
                chunk_order=chunk_order,
                start_offset=current_start,
                end_offset=current_end,
            )
            if not _is_low_signal_chunk(chunk):
                results.append(chunk)
                chunk_order += 1
                overlap_text = _tail_overlap(chunk.content.removeprefix(prefix), overlap_tokens, tokenizer)
            else:
                overlap_text = ""
            current_texts = []
            current_start = None
            current_end = None

        paragraph_budget = max_tokens - prefix_tokens - tokenizer.count_tokens(overlap_text)
        paragraph_fit = _truncate_to_fit(paragraph_text, max(paragraph_budget, 1), tokenizer)
        current_texts = [paragraph_fit] if paragraph_fit else []
        current_start = paragraph.start_offset if paragraph_fit else None
        current_end = paragraph.end_offset if paragraph_fit else None

    if current_texts:
        chunk = _build_chunk(
            document=document,
            section=section,
            tokenizer=tokenizer,
            prefix=prefix,
            body="\n\n".join([part for part in [overlap_text, *current_texts] if part]),
            chunk_order=chunk_order,
            start_offset=current_start,
            end_offset=current_end,
        )
        if not _is_low_signal_chunk(chunk):
            results.append(chunk)

    return results


def _tail_overlap(text: str, overlap_tokens: int, tokenizer: Tokenizer) -> str:
    if overlap_tokens <= 0 or not text.strip():
        return ""
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[-overlap_tokens:]).strip()


def _build_chunk(
    document: DocumentModel,
    section: SectionModel,
    tokenizer: Tokenizer,
    prefix: str,
    body: str,
    chunk_order: int,
    start_offset: int | None,
    end_offset: int | None,
) -> ChunkModel:
    content = f"{prefix}{body}".strip()
    chunk_id = stable_id("chunk", section.id, str(chunk_order), content)
    metadata = ChunkMetadata(
        document_id=document.id,
        document_title=document.title,
        source_path=document.metadata.source_path,
        source_type=document.metadata.source_type,
        source_name=document.metadata.source_name,
        source_url=document.metadata.source_url,
        canonical_url=document.metadata.canonical_url,
        repo_commit_hash=document.metadata.repo_commit_hash,
        scrape_timestamp=document.metadata.scrape_timestamp,
        section_title=section.section_title,
        heading_path=section.heading_path,
    )
    return ChunkModel(
        id=chunk_id,
        section_id=section.id,
        chunk_order=chunk_order,
        content=content,
        token_count=tokenizer.count_tokens(content),
        start_offset=start_offset,
        end_offset=end_offset,
        metadata=metadata,
    )


def _is_low_signal_chunk(chunk: ChunkModel) -> bool:
    body = chunk.content
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return True
    if all(LOW_SIGNAL_LINE_PATTERN.match(line) for line in lines):
        return True
    word_count = len(WORD_PATTERN.findall(body))
    if chunk.token_count < 12 and word_count < 4:
        return True
    return False
