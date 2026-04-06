"""Retrieval-facing query helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from agentic_docs.models import ContextBundle, ContextBundleChunk, QueryResult
from agentic_docs.storage import SQLiteStore
from agentic_docs.tokenizers import get_tokenizer


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "work",
    "works",
}
TOKEN_ALIASES = {
    "add": ["adding"],
    "define": ["definition", "writing", "register"],
    "language": ["lang"],
    "strings": ["string"],
    "services": ["service", "external"],
    "tasks": ["task"],
    "tests": ["test"],
    "forms": ["form"],
    "plugins": ["plugin"],
    "write": ["writing"],
}
HEADING_PREFIX_PATTERN = re.compile(r"^Heading:\s.*?(?:\n\n|\n)", re.DOTALL)
GENERIC_PATH_SUFFIXES = {"docs/apis.md", "index.md"}
CONCEPTUAL_QUERY_PATTERN = re.compile(
    r"\b(how do|how does|what is|what are|define|write|add|register|work|works)\b"
)


@dataclass(frozen=True, slots=True)
class QueryProfile:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    expanded_tokens: list[str]
    intent: str
    concept_families: list[str]


def canonical_path_key(path: str) -> str:
    """Return a canonicalized path key with versioned docs folded into their base path."""

    normalized = path.strip().lstrip("./")
    match = re.match(r"versioned_docs/version-[^/]+/(.+)", normalized)
    if match:
        return match.group(1)
    return normalized


def is_canonical_path(path: str) -> bool:
    """Return whether a path is from the non-versioned canonical corpus."""

    normalized = path.strip().lstrip("./")
    return not normalized.startswith("versioned_docs/")


def normalize_query_text(query_text: str) -> tuple[str, list[str]]:
    """Normalize a developer query into FTS-friendly tokens."""

    raw_tokens = [token.lower() for token in TOKEN_PATTERN.findall(query_text)]
    filtered_tokens = [token for token in raw_tokens if token not in STOPWORDS]
    source_tokens = filtered_tokens or raw_tokens
    tokens: list[str] = []
    for token in source_tokens:
        if token not in tokens:
            tokens.append(token)
    normalized = " ".join(tokens) if tokens else query_text.strip()
    return normalized, tokens


def classify_query_intent(query_text: str, tokens: list[str]) -> str:
    """Classify the query into a small explicit intent set for reranking."""

    lowered = query_text.lower()
    if CONCEPTUAL_QUERY_PATTERN.search(lowered):
        return "conceptual"
    if any(token in {"api", "guide", "docs", "documentation"} for token in tokens):
        return "conceptual"
    return "keyword"


def _expanded_tokens(tokens: list[str]) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        if token not in expanded:
            expanded.append(token)
        if token.endswith("s") and len(token) > 3:
            singular = token[:-1]
            if singular not in expanded:
                expanded.append(singular)
        for alias in TOKEN_ALIASES.get(token, []):
            if alias not in expanded:
                expanded.append(alias)
    return expanded


def _concept_families(tokens: list[str]) -> list[str]:
    expanded = set(_expanded_tokens(tokens))
    families: list[str] = []
    if "upgrade" in expanded or "upgrade.php" in expanded:
        families.append("upgrade")
    if {"language", "lang"} & expanded and {"strings", "string"} & expanded:
        families.append("language_strings")
    if {"mustache", "template", "templates", "output", "renderer", "renderers"} & expanded:
        families.append("output_templates")
    if {"events", "event"} & expanded:
        families.append("events")
    return families


def build_query_profile(query_text: str) -> QueryProfile:
    """Build a reusable query profile for reranking and diagnostics."""

    normalized_query, tokens = normalize_query_text(query_text)
    expanded_tokens = _expanded_tokens(tokens)
    return QueryProfile(
        raw_query=query_text,
        normalized_query=normalized_query,
        tokens=tokens,
        expanded_tokens=expanded_tokens,
        intent=classify_query_intent(query_text, tokens),
        concept_families=_concept_families(tokens),
    )


def _build_fts_queries(query_text: str) -> list[str]:
    normalized, tokens = normalize_query_text(query_text)
    if not tokens:
        return [f'"{query_text.strip()}"'] if query_text.strip() else []

    and_query = " ".join(f'"{token}"' for token in tokens)
    if len(tokens) == 1:
        return [and_query]
    expanded = _expanded_tokens(tokens)
    or_query = " OR ".join(f'"{token}"' for token in expanded)
    phrase_query = f"\"{normalized}\""
    return [and_query, phrase_query, or_query]


def _context_hits(result: QueryResult, tokens: list[str]) -> int:
    fields = [
        result.source_file_path.lower(),
        result.document_title.lower(),
        (result.section_title or "").lower(),
        " > ".join(result.heading_path).lower(),
    ]
    return sum(1 for token in tokens if any(token in field for field in fields))


def _field_overlap(text: str, tokens: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for token in tokens if token in lowered)


def _query_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    if len(tokens) < 2:
        return phrases
    for size in (3, 2):
        if len(tokens) < size:
            continue
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start : start + size])
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _near_exact_match(text: str, normalized_query: str) -> bool:
    if not normalized_query:
        return False
    return normalized_query in normalize_query_text(text)[0]


def _chunk_quality_adjustment(result: QueryResult) -> float:
    if result.token_count < 20:
        return -2.5
    if result.token_count < 40:
        return -1.0
    if result.token_count <= 240:
        return 1.5
    if result.token_count <= 360:
        return 0.5
    return -1.5


def _subsystem_bonus(result: QueryResult, tokens: list[str]) -> float:
    path = result.source_file_path.lower()
    if "/subsystems/" not in path:
        return 0.0
    if any(token in path for token in tokens):
        return 6.0
    return 2.0


def _concept_page_bonus(result: QueryResult, profile: QueryProfile) -> float:
    if profile.intent != "conceptual":
        return 0.0

    path = result.source_file_path.lower()
    if any(segment in path for segment in ("/subsystems/", "/guides/", "/_files/", "/commonfiles/")):
        return 5.0
    if path == "docs/apis.md":
        return 2.0
    return 0.0


def _plugin_type_penalty(result: QueryResult, profile: QueryProfile) -> float:
    if profile.intent != "conceptual":
        return 0.0

    path = result.source_file_path.lower()
    if "/plugintypes/" in path and "/_files/" not in path:
        return -6.0
    return 0.0


def _family_specific_bonus(result: QueryResult, profile: QueryProfile) -> float:
    if not profile.concept_families:
        return 0.0

    path = result.source_file_path.lower()
    title = result.document_title.lower()
    heading = " > ".join(result.heading_path).lower()
    bonus = 0.0

    if "upgrade" in profile.concept_families:
        if any(fragment in path for fragment in ("/guides/upgrade/", "/_files/db-upgrade", "/_files/upgrade-php")):
            bonus += 14.0
        if "upgrade helpers" in heading or "upgrade.php" in heading:
            bonus += 4.0
        if "/plugintypes/" in path:
            bonus -= 8.0

    if "language_strings" in profile.concept_families:
        if path.endswith("docs/apis/_files/lang.md") or "/_files/lang.md" in path:
            bonus += 12.0
        if "language files" in heading or "language files" in title:
            bonus += 4.0
        if "/plugintypes/" in path:
            bonus -= 5.0

    if "output_templates" in profile.concept_families:
        if "/guides/templates/" in path:
            bonus += 18.0
        elif "/subsystems/output/" in path:
            bonus += 10.0
        if title == "templates":
            bonus += 6.0
        elif title == "output api":
            bonus += 2.0
        if "templates" in heading or "mustache" in heading:
            bonus += 6.0
        elif "renderers" in heading:
            bonus += 4.0
        elif "renderable" in heading:
            bonus += 4.0
        if "/plugintypes/format/" in path:
            bonus -= 8.0

    if "events" in profile.concept_families:
        if any(fragment in path for fragment in ("/subsystems/events/", "/_files/db-events")):
            bonus += 18.0
        if "events api" in heading or "events api" in title:
            bonus += 6.0
        if path == "docs/apis.md":
            bonus += 1.0
        if "/_files/db-events" in path and any(token in profile.expanded_tokens for token in ("plugin", "plugins")):
            bonus += 4.0
        if any(fragment in path for fragment in ("/plugintypes/", "/calendar/", "/xapi/")):
            bonus -= 8.0

    return bonus


def _example_penalty(result: QueryResult) -> float:
    path = result.source_file_path.lower()
    heading = " > ".join(result.heading_path).lower()
    section_title = (result.section_title or "").lower()
    if "/_examples/" in path or "example" in path:
        return -6.0
    if heading.startswith("examples") or " > examples" in heading:
        return -6.0
    if "example" in section_title:
        return -4.0
    return 0.0


def _generic_chunk_penalty(result: QueryResult, heading_overlap: int, title_overlap: int, path_overlap: int) -> float:
    path = result.source_file_path.lower()
    if heading_overlap or title_overlap:
        return 0.0
    if any(path.endswith(suffix) for suffix in GENERIC_PATH_SUFFIXES):
        return -2.0
    if path.count("/") <= 1:
        return -1.0
    if path_overlap == 0:
        return -1.0
    return 0.0


def _score_result(result: QueryResult, profile: QueryProfile) -> tuple[float, dict[str, float | int | str | list[str]]]:
    expanded_tokens = profile.expanded_tokens
    phrases = _query_phrases(profile.tokens)
    title_text = result.document_title.lower()
    section_text = (result.section_title or "").lower()
    heading_text = " > ".join(result.heading_path).lower()
    path_text = result.source_file_path.lower()

    lexical_score = max(0.0, -result.score)
    title_overlap = _field_overlap(title_text, expanded_tokens)
    section_overlap = _field_overlap(section_text, expanded_tokens)
    heading_overlap = _field_overlap(heading_text, expanded_tokens)
    path_overlap = _field_overlap(path_text, expanded_tokens)
    title_phrase_hits = _field_overlap(title_text, phrases)
    section_phrase_hits = _field_overlap(section_text, phrases)
    heading_phrase_hits = _field_overlap(heading_text, phrases)
    path_phrase_hits = _field_overlap(path_text, phrases)
    content_context_hits = _context_hits(result, expanded_tokens)
    exact_heading = _near_exact_match(" ".join(result.heading_path), profile.normalized_query)
    exact_title = _near_exact_match(result.document_title, profile.normalized_query)
    canonical_bonus = 2.0 if is_canonical_path(result.source_file_path) else -1.5
    subsystem_bonus = _subsystem_bonus(result, expanded_tokens)
    concept_page_bonus = _concept_page_bonus(result, profile)
    plugin_type_penalty = _plugin_type_penalty(result, profile)
    family_specific_bonus = _family_specific_bonus(result, profile)
    example_penalty = _example_penalty(result)
    quality_adjustment = _chunk_quality_adjustment(result)
    generic_penalty = _generic_chunk_penalty(result, heading_overlap, title_overlap, path_overlap)

    weighted_score = (
        lexical_score
        + (title_overlap * 3.0)
        + (section_overlap * 4.0)
        + (heading_overlap * 4.5)
        + (path_overlap * 2.0)
        + (title_phrase_hits * 6.0)
        + (section_phrase_hits * 5.0)
        + (heading_phrase_hits * 1.5)
        + (path_phrase_hits * 1.0)
        + (content_context_hits * 0.5)
        + (8.0 if exact_heading else 0.0)
        + (5.0 if exact_title else 0.0)
        + canonical_bonus
        + subsystem_bonus
        + concept_page_bonus
        + plugin_type_penalty
        + family_specific_bonus
        + example_penalty
        + quality_adjustment
        + generic_penalty
    )
    breakdown: dict[str, float | int | str | list[str]] = {
        "lexical_score": round(lexical_score, 3),
        "title_overlap": title_overlap,
        "section_overlap": section_overlap,
        "heading_overlap": heading_overlap,
        "path_overlap": path_overlap,
        "title_phrase_hits": title_phrase_hits,
        "section_phrase_hits": section_phrase_hits,
        "heading_phrase_hits": heading_phrase_hits,
        "path_phrase_hits": path_phrase_hits,
        "context_hits": content_context_hits,
        "exact_heading_phrase": 1 if exact_heading else 0,
        "exact_title_phrase": 1 if exact_title else 0,
        "query_intent": profile.intent,
        "concept_families": profile.concept_families,
        "canonical_bonus": canonical_bonus,
        "subsystem_bonus": subsystem_bonus,
        "concept_page_bonus": concept_page_bonus,
        "plugin_type_penalty": plugin_type_penalty,
        "family_specific_bonus": family_specific_bonus,
        "example_penalty": example_penalty,
        "quality_adjustment": quality_adjustment,
        "generic_penalty": generic_penalty,
        "expanded_tokens": expanded_tokens,
        "weighted_score": round(weighted_score, 3),
    }
    return weighted_score, breakdown


def _rerank_results(results: list[QueryResult], profile: QueryProfile, top_k: int) -> list[QueryResult]:
    scored: list[QueryResult] = []
    for result in results:
        weighted_score, breakdown = _score_result(result, profile)
        result.normalized_query = profile.normalized_query
        result.rerank_score = weighted_score
        result.rerank_breakdown = breakdown
        scored.append(result)

    ranked = sorted(
        scored,
        key=lambda result: (
            -(result.rerank_score or 0.0),
            not is_canonical_path(result.source_file_path),
            result.score,
            result.token_count,
            result.chunk_id,
        ),
    )
    deduped: list[QueryResult] = []
    seen_signatures: set[tuple[str, tuple[str, ...], str]] = set()
    for result in ranked:
        signature = (
            canonical_path_key(result.source_file_path),
            tuple(result.heading_path),
            (result.section_title or "").lower(),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped.append(result)
    return deduped[:top_k]


def query_chunks(db_path: Path, query_text: str, top_k: int) -> list[QueryResult]:
    """Query indexed chunks from the SQLite store with light query normalization."""

    store = SQLiteStore(db_path)
    store.initialize()
    profile = build_query_profile(query_text)
    candidates: list[QueryResult] = []
    seen_chunk_ids: set[str] = set()
    candidate_limit = max(top_k * 10, 40)
    for fts_query in _build_fts_queries(query_text):
        if not fts_query:
            continue
        try:
            current_results = store.query(query_text=fts_query, top_k=candidate_limit)
        except Exception:
            current_results = []
        for result in current_results:
            if result.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(result.chunk_id)
            candidates.append(result)
    return _rerank_results(candidates, profile, top_k)


def build_context_bundles(
    db_path: Path,
    results: list[QueryResult],
    include_previous: bool = False,
    include_next: bool = False,
    bundle_max_tokens: int = 600,
    tokenizer_name: str = "openai",
) -> list[ContextBundle]:
    """Build compact, traceable context bundles from query results."""

    store = SQLiteStore(db_path)
    store.initialize()
    tokenizer = get_tokenizer(tokenizer_name)

    bundles: list[ContextBundle] = []
    for rank, result in enumerate(results, start=1):
        section_chunks = store.get_section_chunks(result.section_id)
        index_by_id = {chunk.chunk_id: idx for idx, chunk in enumerate(section_chunks)}
        pivot_index = index_by_id.get(result.chunk_id)
        match_content = result.content
        match_token_count = result.token_count
        selection_strategy = "match_only"
        if bundle_max_tokens > 0 and match_token_count > bundle_max_tokens:
            match_content = _truncate_text_to_tokens(result.content, bundle_max_tokens, tokenizer)
            match_token_count = tokenizer.count_tokens(match_content)
            selection_strategy = "truncated_match"
        ordered_chunks: list[ContextBundleChunk] = [
            ContextBundleChunk(
                chunk_id=result.chunk_id,
                role="match",
                content=match_content,
                token_count=match_token_count,
            )
        ]
        total_tokens = match_token_count
        if pivot_index is not None and bundle_max_tokens > match_token_count and (include_previous or include_next):
            left = pivot_index - 1
            right = pivot_index + 1
            while left >= 0 or right < len(section_chunks):
                added = False
                if include_previous and left >= 0:
                    chunk = section_chunks[left]
                    compact_content = _compact_bundle_chunk_content(chunk.content)
                    compact_tokens = tokenizer.count_tokens(compact_content)
                    if total_tokens + compact_tokens <= bundle_max_tokens:
                        ordered_chunks.insert(
                            0,
                            ContextBundleChunk(
                                chunk_id=chunk.chunk_id,
                                role="previous",
                                content=compact_content,
                                token_count=compact_tokens,
                            ),
                        )
                        total_tokens += compact_tokens
                        selection_strategy = "section_window"
                        added = True
                    left -= 1
                if include_next and right < len(section_chunks):
                    chunk = section_chunks[right]
                    compact_content = _compact_bundle_chunk_content(chunk.content)
                    compact_tokens = tokenizer.count_tokens(compact_content)
                    if total_tokens + compact_tokens <= bundle_max_tokens:
                        ordered_chunks.append(
                            ContextBundleChunk(
                                chunk_id=chunk.chunk_id,
                                role="next",
                                content=compact_content,
                                token_count=compact_tokens,
                            )
                        )
                        total_tokens += compact_tokens
                        selection_strategy = "section_window"
                        added = True
                    right += 1
                if not added:
                    break
        bundles.append(
            ContextBundle(
                rank=rank,
                score=result.score,
                bundle_token_count=sum(chunk.token_count for chunk in ordered_chunks),
                source_file_path=result.source_file_path,
                document_title=result.document_title,
                section_title=result.section_title,
                heading_path=result.heading_path,
                repo_commit_hash=result.repo_commit_hash,
                snippet=result.snippet,
                chunks=ordered_chunks,
                selection_strategy=selection_strategy,
            )
        )
    return bundles


def _compact_bundle_chunk_content(content: str) -> str:
    compact = HEADING_PREFIX_PATTERN.sub("", content, count=1).strip()
    return compact or content.strip()


def _truncate_text_to_tokens(content: str, max_tokens: int, tokenizer) -> str:
    if max_tokens <= 0:
        return ""
    encoded = tokenizer.encode(content)
    if len(encoded) <= max_tokens:
        return content
    return tokenizer.decode(encoded[:max_tokens]).strip()
