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
    "wire": ["register", "declare", "external", "service"],
    "wiring": ["register", "declare", "external", "service"],
    "language": ["lang"],
    "strings": ["string"],
    "string": ["strings", "lang", "pluginname", "get_string"],
    "services": ["service", "external"],
    "tasks": ["task"],
    "tests": ["test"],
    "test": ["tests", "testing"],
    "forms": ["form"],
    "form": ["forms"],
    "plugins": ["plugin"],
    "write": ["writing"],
    "validation": ["validate", "addrule", "rule", "rules"],
    "validate": ["validation", "addrule", "rule", "rules"],
    "renderer": ["renderers", "rendering", "output"],
    "renderers": ["renderer", "rendering", "output"],
    "rendering": ["renderers", "renderable", "output"],
    "flow": ["journey", "renderable", "renderers"],
    "template": ["templates", "mustache"],
    "templates": ["template", "mustache"],
    "privacy": ["provider", "providers", "metadata"],
    "provider": ["providers", "privacy", "metadata"],
    "behat": ["acceptance", "tests", "testing"],
    "phpunit": ["unit", "tests", "testing"],
}
HEADING_PREFIX_PATTERN = re.compile(r"^Heading:\s.*?(?:\n\n|\n)", re.DOTALL)
GENERIC_PATH_SUFFIXES = {"docs/apis.md", "index.md"}
CONCEPTUAL_QUERY_PATTERN = re.compile(
    r"\b(how do|how does|what is|what are|where do|where does|where are|where should|where can|which|find|define|write|add|register|documented|docs|documentation|work|works)\b"
)


@dataclass(frozen=True, slots=True)
class QueryProfile:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    expanded_tokens: list[str]
    intent: str
    concept_families: list[str]
    task_intent: str


def canonical_path_key(path: str) -> str:
    """Return a canonicalized path key with versioned docs folded into their base path."""

    normalized = path.strip().lstrip("./")
    match = re.match(r"versioned_docs/version-[^/]+/(.+)", normalized)
    if match:
        return match.group(1)
    return normalized


def _result_source_metadata(result: QueryResult) -> dict[str, str | None]:
    metadata = result.metadata_json or {}
    source_type = metadata.get("source_type")
    source_name = metadata.get("source_name")
    if not source_name:
        if result.source_file_path.lower().startswith("design_system/"):
            source_name = "design_system"
            source_type = source_type or "scraped_web"
        elif source_type == "repo_markdown" or (
            source_type is None
            and not metadata.get("source_url")
            and not metadata.get("canonical_url")
        ):
            source_name = "devdocs_repo"
            source_type = source_type or "repo_markdown"
    return {
        "source_name": source_name,
        "source_type": source_type,
        "source_url": metadata.get("source_url"),
        "canonical_url": metadata.get("canonical_url"),
    }


def is_canonical_path(path: str) -> bool:
    """Return whether a path is from the non-versioned canonical corpus."""

    normalized = path.strip().lstrip("./")
    return not normalized.startswith("versioned_docs/")


def _path_contains(path: str, fragment: str) -> bool:
    normalized_path = path.strip().lstrip("./").lower()
    normalized_fragment = fragment.strip().strip("/").lower()
    return normalized_fragment in normalized_path


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


FILE_LOCATION_QUERY_PATTERN = re.compile(
    r"\b(where does|where do|where should|what file|where is this defined|where is this registered|where are|file location|location)\b"
)
FLOW_QUERY_PATTERN = re.compile(
    r"\b(fit in the flow|output flow|render(?:ing)? flow|how is .*rendered|path from|flow from|render flow|how does moodle expect .*work)\b"
)
IMPLEMENTATION_QUERY_PATTERN = re.compile(
    r"\b(how do i implement|how do i write|how do i add|how do i configure|how do i define|how do i wire up|what do i need to implement|how do .*registered|how do .*declare)\b"
)
FILE_ANCHOR_PATTERN = re.compile(r"`?([A-Za-z0-9_./-]+\.(?:php|mdx|md|js|mustache|yml))`?")


def classify_task_intent(query_text: str, tokens: list[str]) -> str:
    lowered = query_text.lower()
    if FLOW_QUERY_PATTERN.search(lowered):
        return "flow_explainer"
    if FILE_LOCATION_QUERY_PATTERN.search(lowered):
        return "file_location"
    if IMPLEMENTATION_QUERY_PATTERN.search(lowered):
        return "implementation_guide"
    if any(token in {"file", "registered", "defined", "location", "write", "implement", "configure", "wire"} for token in tokens):
        return "implementation_guide"
    return "general"


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
    if {"design", "system"} <= expanded and {
        "token",
        "tokens",
        "css",
        "scss",
        "semantic",
        "colour",
        "color",
        "icon",
        "icons",
        "layout",
        "breakpoints",
        "breakpoint",
    } & expanded:
        families.append("design_system")
    if {"setting", "settings", "admin"} & expanded and {"plugin", "plugins"} & expanded:
        families.append("admin_settings")
    if "upgrade" in expanded or "upgrade.php" in expanded:
        families.append("upgrade")
    if {"web", "service", "services", "external"} & expanded and {"service", "services", "external"} & expanded:
        families.append("web_services")
    if {"strings", "string"} & expanded and {
        "language",
        "lang",
        "localisation",
        "localization",
        "plugin",
        "plugins",
        "get_string",
        "pluginname",
    } & expanded:
        families.append("language_strings")
    if {"mustache", "template", "templates", "output", "renderer", "renderers"} & expanded:
        families.append("output_templates")
    if {"phpunit", "behat", "testing", "test", "tests", "acceptance", "unit"} & expanded:
        families.append("testing")
    if {"privacy", "provider", "providers", "metadata"} & expanded:
        families.append("privacy")
    if {"events", "event"} & expanded:
        families.append("events")
    if {"form", "forms"} & expanded and {"validation", "validate", "addrule", "rule", "rules"} & expanded:
        families.append("forms_validation")
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
        task_intent=classify_task_intent(query_text, tokens),
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
    if any(_path_contains(path, segment) for segment in ("subsystems", "guides", "_files", "commonfiles")):
        return 5.0
    if path == "docs/apis.md":
        return 2.0
    return 0.0


def _is_docs_location_query(profile: QueryProfile) -> bool:
    lowered = profile.raw_query.lower()
    return any(
        phrase in lowered
        for phrase in ("where ", "where are", "where do", "where should", "find ", "docs", "documentation", "documented")
    )


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
    section_text = (result.section_title or "").lower()
    heading = " > ".join(result.heading_path).lower()
    expanded_tokens = set(profile.expanded_tokens)
    bonus = 0.0

    if "upgrade" in profile.concept_families:
        if _path_contains(path, "guides/upgrade"):
            bonus += 20.0
        elif any(_path_contains(path, fragment) for fragment in ("_files/db-upgrade", "_files/upgrade-php")):
            bonus += 14.0
        if "upgrade helpers" in heading or "upgrade.php" in heading:
            bonus += 4.0
        if _path_contains(path, "plugintypes"):
            bonus -= 12.0

    if "admin_settings" in profile.concept_families:
        if any(_path_contains(path, fragment) for fragment in ("subsystems/admin", "_files/settings-php")):
            bonus += 18.0
        if any(text in heading or text in title for text in ("settings.php", "admin settings", "settings")):
            bonus += 8.0
        if _path_contains(path, "plugintypes"):
            bonus -= 10.0

    if "web_services" in profile.concept_families:
        if any(_path_contains(path, fragment) for fragment in ("subsystems/external/writing-a-service", "_files/db-services")):
            bonus += 20.0
        elif _path_contains(path, "subsystems/external/functions"):
            bonus += 10.0
        elif _path_contains(path, "subsystems/external/index"):
            bonus += 8.0
        if any(text in heading or text in title for text in ("writing a new service", "declare the web service function", "external functions")):
            bonus += 8.0
        if "db/services.php" in heading:
            bonus += 8.0
        if profile.task_intent == "implementation_guide":
            if any(text in heading or text in title for text in ("writing", "declare", "external functions", "function descriptions")):
                bonus += 10.0
            if path == "docs/apis.md":
                bonus -= 8.0
            if _path_contains(path, "subsystems/external/index"):
                bonus -= 4.0
            if "version" in heading or "version" in section_text:
                bonus -= 10.0

    if "language_strings" in profile.concept_families:
        if path.endswith("docs/apis/_files/lang.md") or _path_contains(path, "_files/lang.md"):
            bonus += 18.0
            if profile.task_intent == "file_location":
                bonus += 12.0
        if any(text in heading or text in title for text in ("language files", "language strings", "lang", "string api")):
            bonus += 6.0
        if profile.task_intent == "file_location" and any(text in heading or text in section_text for text in ("language files", "lang files", "lang")):
            bonus += 6.0
        if any(text in heading for text in ("dynamic strings", "working with strings")):
            bonus -= 8.0
        if any(_path_contains(path, fragment) for fragment in ("javascript", "string-deprecation", "plugincontribution/checklist")):
            bonus -= 8.0
        if profile.task_intent == "file_location" and any(
            _path_contains(path, fragment)
            for fragment in ("general/app/development", "policies/codingstyle", "plugincontribution/checklist")
        ):
            bonus -= 14.0
        if {"plugin", "plugins"} & expanded_tokens and _path_contains(path, "general/app"):
            bonus -= 12.0
        if _path_contains(path, "plugintypes"):
            bonus -= 5.0

    if "output_templates" in profile.concept_families:
        renderer_only_query = "renderer" in expanded_tokens and not ({"mustache", "template", "templates"} & expanded_tokens)
        if _path_contains(path, "guides/templates"):
            bonus += 18.0
            if renderer_only_query:
                bonus -= 8.0
        elif _path_contains(path, "subsystems/output"):
            bonus += 10.0
            if renderer_only_query:
                bonus += 12.0
            if profile.task_intent == "flow_explainer":
                bonus += 14.0
        if title == "templates":
            bonus += 6.0
        elif title == "output api":
            bonus += 8.0 if renderer_only_query else 2.0
        if "templates" in heading or "mustache" in heading:
            bonus += 6.0
        elif "renderers" in heading:
            bonus += 8.0
        elif "renderable" in heading:
            bonus += 4.0
        if profile.task_intent == "flow_explainer":
            if any(text in heading or text in section_text for text in ("renderers", "renderable", "page output journey", "output api")):
                bonus += 10.0
            if _path_contains(path, "subsystems/output") and any(
                text in heading or text in section_text for text in ("page output journey", "renderable")
            ):
                bonus += 14.0
            if {"output", "flow"} <= expanded_tokens and _path_contains(path, "subsystems/output"):
                bonus += 18.0
            if _path_contains(path, "guides/templates") and "templates" in heading and "render" not in heading:
                bonus -= 6.0
            if _path_contains(path, "guides/templates") and any(
                text in heading or text in section_text for text in ("renderers", "rendering in php")
            ):
                bonus -= 8.0
        if renderer_only_query and "dependency injection" in heading:
            bonus -= 4.0
        if renderer_only_query and "output api" in heading:
            bonus += 8.0
        if _path_contains(path, "plugintypes/format"):
            bonus -= 8.0
        if "renderer" in expanded_tokens:
            if _path_contains(path, "subsystems/output") and "renderers" in heading:
                bonus += 10.0
            if _path_contains(path, "guides/templates") and "renderers" in heading:
                bonus += 6.0

    if "testing" in profile.concept_families:
        if "behat" in expanded_tokens:
            if _path_contains(path, "tools/behat"):
                bonus += 18.0
            if any(text in heading or text in title for text in ("behat", "writing", "running")):
                bonus += 6.0
            if _is_docs_location_query(profile):
                if path.endswith("general/development/tools/behat/index.md") or path.endswith("general/development/tools/behat/writing.md"):
                    bonus += 10.0
                if path.endswith("general/development/tools/behat/running.md"):
                    bonus -= 4.0
            if _path_contains(path, "tools/mdk"):
                bonus -= 10.0
        if "phpunit" in expanded_tokens:
            if _path_contains(path, "tools/phpunit"):
                bonus += 18.0
            if any(text in heading or text in title for text in ("phpunit", "unit testing", "writing")):
                bonus += 6.0
        if any(_path_contains(path, fragment) for fragment in ("accessibility", "deprecation", "policies")):
            bonus -= 8.0

    if "privacy" in profile.concept_families:
        if any(_path_contains(path, fragment) for fragment in ("subsystems/privacy", "privacy")):
            bonus += 18.0
        if any(text in heading or text in title for text in ("privacy provider", "metadata provider", "request provider")):
            bonus += 8.0
        if "metadata" in expanded_tokens and any(text in heading or text in section_text for text in ("metadata provider", "implementation requirements")):
            bonus += 10.0
        if "provider" in expanded_tokens and _path_contains(path, "providers") and not _path_contains(path, "privacy"):
            bonus -= 8.0

    if "events" in profile.concept_families:
        if any(_path_contains(path, fragment) for fragment in ("subsystems/events", "_files/db-events")):
            bonus += 18.0
        if "events api" in heading or "events api" in title:
            bonus += 6.0
        if any(text in heading for text in ("observers", "triggering events", "event observers")):
            bonus += 6.0
        if path == "docs/apis.md":
            bonus += 1.0
        if _path_contains(path, "_files/db-events") and any(token in profile.expanded_tokens for token in ("plugin", "plugins")):
            bonus += 12.0
        if path == "docs/apis.md" and any(token in profile.expanded_tokens for token in ("plugin", "plugins")):
            bonus -= 4.0
        if any(_path_contains(path, fragment) for fragment in ("plugintypes", "calendar", "xapi")):
            bonus -= 8.0

    if "forms_validation" in profile.concept_families:
        if any(_path_contains(path, fragment) for fragment in ("subsystems/form", "subsystems/forms")):
            bonus += 12.0
        if any(text in heading or text in title for text in ("validation", "addrule", "forms api")):
            bonus += 10.0
        if _path_contains(path, "subsystems/form/fields") and "validation" not in heading:
            bonus -= 10.0

    if "design_system" in profile.concept_families:
        if path.startswith("design_system/"):
            bonus += 6.0
        if _path_contains(path, "for-developers") or "for developers" in heading or "for developers" in title:
            bonus += 8.0
        if _path_contains(path, "for-designers") or "for designers" in heading or "for designers" in title:
            bonus -= 4.0
        if {"css", "token"} <= expanded_tokens or {"css", "tokens"} <= expanded_tokens:
            if any(text in heading or text in section_text for text in ("token consumption", "css tokens")):
                bonus += 18.0
            if any(text in heading or text in section_text for text in ("library", "community", "figma")):
                bonus -= 16.0
        if {"semantic", "colour"} <= expanded_tokens or {"semantic", "color"} <= expanded_tokens:
            if "semantic colour tokens" in heading or "semantic colour tokens" in section_text:
                bonus += 14.0
        if "icon" in expanded_tokens or "icons" in expanded_tokens:
            if "icon library" in heading or "icon library" in section_text:
                bonus += 14.0
        if "breakpoint" in expanded_tokens or "breakpoints" in expanded_tokens:
            if "breakpoints" in heading or "breakpoints" in section_text:
                bonus += 14.0

    return bonus


def _section_focus_bonus(
    result: QueryResult,
    profile: QueryProfile,
    section_overlap: int,
    heading_overlap: int,
    title_overlap: int,
    section_phrase_hits: int,
    heading_phrase_hits: int,
) -> float:
    section_focus = section_overlap + heading_overlap
    phrase_focus = section_phrase_hits + heading_phrase_hits
    bonus = 0.0

    if profile.intent == "conceptual" and section_focus >= max(2, title_overlap + 1):
        bonus += 4.0
    if profile.task_intent == "flow_explainer" and (heading_overlap + section_phrase_hits + heading_phrase_hits) >= 2:
        bonus += 5.0
    if phrase_focus > 0 and section_focus > 0:
        bonus += 3.0

    heading = " > ".join(result.heading_path).lower()
    section = (result.section_title or "").lower()
    if any(text in heading or text in section for text in ("api", "validation", "renderers", "mustache", "language files", "phpunit", "behat", "privacy provider", "events", "page output journey", "implementation requirements", "settings.php")):
        bonus += 2.0

    return bonus


def _incidental_match_penalty(
    result: QueryResult,
    profile: QueryProfile,
    section_overlap: int,
    heading_overlap: int,
) -> float:
    if profile.intent != "conceptual":
        return 0.0

    heading = " > ".join(result.heading_path).lower()
    section = (result.section_title or "").lower()
    path = result.source_file_path.lower()

    if any(text in heading or text in section for text in ("typical workflows", "getting started", "other features", "checklist", "coding")):
        return -6.0
    if any(text in heading for text in ("dynamic strings", "working with strings", "executing behat tests")):
        return -8.0
    if profile.task_intent == "implementation_guide" and any(text in heading for text in ("bump the plugin version", "version.php")):
        return -8.0
    if _path_contains(path, "subsystems/form/fields") and "forms_validation" in profile.concept_families and heading_overlap <= 1:
        return -6.0
    if section_overlap == 0 and heading_overlap == 0 and profile.concept_families:
        return -3.0
    return 0.0


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
    section_focus_bonus = _section_focus_bonus(
        result,
        profile,
        section_overlap,
        heading_overlap,
        title_overlap,
        section_phrase_hits,
        heading_phrase_hits,
    )
    incidental_penalty = _incidental_match_penalty(result, profile, section_overlap, heading_overlap)
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
        + section_focus_bonus
        + incidental_penalty
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
        "section_focus_bonus": section_focus_bonus,
        "incidental_penalty": incidental_penalty,
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
    support_results: list[QueryResult] | None = None,
    query_text: str | None = None,
    include_previous: bool = False,
    include_next: bool = False,
    bundle_max_tokens: int = 600,
    tokenizer_name: str = "openai",
) -> list[ContextBundle]:
    """Build compact, traceable context bundles from query results."""

    store = SQLiteStore(db_path)
    store.initialize()
    tokenizer = get_tokenizer(tokenizer_name)
    profile = build_query_profile(query_text) if query_text else None
    support_pool = support_results or results

    bundles: list[ContextBundle] = []
    for rank, result in enumerate(results, start=1):
        result_source = _result_source_metadata(result)
        section_chunks = store.get_section_chunks(result.section_id)
        document_chunks = store.get_document_chunks(result.document_id)
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
                source_file_path=result.source_file_path,
                source_name=result_source["source_name"],
                source_type=result_source["source_type"],
                source_url=result_source["source_url"],
                canonical_url=result_source["canonical_url"],
                section_title=result.section_title,
                heading_path=result.heading_path,
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
                        chunk_source = _result_source_metadata(chunk)
                        ordered_chunks.insert(
                            0,
                            ContextBundleChunk(
                                chunk_id=chunk.chunk_id,
                                role="previous",
                                content=compact_content,
                                token_count=compact_tokens,
                                source_file_path=chunk.source_file_path,
                                source_name=chunk_source["source_name"],
                                source_type=chunk_source["source_type"],
                                source_url=chunk_source["source_url"],
                                canonical_url=chunk_source["canonical_url"],
                                section_title=chunk.section_title,
                                heading_path=chunk.heading_path,
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
                        chunk_source = _result_source_metadata(chunk)
                        ordered_chunks.append(
                            ContextBundleChunk(
                                chunk_id=chunk.chunk_id,
                                role="next",
                                content=compact_content,
                                token_count=compact_tokens,
                                source_file_path=chunk.source_file_path,
                                source_name=chunk_source["source_name"],
                                source_type=chunk_source["source_type"],
                                source_url=chunk_source["source_url"],
                                canonical_url=chunk_source["canonical_url"],
                                section_title=chunk.section_title,
                                heading_path=chunk.heading_path,
                            )
                        )
                        total_tokens += compact_tokens
                        selection_strategy = "section_window"
                        added = True
                    right += 1
                if not added:
                    break
        support_diagnostic: dict[str, str | int | list[str] | bool] = {}
        if _should_add_support_chunk(profile, result, bundle_max_tokens, total_tokens):
            extra_support_chunks: list[QueryResult] = []
            support_doc_ids: set[str] = set()
            for candidate in support_pool[: min(len(support_pool), 5)]:
                if candidate.document_id == result.document_id or candidate.document_id in support_doc_ids:
                    continue
                support_doc_ids.add(candidate.document_id)
                extra_support_chunks.extend(store.get_document_chunks(candidate.document_id))
            support_candidate = _select_task_support_chunk(
                primary=result,
                candidate_results=support_pool,
                document_chunks=document_chunks + extra_support_chunks,
                profile=profile,
                existing_chunk_ids={chunk.chunk_id for chunk in ordered_chunks},
                preferred_support_budget=max(bundle_max_tokens - 80, 0),
            )
            if support_candidate is not None:
                support_content = _compact_bundle_chunk_content(support_candidate.content)
                support_tokens = tokenizer.count_tokens(support_content)
                if total_tokens + support_tokens <= bundle_max_tokens:
                    support_source = _result_source_metadata(support_candidate)
                    ordered_chunks.append(
                        ContextBundleChunk(
                            chunk_id=support_candidate.chunk_id,
                            role="support",
                            content=support_content,
                            token_count=support_tokens,
                            source_file_path=support_candidate.source_file_path,
                            source_name=support_source["source_name"],
                            source_type=support_source["source_type"],
                            source_url=support_source["source_url"],
                            canonical_url=support_source["canonical_url"],
                            section_title=support_candidate.section_title,
                            heading_path=support_candidate.heading_path,
                        )
                    )
                    total_tokens += support_tokens
                    selection_strategy = "task_support"
                    support_diagnostic = {
                        "task_intent": profile.task_intent,
                        "support_added": True,
                        "support_chunk_id": support_candidate.chunk_id,
                        "support_source_file_path": support_candidate.source_file_path,
                        "support_heading_path": support_candidate.heading_path,
                        "support_file_anchors": _extract_file_anchors(support_candidate.content),
                    }
                else:
                    available_for_match = bundle_max_tokens - support_tokens
                    minimum_match_tokens = 80
                    if available_for_match >= minimum_match_tokens and ordered_chunks:
                        primary_chunk = ordered_chunks[0]
                        trimmed_content = _truncate_text_to_tokens(primary_chunk.content, available_for_match, tokenizer)
                        trimmed_tokens = tokenizer.count_tokens(trimmed_content)
                        if trimmed_tokens >= minimum_match_tokens:
                            ordered_chunks[0] = ContextBundleChunk(
                                chunk_id=primary_chunk.chunk_id,
                                role=primary_chunk.role,
                                content=trimmed_content,
                                token_count=trimmed_tokens,
                                source_file_path=primary_chunk.source_file_path,
                                source_name=primary_chunk.source_name,
                                source_type=primary_chunk.source_type,
                                source_url=primary_chunk.source_url,
                                canonical_url=primary_chunk.canonical_url,
                                section_title=primary_chunk.section_title,
                                heading_path=primary_chunk.heading_path,
                            )
                            support_source = _result_source_metadata(support_candidate)
                            ordered_chunks.append(
                                ContextBundleChunk(
                                    chunk_id=support_candidate.chunk_id,
                                    role="support",
                                    content=support_content,
                                    token_count=support_tokens,
                                    source_file_path=support_candidate.source_file_path,
                                    source_name=support_source["source_name"],
                                    source_type=support_source["source_type"],
                                    source_url=support_source["source_url"],
                                    canonical_url=support_source["canonical_url"],
                                    section_title=support_candidate.section_title,
                                    heading_path=support_candidate.heading_path,
                                )
                            )
                            total_tokens = sum(chunk.token_count for chunk in ordered_chunks)
                            selection_strategy = "task_support_truncated"
                            support_diagnostic = {
                                "task_intent": profile.task_intent,
                                "support_added": True,
                                "support_chunk_id": support_candidate.chunk_id,
                                "support_source_file_path": support_candidate.source_file_path,
                                "support_heading_path": support_candidate.heading_path,
                                "support_file_anchors": _extract_file_anchors(support_candidate.content),
                                "support_trimmed_match": True,
                            }
                        else:
                            support_diagnostic = {
                                "task_intent": profile.task_intent,
                                "support_added": False,
                                "support_rejected": "budget",
                            }
                    else:
                        support_diagnostic = {
                            "task_intent": profile.task_intent,
                            "support_added": False,
                            "support_rejected": "budget",
                        }
            else:
                support_diagnostic = {
                    "task_intent": profile.task_intent,
                    "support_added": False,
                    "support_rejected": "no_candidate",
                }
        bundles.append(
            ContextBundle(
                rank=rank,
                score=result.score,
                bundle_token_count=sum(chunk.token_count for chunk in ordered_chunks),
                source_file_path=result.source_file_path,
                source_name=result_source["source_name"],
                source_type=result_source["source_type"],
                source_url=result_source["source_url"],
                canonical_url=result_source["canonical_url"],
                document_title=result.document_title,
                section_title=result.section_title,
                heading_path=result.heading_path,
                repo_commit_hash=result.repo_commit_hash,
                snippet=result.snippet,
                chunks=ordered_chunks,
                selection_strategy=selection_strategy,
                diagnostics={
                    "bundle_max_tokens": bundle_max_tokens,
                    "within_budget": bundle_max_tokens <= 0 or total_tokens <= bundle_max_tokens,
                    "chunk_count": len(ordered_chunks),
                    "chunk_roles": [chunk.role for chunk in ordered_chunks],
                    "truncated": selection_strategy == "truncated_match",
                    "duplicate_chunk_count": _bundle_duplicate_count(ordered_chunks),
                    "task_intent": profile.task_intent if profile is not None else "general",
                    "concept_families": profile.concept_families if profile is not None else [],
                    **support_diagnostic,
                },
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


def _bundle_duplicate_count(chunks: list[ContextBundleChunk]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for chunk in chunks:
        normalized = " ".join(chunk.content.lower().split())
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
    return duplicates


def _extract_file_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    for match in FILE_ANCHOR_PATTERN.findall(text):
        if match not in anchors:
            anchors.append(match)
    return anchors


def _expected_anchor_terms(profile: QueryProfile) -> list[str]:
    anchors: list[str] = []
    expanded = set(profile.expanded_tokens)
    if {"setting", "settings", "admin"} & expanded:
        anchors.append("settings.php")
    if {"task", "tasks", "scheduled"} & expanded:
        anchors.append("db/tasks.php")
    if {"service", "services", "external", "web"} & expanded:
        anchors.append("db/services.php")
        anchors.append("External functions")
    if {"behat", "acceptance"} & expanded:
        anchors.append("Writing acceptance tests")
    if {"phpunit", "unit"} & expanded:
        anchors.append("Writing PHPUnit tests")
    if {"event", "events"} & expanded:
        anchors.append("db/events.php")
    if {"privacy", "metadata"} <= expanded or ({"privacy", "metadata"} & expanded and "privacy" in expanded):
        anchors.append("metadata provider")
    return anchors


def _support_keyword_overlap(candidate: QueryResult, profile: QueryProfile) -> int:
    heading_text = " ".join(candidate.heading_path).lower()
    section_text = (candidate.section_title or "").lower()
    title_text = candidate.document_title.lower()
    content_text = candidate.content.lower()
    overlap = 0
    for token in profile.expanded_tokens:
        if len(token) < 3:
            continue
        if token in heading_text:
            overlap += 3
        elif token in section_text or token in title_text:
            overlap += 2
        elif token in content_text:
            overlap += 1
    return overlap


def _candidate_support_score(candidate: QueryResult, primary: QueryResult, profile: QueryProfile) -> float:
    score = 0.0
    anchors = _extract_file_anchors(candidate.content)
    heading_text = " > ".join(candidate.heading_path).lower()
    section_text = (candidate.section_title or "").lower()
    title_text = candidate.document_title.lower()
    path_text = candidate.source_file_path.lower()
    content_text = candidate.content.lower()
    expected = _expected_anchor_terms(profile)
    keyword_overlap = _support_keyword_overlap(candidate, profile)

    if "web_services" in profile.concept_families and profile.task_intent == "implementation_guide" and "version" in heading_text:
        return -100.0

    if candidate.document_id == primary.document_id:
        score += 4.0
    elif canonical_path_key(candidate.source_file_path) == canonical_path_key(primary.source_file_path):
        score += 2.0

    score += float(keyword_overlap)

    if profile.task_intent == "file_location":
        if anchors:
            score += 8.0
        if "where to find the code" in heading_text:
            score += 4.0
    if profile.task_intent == "implementation_guide":
        if any(term.lower() in heading_text or term.lower() in section_text or term.lower() in title_text for term in ("writing", "creating", "declare", "register", "settings", "acceptance tests", "external functions")):
            score += 6.0
        if anchors:
            score += 4.0
    if profile.task_intent == "flow_explainer":
        if any(term in heading_text or term in section_text or term in title_text for term in ("renderers", "renderable", "page output journey", "output api", "templates")):
            score += 8.0

    for expected_anchor in expected:
        if expected_anchor.lower() in candidate.content.lower():
            score += 16.0
        if expected_anchor.lower() in heading_text or expected_anchor.lower() in section_text:
            score += 10.0

    if "web_services" in profile.concept_families:
        if any(term in heading_text or term in section_text or term in title_text or term in path_text for term in ("service", "services", "external")):
            score += 8.0
        if any(term in heading_text or term in section_text for term in ("function descriptions", "declare the web service function")):
            score += 10.0
        if "version" in heading_text:
            score -= 20.0
    if "testing" in profile.concept_families:
        if any(term in heading_text or term in section_text or term in title_text for term in ("behat", "phpunit", "acceptance tests")):
            score += 8.0
    if "privacy" in profile.concept_families and "metadata" in profile.expanded_tokens:
        if any(term in heading_text or term in section_text or term in title_text for term in ("metadata provider", "implementation requirements", "null_provider")):
            score += 12.0
    if "output_templates" in profile.concept_families and profile.task_intent == "flow_explainer":
        if any(term in heading_text or term in section_text for term in ("renderers", "renderable", "page output journey", "output api")):
            score += 10.0
    if "design_system" in profile.concept_families:
        if path_text.startswith("design_system/"):
            score += 4.0
        if any(
            term in heading_text or term in section_text
            for term in (
                "token consumption",
                "css tokens",
                "usage",
                "import",
                "benefits",
                "semantic colour tokens",
                "icon library",
                "breakpoints",
            )
        ):
            score += 8.0
        if any(term in heading_text or term in section_text for term in ("library", "community")) or "figma" in content_text:
            score -= 12.0
        if {"css", "token"} <= set(profile.expanded_tokens) or {"css", "tokens"} <= set(profile.expanded_tokens):
            if "css tokens" in heading_text or "css tokens" in section_text:
                score += 10.0
            if "scss tokens" in heading_text or "scss tokens" in section_text:
                score -= 6.0

    if candidate.source_file_path == primary.source_file_path and candidate.section_id != primary.section_id:
        score += 4.0
    if "example" in heading_text or "examples" in heading_text:
        score -= 6.0
    if keyword_overlap == 0:
        score -= 8.0

    return score


def _select_task_support_chunk(
    primary: QueryResult,
    candidate_results: list[QueryResult],
    document_chunks: list[QueryResult],
    profile: QueryProfile,
    existing_chunk_ids: set[str],
    preferred_support_budget: int | None = None,
) -> QueryResult | None:
    support_pool: list[QueryResult] = []
    seen_ids = set(existing_chunk_ids)

    for candidate in candidate_results:
        if candidate.chunk_id in seen_ids or candidate.chunk_id == primary.chunk_id:
            continue
        seen_ids.add(candidate.chunk_id)
        support_pool.append(candidate)

    for chunk in document_chunks:
        if chunk.chunk_id in seen_ids or chunk.chunk_id == primary.chunk_id:
            continue
        seen_ids.add(chunk.chunk_id)
        support_pool.append(chunk)

    primary_text = primary.content.lower()
    expected = _expected_anchor_terms(profile)
    if expected and any(anchor.lower() in primary_text for anchor in expected) and profile.task_intent != "implementation_guide":
        return None

    ranked = sorted(
        support_pool,
        key=lambda candidate: (
            0 if preferred_support_budget is None or candidate.token_count <= preferred_support_budget else 1,
            -_candidate_support_score(candidate, primary, profile),
            candidate.token_count,
            candidate.chunk_id,
        ),
    )
    if not ranked:
        return None
    best = ranked[0]
    if _candidate_support_score(best, primary, profile) <= 0:
        return None
    return best


def _should_add_support_chunk(profile: QueryProfile | None, primary: QueryResult, bundle_max_tokens: int, total_tokens: int) -> bool:
    if profile is None or bundle_max_tokens <= total_tokens:
        return False
    if profile.task_intent != "general":
        return True
    return (
        "design_system" in profile.concept_families
        and primary.source_file_path.lower().startswith("design_system/")
        and primary.token_count < 40
    )
