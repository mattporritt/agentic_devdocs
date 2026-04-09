"""Evaluation, reporting, and baseline comparison for retrieval and bundles.

The guiding design constraint here is trustworthiness: the same canonical report object
drives JSON, text, and markdown outputs so current-run quality and baseline comparison
cannot silently diverge.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import yaml

from agentic_docs.models import (
    BaselineBucketChange,
    BaselineCaseChange,
    BaselineComparison,
    BaselineMetricDelta,
    BundleGradeStats,
    ContextBundle,
    EvalCase,
    EvalGroupReport,
    EvalMatch,
    EvalOutcome,
    EvalReport,
    EvalWindowStats,
    QueryResult,
    SourceConfusionCase,
)
from agentic_docs.provenance import result_source_metadata
from agentic_docs.query_service import build_context_bundles, canonical_path_key, query_chunks


def load_eval_cases(eval_file: Path) -> list[EvalCase]:
    """Load retrieval eval cases from YAML or JSON."""

    raw_text = eval_file.read_text(encoding="utf-8")
    if eval_file.suffix.lower() == ".json":
        loaded = json.loads(raw_text)
    else:
        loaded = yaml.safe_load(raw_text)

    if isinstance(loaded, dict) and "cases" in loaded:
        cases = loaded["cases"]
    else:
        cases = loaded
    if not isinstance(cases, list):
        msg = f"Eval file must contain a list of cases or a top-level 'cases' key: {eval_file}"
        raise ValueError(msg)
    return [EvalCase.model_validate(case) for case in cases]


def load_eval_report_artifact(artifact_path: Path) -> EvalReport:
    """Load an eval report from either a raw eval artifact or a verify-devdocs artifact."""

    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = loaded["eval"] if isinstance(loaded, dict) and "eval" in loaded else loaded
    if not isinstance(payload, dict):
        msg = f"Baseline artifact must be an eval JSON object or a verify-devdocs payload: {artifact_path}"
        raise ValueError(msg)
    return EvalReport.model_validate(payload)


def _matches_any(substrings: list[str], haystacks: list[str]) -> list[str]:
    matches: list[str] = []
    for needle in substrings:
        lowered = needle.lower()
        if any(lowered in haystack for haystack in haystacks):
            matches.append(needle)
    return matches


def _matches_sources(expected_sources: list[str], actual_source: str | None) -> list[str]:
    if not actual_source:
        return []
    lowered_source = actual_source.lower()
    return [source for source in expected_sources if source.lower() == lowered_source]


def _expected_source_name(case: EvalCase) -> str | None:
    if case.preferred_source_names:
        return case.preferred_source_names[0]
    if case.acceptable_source_names:
        return case.acceptable_source_names[0]
    return None


def _matching_paths(expected_paths: list[str], actual_path: str) -> list[str]:
    lowered_path = actual_path.lower()
    lowered_canonical = canonical_path_key(lowered_path)
    return _matches_any(expected_paths, [lowered_path, lowered_canonical])


def _grade_result(case: EvalCase, result: QueryResult) -> tuple[str, list[str], str | None]:
    """Grade one retrieval hit against an eval case with explicit source-aware rules."""

    lower_path = result.source_file_path.lower()
    heading_text = " > ".join(result.heading_path).lower()
    section_text = (result.section_title or "").lower()
    document_text = result.document_title.lower()
    haystacks = [heading_text, section_text, document_text]
    source = result_source_metadata(result)
    source_name = source["source_name"]

    disallowed_paths = _matches_any(case.disallowed_document_paths, [lower_path, canonical_path_key(lower_path)])
    if disallowed_paths:
        return "MISS", [f"disallowed_path:{path}" for path in disallowed_paths], None

    preferred_sources = _matches_sources(case.preferred_source_names, source_name)
    acceptable_sources = _matches_sources(case.acceptable_source_names, source_name)
    preferred_paths = _matching_paths(case.preferred_document_paths, lower_path)
    acceptable_paths = _matching_paths(case.acceptable_document_paths, lower_path)
    preferred_headings = _matches_any(case.preferred_heading_substrings, haystacks)
    acceptable_headings = _matches_any(case.acceptable_heading_substrings, haystacks)

    source_blocks_strong = bool(case.preferred_source_names) and not preferred_sources
    source_blocks_weak = bool(case.acceptable_source_names) and not acceptable_sources

    preferred_signal = bool(preferred_paths or preferred_headings or (case.preferred_source_names and preferred_sources))
    acceptable_signal = bool(acceptable_paths or acceptable_headings or (case.acceptable_source_names and acceptable_sources))

    if preferred_signal and not source_blocks_strong:
        matched_on = [f"preferred_source:{source}" for source in preferred_sources] + [f"preferred_path:{path}" for path in preferred_paths] + [
            f"preferred_heading:{heading}" for heading in preferred_headings
        ]
        preferred_components = sum(bool(component) for component in (preferred_sources, preferred_paths, preferred_headings))
        if preferred_components >= 2:
            rule_type = "preferred_source+target"
        elif preferred_sources:
            rule_type = "preferred_source"
        elif preferred_paths:
            rule_type = "preferred_path"
        else:
            rule_type = "preferred_heading"
        return "STRONG PASS", matched_on, rule_type

    if acceptable_signal and not source_blocks_weak:
        matched_on = [f"acceptable_source:{source}" for source in acceptable_sources] + [f"acceptable_path:{path}" for path in acceptable_paths] + [
            f"acceptable_heading:{heading}" for heading in acceptable_headings
        ]
        acceptable_components = sum(bool(component) for component in (acceptable_sources, acceptable_paths, acceptable_headings))
        if acceptable_components >= 2:
            rule_type = "acceptable_source+target"
        elif acceptable_sources:
            rule_type = "acceptable_source"
        elif acceptable_paths:
            rule_type = "acceptable_path"
        else:
            rule_type = "acceptable_heading"
        return "WEAK PASS", matched_on, rule_type

    return "MISS", [], None


def _bundle_expected_paths(case: EvalCase) -> list[str]:
    return case.preferred_bundle_paths or case.preferred_document_paths or case.acceptable_document_paths


def _bundle_expected_sources(case: EvalCase) -> tuple[list[str], list[str]]:
    preferred = case.preferred_bundle_source_names or case.preferred_source_names
    acceptable = case.acceptable_bundle_source_names or case.acceptable_source_names
    return preferred, acceptable


def _bundle_expected_headings(case: EvalCase) -> list[str]:
    preferred = case.preferred_heading_substrings_for_bundle or case.preferred_heading_substrings
    required = case.required_heading_substrings_for_bundle
    combined: list[str] = []
    for heading in preferred + required:
        if heading not in combined:
            combined.append(heading)
    return combined


def _bundle_source_names(bundle: ContextBundle) -> list[str]:
    names: list[str] = []
    for name in [bundle.source_name, *[chunk.source_name for chunk in bundle.chunks]]:
        if not name:
            continue
        if name not in names:
            names.append(name)
    return names


def _bundle_haystacks(bundle: ContextBundle) -> list[str]:
    haystacks = [
        bundle.source_file_path.lower(),
        bundle.document_title.lower(),
        (bundle.section_title or "").lower(),
        " > ".join(bundle.heading_path).lower(),
    ]
    for chunk in bundle.chunks:
        haystacks.append(chunk.source_file_path.lower())
        haystacks.append((chunk.section_title or "").lower())
        haystacks.append(" > ".join(chunk.heading_path).lower())
        haystacks.append(chunk.content.lower())
    return haystacks


def _bundle_redundancy_count(bundle: ContextBundle) -> int:
    seen: set[str] = set()
    duplicates = 0
    for chunk in bundle.chunks:
        normalized = " ".join(chunk.content.lower().split())
        if normalized in seen:
            duplicates += 1
            continue
        seen.add(normalized)
    return duplicates


def _bundle_budget(case: EvalCase, bundle_max_tokens: int) -> int:
    if case.max_reasonable_bundle_tokens is not None:
        return case.max_reasonable_bundle_tokens
    return bundle_max_tokens


def _grade_bundle(
    case: EvalCase,
    bundle: ContextBundle | None,
    retrieval_grade: str,
    bundle_max_tokens: int,
    evaluate_bundle: bool,
) -> tuple[str | None, dict[str, object]]:
    """Grade a built context bundle independently from retrieval correctness."""

    if not evaluate_bundle:
        return None, {}
    if bundle is None:
        return "INSUFFICIENT", {"bundle_diagnostic": "no context bundle was produced"}

    budget = _bundle_budget(case, bundle_max_tokens)
    expected_paths = _bundle_expected_paths(case)
    preferred_bundle_sources, acceptable_bundle_sources = _bundle_expected_sources(case)
    required_headings = case.required_heading_substrings_for_bundle
    preferred_headings = _bundle_expected_headings(case)
    haystacks = _bundle_haystacks(bundle)
    bundle_paths = [bundle.source_file_path] + [chunk.source_file_path for chunk in bundle.chunks]
    bundle_source_names = _bundle_source_names(bundle)
    bundle_source_coherent = len(bundle_source_names) <= 1
    primary_bundle_source = bundle.source_name or (bundle_source_names[0] if bundle_source_names else None)
    matched_paths: list[str] = []
    for path in bundle_paths:
        for match in _matching_paths(expected_paths, path):
            if match not in matched_paths:
                matched_paths.append(match)
    matched_preferred_sources = [source for source in bundle_source_names if source in preferred_bundle_sources]
    matched_acceptable_sources = [source for source in bundle_source_names if source in acceptable_bundle_sources]
    required_present = _matches_any(required_headings, haystacks)
    preferred_present = _matches_any(preferred_headings, haystacks)
    missing_required = [heading for heading in required_headings if heading not in required_present]
    within_budget = budget <= 0 or bundle.bundle_token_count <= budget
    far_over_budget = budget > 0 and bundle.bundle_token_count > int(budget * 1.35)
    redundancy_count = _bundle_redundancy_count(bundle)
    chunk_count = len(bundle.chunks)
    too_thin = bundle.bundle_token_count < 40 and chunk_count <= 1
    truncated = bundle.selection_strategy == "truncated_match"

    diagnostic_parts: list[str] = []
    if matched_paths:
        diagnostic_parts.append("bundle path matched preferred target")
    elif expected_paths:
        diagnostic_parts.append("bundle path did not match preferred target")
    if primary_bundle_source:
        diagnostic_parts.append(f"bundle primary source: {primary_bundle_source}")
    if preferred_bundle_sources:
        if matched_preferred_sources:
            diagnostic_parts.append(f"bundle source matched preferred source: {', '.join(matched_preferred_sources)}")
        else:
            diagnostic_parts.append("bundle source did not match preferred source")
    elif acceptable_bundle_sources and matched_acceptable_sources:
        diagnostic_parts.append(f"bundle source matched acceptable source: {', '.join(matched_acceptable_sources)}")
    if not case.allow_mixed_bundle_sources and not bundle_source_coherent:
        diagnostic_parts.append(f"bundle mixed sources: {', '.join(bundle_source_names)}")
    if required_present:
        diagnostic_parts.append(f"required headings present: {', '.join(required_present)}")
    if missing_required:
        diagnostic_parts.append(f"missing required headings: {', '.join(missing_required)}")
    if not within_budget:
        diagnostic_parts.append(f"bundle exceeded token budget ({bundle.bundle_token_count}>{budget})")
    if redundancy_count:
        diagnostic_parts.append(f"bundle contains {redundancy_count} duplicate chunk(s)")
    if too_thin:
        diagnostic_parts.append("bundle is very small")
    if truncated:
        diagnostic_parts.append("bundle was truncated to fit the budget")

    if retrieval_grade == "MISS":
        grade = "INSUFFICIENT"
    elif missing_required:
        grade = "INSUFFICIENT"
    elif far_over_budget:
        grade = "INSUFFICIENT"
    elif preferred_bundle_sources and not matched_preferred_sources and not matched_acceptable_sources:
        grade = "INSUFFICIENT"
    elif expected_paths and not matched_paths:
        grade = "PARTIAL"
    elif not case.allow_mixed_bundle_sources and not bundle_source_coherent:
        grade = "PARTIAL"
    elif preferred_bundle_sources and not matched_preferred_sources and matched_acceptable_sources:
        grade = "PARTIAL"
    elif too_thin or truncated or redundancy_count > 0 or not within_budget:
        grade = "PARTIAL"
    elif preferred_headings and not preferred_present:
        grade = "PARTIAL"
    else:
        grade = "COMPLETE"

    if not diagnostic_parts:
        diagnostic_parts.append("bundle contained the primary section within budget")

    return grade, {
        "bundle_path": bundle.source_file_path,
        "bundle_source_name": primary_bundle_source,
        "bundle_source_type": bundle.source_type,
        "bundle_source_names": bundle_source_names,
        "bundle_source_coherent": bundle_source_coherent,
        "bundle_token_count": bundle.bundle_token_count,
        "bundle_chunk_count": chunk_count,
        "bundle_selection_strategy": bundle.selection_strategy,
        "bundle_within_budget": within_budget,
        "bundle_matched_path": bool(matched_paths),
        "bundle_required_headings_present": required_present,
        "bundle_required_headings_missing": missing_required,
        "bundle_diagnostic": "; ".join(diagnostic_parts),
    }


def _failure_summary(case: EvalCase, results: list[QueryResult]) -> str:
    if not results:
        return "No retrieval results returned"

    top = results[0]
    top_path = top.source_file_path
    top_content = top.content
    top_source = result_source_metadata(top)["source_name"]
    if case.preferred_source_names and top_source and top_source not in case.preferred_source_names:
        return f"Top result came from the wrong source ({top_source}) for this query"
    if top_path.startswith("versioned_docs/"):
        return f"Top result was versioned duplicate noise: {top_path}"
    if "import " in top_content or "<" in top_content and "/>" in top_content:
        return f"Top result appears dominated by MDX wrapper content: {top_path}"
    if case.disallowed_document_paths and any(path.lower() in top_path.lower() for path in case.disallowed_document_paths):
        return f"Top result came from an explicitly disallowed path: {top_path}"
    if case.preferred_document_paths or case.acceptable_document_paths:
        return f"Wrong chunk surfaced for the query; none of the preferred or acceptable paths matched within top {len(results)}"
    return "Weak lexical match; no preferred or acceptable target matched"


def _ranking_diagnostic(
    case: EvalCase,
    grade: str,
    matched_rank: int | None,
    preferred_match: EvalMatch | None,
    results: list[QueryResult],
    preferred_source_rank: int | None,
) -> str | None:
    if case.preferred_source_names and results:
        top_source = result_source_metadata(results[0])["source_name"]
        if top_source not in case.preferred_source_names:
            if preferred_source_rank is not None:
                return (
                    f"Preferred source {case.preferred_source_names[0]} was present at rank {preferred_source_rank}, "
                    f"but rank 1 came from {top_source}. This is a source-selection issue, not a recall issue."
                )
            return f"Top result came from {top_source}, but the preferred source is {case.preferred_source_names[0]}."
    if preferred_match is not None and grade == "WEAK PASS":
        top_path = results[0].source_file_path if results else "no result"
        return (
            f"Preferred result was present at rank {preferred_match.rank} but lost to a weaker hit at rank 1 "
            f"({top_path}). This looks like a ranking issue, not a recall issue."
        )
    if preferred_match is not None and preferred_match.rank > 1:
        return f"Preferred result was retrieved at rank {preferred_match.rank}; top-1 ranking still needs improvement."
    if grade == "MISS" and results:
        return "No preferred or acceptable target matched in the retrieved set; inspect top lexical hits for query drift."
    if grade == "MISS":
        return "No candidates were retrieved."
    if matched_rank is not None and matched_rank > 1:
        return f"Correct target was retrieved at rank {matched_rank}, so ranking headroom remains."
    return None


def _score_case(
    case: EvalCase,
    results: list[QueryResult],
    bundle: ContextBundle | None = None,
    bundle_max_tokens: int = 0,
    evaluate_bundle: bool = False,
) -> EvalOutcome:
    matched_result: EvalMatch | None = None
    matched_rank: int | None = None
    preferred_result: EvalMatch | None = None
    preferred_source_rank: int | None = None
    grade = "MISS"
    matched_rule_type: str | None = None

    if case.preferred_source_names:
        for rank, result in enumerate(results, start=1):
            source = result_source_metadata(result)
            if source["source_name"] in case.preferred_source_names:
                preferred_source_rank = rank
                break

    for rank, result in enumerate(results, start=1):
        source = result_source_metadata(result)
        result_grade, matched_on, rule_type = _grade_result(case, result)
        if result_grade == "MISS":
            continue
        candidate_match = EvalMatch(
            rank=rank,
            chunk_id=result.chunk_id,
            source_file_path=result.source_file_path,
            source_name=source["source_name"],
            source_type=source["source_type"],
            source_url=source["source_url"],
            canonical_url=source["canonical_url"],
            document_title=result.document_title,
            section_title=result.section_title,
            heading_path=result.heading_path,
            score=result.score,
            snippet=result.snippet,
            matched_on=matched_on,
            grade=result_grade,
            matched_rule_type=rule_type or "",
        )
        if result_grade == "STRONG PASS" and preferred_result is None:
            preferred_result = candidate_match
        matched_rank = rank
        grade = result_grade
        matched_rule_type = rule_type
        matched_result = candidate_match
        if result_grade == "STRONG PASS":
            break
        if result_grade == "WEAK PASS":
            break

    strong_pass_top_1 = grade == "STRONG PASS" and matched_rank == 1
    strong_pass_top_3 = grade == "STRONG PASS" and matched_rank is not None and matched_rank <= 3
    strong_pass_top_5 = grade == "STRONG PASS" and matched_rank is not None and matched_rank <= 5
    weak_pass_top_1 = grade == "WEAK PASS" and matched_rank == 1
    weak_pass_top_3 = grade == "WEAK PASS" and matched_rank is not None and matched_rank <= 3
    weak_pass_top_5 = grade == "WEAK PASS" and matched_rank is not None and matched_rank <= 5

    bundle_grade, bundle_details = _grade_bundle(case, bundle, grade, bundle_max_tokens, evaluate_bundle)
    expected_source_name = _expected_source_name(case)

    return EvalOutcome(
        case_id=case.id,
        query=case.query,
        bucket=case.bucket,
        query_style=case.query_style,
        concept_id=case.concept_id,
        expected_source_name=expected_source_name,
        acceptable_source_names=case.acceptable_source_names,
        top_k=case.top_k,
        grade=grade,
        strong_pass_top_1=strong_pass_top_1,
        strong_pass_top_3=strong_pass_top_3,
        strong_pass_top_5=strong_pass_top_5,
        weak_pass_top_1=weak_pass_top_1,
        weak_pass_top_3=weak_pass_top_3,
        weak_pass_top_5=weak_pass_top_5,
        matched_result_rank=matched_rank,
        matched_result_path=matched_result.source_file_path if matched_result else None,
        matched_result_heading=" > ".join(matched_result.heading_path) if matched_result else None,
        matched_result_source_name=matched_result.source_name if matched_result else None,
        matched_result_source_type=matched_result.source_type if matched_result else None,
        matched_rule_type=matched_rule_type,
        matched_result=matched_result,
        failure_summary=None if grade != "MISS" else _failure_summary(case, results),
        preferred_result_rank=preferred_result.rank if preferred_result else None,
        preferred_result_path=preferred_result.source_file_path if preferred_result else None,
        preferred_result_heading=" > ".join(preferred_result.heading_path) if preferred_result else None,
        preferred_source_rank=preferred_source_rank,
        ranking_diagnostic=_ranking_diagnostic(case, grade, matched_rank, preferred_result, results, preferred_source_rank),
        bundle_grade=bundle_grade,
        bundle_path=bundle_details.get("bundle_path"),
        bundle_source_name=bundle_details.get("bundle_source_name"),
        bundle_source_type=bundle_details.get("bundle_source_type"),
        bundle_source_names=bundle_details.get("bundle_source_names", []),
        bundle_source_coherent=bundle_details.get("bundle_source_coherent"),
        bundle_token_count=bundle_details.get("bundle_token_count"),
        bundle_chunk_count=bundle_details.get("bundle_chunk_count"),
        bundle_selection_strategy=bundle_details.get("bundle_selection_strategy"),
        bundle_within_budget=bundle_details.get("bundle_within_budget"),
        bundle_matched_path=bundle_details.get("bundle_matched_path"),
        bundle_required_headings_present=bundle_details.get("bundle_required_headings_present", []),
        bundle_required_headings_missing=bundle_details.get("bundle_required_headings_missing", []),
        bundle_diagnostic=bundle_details.get("bundle_diagnostic"),
    )


def _window_stats(outcomes: list[EvalOutcome], window: int) -> EvalWindowStats:
    strong = 0
    weak = 0
    for outcome in outcomes:
        if outcome.grade == "STRONG PASS" and outcome.matched_result_rank is not None and outcome.matched_result_rank <= window:
            strong += 1
        elif outcome.grade == "WEAK PASS" and outcome.matched_result_rank is not None and outcome.matched_result_rank <= window:
            weak += 1
    total = len(outcomes)
    misses = total - strong - weak
    return EvalWindowStats(
        strong_passes=strong,
        weak_passes=weak,
        misses=misses,
        strong_pass_rate=(strong / total) if total else 0.0,
        weak_pass_rate=(weak / total) if total else 0.0,
    )


def _build_report(outcomes: list[EvalOutcome]) -> EvalReport:
    total = len(outcomes)
    if total == 0:
        empty_window = EvalWindowStats(
            strong_passes=0,
            weak_passes=0,
            misses=0,
            strong_pass_rate=0.0,
            weak_pass_rate=0.0,
        )
        return EvalReport(
            total_queries=0,
            strong_passes=0,
            weak_passes=0,
            misses=0,
            top_1=empty_window,
            top_3=empty_window,
            top_5=empty_window,
            buckets={},
            expected_sources={},
            query_styles={},
            concepts={},
            bundle_overall=None,
            bundle_buckets={},
            bundle_expected_sources={},
            source_confusions=[],
            outcomes=[],
        )

    strong_passes = sum(1 for outcome in outcomes if outcome.grade == "STRONG PASS")
    weak_passes = sum(1 for outcome in outcomes if outcome.grade == "WEAK PASS")
    misses = total - strong_passes - weak_passes
    buckets = _group_reports(outcomes, key_fn=lambda outcome: outcome.bucket)
    expected_sources = _group_reports(
        outcomes,
        key_fn=lambda outcome: outcome.expected_source_name,
        include_empty=False,
    )
    query_styles = _group_reports(
        outcomes,
        key_fn=lambda outcome: outcome.query_style,
        include_empty=False,
    )
    concepts = _group_reports(
        outcomes,
        key_fn=lambda outcome: outcome.concept_id,
        include_empty=False,
    )
    bundle_overall = _bundle_stats(outcomes)
    bundle_buckets = _bundle_group_stats(outcomes, key_fn=lambda outcome: outcome.bucket)
    bundle_expected_sources = _bundle_group_stats(
        outcomes,
        key_fn=lambda outcome: outcome.expected_source_name,
    )
    return EvalReport(
        total_queries=total,
        strong_passes=strong_passes,
        weak_passes=weak_passes,
        misses=misses,
        top_1=_window_stats(outcomes, 1),
        top_3=_window_stats(outcomes, 3),
        top_5=_window_stats(outcomes, 5),
        buckets=buckets,
        expected_sources=expected_sources,
        query_styles=query_styles,
        concepts=concepts,
        bundle_overall=bundle_overall,
        bundle_buckets=bundle_buckets,
        bundle_expected_sources=bundle_expected_sources,
        source_confusions=_source_confusion_cases(outcomes),
        outcomes=outcomes,
    )


def _bundle_stats(outcomes: list[EvalOutcome]) -> BundleGradeStats | None:
    graded = [outcome for outcome in outcomes if outcome.bundle_grade is not None]
    if not graded:
        return None
    total = len(graded)
    complete = sum(1 for outcome in graded if outcome.bundle_grade == "COMPLETE")
    partial = sum(1 for outcome in graded if outcome.bundle_grade == "PARTIAL")
    insufficient = total - complete - partial
    return BundleGradeStats(
        total_evaluated=total,
        complete=complete,
        partial=partial,
        insufficient=insufficient,
        complete_rate=complete / total,
        partial_rate=partial / total,
        insufficient_rate=insufficient / total,
    )


def _comparison_status_from_flags(improved: bool, regressed: bool) -> str:
    if improved and regressed:
        return "mixed"
    if regressed:
        return "regressed"
    if improved:
        return "improved"
    return "unchanged"


def _metric_delta(current: float | int, baseline: float | int) -> BaselineMetricDelta:
    return BaselineMetricDelta(current=current, baseline=baseline, delta=float(current) - float(baseline))


def _compare_metric_set(current_metrics: dict[str, float | int], baseline_metrics: dict[str, float | int], better_when_higher: set[str]) -> tuple[str, dict[str, BaselineMetricDelta]]:
    deltas: dict[str, BaselineMetricDelta] = {}
    improved = False
    regressed = False
    for key, current_value in current_metrics.items():
        baseline_value = baseline_metrics[key]
        delta = _metric_delta(current_value, baseline_value)
        deltas[key] = delta
        if current_value == baseline_value:
            continue
        if key in better_when_higher:
            improved = improved or current_value > baseline_value
            regressed = regressed or current_value < baseline_value
        else:
            improved = improved or current_value < baseline_value
            regressed = regressed or current_value > baseline_value
    return _comparison_status_from_flags(improved, regressed), deltas


def _retrieval_metrics(report: EvalReport) -> dict[str, float | int]:
    return {
        "strong_passes": report.strong_passes,
        "weak_passes": report.weak_passes,
        "misses": report.misses,
        "top_1_strong_pass_rate": report.top_1.strong_pass_rate,
        "top_3_strong_pass_rate": report.top_3.strong_pass_rate,
        "top_5_strong_pass_rate": report.top_5.strong_pass_rate,
    }


def _bundle_metrics(stats: BundleGradeStats) -> dict[str, float | int]:
    return {
        "complete": stats.complete,
        "partial": stats.partial,
        "insufficient": stats.insufficient,
        "complete_rate": stats.complete_rate,
    }


def _group_retrieval_metrics(report: EvalGroupReport) -> dict[str, float | int]:
    return {
        "strong_passes": report.strong_passes,
        "weak_passes": report.weak_passes,
        "misses": report.misses,
        "top_1_strong_pass_rate": report.top_1.strong_pass_rate,
    }


def _compare_bucket_maps(
    current: dict[str, EvalGroupReport],
    baseline: dict[str, EvalGroupReport],
) -> list[BaselineBucketChange]:
    labels = sorted(set(current) | set(baseline))
    changes: list[BaselineBucketChange] = []
    better_when_higher = {"strong_passes", "top_1_strong_pass_rate"}
    for label in labels:
        current_bucket = current.get(label)
        baseline_bucket = baseline.get(label)
        if current_bucket is None or baseline_bucket is None:
            continue
        status, _ = _compare_metric_set(
            _group_retrieval_metrics(current_bucket),
            _group_retrieval_metrics(baseline_bucket),
            better_when_higher=better_when_higher,
        )
        if status == "unchanged":
            continue
        changes.append(
            BaselineBucketChange(
                metric_family="retrieval",
                label=label,
                status=status,
                current=_group_retrieval_metrics(current_bucket),
                baseline=_group_retrieval_metrics(baseline_bucket),
            )
        )
    return changes


def _compare_bundle_bucket_maps(
    current: dict[str, BundleGradeStats],
    baseline: dict[str, BundleGradeStats],
) -> list[BaselineBucketChange]:
    labels = sorted(set(current) | set(baseline))
    changes: list[BaselineBucketChange] = []
    better_when_higher = {"complete", "complete_rate"}
    for label in labels:
        current_bucket = current.get(label)
        baseline_bucket = baseline.get(label)
        if current_bucket is None or baseline_bucket is None:
            continue
        current_metrics = _bundle_metrics(current_bucket)
        baseline_metrics = _bundle_metrics(baseline_bucket)
        status, _ = _compare_metric_set(current_metrics, baseline_metrics, better_when_higher=better_when_higher)
        if status == "unchanged":
            continue
        changes.append(
            BaselineBucketChange(
                metric_family="bundle",
                label=label,
                status=status,
                current=current_metrics,
                baseline=baseline_metrics,
            )
        )
    return changes


def _compare_case_changes(current: EvalReport, baseline: EvalReport) -> list[BaselineCaseChange]:
    baseline_outcomes = {outcome.case_id: outcome for outcome in baseline.outcomes}
    changes: list[BaselineCaseChange] = []
    for outcome in current.outcomes:
        baseline_outcome = baseline_outcomes.get(outcome.case_id)
        if baseline_outcome is None:
            continue
        retrieval_changed = outcome.grade != baseline_outcome.grade
        bundle_changed = outcome.bundle_grade != baseline_outcome.bundle_grade
        if not retrieval_changed and not bundle_changed:
            continue
        changes.append(
            BaselineCaseChange(
                case_id=outcome.case_id,
                query=outcome.query,
                retrieval_from=baseline_outcome.grade,
                retrieval_to=outcome.grade,
                bundle_from=baseline_outcome.bundle_grade,
                bundle_to=outcome.bundle_grade,
            )
        )
    return changes


def compare_eval_reports(current: EvalReport, baseline: EvalReport, baseline_path: Path) -> BaselineComparison:
    """Compare a current eval report against a supplied baseline report."""

    retrieval_status, retrieval_deltas = _compare_metric_set(
        _retrieval_metrics(current),
        _retrieval_metrics(baseline),
        better_when_higher={"strong_passes", "top_1_strong_pass_rate", "top_3_strong_pass_rate", "top_5_strong_pass_rate"},
    )

    bundle_status: str | None = None
    bundle_deltas: dict[str, BaselineMetricDelta] = {}
    if current.bundle_overall is not None and baseline.bundle_overall is not None:
        bundle_status, bundle_deltas = _compare_metric_set(
            _bundle_metrics(current.bundle_overall),
            _bundle_metrics(baseline.bundle_overall),
            better_when_higher={"complete", "complete_rate"},
        )

    overall_flags = {retrieval_status}
    if bundle_status is not None:
        overall_flags.add(bundle_status)
    if "regressed" in overall_flags and "improved" in overall_flags:
        status = "mixed"
    elif "mixed" in overall_flags:
        status = "mixed"
    elif "regressed" in overall_flags:
        status = "regressed"
    elif "improved" in overall_flags:
        status = "improved"
    else:
        status = "unchanged"

    return BaselineComparison(
        status=status,
        baseline_provided=True,
        baseline_path=str(baseline_path),
        retrieval_status=retrieval_status,
        bundle_status=bundle_status,
        retrieval_deltas=retrieval_deltas,
        bundle_deltas=bundle_deltas,
        changed_retrieval_buckets=_compare_bucket_maps(current.buckets, baseline.buckets),
        changed_bundle_buckets=_compare_bundle_bucket_maps(current.bundle_buckets, baseline.bundle_buckets),
        changed_cases=_compare_case_changes(current, baseline),
    )


def _bundle_group_stats(outcomes: list[EvalOutcome], key_fn) -> dict[str, BundleGradeStats]:
    grouped: dict[str, list[EvalOutcome]] = {}
    for outcome in outcomes:
        if outcome.bundle_grade is None:
            continue
        label = str(key_fn(outcome) or "uncategorized")
        grouped.setdefault(label, []).append(outcome)
    reports: dict[str, BundleGradeStats] = {}
    for label, grouped_outcomes in sorted(grouped.items()):
        stats = _bundle_stats(grouped_outcomes)
        if stats is not None:
            reports[label] = stats
    return reports


def _source_confusion_cases(outcomes: list[EvalOutcome]) -> list[SourceConfusionCase]:
    confusions: list[SourceConfusionCase] = []
    for outcome in outcomes:
        if outcome.expected_source_name is None:
            continue
        retrieval_confused = outcome.matched_result_source_name not in {
            outcome.expected_source_name,
            *outcome.acceptable_source_names,
        }
        bundle_confused = bool(outcome.bundle_source_names) and (
            not outcome.bundle_source_coherent
            or any(source not in {outcome.expected_source_name, *outcome.acceptable_source_names} for source in outcome.bundle_source_names)
        )
        if not retrieval_confused and not bundle_confused:
            continue
        confusions.append(
            SourceConfusionCase(
                case_id=outcome.case_id,
                query=outcome.query,
                expected_source_name=outcome.expected_source_name,
                acceptable_source_names=outcome.acceptable_source_names,
                matched_result_source_name=outcome.matched_result_source_name,
                matched_result_path=outcome.matched_result_path,
                bundle_source_names=outcome.bundle_source_names,
                grade=outcome.grade,
                bundle_grade=outcome.bundle_grade,
            )
        )
    return confusions


def _group_reports(
    outcomes: list[EvalOutcome],
    key_fn,
    include_empty: bool = True,
) -> dict[str, EvalGroupReport]:
    grouped: dict[str, list[EvalOutcome]] = {}
    for outcome in outcomes:
        label = key_fn(outcome)
        if label is None and not include_empty:
            continue
        group_key = str(label or "uncategorized")
        grouped.setdefault(group_key, []).append(outcome)

    reports: dict[str, EvalGroupReport] = {}
    for label, group_outcomes in sorted(grouped.items()):
        total = len(group_outcomes)
        strong_passes = sum(1 for outcome in group_outcomes if outcome.grade == "STRONG PASS")
        weak_passes = sum(1 for outcome in group_outcomes if outcome.grade == "WEAK PASS")
        misses = total - strong_passes - weak_passes
        reports[label] = EvalGroupReport(
            label=label,
            total_queries=total,
            strong_passes=strong_passes,
            weak_passes=weak_passes,
            misses=misses,
            top_1=_window_stats(group_outcomes, 1),
            top_3=_window_stats(group_outcomes, 3),
            top_5=_window_stats(group_outcomes, 5),
            case_ids=[outcome.case_id for outcome in group_outcomes],
        )
    return reports


def assert_report_consistent(report: EvalReport) -> None:
    """Fail fast if aggregate report fields diverge from per-query outcomes."""

    recomputed = _build_report(report.outcomes)
    if report.model_dump(exclude={"baseline_comparison"}) != recomputed.model_dump(exclude={"baseline_comparison"}):
        msg = "Eval report aggregates diverged from per-query outcomes"
        raise ValueError(msg)


def render_eval_text(report: EvalReport, show_weak_details: bool = False) -> str:
    """Render human-readable eval output from the canonical report object."""

    assert_report_consistent(report)
    lines = [
        f"total_queries: {report.total_queries}",
        f"strong_passes: {report.strong_passes}",
        f"weak_passes: {report.weak_passes}",
        f"misses: {report.misses}",
        f"top_1_strong_pass_rate: {report.top_1.strong_pass_rate:.3f}",
        f"top_1_weak_pass_rate: {report.top_1.weak_pass_rate:.3f}",
        f"top_3_strong_pass_rate: {report.top_3.strong_pass_rate:.3f}",
        f"top_3_weak_pass_rate: {report.top_3.weak_pass_rate:.3f}",
        f"top_5_strong_pass_rate: {report.top_5.strong_pass_rate:.3f}",
        f"top_5_weak_pass_rate: {report.top_5.weak_pass_rate:.3f}",
        "",
    ]
    if report.bundle_overall is not None:
        lines.extend(
            [
                f"bundle_complete: {report.bundle_overall.complete}",
                f"bundle_partial: {report.bundle_overall.partial}",
                f"bundle_insufficient: {report.bundle_overall.insufficient}",
                f"bundle_complete_rate: {report.bundle_overall.complete_rate:.3f}",
                f"bundle_partial_rate: {report.bundle_overall.partial_rate:.3f}",
                f"bundle_insufficient_rate: {report.bundle_overall.insufficient_rate:.3f}",
                "",
            ]
        )
    if report.baseline_comparison is not None:
        lines.extend(
            [
                f"baseline_comparison_status: {report.baseline_comparison.status}",
                f"baseline_path: {report.baseline_comparison.baseline_path}",
                f"retrieval_comparison_status: {report.baseline_comparison.retrieval_status}",
                f"bundle_comparison_status: {report.baseline_comparison.bundle_status}",
                "",
            ]
        )
        if report.baseline_comparison.retrieval_deltas:
            lines.append("retrieval_deltas:")
            for key, delta in report.baseline_comparison.retrieval_deltas.items():
                lines.append(f"  {key}: current={delta.current} baseline={delta.baseline} delta={delta.delta:.3f}")
            lines.append("")
        if report.baseline_comparison.bundle_deltas:
            lines.append("bundle_deltas:")
            for key, delta in report.baseline_comparison.bundle_deltas.items():
                lines.append(f"  {key}: current={delta.current} baseline={delta.baseline} delta={delta.delta:.3f}")
            lines.append("")
        if report.baseline_comparison.changed_retrieval_buckets:
            lines.append("changed_retrieval_buckets:")
            for change in report.baseline_comparison.changed_retrieval_buckets:
                lines.append(f"  {change.label}: {change.status}")
            lines.append("")
        if report.baseline_comparison.changed_bundle_buckets:
            lines.append("changed_bundle_buckets:")
            for change in report.baseline_comparison.changed_bundle_buckets:
                lines.append(f"  {change.label}: {change.status}")
            lines.append("")
        if report.baseline_comparison.changed_cases:
            lines.append("changed_cases:")
            for change in report.baseline_comparison.changed_cases:
                lines.append(
                    f"  {change.case_id}: retrieval {change.retrieval_from}->{change.retrieval_to}, "
                    f"bundle {change.bundle_from}->{change.bundle_to}"
                )
            lines.append("")
    if report.buckets:
        lines.append("bucket_summary:")
        for bucket, bucket_report in report.buckets.items():
            lines.append(
                f"  {bucket}: strong={bucket_report.strong_passes} weak={bucket_report.weak_passes} "
                f"miss={bucket_report.misses} top_1_strong={bucket_report.top_1.strong_pass_rate:.3f}"
            )
        lines.append("")
    if report.expected_sources:
        lines.append("expected_source_summary:")
        for source_name, source_report in report.expected_sources.items():
            lines.append(
                f"  {source_name}: strong={source_report.strong_passes} weak={source_report.weak_passes} "
                f"miss={source_report.misses} top_1_strong={source_report.top_1.strong_pass_rate:.3f}"
            )
        lines.append("")
    if report.query_styles:
        lines.append("query_style_summary:")
        for style, style_report in report.query_styles.items():
            lines.append(
                f"  {style}: strong={style_report.strong_passes} weak={style_report.weak_passes} "
                f"miss={style_report.misses} top_1_strong={style_report.top_1.strong_pass_rate:.3f}"
            )
        lines.append("")
    if report.bundle_buckets:
        lines.append("bundle_bucket_summary:")
        for bucket, bucket_report in report.bundle_buckets.items():
            lines.append(
                f"  {bucket}: complete={bucket_report.complete} partial={bucket_report.partial} "
                f"insufficient={bucket_report.insufficient} complete_rate={bucket_report.complete_rate:.3f}"
            )
        lines.append("")
    if report.bundle_expected_sources:
        lines.append("bundle_expected_source_summary:")
        for source_name, source_report in report.bundle_expected_sources.items():
            lines.append(
                f"  {source_name}: complete={source_report.complete} partial={source_report.partial} "
                f"insufficient={source_report.insufficient} complete_rate={source_report.complete_rate:.3f}"
            )
        lines.append("")
    if report.source_confusions:
        lines.append("source_confusions:")
        for confusion in report.source_confusions:
            lines.append(
                f"  {confusion.case_id}: expected={confusion.expected_source_name} actual={confusion.matched_result_source_name} "
                f"bundle_sources={','.join(confusion.bundle_source_names) or '-'} grade={confusion.grade} bundle={confusion.bundle_grade}"
            )
        lines.append("")
    if report.concepts:
        lines.append("concept_summary:")
        for concept, concept_report in report.concepts.items():
            lines.append(
                f"  {concept}: strong={concept_report.strong_passes} weak={concept_report.weak_passes} "
                f"miss={concept_report.misses} top_1_strong={concept_report.top_1.strong_pass_rate:.3f}"
            )
        lines.append("")
    for outcome in report.outcomes:
        lines.append(f"{outcome.grade} {outcome.case_id}: {outcome.query}")
        lines.append(
            f"  bucket={outcome.bucket} style={outcome.query_style or 'unspecified'} concept={outcome.concept_id or outcome.case_id} "
            f"expected_source={outcome.expected_source_name or 'unspecified'}"
        )
        if outcome.matched_result is not None:
            lines.append(
                f"  best_match_rank={outcome.matched_result.rank} path={outcome.matched_result.source_file_path} "
                f"source={outcome.matched_result.source_name or '-'} "
                f"rule={outcome.matched_result.matched_rule_type} matched_on={', '.join(outcome.matched_result.matched_on)}"
            )
        elif outcome.matched_result_rank is not None:
            lines.append(
                f"  best_match_rank={outcome.matched_result_rank} path={outcome.matched_result_path} "
                f"source={outcome.matched_result_source_name or '-'}"
            )
        if show_weak_details and outcome.preferred_result_rank is not None:
            lines.append(
                f"  preferred_result_rank={outcome.preferred_result_rank} path={outcome.preferred_result_path} "
                f"heading={outcome.preferred_result_heading}"
            )
        if show_weak_details and outcome.preferred_source_rank is not None:
            lines.append(f"  preferred_source_rank={outcome.preferred_source_rank}")
        if show_weak_details and outcome.ranking_diagnostic:
            lines.append(f"  ranking={outcome.ranking_diagnostic}")
        if outcome.bundle_grade is not None:
            lines.append(
                f"  bundle_grade={outcome.bundle_grade} path={outcome.bundle_path} "
                f"source={outcome.bundle_source_name or '-'} "
                f"tokens={outcome.bundle_token_count} strategy={outcome.bundle_selection_strategy}"
            )
        if show_weak_details and outcome.bundle_diagnostic:
            lines.append(f"  bundle={outcome.bundle_diagnostic}")
        if outcome.failure_summary:
            lines.append(f"  failure={outcome.failure_summary}")
    return "\n".join(lines)


def render_eval_summary_markdown(report: EvalReport) -> str:
    """Render a markdown summary from the canonical eval report."""

    assert_report_consistent(report)
    summary = dedent(
        f"""
        # Eval Summary

        - Total queries: `{report.total_queries}`
        - Strong passes: `{report.strong_passes}`
        - Weak passes: `{report.weak_passes}`
        - Misses: `{report.misses}`
        - Top-1 strong-pass rate: `{report.top_1.strong_pass_rate:.3f}`
        - Top-1 weak-pass rate: `{report.top_1.weak_pass_rate:.3f}`
        - Top-3 strong-pass rate: `{report.top_3.strong_pass_rate:.3f}`
        - Top-3 weak-pass rate: `{report.top_3.weak_pass_rate:.3f}`
        - Top-5 strong-pass rate: `{report.top_5.strong_pass_rate:.3f}`
        - Top-5 weak-pass rate: `{report.top_5.weak_pass_rate:.3f}`
        """
    ).strip()

    if report.bundle_overall is not None:
        summary = "\n".join(
            [
                summary,
                f"- Bundle complete: `{report.bundle_overall.complete}`",
                f"- Bundle partial: `{report.bundle_overall.partial}`",
                f"- Bundle insufficient: `{report.bundle_overall.insufficient}`",
                f"- Bundle complete rate: `{report.bundle_overall.complete_rate:.3f}`",
            ]
        )
    if report.baseline_comparison is not None:
        summary = "\n".join(
            [
                summary,
                f"- Baseline comparison: `{report.baseline_comparison.status}`",
                f"- Baseline path: `{report.baseline_comparison.baseline_path}`",
                f"- Retrieval comparison: `{report.baseline_comparison.retrieval_status}`",
                f"- Bundle comparison: `{report.baseline_comparison.bundle_status}`",
            ]
        )

    if report.buckets:
        lines = [summary, "", "## Buckets", ""]
        for bucket, bucket_report in report.buckets.items():
            lines.append(
                f"- Bucket `{bucket}`: `{bucket_report.strong_passes}` strong / `{bucket_report.weak_passes}` weak / "
                f"`{bucket_report.misses}` miss, top-1 strong `{bucket_report.top_1.strong_pass_rate:.3f}`"
            )
        if report.expected_sources:
            lines.extend(["", "## Expected Sources", ""])
            for source_name, source_report in report.expected_sources.items():
                lines.append(
                    f"- Source `{source_name}`: `{source_report.strong_passes}` strong / `{source_report.weak_passes}` weak / "
                    f"`{source_report.misses}` miss across `{source_report.total_queries}` queries"
                )
        if report.query_styles:
            lines.extend(["", "## Query Styles", ""])
            for style, style_report in report.query_styles.items():
                lines.append(
                    f"- Style `{style}`: `{style_report.strong_passes}` strong / `{style_report.weak_passes}` weak / "
                    f"`{style_report.misses}` miss across `{style_report.total_queries}` queries"
                )
        if report.concepts:
            lines.extend(["", "## Concepts", ""])
            for concept, concept_report in report.concepts.items():
                lines.append(
                    f"- Concept `{concept}`: `{concept_report.strong_passes}` strong / `{concept_report.weak_passes}` weak / "
                    f"`{concept_report.misses}` miss across `{concept_report.total_queries}` queries"
                )
        if report.bundle_buckets:
            lines.extend(["", "## Bundle Buckets", ""])
            for bucket, bucket_report in report.bundle_buckets.items():
                lines.append(
                    f"- Bundle bucket `{bucket}`: `{bucket_report.complete}` complete / `{bucket_report.partial}` partial / "
                    f"`{bucket_report.insufficient}` insufficient, complete `{bucket_report.complete_rate:.3f}`"
                )
        if report.bundle_expected_sources:
            lines.extend(["", "## Bundle Sources", ""])
            for source_name, source_report in report.bundle_expected_sources.items():
                lines.append(
                    f"- Bundle source `{source_name}`: `{source_report.complete}` complete / `{source_report.partial}` partial / "
                    f"`{source_report.insufficient}` insufficient across `{source_report.total_evaluated}` evaluated bundles"
                )
        if report.source_confusions:
            lines.extend(["", "## Source Confusions", ""])
            for confusion in report.source_confusions:
                lines.append(
                    f"- Case `{confusion.case_id}` expected `{confusion.expected_source_name}` but got `{confusion.matched_result_source_name}` "
                    f"with bundle sources `{', '.join(confusion.bundle_source_names) or '-'}`"
                )
        if report.baseline_comparison is not None and report.baseline_comparison.changed_retrieval_buckets:
            lines.extend(["", "## Changed Retrieval Buckets", ""])
            for change in report.baseline_comparison.changed_retrieval_buckets:
                lines.append(f"- Retrieval bucket `{change.label}`: `{change.status}`")
        if report.baseline_comparison is not None and report.baseline_comparison.changed_bundle_buckets:
            lines.extend(["", "## Changed Bundle Buckets", ""])
            for change in report.baseline_comparison.changed_bundle_buckets:
                lines.append(f"- Bundle bucket `{change.label}`: `{change.status}`")
        if report.baseline_comparison is not None and report.baseline_comparison.changed_cases:
            lines.extend(["", "## Changed Cases", ""])
            for change in report.baseline_comparison.changed_cases:
                lines.append(
                    f"- Case `{change.case_id}`: retrieval `{change.retrieval_from}` -> `{change.retrieval_to}`, "
                    f"bundle `{change.bundle_from}` -> `{change.bundle_to}`"
                )
        return "\n".join(lines)
    return summary


def run_eval(
    db_path: Path,
    eval_file: Path,
    with_bundles: bool = False,
    bundle_max_tokens: int = 450,
    include_previous: bool = True,
    include_next: bool = True,
    baseline: Path | None = None,
) -> EvalReport:
    """Run retrieval evaluation over the configured set of queries."""

    cases = load_eval_cases(eval_file)
    outcomes: list[EvalOutcome] = []
    for case in cases:
        results = query_chunks(db_path=db_path, query_text=case.query, top_k=max(case.top_k, 5))
        bundle = None
        if with_bundles and results:
            bundles = build_context_bundles(
                db_path=db_path,
                results=results[:1],
                support_results=results,
                query_text=case.query,
                include_previous=include_previous,
                include_next=include_next,
                bundle_max_tokens=case.max_reasonable_bundle_tokens or bundle_max_tokens,
            )
            bundle = bundles[0] if bundles else None
        outcomes.append(
            _score_case(
                case,
                results,
                bundle=bundle,
                bundle_max_tokens=bundle_max_tokens,
                evaluate_bundle=with_bundles,
            )
        )
    report = _build_report(outcomes)
    if baseline is not None:
        baseline_report = load_eval_report_artifact(baseline)
        report = report.model_copy(update={"baseline_comparison": compare_eval_reports(report, baseline_report, baseline)})
    assert_report_consistent(report)
    return report
