"""Lightweight retrieval evaluation harness with explicit strong/weak grading."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import yaml

from agentic_docs.models import (
    EvalCase,
    EvalMatch,
    EvalOutcome,
    EvalReport,
    EvalWindowStats,
    QueryResult,
)
from agentic_docs.query_service import canonical_path_key, query_chunks


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


def _matches_any(substrings: list[str], haystacks: list[str]) -> list[str]:
    matches: list[str] = []
    for needle in substrings:
        lowered = needle.lower()
        if any(lowered in haystack for haystack in haystacks):
            matches.append(needle)
    return matches


def _grade_result(case: EvalCase, result: QueryResult) -> tuple[str, list[str], str | None]:
    lower_path = result.source_file_path.lower()
    heading_text = " > ".join(result.heading_path).lower()
    section_text = (result.section_title or "").lower()
    document_text = result.document_title.lower()
    haystacks = [heading_text, section_text, document_text]

    disallowed_paths = _matches_any(case.disallowed_document_paths, [lower_path, canonical_path_key(lower_path)])
    if disallowed_paths:
        return "MISS", [f"disallowed_path:{path}" for path in disallowed_paths], None

    preferred_paths = _matches_any(case.preferred_document_paths, [lower_path, canonical_path_key(lower_path)])
    acceptable_paths = _matches_any(case.acceptable_document_paths, [lower_path, canonical_path_key(lower_path)])
    preferred_headings = _matches_any(case.preferred_heading_substrings, haystacks)
    acceptable_headings = _matches_any(case.acceptable_heading_substrings, haystacks)

    if preferred_paths or preferred_headings:
        matched_on = [f"preferred_path:{path}" for path in preferred_paths] + [
            f"preferred_heading:{heading}" for heading in preferred_headings
        ]
        rule_type = "preferred_path+heading" if preferred_paths and preferred_headings else "preferred_path" if preferred_paths else "preferred_heading"
        return "STRONG PASS", matched_on, rule_type

    if acceptable_paths or acceptable_headings:
        matched_on = [f"acceptable_path:{path}" for path in acceptable_paths] + [
            f"acceptable_heading:{heading}" for heading in acceptable_headings
        ]
        rule_type = "acceptable_path+heading" if acceptable_paths and acceptable_headings else "acceptable_path" if acceptable_paths else "acceptable_heading"
        return "WEAK PASS", matched_on, rule_type

    return "MISS", [], None


def _failure_summary(case: EvalCase, results: list[QueryResult]) -> str:
    if not results:
        return "No retrieval results returned"

    top = results[0]
    top_path = top.source_file_path
    top_content = top.content
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
    grade: str,
    matched_rank: int | None,
    preferred_match: EvalMatch | None,
    results: list[QueryResult],
) -> str | None:
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


def _score_case(case: EvalCase, results: list[QueryResult]) -> EvalOutcome:
    matched_result: EvalMatch | None = None
    matched_rank: int | None = None
    preferred_result: EvalMatch | None = None
    grade = "MISS"
    matched_rule_type: str | None = None

    for rank, result in enumerate(results, start=1):
        result_grade, matched_on, rule_type = _grade_result(case, result)
        if result_grade == "MISS":
            continue
        candidate_match = EvalMatch(
            rank=rank,
            chunk_id=result.chunk_id,
            source_file_path=result.source_file_path,
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

    return EvalOutcome(
        case_id=case.id,
        query=case.query,
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
        matched_rule_type=matched_rule_type,
        matched_result=matched_result,
        failure_summary=None if grade != "MISS" else _failure_summary(case, results),
        preferred_result_rank=preferred_result.rank if preferred_result else None,
        preferred_result_path=preferred_result.source_file_path if preferred_result else None,
        preferred_result_heading=" > ".join(preferred_result.heading_path) if preferred_result else None,
        ranking_diagnostic=_ranking_diagnostic(grade, matched_rank, preferred_result, results),
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
            outcomes=[],
        )

    strong_passes = sum(1 for outcome in outcomes if outcome.grade == "STRONG PASS")
    weak_passes = sum(1 for outcome in outcomes if outcome.grade == "WEAK PASS")
    misses = total - strong_passes - weak_passes
    return EvalReport(
        total_queries=total,
        strong_passes=strong_passes,
        weak_passes=weak_passes,
        misses=misses,
        top_1=_window_stats(outcomes, 1),
        top_3=_window_stats(outcomes, 3),
        top_5=_window_stats(outcomes, 5),
        outcomes=outcomes,
    )


def assert_report_consistent(report: EvalReport) -> None:
    """Fail fast if aggregate report fields diverge from per-query outcomes."""

    recomputed = _build_report(report.outcomes)
    if report.model_dump() != recomputed.model_dump():
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
    for outcome in report.outcomes:
        lines.append(f"{outcome.grade} {outcome.case_id}: {outcome.query}")
        if outcome.matched_result is not None:
            lines.append(
                f"  best_match_rank={outcome.matched_result.rank} path={outcome.matched_result.source_file_path} "
                f"rule={outcome.matched_result.matched_rule_type} matched_on={', '.join(outcome.matched_result.matched_on)}"
            )
        if show_weak_details and outcome.preferred_result_rank is not None:
            lines.append(
                f"  preferred_result_rank={outcome.preferred_result_rank} path={outcome.preferred_result_path} "
                f"heading={outcome.preferred_result_heading}"
            )
        if show_weak_details and outcome.ranking_diagnostic:
            lines.append(f"  ranking={outcome.ranking_diagnostic}")
        if outcome.failure_summary:
            lines.append(f"  failure={outcome.failure_summary}")
    return "\n".join(lines)


def render_eval_summary_markdown(report: EvalReport) -> str:
    """Render a markdown summary from the canonical eval report."""

    assert_report_consistent(report)
    return dedent(
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


def run_eval(db_path: Path, eval_file: Path) -> EvalReport:
    """Run retrieval evaluation over the configured set of queries."""

    cases = load_eval_cases(eval_file)
    outcomes: list[EvalOutcome] = []
    for case in cases:
        results = query_chunks(db_path=db_path, query_text=case.query, top_k=max(case.top_k, 5))
        outcomes.append(_score_case(case, results))
    report = _build_report(outcomes)
    assert_report_consistent(report)
    return report
