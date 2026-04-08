from pathlib import Path

import json
import re

import pytest

from agentic_docs.evaluation import (
    _build_report,
    assert_report_consistent,
    compare_eval_reports,
    load_eval_cases,
    load_eval_report_artifact,
    render_eval_summary_markdown,
    render_eval_text,
    run_eval,
)
from agentic_docs.ingest import ingest_source
from agentic_docs.models import BundleGradeStats, EvalOutcome, EvalReport, EvalWindowStats
from agentic_docs.query_service import query_chunks


def _window(strong: int, weak: int, misses: int, total: int) -> EvalWindowStats:
    return EvalWindowStats(
        strong_passes=strong,
        weak_passes=weak,
        misses=misses,
        strong_pass_rate=strong / total if total else 0.0,
        weak_pass_rate=weak / total if total else 0.0,
    )


def _report(
    *,
    strong: int,
    weak: int,
    misses: int,
    outcomes: list[EvalOutcome],
    bundle_overall: BundleGradeStats | None = None,
    buckets: dict | None = None,
    bundle_buckets: dict | None = None,
) -> EvalReport:
    report = _build_report(outcomes)
    return report.model_copy(
        update={
            "strong_passes": strong,
            "weak_passes": weak,
            "misses": misses,
            "top_1": _window(strong, weak, misses, len(outcomes)),
            "top_3": _window(strong, weak, misses, len(outcomes)),
            "top_5": _window(strong, weak, misses, len(outcomes)),
            "buckets": report.buckets if buckets is None else buckets,
            "concepts": report.concepts,
            "bundle_overall": report.bundle_overall if bundle_overall is None else bundle_overall,
            "bundle_buckets": report.bundle_buckets if bundle_buckets is None else bundle_buckets,
        }
    )


def test_load_eval_cases_from_yaml(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings?",
                "    preferred_document_paths:",
                "      - docs/apis/subsystems/admin/index.md",
                "    acceptable_document_paths:",
                "      - docs/apis.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
                "    max_reasonable_bundle_tokens: 220",
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(eval_file)

    assert len(cases) == 1
    assert cases[0].id == "settings"
    assert cases[0].bucket == "plugin-structure"
    assert cases[0].concept_id == "admin-settings"
    assert cases[0].preferred_heading_substrings == ["Admin settings"]
    assert cases[0].acceptable_document_paths == ["docs/apis.md"]
    assert cases[0].required_heading_substrings_for_bundle == ["settings.php"]
    assert cases[0].max_reasonable_bundle_tokens == 220


def test_run_eval_scores_hits(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin admin settings.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - apis/subsystems/admin/index.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)

    assert report.total_queries == 1
    assert report.top_1.strong_pass_rate == 1.0
    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].matched_result is not None
    assert report.buckets["plugin-structure"].strong_passes == 1
    assert report.concepts["admin-settings"].strong_passes == 1


def test_run_eval_matches_direct_query_path_for_forms_validation(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "form").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "form" / "index.md").write_text(
        "---\n"
        "title: Forms API\n"
        "---\n\n"
        "## Forms API\n\nUse addRule() and validation helpers in the Forms API.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: forms-validation",
                "    bucket: apis-subsystems",
                "    concept_id: forms-validation",
                "    query: How does the Forms API validation pattern work?",
                "    preferred_document_paths:",
                "      - apis/subsystems/form/index.md",
                "    acceptable_heading_substrings:",
                "      - Forms API",
                "      - addRule",
            ]
        ),
        encoding="utf-8",
    )

    direct_results = query_chunks(db_path=db_path, query_text="How does the Forms API validation pattern work?", top_k=5)
    report = run_eval(db_path=db_path, eval_file=eval_file)

    assert direct_results[0].source_file_path == "apis/subsystems/form/index.md"
    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].matched_result_path == direct_results[0].source_file_path
    assert report.outcomes[0].bucket == "apis-subsystems"
    assert report.outcomes[0].concept_id == "forms-validation"


def test_run_eval_distinguishes_weak_pass(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "guides").mkdir(parents=True)
    (docs_dir / "guides" / "index.md").write_text(
        "---\n"
        "title: API Guides\n"
        "---\n\n"
        "Admin settings API helps configure plugins.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - apis/subsystems/admin/index.md",
                "    acceptable_document_paths:",
                "      - guides/index.md",
                "    acceptable_heading_substrings:",
                "      - Admin settings",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)

    assert report.outcomes[0].grade == "WEAK PASS"
    assert report.top_1.weak_pass_rate == 1.0


def test_run_eval_reports_preferred_result_rank_for_weak_pass(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apis.md").write_text(
        "---\n"
        "title: API overview\n"
        "---\n\n"
        "Moodle APIs include events, forms, tasks, and output APIs.\n",
        encoding="utf-8",
    )
    (docs_dir / "events.md").write_text(
        "# Events API\n\n## Event observers\n\nRegister plugin observers in db/events.php.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: events",
                "    bucket: events-hooks-integration",
                "    concept_id: events",
                "    query: How do events work in plugins?",
                "    preferred_document_paths:",
                "      - events.md",
                "    acceptable_document_paths:",
                "      - apis.md",
                "    acceptable_heading_substrings:",
                "      - API",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)

    assert report.outcomes[0].grade in {"STRONG PASS", "WEAK PASS"}
    if report.outcomes[0].grade == "WEAK PASS":
        assert report.outcomes[0].preferred_result_rank is not None
        assert report.outcomes[0].ranking_diagnostic is not None


def _parse_summary_markdown(summary: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in summary.splitlines():
        match = re.match(r"- ([^:]+): `([^`]+)`", line.strip())
        if match:
            parsed[match.group(1)] = match.group(2)
    return parsed


def test_eval_renderers_share_single_source_of_truth(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin admin settings.\n",
        encoding="utf-8",
    )
    (docs_dir / "guides.md").write_text(
        "---\n"
        "title: API Guides\n"
        "---\n\n"
        "Admin settings API helps configure plugins.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    acceptable_document_paths:",
                "      - guides.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)
    json_payload = report.model_dump()
    text_output = render_eval_text(report, show_weak_details=True)
    summary_output = render_eval_summary_markdown(report)
    summary_fields = _parse_summary_markdown(summary_output)

    assert f"strong_passes: {json_payload['strong_passes']}" in text_output
    assert f"weak_passes: {json_payload['weak_passes']}" in text_output
    assert f"misses: {json_payload['misses']}" in text_output
    assert summary_fields["Total queries"] == str(json_payload["total_queries"])
    assert summary_fields["Strong passes"] == str(json_payload["strong_passes"])
    assert summary_fields["Weak passes"] == str(json_payload["weak_passes"])
    assert summary_fields["Misses"] == str(json_payload["misses"])
    assert summary_fields["Top-1 strong-pass rate"] == f"{json_payload['top_1']['strong_pass_rate']:.3f}"
    assert summary_fields["Top-1 weak-pass rate"] == f"{json_payload['top_1']['weak_pass_rate']:.3f}"
    assert summary_fields["Top-3 strong-pass rate"] == f"{json_payload['top_3']['strong_pass_rate']:.3f}"
    assert summary_fields["Top-3 weak-pass rate"] == f"{json_payload['top_3']['weak_pass_rate']:.3f}"
    assert summary_fields["Top-5 strong-pass rate"] == f"{json_payload['top_5']['strong_pass_rate']:.3f}"
    assert summary_fields["Top-5 weak-pass rate"] == f"{json_payload['top_5']['weak_pass_rate']:.3f}"
    assert json.loads(json.dumps(json_payload))["strong_passes"] == json_payload["strong_passes"]
    assert "plugin-structure" in json_payload["buckets"]
    assert "admin-settings" in json_payload["concepts"]
    assert "Bucket `plugin-structure`" in summary_output
    assert "Concept `admin-settings`" in summary_output


def test_run_eval_with_bundles_reports_complete_and_bucket_stats(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Admin settings\n\n"
        "Use settings.php to add plugin settings and describe each setting clearly. "
        "Document the settings tree, the admin category, and any defaults so plugin maintainers can follow the pattern reliably.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
                "    max_reasonable_bundle_tokens: 220",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=220)

    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].bundle_grade == "COMPLETE"
    assert report.bundle_overall is not None
    assert report.bundle_overall.complete == 1
    assert report.bundle_buckets["plugin-structure"].complete == 1


def test_run_eval_with_bundles_can_report_partial_for_truncated_bundle(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Admin settings\n\n"
        + ("Use settings.php to add plugin settings with explanatory details. " * 40),
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=200, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
                "    max_reasonable_bundle_tokens: 40",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=40)

    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].bundle_grade == "PARTIAL"
    assert report.outcomes[0].bundle_selection_strategy == "truncated_match"


def test_run_eval_with_bundles_can_report_insufficient_when_required_context_is_missing(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Admin settings\n\n"
        "This page explains plugin configuration without naming the target file explicitly.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=220)

    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].bundle_grade == "INSUFFICIENT"
    assert report.outcomes[0].bundle_required_headings_missing == ["settings.php"]


def test_run_eval_with_bundles_completes_web_service_bundle_using_support_chunk(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apis.md").write_text(
        "# API Guides\n\n## Web services API\n\nThe Web services API lets plugins expose external functions.\n",
        encoding="utf-8",
    )
    (docs_dir / "writing-a-service.md").write_text(
        "# Writing a new service\n\n## Declare the web service function\n\nDeclare the function in `db/services.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=80, overlap_tokens=5)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: web-services",
                "    bucket: web-services",
                "    concept_id: web-services",
                "    query: How do I define web services for a plugin?",
                "    preferred_document_paths:",
                "      - writing-a-service.md",
                "    acceptable_document_paths:",
                "      - apis.md",
                "    acceptable_heading_substrings:",
                "      - Web services API",
                "    required_heading_substrings_for_bundle:",
                "      - db/services.php",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=220)

    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].bundle_grade == "COMPLETE"
    assert "db/services.php" in report.outcomes[0].bundle_required_headings_present


def test_run_eval_with_bundles_completes_admin_file_location_bundle_with_trimmed_support(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Individual settings\n\n"
        + ("Choose a suitable admin setting and configure defaults for your plugin. " * 30)
        + "\n\n## Where to find the code\n\n"
        "Plugin admin settings live in `settings.php` and are managed from `admin/settings.php`.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings-file-location",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: Where do Moodle plugin admin settings go?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
                "    max_reasonable_bundle_tokens: 180",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=180)

    assert report.outcomes[0].grade == "STRONG PASS"
    assert report.outcomes[0].bundle_grade == "COMPLETE"
    assert report.outcomes[0].bundle_selection_strategy in {"task_support", "task_support_truncated"}
    assert "settings.php" in report.outcomes[0].bundle_required_headings_present


def test_bundle_reporting_stays_consistent_across_renderers(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Admin settings\n\n"
        "Use settings.php to add plugin settings. "
        "Document each setting and explain how the admin tree and defaults are registered for the plugin.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file, with_bundles=True, bundle_max_tokens=220)
    json_payload = report.model_dump()
    text_output = render_eval_text(report, show_weak_details=True)
    summary_output = render_eval_summary_markdown(report)
    summary_fields = _parse_summary_markdown(summary_output)

    assert report.bundle_overall is not None
    assert f"bundle_complete: {json_payload['bundle_overall']['complete']}" in text_output
    assert f"bundle_partial: {json_payload['bundle_overall']['partial']}" in text_output
    assert f"bundle_insufficient: {json_payload['bundle_overall']['insufficient']}" in text_output
    assert summary_fields["Bundle complete"] == str(json_payload["bundle_overall"]["complete"])
    assert summary_fields["Bundle partial"] == str(json_payload["bundle_overall"]["partial"])
    assert summary_fields["Bundle insufficient"] == str(json_payload["bundle_overall"]["insufficient"])


def test_assert_report_consistent_fails_on_divergent_aggregates(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin admin settings.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)
    tampered = report.model_copy(update={"strong_passes": report.strong_passes + 1})

    with pytest.raises(ValueError, match="diverged"):
        assert_report_consistent(tampered)


def test_bucket_and_concept_aggregates_remain_consistent(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin admin settings.\n",
        encoding="utf-8",
    )
    (docs_dir / "events.md").write_text(
        "# Events API\n\n## Event observers\n\nRegister plugin observers in db/events.php.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_source(source=docs_dir, db_path=db_path, tokenizer_name="openai", max_tokens=120, overlap_tokens=10)

    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings-a",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "  - id: admin-settings-b",
                "    bucket: plugin-structure",
                "    concept_id: admin-settings",
                "    query: Where do Moodle plugin admin settings go?",
                "    preferred_document_paths:",
                "      - admin.md",
                "  - id: events-a",
                "    bucket: events-hooks-integration",
                "    concept_id: events",
                "    query: How do events work in plugins?",
                "    preferred_document_paths:",
                "      - events.md",
            ]
        ),
        encoding="utf-8",
    )

    report = run_eval(db_path=db_path, eval_file=eval_file)

    assert report.total_queries == 3
    assert report.buckets["plugin-structure"].total_queries == 2
    assert report.buckets["events-hooks-integration"].total_queries == 1
    assert report.concepts["admin-settings"].total_queries == 2
    assert report.concepts["events"].total_queries == 1
    assert (
        report.buckets["plugin-structure"].strong_passes
        + report.buckets["plugin-structure"].weak_passes
        + report.buckets["plugin-structure"].misses
        == report.buckets["plugin-structure"].total_queries
    )


def _outcome(case_id: str, query: str, grade: str, bucket: str = "bucket-a", bundle_grade: str | None = None) -> EvalOutcome:
    return EvalOutcome(
        case_id=case_id,
        query=query,
        bucket=bucket,
        concept_id=None,
        top_k=5,
        grade=grade,
        strong_pass_top_1=grade == "STRONG PASS",
        strong_pass_top_3=grade == "STRONG PASS",
        strong_pass_top_5=grade == "STRONG PASS",
        weak_pass_top_1=grade == "WEAK PASS",
        weak_pass_top_3=grade == "WEAK PASS",
        weak_pass_top_5=grade == "WEAK PASS",
        matched_result_rank=1 if grade != "MISS" else None,
        bundle_grade=bundle_grade,
    )


def test_compare_eval_reports_marks_improved_and_tracks_case_changes(tmp_path: Path) -> None:
    baseline = _report(
        strong=0,
        weak=1,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "WEAK PASS", bundle_grade="PARTIAL")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=0,
            partial=1,
            insufficient=0,
            complete_rate=0.0,
            partial_rate=1.0,
            insufficient_rate=0.0,
        ),
    )
    current = _report(
        strong=1,
        weak=0,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bundle_grade="COMPLETE")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=1,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    )

    comparison = compare_eval_reports(current, baseline, tmp_path / "baseline.json")

    assert comparison.status == "improved"
    assert comparison.retrieval_status == "improved"
    assert comparison.bundle_status == "improved"
    assert comparison.changed_cases[0].retrieval_from == "WEAK PASS"
    assert comparison.changed_cases[0].retrieval_to == "STRONG PASS"
    assert comparison.changed_cases[0].bundle_from == "PARTIAL"
    assert comparison.changed_cases[0].bundle_to == "COMPLETE"


def test_compare_eval_reports_marks_regressed(tmp_path: Path) -> None:
    baseline = _report(
        strong=1,
        weak=0,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bundle_grade="COMPLETE")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=1,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    )
    current = _report(
        strong=0,
        weak=0,
        misses=1,
        outcomes=[_outcome("case-a", "Query A", "MISS", bundle_grade="INSUFFICIENT")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=0,
            partial=0,
            insufficient=1,
            complete_rate=0.0,
            partial_rate=0.0,
            insufficient_rate=1.0,
        ),
    )

    comparison = compare_eval_reports(current, baseline, tmp_path / "baseline.json")

    assert comparison.status == "regressed"
    assert comparison.retrieval_status == "regressed"
    assert comparison.bundle_status == "regressed"


def test_compare_eval_reports_marks_mixed_and_tracks_bucket_changes(tmp_path: Path) -> None:
    baseline = _report(
        strong=1,
        weak=1,
        misses=0,
        outcomes=[
            _outcome("case-a", "Query A", "STRONG PASS", bucket="bucket-a", bundle_grade="COMPLETE"),
            _outcome("case-b", "Query B", "WEAK PASS", bucket="bucket-b", bundle_grade="PARTIAL"),
        ],
        bundle_overall=BundleGradeStats(
            total_evaluated=2,
            complete=1,
            partial=1,
            insufficient=0,
            complete_rate=0.5,
            partial_rate=0.5,
            insufficient_rate=0.0,
        ),
        buckets={
            "bucket-a": _report(strong=1, weak=0, misses=0, outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bucket="bucket-a")]).buckets["bucket-a"],
            "bucket-b": _report(strong=0, weak=1, misses=0, outcomes=[_outcome("case-b", "Query B", "WEAK PASS", bucket="bucket-b")]).buckets["bucket-b"],
        },
        bundle_buckets={
            "bucket-a": BundleGradeStats(total_evaluated=1, complete=1, partial=0, insufficient=0, complete_rate=1.0, partial_rate=0.0, insufficient_rate=0.0),
            "bucket-b": BundleGradeStats(total_evaluated=1, complete=0, partial=1, insufficient=0, complete_rate=0.0, partial_rate=1.0, insufficient_rate=0.0),
        },
    )
    current = _report(
        strong=1,
        weak=0,
        misses=1,
        outcomes=[
            _outcome("case-a", "Query A", "MISS", bucket="bucket-a", bundle_grade="INSUFFICIENT"),
            _outcome("case-b", "Query B", "STRONG PASS", bucket="bucket-b", bundle_grade="COMPLETE"),
        ],
        bundle_overall=BundleGradeStats(
            total_evaluated=2,
            complete=1,
            partial=0,
            insufficient=1,
            complete_rate=0.5,
            partial_rate=0.0,
            insufficient_rate=0.5,
        ),
        buckets={
            "bucket-a": _report(strong=0, weak=0, misses=1, outcomes=[_outcome("case-a", "Query A", "MISS", bucket="bucket-a")]).buckets["bucket-a"],
            "bucket-b": _report(strong=1, weak=0, misses=0, outcomes=[_outcome("case-b", "Query B", "STRONG PASS", bucket="bucket-b")]).buckets["bucket-b"],
        },
        bundle_buckets={
            "bucket-a": BundleGradeStats(total_evaluated=1, complete=0, partial=0, insufficient=1, complete_rate=0.0, partial_rate=0.0, insufficient_rate=1.0),
            "bucket-b": BundleGradeStats(total_evaluated=1, complete=1, partial=0, insufficient=0, complete_rate=1.0, partial_rate=0.0, insufficient_rate=0.0),
        },
    )

    comparison = compare_eval_reports(current, baseline, tmp_path / "baseline.json")

    assert comparison.status == "mixed"
    assert comparison.retrieval_status == "mixed"
    assert comparison.bundle_status == "mixed"
    assert {change.label for change in comparison.changed_retrieval_buckets} == {"bucket-a", "bucket-b"}
    assert {change.label for change in comparison.changed_bundle_buckets} == {"bucket-a", "bucket-b"}


def test_compare_eval_reports_marks_unchanged(tmp_path: Path) -> None:
    report = _report(
        strong=1,
        weak=0,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bundle_grade="COMPLETE")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=1,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    )

    comparison = compare_eval_reports(report, report, tmp_path / "baseline.json")

    assert comparison.status == "unchanged"
    assert comparison.retrieval_status == "unchanged"
    assert comparison.bundle_status == "unchanged"
    assert comparison.changed_cases == []


def test_renderers_include_baseline_comparison(tmp_path: Path) -> None:
    baseline = _report(
        strong=0,
        weak=1,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "WEAK PASS", bundle_grade="PARTIAL")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=0,
            partial=1,
            insufficient=0,
            complete_rate=0.0,
            partial_rate=1.0,
            insufficient_rate=0.0,
        ),
    )
    current = _report(
        strong=1,
        weak=0,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bundle_grade="COMPLETE")],
        bundle_overall=BundleGradeStats(
            total_evaluated=1,
            complete=1,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    ).model_copy(update={"baseline_comparison": compare_eval_reports(
        _report(
            strong=1,
            weak=0,
            misses=0,
            outcomes=[_outcome("case-a", "Query A", "STRONG PASS", bundle_grade="COMPLETE")],
            bundle_overall=BundleGradeStats(
                total_evaluated=1,
                complete=1,
                partial=0,
                insufficient=0,
                complete_rate=1.0,
                partial_rate=0.0,
                insufficient_rate=0.0,
            ),
        ),
        baseline,
        tmp_path / "baseline.json",
    )})

    text_output = render_eval_text(current, show_weak_details=True)
    summary_output = render_eval_summary_markdown(current)
    payload = current.model_dump()

    assert payload["baseline_comparison"]["status"] == "improved"
    assert "baseline_comparison_status: improved" in text_output
    assert "changed_cases:" in text_output
    assert "Baseline comparison: `improved`" in summary_output


def test_load_eval_report_artifact_supports_verify_payload(tmp_path: Path) -> None:
    report = _report(
        strong=1,
        weak=0,
        misses=0,
        outcomes=[_outcome("case-a", "Query A", "STRONG PASS")],
    )
    artifact = tmp_path / "verify.json"
    artifact.write_text(json.dumps({"eval": report.model_dump()}), encoding="utf-8")

    loaded = load_eval_report_artifact(artifact)

    assert loaded.strong_passes == 1
