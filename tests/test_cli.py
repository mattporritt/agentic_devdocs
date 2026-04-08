import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_docs.cli import _validation_summary_status, _validation_worktree_payload, app
from agentic_docs.models import BundleGradeStats, EvalReport, EvalWindowStats, RuntimeContractEnvelope


runner = CliRunner()


def _window(strong: int, weak: int, misses: int, total: int) -> EvalWindowStats:
    return EvalWindowStats(
        strong_passes=strong,
        weak_passes=weak,
        misses=misses,
        strong_pass_rate=strong / total if total else 0.0,
        weak_pass_rate=weak / total if total else 0.0,
    )


def test_cli_ingest_and_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "forms.md").write_text(
        "# Forms\n\n## Form API\n\nMoodle forms help build validated forms.",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"

    ingest_result = runner.invoke(
        app,
        [
            "ingest",
            "--source",
            str(docs_dir),
            "--db-path",
            str(db_path),
            "--json",
        ],
    )
    assert ingest_result.exit_code == 0
    assert '"documents": 1' in ingest_result.stdout

    query_result = runner.invoke(
        app,
        [
            "query",
            "forms",
            "--db-path",
            str(db_path),
            "--json",
        ],
    )
    assert query_result.exit_code == 0
    assert '"document_title": "Forms"' in query_result.stdout


def test_cli_eval_and_context_bundle(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin settings. "
        "Explain the admin tree, defaults, and configuration structure so developers can apply the pattern correctly. "
        "Include enough surrounding explanation that a coding agent can understand where the setting is registered, how the admin tree is built, and how defaults are supplied.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - apis/subsystems/admin/index.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
            ]
        ),
        encoding="utf-8",
    )

    ingest_result = runner.invoke(
        app,
        ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"],
    )
    assert ingest_result.exit_code == 0

    eval_result = runner.invoke(
        app,
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--with-bundles", "--json"],
    )
    assert eval_result.exit_code == 0
    assert '"strong_pass_rate": 1.0' in eval_result.stdout
    assert '"grade": "STRONG PASS"' in eval_result.stdout
    assert '"bundle_grade": "COMPLETE"' in eval_result.stdout

    bundle_result = runner.invoke(
        app,
        [
            "query",
            "admin settings",
            "--db-path",
            str(db_path),
            "--context-bundle",
            "--bundle-max-tokens",
            "120",
            "--explain-bundle",
            "--json",
        ],
    )
    assert bundle_result.exit_code == 0
    assert '"chunks"' in bundle_result.stdout
    assert '"selection_strategy"' in bundle_result.stdout
    assert '"diagnostics"' in bundle_result.stdout


def test_cli_query_explain_ranking_and_eval_weak_details(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "apis.md").write_text(
        "---\n"
        "title: APIs overview\n"
        "---\n\n"
        "Moodle APIs include events, web services, forms, and tasks.\n",
        encoding="utf-8",
    )
    (docs_dir / "events.md").write_text(
        "# Events API\n\n## Event observers\n\nRegister plugin observers in db/events.php.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: events",
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

    ingest_result = runner.invoke(
        app,
        ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"],
    )
    assert ingest_result.exit_code == 0

    query_result = runner.invoke(
        app,
        [
            "query",
            "events plugins",
            "--db-path",
            str(db_path),
            "--explain-ranking",
        ],
    )
    assert query_result.exit_code == 0
    assert "rerank_score:" in query_result.stdout
    assert "rerank_breakdown:" in query_result.stdout

    eval_result = runner.invoke(
        app,
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--show-weak-details"],
    )
    assert eval_result.exit_code == 0
    assert "ranking=" in eval_result.stdout or "preferred_result_rank=" in eval_result.stdout


def test_cli_query_json_contract_for_devdocs_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Individual settings\n\n"
        "Plugin admin settings live in `settings.php` and are registered through the admin tree.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_result = runner.invoke(app, ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"])
    assert ingest_result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "query",
            "Where do Moodle plugin admin settings go?",
            "--db-path",
            str(db_path),
            "--json-contract",
        ],
    )

    assert result.exit_code == 0
    raw = json.loads(result.stdout)
    payload = RuntimeContractEnvelope.model_validate_json(result.stdout)
    assert payload.tool == "agentic_docs"
    assert payload.version == "v1"
    assert raw["results"][0]["id"] == payload.results[0].id
    assert isinstance(raw["results"][0]["content"]["sections"], list)
    assert isinstance(raw["results"][0]["content"]["file_anchors"], list)
    assert isinstance(raw["results"][0]["content"]["key_points"], list)
    assert raw["results"][0]["diagnostics"]["support_reason"] == "file_location"
    assert payload.intent.task_intent == "file_location"
    assert payload.results[0].confidence == "high"
    assert payload.results[0].source.name == "devdocs_repo"
    assert payload.results[0].source.path == "apis/subsystems/admin/index.md"
    assert payload.results[0].content.sections[0].id
    assert payload.results[0].content.sections[0].document_title == "Admin settings"
    assert payload.results[0].content.sections[0].source_path == "apis/subsystems/admin/index.md"
    assert payload.results[0].content.sections[0].source_url is None
    assert payload.results[0].content.sections[0].canonical_url is None
    assert "settings.php" in payload.results[0].content.file_anchors
    assert payload.results[0].diagnostics.token_count > 0


def test_cli_query_json_contract_for_design_system_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "design_system" / "styles").mkdir(parents=True)
    (docs_dir / "design_system" / "styles" / "colours-32c91c.md").write_text(
        "# Colours\n\n"
        "## Tokens\n\n"
        "### Semantic colour tokens\n\n"
        "Semantic colour tokens give colours a role in the interface.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_result = runner.invoke(app, ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"])
    assert ingest_result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "query",
            "What are semantic colour tokens?",
            "--db-path",
            str(db_path),
            "--json-contract",
        ],
    )

    assert result.exit_code == 0
    raw = json.loads(result.stdout)
    payload = RuntimeContractEnvelope.model_validate_json(result.stdout)
    assert payload.intent.query_intent == "conceptual"
    assert payload.results[0].source.name == "design_system"
    assert payload.results[0].source.path == "design_system/styles/colours-32c91c.md"
    assert raw["results"][0]["source"]["url"] is None
    assert raw["results"][0]["source"]["canonical_url"] is None
    assert isinstance(raw["results"][0]["content"]["sections"], list)
    assert payload.results[0].content.sections[0].heading_path[-1] == "Semantic colour tokens"
    assert payload.results[0].content.sections[0].document_title == "Colours"
    assert payload.results[0].content.sections[0].source_path == "design_system/styles/colours-32c91c.md"


def test_cli_query_json_contract_is_stable_for_combined_query(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "output").mkdir(parents=True)
    (docs_dir / "design_system" / "styles").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "output" / "index.md").write_text(
        "---\n"
        "title: Output API\n"
        "---\n\n"
        "## Output API\n\n"
        "The Output API explains how Moodle renderers, renderables, themes and templates all work together.\n\n"
        "## Page Output Journey\n\n"
        "### Renderable\n\n"
        "Renderables can be rendered through templates in Moodle output.\n",
        encoding="utf-8",
    )
    (docs_dir / "design_system" / "styles" / "colours-32c91c.md").write_text(
        "# Colours\n\n"
        "## Overview\n\n"
        "### Core principles\n\n"
        "Use semantic tokens in components.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    ingest_result = runner.invoke(app, ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"])
    assert ingest_result.exit_code == 0

    args = [
        "query",
        "How should this render in Moodle?",
        "--db-path",
        str(db_path),
        "--json-contract",
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0
    assert second.exit_code == 0
    first_payload = RuntimeContractEnvelope.model_validate_json(first.stdout)
    second_payload = RuntimeContractEnvelope.model_validate_json(second.stdout)
    assert first_payload.model_dump() == second_payload.model_dump()
    assert first_payload.results[0].id == second_payload.results[0].id
    assert first_payload.results[0].content.sections[0].id == second_payload.results[0].content.sections[0].id
    assert first_payload.results[0].source.name == "devdocs_repo"
    assert first_payload.results[0].diagnostics.selection_strategy in {"match_only", "task_support", "task_support_truncated"}


def test_cli_query_json_contract_empty_results_shape(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "simple.md").write_text("# Simple\n\n## Intro\n\nOnly one concept lives here.\n", encoding="utf-8")
    db_path = tmp_path / "docs.db"
    ingest_result = runner.invoke(app, ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"])
    assert ingest_result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "query",
            "nonexistent phrase zebra tungsten",
            "--db-path",
            str(db_path),
            "--json-contract",
        ],
    )

    assert result.exit_code == 0
    raw = json.loads(result.stdout)
    assert raw["tool"] == "agentic_docs"
    assert raw["version"] == "v1"
    assert raw["results"] == []
    assert isinstance(raw["intent"]["concept_families"], list)


def test_runtime_contract_schema_artifact_matches_model() -> None:
    schema_path = Path("schemas/runtime_contract_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema == RuntimeContractEnvelope.model_json_schema()


def test_cli_eval_text_and_json_are_consistent(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "Use settings.php to add plugin settings.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
            ]
        ),
        encoding="utf-8",
    )

    ingest_result = runner.invoke(
        app,
        ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"],
    )
    assert ingest_result.exit_code == 0

    eval_json = runner.invoke(
        app,
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--with-bundles", "--json"],
    )
    eval_text = runner.invoke(
        app,
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--with-bundles"],
    )
    assert eval_json.exit_code == 0
    assert eval_text.exit_code == 0

    payload = json.loads(eval_json.stdout)
    assert f"total_queries: {payload['total_queries']}" in eval_text.stdout
    assert f"strong_passes: {payload['strong_passes']}" in eval_text.stdout
    assert f"weak_passes: {payload['weak_passes']}" in eval_text.stdout
    assert f"misses: {payload['misses']}" in eval_text.stdout
    assert f"bundle_complete: {payload['bundle_overall']['complete']}" in eval_text.stdout


def test_cli_eval_show_bundle_details_outputs_bundle_diagnostics(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "admin.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Admin settings\n\n"
        "Use settings.php to add plugin settings.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: admin-settings",
                "    query: How do I add admin settings for a plugin?",
                "    preferred_document_paths:",
                "      - admin.md",
                "    required_heading_substrings_for_bundle:",
                "      - settings.php",
            ]
        ),
        encoding="utf-8",
    )

    ingest_result = runner.invoke(
        app,
        ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"],
    )
    assert ingest_result.exit_code == 0

    eval_result = runner.invoke(
        app,
        [
            "eval",
            "--db-path",
            str(db_path),
            "--eval-file",
            str(eval_file),
            "--with-bundles",
            "--show-bundle-details",
        ],
    )
    assert eval_result.exit_code == 0
    assert "bundle_grade=" in eval_result.stdout
    assert "bundle=" in eval_result.stdout


def test_validation_worktree_payload_rejects_dirty_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "agentic_docs.cli.git_working_tree_status",
        lambda _path: {"is_git_repo": True, "clean": False, "status_lines": [" M README.md"]},
    )
    monkeypatch.setattr("agentic_docs.cli.git_head_commit", lambda _path: "abc123")

    with pytest.raises(Exception, match="clean git working tree"):
        _validation_worktree_payload(tmp_path, allow_dirty=False)


def test_validation_worktree_payload_allows_dirty_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "agentic_docs.cli.git_working_tree_status",
        lambda _path: {"is_git_repo": True, "clean": False, "status_lines": [" M README.md"]},
    )
    monkeypatch.setattr("agentic_docs.cli.git_head_commit", lambda _path: "abc123")

    payload = _validation_worktree_payload(tmp_path, allow_dirty=True)

    assert payload["clean"] is False
    assert payload["mode"] == "dirty_override"
    assert payload["status_lines"] == [" M README.md"]


def test_validation_summary_status_distinguishes_warnings_from_regression() -> None:
    report = EvalReport(
        total_queries=3,
        strong_passes=2,
        weak_passes=1,
        misses=0,
        top_1=_window(2, 1, 0, 3),
        top_3=_window(2, 1, 0, 3),
        top_5=_window(2, 1, 0, 3),
        outcomes=[],
        bundle_overall=BundleGradeStats(
            total_evaluated=3,
            complete=3,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    )

    status = _validation_summary_status(report)

    assert status is not None
    assert status["overall_status"] == "GREEN_WITH_WARNINGS"
    assert status["retrieval_fully_green"] is False
    assert status["weak_or_miss_present"] is True
    assert status["bundle_fully_green"] is True
    assert status["bundle_non_complete_present"] is False
    assert status["baseline_comparison"]["status"] == "not_compared"
    assert status["baseline_comparison"]["baseline_provided"] is False
    assert status["baseline_comparison"]["retrieval_status"] is None


def test_validation_summary_status_marks_fully_green_run() -> None:
    report = EvalReport(
        total_queries=2,
        strong_passes=2,
        weak_passes=0,
        misses=0,
        top_1=_window(2, 0, 0, 2),
        top_3=_window(2, 0, 0, 2),
        top_5=_window(2, 0, 0, 2),
        outcomes=[],
        bundle_overall=BundleGradeStats(
            total_evaluated=2,
            complete=2,
            partial=0,
            insufficient=0,
            complete_rate=1.0,
            partial_rate=0.0,
            insufficient_rate=0.0,
        ),
    )

    status = _validation_summary_status(report)

    assert status is not None
    assert status["overall_status"] == "GREEN"
    assert status["retrieval_fully_green"] is True
    assert status["weak_or_miss_present"] is False
    assert status["bundle_fully_green"] is True
    assert status["bundle_non_complete_present"] is False


def test_validation_summary_status_marks_partial_bundles_as_non_green() -> None:
    report = EvalReport(
        total_queries=2,
        strong_passes=2,
        weak_passes=0,
        misses=0,
        top_1=_window(2, 0, 0, 2),
        top_3=_window(2, 0, 0, 2),
        top_5=_window(2, 0, 0, 2),
        outcomes=[],
        bundle_overall=BundleGradeStats(
            total_evaluated=2,
            complete=1,
            partial=1,
            insufficient=0,
            complete_rate=0.5,
            partial_rate=0.5,
            insufficient_rate=0.0,
        ),
    )

    status = _validation_summary_status(report)

    assert status is not None
    assert status["bundle_fully_green"] is False
    assert status["bundle_non_complete_present"] is True
    assert status["overall_status"] == "GREEN_WITH_WARNINGS"


def test_validation_summary_status_marks_insufficient_bundles_as_non_green() -> None:
    report = EvalReport(
        total_queries=2,
        strong_passes=2,
        weak_passes=0,
        misses=0,
        top_1=_window(2, 0, 0, 2),
        top_3=_window(2, 0, 0, 2),
        top_5=_window(2, 0, 0, 2),
        outcomes=[],
        bundle_overall=BundleGradeStats(
            total_evaluated=2,
            complete=1,
            partial=0,
            insufficient=1,
            complete_rate=0.5,
            partial_rate=0.0,
            insufficient_rate=0.5,
        ),
    )

    status = _validation_summary_status(report)

    assert status is not None
    assert status["bundle_fully_green"] is False
    assert status["bundle_non_complete_present"] is True
    assert status["overall_status"] == "GREEN_WITH_WARNINGS"


def test_validation_summary_status_marks_misses_as_non_green() -> None:
    report = EvalReport(
        total_queries=2,
        strong_passes=1,
        weak_passes=0,
        misses=1,
        top_1=_window(1, 0, 1, 2),
        top_3=_window(1, 0, 1, 2),
        top_5=_window(1, 0, 1, 2),
        outcomes=[],
        bundle_overall=None,
    )

    status = _validation_summary_status(report)

    assert status is not None
    assert status["overall_status"] == "NON_GREEN"
    assert status["retrieval_fully_green"] is False
    assert status["weak_or_miss_present"] is True
    assert status["bundle_fully_green"] is None
    assert status["bundle_non_complete_present"] is None


def test_cli_verify_devdocs_runs_eval_sequentially_and_records_cleanliness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "form").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "form" / "index.md").write_text(
        "---\n"
        "title: Forms API\n"
        "---\n\n"
        "## Forms API\n\nUse the Forms API and addRule() validation helpers.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "docs.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: forms-validation",
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

    monkeypatch.setattr(
        "agentic_docs.cli._validation_worktree_payload",
        lambda _path, allow_dirty: {
            "repo_path": str(tmp_path),
            "repo_head_commit": "tool-commit",
            "is_git_repo": True,
            "clean": True,
            "allow_dirty": allow_dirty,
            "status_lines": [],
            "mode": "clean",
        },
    )
    monkeypatch.setattr("agentic_docs.cli.current_commit_hash", lambda _path: "devdocs-commit")

    result = runner.invoke(
        app,
        [
            "verify-devdocs",
            "--local-path",
            str(docs_dir),
            "--db-path",
            str(db_path),
            "--eval-file",
            str(eval_file),
            "--skip-sync",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["working_tree"]["clean"] is True
    assert payload["working_tree"]["mode"] == "clean"
    assert payload["repo_commit_hash"] == "devdocs-commit"
    assert payload["smoke_queries"]["Forms API validation"][0]["source_file_path"] == "apis/subsystems/form/index.md"
    assert payload["eval"]["strong_passes"] == 1
    assert payload["eval"]["misses"] == 0
    assert payload["eval"]["outcomes"][0]["case_id"] == "forms-validation"
    assert payload["eval"]["outcomes"][0]["grade"] == "STRONG PASS"
    assert "regression_detected" not in payload
    assert payload["validation_status"]["retrieval_fully_green"] is True
    assert payload["validation_status"]["weak_or_miss_present"] is False
    assert payload["validation_status"]["bundle_fully_green"] is (
        payload["eval"]["bundle_overall"]["partial"] == 0 and payload["eval"]["bundle_overall"]["insufficient"] == 0
    )
    assert payload["validation_status"]["bundle_non_complete_present"] is (
        payload["eval"]["bundle_overall"]["partial"] > 0 or payload["eval"]["bundle_overall"]["insufficient"] > 0
    )
    assert payload["validation_status"]["baseline_comparison"]["status"] == "not_compared"


def test_cli_ingest_site_uses_site_ingestion_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "design.db"
    monkeypatch.setattr(
        "agentic_docs.cli.ingest_site_source",
        lambda **kwargs: {
            "documents": 3,
            "sections": 7,
            "chunks": 9,
            "source_type": "scraped_web",
            "source_name": "design_system",
            "base_url": kwargs["base_url"],
        },
    )

    result = runner.invoke(
        app,
        [
            "ingest-site",
            "--base-url",
            "https://design.moodle.com",
            "--db-path",
            str(db_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["documents"] == 3
    assert payload["source_type"] == "scraped_web"
    assert payload["source_name"] == "design_system"


def test_cli_verify_site_runs_eval_sequentially_and_records_source_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "design_system" / "foundation").mkdir(parents=True)
    (docs_dir / "design_system" / "foundation" / "colours-32c91c.site").write_text(
        "placeholder",
        encoding="utf-8",
    )
    db_path = tmp_path / "design.db"
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: colors",
                "    query: Where are semantic colour tokens documented?",
                "    preferred_document_paths:",
                "      - design_system/foundation/colours-32c91c.site",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "agentic_docs.cli._validation_worktree_payload",
        lambda _path, allow_dirty: {
            "repo_path": str(tmp_path),
            "repo_head_commit": "tool-commit",
            "is_git_repo": True,
            "clean": True,
            "allow_dirty": allow_dirty,
            "status_lines": [],
            "mode": "clean",
        },
    )

    def fake_ingest_site_source(**_kwargs: object) -> dict[str, object]:
        from agentic_docs.chunking import chunk_document
        from agentic_docs.models import DocumentMetadata, DocumentModel, SectionModel
        from agentic_docs.storage import SQLiteStore
        from agentic_docs.tokenizers import get_tokenizer

        store = SQLiteStore(db_path)
        store.initialize()
        store.reindex()
        document = DocumentModel(
            id="site-doc",
            title="Colours",
            metadata=DocumentMetadata(
                source_path="design_system/foundation/colours-32c91c.site",
                source_type="scraped_web",
                source_name="design_system",
                source_url="https://design.moodle.com/98292f05f/p/32c91c",
                canonical_url="https://design.moodle.com/98292f05f/p/32c91c",
                file_hash="hash",
                content_hash="hash",
            ),
            sections=[
                SectionModel(
                    id="site-sec",
                    document_id="site-doc",
                    section_order=0,
                    section_title="Semantic colour tokens",
                    heading_level=2,
                    heading_path=["Colours", "Semantic colour tokens"],
                    content="Use semantic tokens.",
                )
            ],
        )
        chunks = chunk_document(
            document=document,
            tokenizer=get_tokenizer("openai"),
            max_tokens=400,
            overlap_tokens=60,
        )
        store.store_document(document, chunks)
        return {
            "documents": 1,
            "sections": 1,
            "chunks": len(chunks),
            "source_type": "scraped_web",
            "source_name": "design_system",
            "base_url": "https://design.moodle.com",
        }

    monkeypatch.setattr("agentic_docs.cli.ingest_site_source", fake_ingest_site_source)

    result = runner.invoke(
        app,
        [
            "verify-site",
            "--base-url",
            "https://design.moodle.com",
            "--db-path",
            str(db_path),
            "--eval-file",
            str(eval_file),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ingest"]["source_type"] == "scraped_web"
    assert payload["stats"]["sources"][0]["chunk_count"] >= 1
    assert payload["eval"]["total_queries"] == 1
