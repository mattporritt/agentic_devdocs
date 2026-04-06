import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentic_docs.cli import _validation_worktree_payload, app


runner = CliRunner()


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
                "      - apis/subsystems/admin/index.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
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
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--json"],
    )
    assert eval_result.exit_code == 0
    assert '"strong_pass_rate": 1.0' in eval_result.stdout
    assert '"grade": "STRONG PASS"' in eval_result.stdout

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
            "--json",
        ],
    )
    assert bundle_result.exit_code == 0
    assert '"chunks"' in bundle_result.stdout
    assert '"selection_strategy"' in bundle_result.stdout


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
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file), "--json"],
    )
    eval_text = runner.invoke(
        app,
        ["eval", "--db-path", str(db_path), "--eval-file", str(eval_file)],
    )
    assert eval_json.exit_code == 0
    assert eval_text.exit_code == 0

    payload = json.loads(eval_json.stdout)
    assert f"total_queries: {payload['total_queries']}" in eval_text.stdout
    assert f"strong_passes: {payload['strong_passes']}" in eval_text.stdout
    assert f"weak_passes: {payload['weak_passes']}" in eval_text.stdout
    assert f"misses: {payload['misses']}" in eval_text.stdout


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
    assert payload["regression_detected"] is False
