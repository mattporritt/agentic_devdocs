from pathlib import Path

from typer.testing import CliRunner

from agentic_docs.cli import app


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
