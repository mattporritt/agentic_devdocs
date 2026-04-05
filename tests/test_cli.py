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

