from pathlib import Path

from agentic_docs.evaluation import load_eval_cases, run_eval
from agentic_docs.ingest import ingest_source


def test_load_eval_cases_from_yaml(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.yaml"
    eval_file.write_text(
        "\n".join(
            [
                "cases:",
                "  - id: settings",
                "    query: How do I add admin settings?",
                "    preferred_document_paths:",
                "      - docs/apis/subsystems/admin/index.md",
                "    acceptable_document_paths:",
                "      - docs/apis.md",
                "    preferred_heading_substrings:",
                "      - Admin settings",
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(eval_file)

    assert len(cases) == 1
    assert cases[0].id == "settings"
    assert cases[0].preferred_heading_substrings == ["Admin settings"]
    assert cases[0].acceptable_document_paths == ["docs/apis.md"]


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
