from pathlib import Path

from agentic_docs.parser import discover_markdown_files, parse_markdown_document


def test_discover_markdown_files_filters_non_markdown(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n", encoding="utf-8")
    (tmp_path / "docs" / "b.txt").write_text("ignore", encoding="utf-8")

    files = discover_markdown_files(tmp_path)

    assert [path.name for path in files] == ["a.md"]


def test_parse_markdown_document_extracts_sections(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "\n".join(
            [
                "# Guide",
                "",
                "Intro paragraph.",
                "",
                "## Installation",
                "",
                "Install steps.",
                "",
                "### Advanced",
                "",
                "Advanced steps.",
            ]
        ),
        encoding="utf-8",
    )

    document = parse_markdown_document(path=path, root=tmp_path, repo_commit_hash="abc123")

    assert document.title == "Guide"
    assert document.metadata.source_path == "guide.md"
    assert [section.section_title for section in document.sections] == ["Guide", "Installation", "Advanced"]
    assert document.sections[1].heading_path == ["Guide", "Installation"]
    assert "Install steps." in document.sections[1].content

