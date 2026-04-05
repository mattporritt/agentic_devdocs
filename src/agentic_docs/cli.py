"""Typer CLI for syncing, ingesting, querying, and inspecting docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentic_docs.config import IngestConfig, QueryConfig
from agentic_docs.git_sync import sync_repository
from agentic_docs.ingest import ingest_source
from agentic_docs.query_service import query_chunks
from agentic_docs.storage import SQLiteStore

app = typer.Typer(help="Ingest markdown docs into agent-friendly retrieval artifacts.")


def _emit(data: object, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True, default=str))
        return
    if isinstance(data, dict):
        for key, value in data.items():
            typer.echo(f"{key}: {value}")
        return
    typer.echo(str(data))


@app.command()
def sync(
    repo_url: Annotated[str, typer.Option(help="Git repository URL to sync.")],
    local_path: Annotated[Path, typer.Option(help="Local path for the cloned repository.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Clone or update a git repository and report the checked-out commit."""

    commit_hash = sync_repository(repo_url=repo_url, local_path=local_path)
    _emit({"repo_url": repo_url, "local_path": str(local_path), "repo_commit_hash": commit_hash}, json_output)


@app.command()
def ingest(
    source: Annotated[Path, typer.Option(help="Directory containing markdown documentation.")],
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    tokenizer: Annotated[str, typer.Option(help="Tokenizer adapter to use.")] = "openai",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
    overlap_tokens: Annotated[int, typer.Option(help="Overlap tokens between adjacent chunks.")] = 60,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Parse markdown docs, chunk them, and index them into SQLite."""

    config = IngestConfig(
        source=source,
        db_path=db_path,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    result = ingest_source(
        source=config.source,
        db_path=config.db_path,
        tokenizer_name=config.tokenizer,
        max_tokens=config.max_tokens,
        overlap_tokens=config.overlap_tokens,
    )
    _emit(result, json_output)


@app.command()
def query(
    query_text: Annotated[str, typer.Argument(help="Free-text query string.")],
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    top_k: Annotated[int, typer.Option(help="Number of chunks to return.")] = 5,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Query the indexed chunks using SQLite FTS5 lexical search."""

    config = QueryConfig(db_path=db_path, top_k=top_k)
    results = query_chunks(db_path=config.db_path, query_text=query_text, top_k=config.top_k)
    if json_output:
        _emit([result.model_dump() for result in results], True)
        return
    for result in results:
        typer.echo(f"chunk_id: {result.chunk_id}")
        typer.echo(f"score: {result.score}")
        typer.echo(f"source_file_path: {result.source_file_path}")
        typer.echo(f"document_title: {result.document_title}")
        typer.echo(f"section_title: {result.section_title}")
        typer.echo(f"heading_path: {' > '.join(result.heading_path)}")
        typer.echo(f"token_count: {result.token_count}")
        typer.echo(f"repo_commit_hash: {result.repo_commit_hash}")
        typer.echo(result.content)
        typer.echo("")


@app.command()
def stats(
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Show document, section, and chunk counts for a database."""

    store = SQLiteStore(db_path)
    store.initialize()
    _emit(store.stats(), json_output)


@app.command("inspect-chunk")
def inspect_chunk(
    chunk_id: Annotated[str, typer.Argument(help="Chunk identifier.")],
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Inspect a stored chunk and its trace metadata."""

    store = SQLiteStore(db_path)
    store.initialize()
    result = store.inspect_chunk(chunk_id)
    if result is None:
        raise typer.Exit(code=1)
    _emit(result, json_output)


@app.command("inspect-doc")
def inspect_doc(
    document_id: Annotated[str, typer.Argument(help="Document identifier.")],
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Inspect a stored document and its section hierarchy."""

    store = SQLiteStore(db_path)
    store.initialize()
    result = store.inspect_document(document_id)
    if result is None:
        raise typer.Exit(code=1)
    _emit(result, json_output)


if __name__ == "__main__":
    app()
