"""Typer CLI for syncing, ingesting, querying, and inspecting docs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from agentic_docs.config import IngestConfig, QueryConfig
from agentic_docs.evaluation import run_eval
from agentic_docs.git_sync import current_commit_hash, sync_repository
from agentic_docs.ingest import ingest_source
from agentic_docs.query_service import build_context_bundles, query_chunks
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
    context_bundle: Annotated[bool, typer.Option("--context-bundle", help="Return compact agent-oriented context bundles.")] = False,
    include_previous: Annotated[bool, typer.Option(help="Include the previous chunk in context bundles.")] = False,
    include_next: Annotated[bool, typer.Option(help="Include the next chunk in context bundles.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Query the indexed chunks using SQLite FTS5 lexical search."""

    config = QueryConfig(db_path=db_path, top_k=top_k)
    results = query_chunks(db_path=config.db_path, query_text=query_text, top_k=config.top_k)
    if context_bundle:
        bundles = build_context_bundles(
            db_path=config.db_path,
            results=results,
            include_previous=include_previous,
            include_next=include_next,
        )
        if json_output:
            _emit([bundle.model_dump() for bundle in bundles], True)
            return
        for bundle in bundles:
            typer.echo(f"rank: {bundle.rank}")
            typer.echo(f"score: {bundle.score}")
            typer.echo(f"source_file_path: {bundle.source_file_path}")
            typer.echo(f"document_title: {bundle.document_title}")
            typer.echo(f"section_title: {bundle.section_title}")
            typer.echo(f"heading_path: {' > '.join(bundle.heading_path)}")
            typer.echo(f"bundle_token_count: {bundle.bundle_token_count}")
            typer.echo(f"repo_commit_hash: {bundle.repo_commit_hash}")
            if bundle.snippet:
                typer.echo(f"snippet: {bundle.snippet}")
            for chunk in bundle.chunks:
                typer.echo(f"[{chunk.role}] {chunk.chunk_id} ({chunk.token_count} tokens)")
                typer.echo(chunk.content)
                typer.echo("")
        return
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
        typer.echo(f"normalized_query: {result.normalized_query}")
        typer.echo(f"snippet: {result.snippet}")
        typer.echo(result.content)
        typer.echo("")


@app.command()
def stats(
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    limit: Annotated[int, typer.Option(help="Number of diagnostic rows to display.")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Show document, section, and chunk counts for a database."""

    store = SQLiteStore(db_path)
    store.initialize()
    data = store.detailed_stats(limit=limit)
    if json_output:
        _emit(data, True)
        return
    overview = data["overview"]
    _emit(overview, False)
    typer.echo("")
    typer.echo("top_documents_by_chunk_count:")
    for row in data["chunks_per_document"]:
        typer.echo(f"- {row['chunk_count']}  {row['source_path']} ({row['title']})")
    typer.echo("")
    typer.echo("largest_chunks:")
    for row in data["largest_chunks"]:
        typer.echo(f"- {row['token_count']}  {row['chunk_id']}  {row['source_path']}  {row['heading_path']}")
    typer.echo("")
    typer.echo("small_chunks:")
    for row in data["small_chunks"]:
        typer.echo(f"- {row['token_count']}  {row['chunk_id']}  {row['source_path']}  {row['heading_path']}")
    if data["duplicate_chunk_candidates"]:
        typer.echo("")
        typer.echo("duplicate_chunk_candidates:")
        for row in data["duplicate_chunk_candidates"]:
            typer.echo(
                f"- {row['duplicate_count']}  sample={row['sample_chunk_id']}  source={row['sample_source_path']}"
            )


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


@app.command("verify-devdocs")
def verify_devdocs(
    repo_url: Annotated[str, typer.Option(help="Git repository URL to sync.")] = "https://github.com/moodle/devdocs/",
    local_path: Annotated[Path, typer.Option(help="Local path for the cloned repository.")] = Path("./_smoke_test/devdocs"),
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")] = Path("./_smoke_test/agentic-docs.db"),
    tokenizer: Annotated[str, typer.Option(help="Tokenizer adapter to use.")] = "openai",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
    overlap_tokens: Annotated[int, typer.Option(help="Overlap tokens between adjacent chunks.")] = 60,
    skip_sync: Annotated[bool, typer.Option(help="Use the existing local checkout without attempting a network sync.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a repeatable sync, ingest, stats, and smoke-query workflow for Moodle devdocs."""

    repo_commit_hash = sync_repository(repo_url=repo_url, local_path=local_path) if not skip_sync else current_commit_hash(local_path)
    ingest_result = ingest_source(
        source=local_path,
        db_path=db_path,
        tokenizer_name=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    smoke_queries = [
        "Forms API validation",
        "settings.php plugin admin settings",
        "db/tasks.php scheduled tasks",
    ]
    smoke_results = {
        query_text: [result.model_dump() for result in query_chunks(db_path=db_path, query_text=query_text, top_k=3)]
        for query_text in smoke_queries
    }
    payload = {
        "repo_url": repo_url,
        "local_path": str(local_path),
        "db_path": str(db_path),
        "repo_commit_hash": repo_commit_hash,
        "ingest": ingest_result,
        "stats": SQLiteStore(db_path).detailed_stats(limit=5),
        "smoke_queries": smoke_results,
    }
    _emit(payload, json_output)


@app.command()
def eval(
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    eval_file: Annotated[Path, typer.Option(help="Path to the retrieval eval YAML or JSON file.")],
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run the lightweight retrieval evaluation harness."""

    report = run_eval(db_path=db_path, eval_file=eval_file)
    if json_output:
        _emit(report.model_dump(), True)
        return
    typer.echo(f"total_queries: {report.total_queries}")
    typer.echo(f"strong_passes: {report.strong_passes}")
    typer.echo(f"weak_passes: {report.weak_passes}")
    typer.echo(f"misses: {report.misses}")
    typer.echo(f"top_1_strong_pass_rate: {report.top_1.strong_pass_rate:.3f}")
    typer.echo(f"top_1_weak_pass_rate: {report.top_1.weak_pass_rate:.3f}")
    typer.echo(f"top_3_strong_pass_rate: {report.top_3.strong_pass_rate:.3f}")
    typer.echo(f"top_3_weak_pass_rate: {report.top_3.weak_pass_rate:.3f}")
    typer.echo(f"top_5_strong_pass_rate: {report.top_5.strong_pass_rate:.3f}")
    typer.echo(f"top_5_weak_pass_rate: {report.top_5.weak_pass_rate:.3f}")
    typer.echo("")
    for outcome in report.outcomes:
        typer.echo(f"{outcome.grade} {outcome.case_id}: {outcome.query}")
        if outcome.matched_result is not None:
            typer.echo(
                f"  best_match_rank={outcome.matched_result.rank} path={outcome.matched_result.source_file_path} rule={outcome.matched_result.matched_rule_type} matched_on={', '.join(outcome.matched_result.matched_on)}"
            )
        if outcome.failure_summary:
            typer.echo(f"  failure={outcome.failure_summary}")


if __name__ == "__main__":
    app()
