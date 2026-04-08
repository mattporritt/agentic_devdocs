"""Typer CLI for syncing, ingesting, querying, and inspecting docs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from agentic_docs.config import IngestConfig, QueryConfig
from agentic_docs.evaluation import render_eval_text, run_eval
from agentic_docs.git_sync import current_commit_hash, git_head_commit, git_working_tree_status, sync_repository
from agentic_docs.ingest import ingest_source
from agentic_docs.query_service import build_context_bundles, query_chunks
from agentic_docs.site_ingest import ingest_site_source
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


def _source_fields(metadata_json: dict[str, object] | None) -> tuple[str | None, str | None, str | None, str | None]:
    if not metadata_json:
        return None, None, None, None
    source_name = metadata_json.get("source_name")
    source_type = metadata_json.get("source_type")
    source_url = metadata_json.get("source_url")
    canonical_url = metadata_json.get("canonical_url")
    if not source_name and source_type == "repo_markdown":
        source_name = "devdocs_repo"
    return (
        str(source_name) if source_name is not None else None,
        str(source_type) if source_type is not None else None,
        str(source_url) if source_url is not None else None,
        str(canonical_url) if canonical_url is not None else None,
    )


def _validation_worktree_payload(repo_path: Path, allow_dirty: bool) -> dict[str, object]:
    """Validate and describe the current project worktree for trustworthy validation runs."""

    status = git_working_tree_status(repo_path)
    if status["is_git_repo"] is False:
        mode = "not_git_repo"
    elif status["clean"] is False and allow_dirty:
        mode = "dirty_override"
    else:
        mode = "clean"
    payload = {
        "repo_path": str(repo_path),
        "repo_head_commit": git_head_commit(repo_path),
        "is_git_repo": status["is_git_repo"],
        "clean": status["clean"],
        "allow_dirty": allow_dirty,
        "status_lines": status["status_lines"],
        "mode": mode,
    }
    if status["is_git_repo"] and status["clean"] is False and not allow_dirty:
        message = "Validation requires a clean git working tree. Commit or stash changes, or pass --allow-dirty."
        raise typer.BadParameter(message)
    return payload


def _validation_summary_status(eval_report: object | None) -> dict[str, object] | None:
    """Describe current-run validation quality without implying a baseline regression."""

    if eval_report is None:
        return None

    retrieval_fully_green = eval_report.weak_passes == 0 and eval_report.misses == 0
    weak_or_miss_present = eval_report.weak_passes > 0 or eval_report.misses > 0

    bundle_overall = eval_report.bundle_overall
    bundle_fully_green = None
    bundle_non_complete_present = None
    if bundle_overall is not None:
        bundle_fully_green = bundle_overall.partial == 0 and bundle_overall.insufficient == 0
        bundle_non_complete_present = bundle_overall.partial > 0 or bundle_overall.insufficient > 0

    if eval_report.misses > 0:
        overall_status = "NON_GREEN"
    elif weak_or_miss_present or bundle_non_complete_present:
        overall_status = "GREEN_WITH_WARNINGS"
    else:
        overall_status = "GREEN"

    return {
        "overall_status": overall_status,
        "retrieval_fully_green": retrieval_fully_green,
        "weak_or_miss_present": weak_or_miss_present,
        "bundle_fully_green": bundle_fully_green,
        "bundle_non_complete_present": bundle_non_complete_present,
        "baseline_comparison": (
            eval_report.baseline_comparison.model_dump()
            if getattr(eval_report, "baseline_comparison", None) is not None
            else {
                "status": "not_compared",
                "baseline_provided": False,
                "baseline_path": None,
                "retrieval_status": None,
                "bundle_status": None,
                "retrieval_deltas": {},
                "bundle_deltas": {},
                "changed_retrieval_buckets": [],
                "changed_bundle_buckets": [],
                "changed_cases": [],
            }
        ),
    }


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


@app.command("ingest-site")
def ingest_site(
    base_url: Annotated[str, typer.Option(help="Website base URL to scrape.")] = "https://design.moodle.com",
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")] = Path("./_smoke_test/design-system.db"),
    tokenizer: Annotated[str, typer.Option(help="Tokenizer adapter to use.")] = "openai",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
    overlap_tokens: Annotated[int, typer.Option(help="Overlap tokens between adjacent chunks.")] = 60,
    max_pages: Annotated[int | None, typer.Option(help="Optional limit for the number of in-scope pages to ingest.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Scrape a bounded website source and index it into SQLite."""

    result = ingest_site_source(
        base_url=base_url,
        db_path=db_path,
        tokenizer_name=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        max_pages=max_pages,
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
    bundle_max_tokens: Annotated[int, typer.Option(help="Maximum tokens to include in a context bundle.")] = 600,
    explain_ranking: Annotated[bool, typer.Option("--explain-ranking", help="Include reranking signal breakdowns in the output.")] = False,
    explain_bundle: Annotated[bool, typer.Option("--explain-bundle", help="Include bundle diagnostics in the output.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Query the indexed chunks using SQLite FTS5 lexical search."""

    config = QueryConfig(db_path=db_path, top_k=top_k)
    results = query_chunks(db_path=config.db_path, query_text=query_text, top_k=config.top_k)
    if context_bundle:
        bundles = build_context_bundles(
            db_path=config.db_path,
            results=results,
            query_text=query_text,
            include_previous=include_previous,
            include_next=include_next,
            bundle_max_tokens=bundle_max_tokens,
        )
        if json_output:
            _emit([bundle.model_dump() for bundle in bundles], True)
            return
        for bundle in bundles:
            typer.echo(f"rank: {bundle.rank}")
            typer.echo(f"score: {bundle.score}")
            typer.echo(f"source_file_path: {bundle.source_file_path}")
            typer.echo(f"source_name: {bundle.source_name}")
            typer.echo(f"source_type: {bundle.source_type}")
            typer.echo(f"source_url: {bundle.source_url}")
            typer.echo(f"canonical_url: {bundle.canonical_url}")
            typer.echo(f"document_title: {bundle.document_title}")
            typer.echo(f"section_title: {bundle.section_title}")
            typer.echo(f"heading_path: {' > '.join(bundle.heading_path)}")
            typer.echo(f"bundle_token_count: {bundle.bundle_token_count}")
            typer.echo(f"selection_strategy: {bundle.selection_strategy}")
            typer.echo(f"repo_commit_hash: {bundle.repo_commit_hash}")
            if explain_bundle and bundle.diagnostics:
                typer.echo(f"bundle_diagnostics: {json.dumps(bundle.diagnostics, sort_keys=True)}")
            if bundle.snippet:
                typer.echo(f"snippet: {bundle.snippet}")
            for chunk in bundle.chunks:
                typer.echo(
                    f"[{chunk.role}] {chunk.chunk_id} ({chunk.token_count} tokens) {chunk.source_file_path} "
                    f"{chunk.source_name or '-'} {' > '.join(chunk.heading_path)}"
                )
                typer.echo(chunk.content)
                typer.echo("")
        return
    if json_output:
        _emit([result.model_dump() for result in results], True)
        return
    for result in results:
        source_name, source_type, source_url, canonical_url = _source_fields(result.metadata_json)
        typer.echo(f"chunk_id: {result.chunk_id}")
        typer.echo(f"score: {result.score}")
        typer.echo(f"source_file_path: {result.source_file_path}")
        typer.echo(f"source_name: {source_name}")
        typer.echo(f"source_type: {source_type}")
        typer.echo(f"source_url: {source_url}")
        typer.echo(f"canonical_url: {canonical_url}")
        typer.echo(f"document_title: {result.document_title}")
        typer.echo(f"section_title: {result.section_title}")
        typer.echo(f"heading_path: {' > '.join(result.heading_path)}")
        typer.echo(f"token_count: {result.token_count}")
        typer.echo(f"repo_commit_hash: {result.repo_commit_hash}")
        typer.echo(f"normalized_query: {result.normalized_query}")
        typer.echo(f"snippet: {result.snippet}")
        if explain_ranking and result.rerank_score is not None:
            typer.echo(f"rerank_score: {result.rerank_score:.3f}")
            typer.echo(f"rerank_breakdown: {json.dumps(result.rerank_breakdown, sort_keys=True)}")
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
    if data.get("sources"):
        typer.echo("")
        typer.echo("sources:")
        for row in data["sources"]:
            source_name = row["source_name"] or "-"
            typer.echo(
                f"- {row['source_type']} / {source_name}: docs={row['document_count']} sections={row['section_count']} chunks={row['chunk_count']}"
            )
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
    eval_file: Annotated[Path | None, typer.Option(help="Optional eval YAML/JSON to run after ingest for a single-source-of-truth validation payload.")] = None,
    tokenizer: Annotated[str, typer.Option(help="Tokenizer adapter to use.")] = "openai",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
    overlap_tokens: Annotated[int, typer.Option(help="Overlap tokens between adjacent chunks.")] = 60,
    with_bundles: Annotated[bool, typer.Option(help="Evaluate bundle usefulness as part of the validation payload.")] = True,
    bundle_max_tokens: Annotated[int, typer.Option(help="Maximum tokens for evaluated context bundles in validation.")] = 450,
    baseline: Annotated[Path | None, typer.Option(help="Optional prior eval.json or verify_devdocs.json artifact to compare against.")] = None,
    skip_sync: Annotated[bool, typer.Option(help="Use the existing local checkout without attempting a network sync.")] = False,
    allow_dirty: Annotated[bool, typer.Option(help="Allow validation to run from a dirty git working tree.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a repeatable sync, ingest, stats, and smoke-query workflow for Moodle devdocs."""

    workspace_path = Path(os.getcwd())
    working_tree = _validation_worktree_payload(workspace_path, allow_dirty=allow_dirty)
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
    eval_report = (
        run_eval(
            db_path=db_path,
            eval_file=eval_file,
            with_bundles=with_bundles,
            bundle_max_tokens=bundle_max_tokens,
            baseline=baseline,
        )
        if eval_file is not None
        else None
    )
    validation_status = _validation_summary_status(eval_report)
    payload = {
        "repo_url": repo_url,
        "local_path": str(local_path),
        "db_path": str(db_path),
        "working_tree": working_tree,
        "repo_commit_hash": repo_commit_hash,
        "ingest": ingest_result,
        "stats": SQLiteStore(db_path).detailed_stats(limit=5),
        "smoke_queries": smoke_results,
        "eval": eval_report.model_dump() if eval_report is not None else None,
        "validation_status": validation_status,
    }
    _emit(payload, json_output)


@app.command("verify-site")
def verify_site(
    base_url: Annotated[str, typer.Option(help="Website base URL to scrape.")] = "https://design.moodle.com",
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")] = Path("./_smoke_test/design-system.db"),
    eval_file: Annotated[Path | None, typer.Option(help="Optional eval YAML/JSON to run after ingest for a single-source-of-truth validation payload.")] = None,
    tokenizer: Annotated[str, typer.Option(help="Tokenizer adapter to use.")] = "openai",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
    overlap_tokens: Annotated[int, typer.Option(help="Overlap tokens between adjacent chunks.")] = 60,
    max_pages: Annotated[int | None, typer.Option(help="Optional limit for the number of in-scope pages to ingest.")] = None,
    with_bundles: Annotated[bool, typer.Option(help="Evaluate bundle usefulness as part of the validation payload.")] = True,
    bundle_max_tokens: Annotated[int, typer.Option(help="Maximum tokens for evaluated context bundles in validation.")] = 450,
    baseline: Annotated[Path | None, typer.Option(help="Optional prior eval.json or verify artifact to compare against.")] = None,
    allow_dirty: Annotated[bool, typer.Option(help="Allow validation to run from a dirty git working tree.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run a repeatable ingest, stats, and smoke-query workflow for the design-system site."""

    workspace_path = Path(os.getcwd())
    working_tree = _validation_worktree_payload(workspace_path, allow_dirty=allow_dirty)
    ingest_result = ingest_site_source(
        base_url=base_url,
        db_path=db_path,
        tokenizer_name=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        max_pages=max_pages,
    )
    smoke_queries = [
        "design tokens for developers",
        "semantic colour tokens",
        "icon library",
    ]
    smoke_results = {
        query_text: [result.model_dump() for result in query_chunks(db_path=db_path, query_text=query_text, top_k=3)]
        for query_text in smoke_queries
    }
    eval_report = (
        run_eval(
            db_path=db_path,
            eval_file=eval_file,
            with_bundles=with_bundles,
            bundle_max_tokens=bundle_max_tokens,
            baseline=baseline,
        )
        if eval_file is not None
        else None
    )
    validation_status = _validation_summary_status(eval_report)
    payload = {
        "base_url": base_url,
        "db_path": str(db_path),
        "working_tree": working_tree,
        "ingest": ingest_result,
        "stats": SQLiteStore(db_path).detailed_stats(limit=5),
        "smoke_queries": smoke_results,
        "eval": eval_report.model_dump() if eval_report is not None else None,
        "validation_status": validation_status,
    }
    _emit(payload, json_output)


@app.command()
def eval(
    db_path: Annotated[Path, typer.Option(help="SQLite database path.")],
    eval_file: Annotated[Path, typer.Option(help="Path to the retrieval eval YAML or JSON file.")],
    show_weak_details: Annotated[bool, typer.Option(help="Show extra diagnostics for weak passes and ranking misses.")] = False,
    with_bundles: Annotated[bool, typer.Option(help="Evaluate agent-facing context bundle usefulness alongside retrieval.")] = False,
    show_bundle_details: Annotated[bool, typer.Option(help="Show extra diagnostics for bundle usefulness outcomes.")] = False,
    bundle_max_tokens: Annotated[int, typer.Option(help="Maximum tokens for evaluated context bundles.")] = 450,
    baseline: Annotated[Path | None, typer.Option(help="Optional prior eval.json or verify_devdocs.json artifact to compare against.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON output.")] = False,
) -> None:
    """Run the lightweight retrieval evaluation harness."""

    report = run_eval(
        db_path=db_path,
        eval_file=eval_file,
        with_bundles=with_bundles,
        bundle_max_tokens=bundle_max_tokens,
        baseline=baseline,
    )
    if json_output:
        _emit(report.model_dump(), True)
        return
    typer.echo(render_eval_text(report, show_weak_details=(show_weak_details or show_bundle_details)))


if __name__ == "__main__":
    app()
