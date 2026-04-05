# agentic-docs

`agentic-docs` is an internal Python CLI for turning developer documentation into compact, inspectable retrieval artifacts for AI coding agents.

Its v1 focus is narrow on purpose: ingest markdown documentation from a git repository, preserve structure, chunk it with model-aware token limits, and index it into SQLite with FTS5 so agents can retrieve high-signal Moodle documentation without wasting context on whole pages.

## Why this exists

AI-assisted Moodle development gets better when agents can fetch the right documentation context quickly and cheaply.

Naive approaches tend to:

- stuff entire markdown files into context windows
- split text on characters instead of model tokens
- lose heading structure and provenance
- make retrieval quality hard to debug

`agentic-docs` is designed to improve that by staying:

- local-first
- CLI-first
- token-aware
- traceable back to source files and headings
- easy to inspect when retrieval quality is poor

## V1 scope

Included in v1:

- git repository sync
- markdown discovery and parsing
- canonical document/section models
- token-aware chunking with overlap
- SQLite persistence with explicit tables
- FTS5 lexical retrieval
- inspect and stats commands
- pytest coverage for the core pipeline

Deferred in v1:

- HTTP API
- embeddings or vector search
- reranking
- non-markdown formats
- Moodle-specific semantic enrichment

## Architecture overview

Pipeline:

1. `sync` clones or updates the source git repository.
2. `ingest` discovers markdown files and parses them into canonical documents and sections.
3. Sections are chunked using token counts from a pluggable tokenizer interface.
4. Documents, sections, and chunks are written into SQLite.
5. Chunk content is indexed with FTS5 for retrieval.
6. `query`, `stats`, and inspect commands expose the indexed corpus in a debuggable way.

Core modules:

- `agentic_docs.git_sync`: clone/update repositories and record commit hashes
- `agentic_docs.parser`: markdown discovery and heading-aware section extraction
- `agentic_docs.models`: canonical Pydantic models
- `agentic_docs.tokenizers`: tokenizer abstraction plus OpenAI/tiktoken adapter
- `agentic_docs.chunking`: token-aware retrieval chunk construction
- `agentic_docs.storage`: SQLite schema, persistence, FTS5, and inspection helpers
- `agentic_docs.cli`: Typer CLI

## Installation

Create a virtual environment and install the package:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Example workflow

Sync Moodle developer docs:

```bash
agentic-docs sync \
  --repo-url https://github.com/moodle/devdocs/ \
  --local-path ./_data/devdocs
```

Ingest into SQLite:

```bash
agentic-docs ingest \
  --source ./_data/devdocs \
  --db-path ./_data/devdocs.db \
  --tokenizer openai \
  --max-tokens 400 \
  --overlap-tokens 60
```

Inspect counts:

```bash
agentic-docs stats --db-path ./_data/devdocs.db
```

Run a retrieval query:

```bash
agentic-docs query "form api moodle" --db-path ./_data/devdocs.db --top-k 5
```

Machine-readable output:

```bash
agentic-docs query "form api moodle" --db-path ./_data/devdocs.db --top-k 5 --json
```

## CLI commands

### `sync`

Clone or update a source git repository.

```bash
agentic-docs sync --repo-url <url> --local-path <path>
```

Returns the local path and the commit hash now available for ingestion.

### `ingest`

Parse markdown docs, build canonical sections, chunk them by tokens, and index them into SQLite.

```bash
agentic-docs ingest --source <path> --db-path <path> --tokenizer openai
```

Useful flags:

- `--max-tokens`: target upper bound for each retrieval chunk
- `--overlap-tokens`: shared tail context between adjacent chunks
- `--json`: machine-readable output

### `query`

Run lexical retrieval against indexed chunks.

```bash
agentic-docs query "form api moodle" --db-path <path> --top-k 5
```

Returned data includes:

- chunk id
- score
- content
- source file path
- document title
- section title
- heading path
- token count
- repo commit hash

### `stats`

Show counts for documents, sections, and chunks.

```bash
agentic-docs stats --db-path <path>
```

### `inspect-chunk`

Inspect an individual chunk and its metadata.

```bash
agentic-docs inspect-chunk <chunk-id> --db-path <path>
```

### `inspect-doc`

Inspect a document and its extracted sections.

```bash
agentic-docs inspect-doc <document-id> --db-path <path>
```

## Schema

The SQLite schema stays intentionally explicit:

- `documents`
- `sections`
- `chunks`
- `chunks_fts`

Key design choices:

- document ids are stable from source path
- section ids are stable from document identity plus heading path/order
- chunk ids are deterministic from section identity, order, and content
- repo commit hashes are stored so retrieval results can be traced back to a concrete source revision

## Token-aware chunking

Chunking is designed for retrieval efficiency rather than long-form reading.

The current strategy:

- respects markdown heading boundaries by chunking within extracted sections
- prepends heading context so isolated chunks still make sense
- uses tokenizer counts instead of characters
- supports configurable overlap
- truncates oversize paragraphs only when needed

This gives agents compact chunks that are usually easier to rank and cheaper to pass into a model context window.

## Testing

Run the test suite with:

```bash
pytest
```

Coverage currently includes:

- markdown discovery and section extraction
- tokenizer behavior
- token-aware chunking
- metadata preservation
- SQLite indexing and retrieval basics
- CLI ingest/query sanity

## Current limitations

- Retrieval is lexical FTS-only in v1.
- The markdown parser extracts section structure generically and does not yet add Moodle-specific semantics.
- Anthropic support is designed for via the tokenizer abstraction, but only the OpenAI/tiktoken adapter is implemented.
- Section extraction is heading-oriented and intentionally simple; deeper markdown semantics can be layered on in a later version.

## Future roadmap

Likely next steps after v1:

- embeddings and hybrid retrieval
- lightweight reranking
- Moodle-specific enrichment for common doc types
- result formatting tuned for downstream agent prompts
- incremental reindexing instead of full rebuilds

## Notes for extension

When extending this tool, prefer preserving the current design principles:

- explicit schema over hidden abstractions
- traceability over cleverness
- compact chunks over broad stuffing
- inspectability over opaque pipelines
