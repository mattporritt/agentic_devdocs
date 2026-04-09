# Architecture Guide

`agentic-docs` is a local, CLI-first retrieval system for authoritative Moodle documentation sources. The codebase is intentionally conservative: most important behavior is implemented as explicit rules rather than opaque model-driven logic.

This guide is for maintainers who need to understand the current architecture, data flow, and invariants before changing code.

## System Overview

There are two supported source families today:

- Moodle developer docs from the public git repository
- Moodle Design System docs from the public `design.moodle.com` site

Both sources end up in the same canonical internal model:

1. source document
2. logical sections extracted from headings
3. token-aware retrieval chunks
4. SQLite persistence and FTS5 indexing
5. query-time reranking
6. context bundle assembly
7. evaluation and validation reporting

The main architectural choice is that ingestion paths may differ, but retrieval operates over one shared schema and one shared set of query/bundle primitives.

## Module Responsibilities

### `agentic_docs.cli`

The Typer CLI entrypoint. It should stay thin where practical:

- parse CLI options
- delegate ingestion/query/eval work
- render human or machine outputs
- enforce validation workflow rules such as clean-worktree checks

Dense helper logic that is specific to runtime contracts or provenance should not accumulate here indefinitely.

### `agentic_docs.ingest`

Repo-based ingestion orchestration:

- discover markdown and MDX source files
- parse them into canonical documents and sections
- chunk them with the configured tokenizer
- store them into SQLite

This module is intentionally small because most real logic lives in parser, chunking, and storage.

### `agentic_docs.site_ingest`

Bounded ingestion for the Moodle Design System site.

Important design constraints:

- it is not a general crawler
- it is grounded in the current Zeroheight-backed site structure
- it converts scraped pages into the same canonical schema as repo docs

If the site changes shape in the future, update this module conservatively and keep the source metadata explicit.

### `agentic_docs.parser`

Heading-aware parsing for repo-based markdown and MDX.

Important responsibilities:

- discover source files
- extract frontmatter title when available
- preserve heading structure
- produce stable document and section identifiers
- avoid over-normalizing content in ways that make provenance or debugging harder

### `agentic_docs.chunking`

Token-aware section chunking.

Important invariants:

- chunk ids must be deterministic
- chunk order must be stable
- heading context should be preserved so isolated chunks remain understandable
- overlap is allowed, but chunk creation should remain inspectable

### `agentic_docs.storage`

Explicit SQLite schema and storage helper layer.

This module owns:

- schema creation and compatibility upgrades
- storing documents, sections, and chunks
- FTS5 index population
- corpus stats and inspect helpers
- retrieval row hydration back into `QueryResult`

Important invariant:

- storage is the source of stable chunk identifiers used elsewhere, including the runtime contract

### `agentic_docs.query_service`

The core retrieval pipeline.

Responsibilities:

- query normalization
- query profiling and intent detection
- candidate retrieval from SQLite FTS5
- explicit reranking
- context bundle assembly
- bundle support-chunk selection

This is the most tuning-heavy module in the repository. It should stay explicit and debuggable. When adding rules:

- prefer named, inspectable heuristics
- keep concept-family handling grouped and readable
- avoid hidden behavior branches that only exist for a single fixture

### `agentic_docs.provenance`

Small shared helpers for normalizing source identity across:

- retrieval rows
- evaluation
- CLI output
- runtime contract generation

This module exists specifically to avoid subtle drift between "what source won?" in different reporting paths.

### `agentic_docs.evaluation`

The evaluation and reporting layer.

Responsibilities:

- load eval fixtures
- grade retrieval results and bundles
- build aggregate reports
- compare reports against baselines
- render human-readable output from the same canonical report object

The most important invariant here is consistency: JSON, text, and markdown summaries must describe the same report object rather than recomputing metrics independently.

### `agentic_docs.models`

Shared Pydantic models for:

- canonical documents/sections/chunks
- query results and bundles
- evaluation reports
- runtime contract payloads

These models define the internal and external contracts of the codebase. Changes here should be made deliberately and accompanied by docs and tests.

### `agentic_docs.runtime_contract`

Maps context bundles into the stable runtime-facing `v1` JSON envelope.

This module should stay focused on:

- contract-shape normalization
- deterministic ids
- compact structured sections
- provenance consistency

It should not become a second retrieval pipeline.

## Data Lifecycle

### Ingestion lifecycle

1. fetch or open source content
2. normalize into `DocumentModel` and `SectionModel`
3. chunk each section into `ChunkModel`
4. store rows in SQLite
5. index chunks in `chunks_fts`

### Query lifecycle

1. normalize raw query text
2. build a `QueryProfile`
3. retrieve a broad lexical candidate set from FTS5
4. apply explicit reranking
5. optionally build context bundles from the top results
6. optionally map bundles into the runtime contract

### Validation lifecycle

1. create or refresh a corpus
2. run smoke queries and stats
3. run eval sequentially against the same database snapshot
4. generate machine-readable and human-readable artifacts
5. optionally compare against a supplied baseline artifact

## Important Data Models

### Canonical corpus models

- `DocumentModel`
- `SectionModel`
- `ChunkModel`

Important invariant:

- these models are source-agnostic enough to support multiple authoritative sources without flattening provenance away

### Retrieval models

- `QueryResult`
- `ContextBundle`
- `ContextBundleChunk`

Important invariant:

- retrieval and bundle assembly should preserve enough source and heading context to explain every returned result

### Runtime contract models

- `RuntimeContractEnvelope`
- `RuntimeContractResult`
- `RuntimeContractSource`
- `RuntimeContractContent`
- `RuntimeContractSection`
- `RuntimeContractDiagnostics`

Important invariant:

- the runtime contract must remain stable, explicit, and easy for downstream code to consume without defensive guessing

### Evaluation models

- `EvalCase`
- `EvalOutcome`
- `EvalReport`
- baseline comparison models

Important invariant:

- evaluation strictness should not be relaxed to hide quality problems

## Source Provenance Rules

Source provenance is first-class throughout the system.

Every stored source should preserve as much of the following as makes sense:

- `source_type`
- `source_name`
- `source_url`
- `canonical_url`
- `source_path`
- `document_title`
- `section_title`
- `heading_path`

The system should not silently collapse multiple authoritative sources into a source-less corpus.

## Evaluation And Reporting Invariants

These invariants are important and should be preserved:

- retrieval grading remains `STRONG PASS`, `WEAK PASS`, `MISS`
- bundle grading remains `COMPLETE`, `PARTIAL`, `INSUFFICIENT`
- current-run quality status is separate from baseline comparison status
- "not fully green" does not mean "regressed"
- source-aware evaluation must keep expected source semantics explicit
- `eval.json` is the canonical machine-readable report for a run

## Runtime Contract Invariants

The runtime contract is now part of the external surface area of the project.

Important invariants for `v1`:

- stable envelope fields are always present
- list fields are always present and may be empty
- nullable fields stay present as `null`
- `content.sections` is always present
- result and section ids are deterministic
- provenance shape stays consistent across sources

The checked-in schema at `schemas/runtime_contract_v1.json` should match the live Pydantic model.

## Known Conservative Choices

The codebase intentionally prefers conservative choices in a few places:

- lexical retrieval plus explicit reranking instead of embeddings
- bounded site scraping instead of a general crawler
- local SQLite rather than a service
- explicit heuristics instead of opaque semantic models
- explicit eval fixtures instead of model-based judges

These choices should not be "cleaned up" away accidentally during refactors.

## Where To Be Careful

Pay special attention when changing:

- provenance normalization
- query profiling and intent rules
- bundle support selection
- evaluation aggregation
- runtime contract models
- validation workflow sequencing

These areas are easy to destabilize because they connect many pieces of the system.
