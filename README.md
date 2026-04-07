# agentic-docs

`agentic-docs` is an internal Python CLI for turning Moodle developer documentation into compact, inspectable retrieval artifacts for AI coding agents.

It is built to answer a practical question: when an agent is working on Moodle LMS code, can we retrieve the smallest useful documentation context instead of flooding the model with whole pages?

The current phase focuses on three things:

- concept-aware retrieval that can surface the right explanatory section
- compact context bundles that are useful to downstream coding agents
- bundle and retrieval diagnostics that make tuning more actionable

The current evaluation flow is intentionally stricter than earlier iterations:

- `STRONG PASS` means retrieval found a clearly correct target
- `WEAK PASS` means retrieval found a related fallback but not the best target
- `MISS` means retrieval failed to surface an acceptably correct target within the configured top-k

Bundle usefulness is now graded separately:

- `COMPLETE` means the returned bundle contains the key explanatory section, stays reasonably compact, and is usable as-is
- `PARTIAL` means the bundle is usable but thin, noisy, truncated, or otherwise suboptimal
- `INSUFFICIENT` means the bundle is missing critical context or is not practically usable for an agent

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

## Scope

Included currently:

- git repository sync
- markdown discovery and parsing
- canonical document/section models
- token-aware chunking with overlap
- SQLite persistence with explicit tables
- FTS5 lexical retrieval
- inspect and richer stats commands
- eval harness with curated Moodle retrieval cases
- agent-oriented context bundle output
- repeatable Moodle devdocs verification workflow
- canonical-doc preference over equivalent versioned duplicates
- conservative MDX wrapper-noise reduction
- pytest coverage for the core pipeline

Still deferred:

- HTTP API
- embeddings or vector search
- non-markdown formats
- Moodle-specific semantic enrichment

## Architecture overview

Pipeline:

1. `sync` clones or updates the source git repository.
2. `ingest` discovers markdown files and parses them into canonical documents and sections.
3. Sections are chunked using token counts from a pluggable tokenizer interface.
4. Documents, sections, and chunks are written into SQLite.
5. Chunk content is indexed with FTS5 for retrieval.
6. Queries are normalized for safer FTS5 matching, then reranked with explicit field-aware scoring.
7. Canonical docs are preferred over equivalent versioned docs when the match quality is otherwise similar.
8. Concept-heavy queries can receive concept-family and section-focus boosts so explanatory sections beat incidental mentions more often.
9. `query`, `stats`, inspect commands, and `eval` expose the corpus in a debuggable way, including ranking diagnostics for weak passes.

Core modules:

- `agentic_docs.git_sync`: clone/update repositories and record commit hashes
- `agentic_docs.parser`: markdown discovery and heading-aware section extraction
- `agentic_docs.models`: canonical Pydantic models
- `agentic_docs.tokenizers`: tokenizer abstraction plus OpenAI/tiktoken adapter
- `agentic_docs.chunking`: token-aware retrieval chunk construction
- `agentic_docs.storage`: SQLite schema, persistence, FTS5, and inspection helpers
- `agentic_docs.query_service`: query normalization, retrieval, and context bundle assembly
- `agentic_docs.evaluation`: eval loading and scoring
- `agentic_docs.cli`: Typer CLI

## Installation

Create a virtual environment and install the package:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Verify Against Moodle Devdocs

This is the recommended end-to-end workflow against the live public repo:

```bash
agentic-docs verify-devdocs \
  --repo-url https://github.com/moodle/devdocs/ \
  --local-path ./_smoke_test/devdocs \
  --db-path ./_smoke_test/agentic-docs.db \
  --eval-file ./evals/moodle_devdocs_eval.yaml \
  --tokenizer openai \
  --max-tokens 400 \
  --overlap-tokens 60
```

This command:

- syncs the repo locally
- ingests the docs into SQLite
- prints stats and diagnostic summaries
- runs a few smoke-test queries
- optionally runs the strict eval sequentially on the same freshly ingested DB when `--eval-file` is provided

Validation now checks the current project git worktree before running:

- clean worktrees are accepted normally
- dirty worktrees fail fast by default
- `--allow-dirty` records that the run was forced from a dirty tree

This guard exists because validation artifacts are only trustworthy when they can be traced to a specific committed tool state.

The current public Moodle devdocs repo contains markdown and MDX content across directories such as `docs/`, `versioned_docs/`, and `general/`, and the current ingestion path is intentionally broad enough to cover that real structure.

When the corpus contains both canonical and versioned copies of substantively similar docs, retrieval prefers the canonical non-versioned path but still allows versioned docs through when they remain the best available match.

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

Agent-oriented context bundles:

```bash
agentic-docs query \
  "Forms API validation" \
  --db-path ./_data/devdocs.db \
  --top-k 3 \
  --context-bundle \
  --include-next \
  --bundle-max-tokens 350 \
  --explain-bundle \
  --json
```

Run the starter retrieval evaluation set:

```bash
agentic-docs eval \
  --db-path ./_data/devdocs.db \
  --eval-file ./evals/moodle_devdocs_eval.yaml \
  --with-bundles \
  --show-bundle-details
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

Useful flags:

- `--context-bundle`: return compact agent-oriented bundles instead of bare chunks
- `--include-previous`: include the previous chunk in the bundle
- `--include-next`: include the next chunk in the bundle
- `--bundle-max-tokens`: cap the context bundle size, truncating oversize matches if necessary
- `--explain-ranking`: include the explicit reranking score breakdown
- `--explain-bundle`: include bundle diagnostics such as budget fit, chunk roles, and truncation
- `--json`: machine-readable output

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
- normalized query
- content snippet
- rerank score and score breakdown when requested

### `stats`

Show counts for documents, sections, and chunks.

```bash
agentic-docs stats --db-path <path>
```

Stats now include:

- corpus overview counts
- average chunk token count
- top documents by chunk count
- largest chunks
- very small chunks
- exact duplicate chunk candidates

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

### `verify-devdocs`

Run the repeatable public Moodle devdocs verification workflow.

```bash
agentic-docs verify-devdocs \
  --repo-url https://github.com/moodle/devdocs/ \
  --local-path ./_smoke_test/devdocs \
  --db-path ./_smoke_test/agentic-docs.db \
  --eval-file ./evals/moodle_devdocs_eval.yaml
```

Useful flags:

- `--eval-file`: run the strict eval as part of the same sequential validation workflow
- `--allow-dirty`: override the clean-worktree requirement, while recording that the run came from a dirty tree

### `eval`

Run the lightweight retrieval evaluation harness.

```bash
agentic-docs eval --db-path <path> --eval-file ./evals/moodle_devdocs_eval.yaml --show-weak-details
```

The eval command reports:

- total query count
- strong pass, weak pass, and miss counts
- top-1, top-3, and top-5 strong-pass rates
- top-1, top-3, and top-5 weak-pass rates
- bundle complete / partial / insufficient counts and rates when `--with-bundles` is enabled
- per-query strong pass / weak pass / miss
- per-query bundle usefulness grade when enabled
- the highest-ranked matching result
- preferred-result rank and ranking diagnostics for weak passes when requested
- bundle diagnostics for missing headings, oversize bundles, or thin/noisy context when requested
- failure summaries for misses

Use `--json` for machine-readable output suitable for automation or later comparison between runs.

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

## Canonical Preference

The Moodle devdocs corpus includes both canonical docs and versioned docs.

To reduce retrieval duplication and noise:

- canonical non-versioned docs are preferred over versioned docs when they are otherwise similar
- near-duplicate results with the same canonical path and heading signature are suppressed conservatively
- versioned docs are still allowed when no canonical result is as good

This preference is now part of the explicit reranking layer, alongside title, section, heading, path, and chunk-quality signals.

## Field-Aware Ranking

SQLite FTS5 remains the lexical retrieval base, but final ordering is adjusted with a small explicit reranker.

Current scoring signals include:

- lexical FTS score from the candidate set
- overlap with document title, section title, heading path, and source path
- phrase hits in document and section titles
- conceptual-query preference for explainer and subsystem pages when the query is clearly asking how something works
- canonical-doc preference
- concept-page preference over broader plugin-type pages for conceptual queries
- section-focus bonuses when the heading path and section title align better than the broader page title
- targeted disambiguation for upgrade, language-string, output/template, testing, privacy, forms-validation, and events query families
- subsystem-path preference when the query clearly targets a subsystem page
- conservative penalties for example-heavy, workflow-heavy, or otherwise incidental chunks
- chunk-size quality heuristics

Use `agentic-docs query ... --explain-ranking` to inspect the weighted score breakdown for each result.

The current concept-aware reranker is still deliberately small and explicit. For example:

- conceptual Mustache or rendering queries prefer template and output explainer pages over narrower plugin-specific examples
- renderer-only and docs-location queries prefer Output API sections over template examples when the user is really asking where renderers are documented
- conceptual events queries prefer the events concept pages and `db/events.php` explainer over incidental references such as calendar or xAPI pages
- concept-heavy strings, privacy-provider, PHPUnit, and Behat queries prefer subsystem/testing explainers over broad checklists, workflow pages, or incidental mentions
- broad plugin-type overview pages are demoted when a cleaner canonical explainer page is already in the candidate set

## Retrieval Evaluation

The evaluation harness is intentionally lightweight. It is not a benchmark framework and it does not try to be statistically perfect.

It is meant to answer practical questions during tuning:

- do common Moodle developer questions retrieve the right docs?
- are chunks compact but still useful?
- is lexical retrieval good enough for now?
- which misses point to chunking, indexing, query handling, or packaging problems?

The eval file lives at [evals/moodle_devdocs_eval.yaml](/Users/mattp/projects/agentic_devdocs/evals/moodle_devdocs_eval.yaml) and now aims to be broader than the original narrow benchmark. It mixes:

- conceptual phrasing
- implementation phrasing
- file-location phrasing
- troubleshooting-style phrasing
- multiple phrasings for the same underlying Moodle development concept

Each case still uses explicit grading targets:

- `preferred_document_paths` and `preferred_heading_substrings` for strong passes
- `acceptable_document_paths` and `acceptable_heading_substrings` for weak passes
- `disallowed_document_paths` for obviously wrong-but-lexically-close results
- `bucket` to group related areas such as testing, privacy, output/rendering, or upgrade/schema work
- `concept_id` to group multiple query phrasings for the same underlying task
- `required_heading_substrings_for_bundle` and `max_reasonable_bundle_tokens` for explicit bundle usefulness checks

It covers representative questions such as:

- plugin admin settings
- plugin upgrade steps
- capabilities in `db/access.php`
- scheduled tasks in `db/tasks.php`
- web services
- Forms API validation
- language strings
- output and templates
- privacy providers
- events
- PHPUnit
- Behat

The grading logic is intentionally explicit:

- strong pass: best acceptable hit is a preferred target
- weak pass: best acceptable hit is fallback-only
- miss: no preferred or acceptable target appears within the configured top-k

The report exposes:

- strong-pass counts and rates
- weak-pass counts and rates
- misses
- bucket-level breakdowns
- concept-level breakdowns across related phrasings
- bundle bucket breakdowns when bundle evaluation is enabled
- matched rule type and matched result rank per query
- preferred-result rank when a better target was retrieved but ranked too low
- concise ranking diagnostics for weak passes and ranking misses
- concise bundle diagnostics explaining why a bundle was complete, partial, or insufficient

This keeps failures understandable and easy to tune against.

`eval.json` is the canonical source of truth for a run. The plain-text eval output and any markdown summary should be rendered from the same in-memory `EvalReport`, and the code now performs an internal consistency check to fail fast if aggregate counts or rates ever diverge from the per-query outcomes.

One regression investigation found that validation artifacts could look red even when retrieval itself was healthy if ingest and eval were run in parallel against the same SQLite database. The recommended workflow is now to use `verify-devdocs --eval-file ...`, which performs ingest, smoke queries, stats, and eval sequentially on the same DB snapshot.

It is normal and acceptable for a broader benchmark to reduce headline scores. A perfect score on a small hand-tuned set is not enough; the more useful signal is whether retrieval remains strong across realistic wording variation and whether weak spots cluster in a meaningful bucket or concept group.

Bucket results help answer where retrieval is weakest.
Concept results help answer whether retrieval is brittle to alternate phrasings of the same task.
Bundle results help answer whether the retrieved context is actually actionable for an agent, especially on file-location and implementation-guide queries.

## Context Bundles

Bare chunk retrieval is useful for debugging, but agent workflows usually need a slightly richer structure.

`--context-bundle` returns compact bundles containing:

- matched chunk content
- source file path
- document title
- section title
- heading path
- token counts
- repo commit hash
- a selection strategy marker such as `match_only`, `section_window`, `task_support`, `task_support_truncated`, or `truncated_match`
- optional adjacent chunks from the same section when they fit within the token budget

Bundles are assembled conservatively:

- the matched chunk is always included
- oversize matched chunks are truncated to the requested bundle budget
- same-section adjacent chunks are added only when explicitly requested and they still fit the budget
- file-location and implementation-guide queries may add one support chunk when it contributes a concrete implementation anchor such as `settings.php`, `db/tasks.php`, `db/services.php`, or nearby writing-guide context
- if that support chunk is more valuable than the full matched chunk, the matched chunk may be trimmed so the final bundle still fits the budget
- repeated heading prefixes are stripped from adjacent context to reduce prompt noise

Task-oriented bundle completion is rule-based and inspectable:

- file-location intent is detected from phrasing like `where do`, `what file`, `where is this defined`, or `where is this registered`
- implementation-guide intent is detected from phrasing like `how do I implement`, `how do I write`, `how do I define`, or `how do I configure`
- support chunks are chosen with explicit file-anchor and concept-overlap heuristics rather than opaque semantic scoring
- `--explain-bundle` reports the detected task intent, whether a support chunk was added, which file anchors were surfaced, and whether the match had to be trimmed to stay inside budget

Bundle usefulness is now measured explicitly in eval runs. The current heuristics are deliberately simple and inspectable:

- `COMPLETE`: preferred bundle path present, required headings present when specified, and bundle remains within budget
- `PARTIAL`: key section is present but the bundle is truncated, too thin, slightly over budget, or otherwise fallback-quality
- `INSUFFICIENT`: critical required heading is missing, no usable bundle was produced, or the bundle is significantly over budget

This lets the benchmark expose cases where retrieval is already strong but the returned context package is still weak for an agent.

This matters most for concrete task queries such as:

- where do plugin admin settings go
- what file defines scheduled tasks
- where are external service functions declared
- where do I find the Behat writing guide

This is still intentionally simple. It is not a full prompt-assembly system, but it is a practical bridge toward agent-ready retrieval.

## Inspecting Weak Passes

The strict eval loop is the tuning source of truth.

When a query stays weak, use:

```bash
agentic-docs eval --db-path <path> --eval-file ./evals/moodle_devdocs_eval.yaml --show-weak-details
```

and:

```bash
agentic-docs query "How do events work in plugins?" --db-path <path> --top-k 5 --explain-ranking --json
```

and for bundle usefulness:

```bash
agentic-docs eval --db-path <path> --eval-file ./evals/moodle_devdocs_eval.yaml --with-bundles --show-bundle-details
```

These diagnostics show:

- the actual top-ranked result
- the preferred-result rank if a better target was already retrieved
- the explicit score breakdown that made one result outrank another
- whether the remaining issue is broad-page dominance, plugin-type noise, or weak lexical evidence
- whether a bundle missed a required heading, exceeded budget, or stayed too thin to be fully useful

## MDX Noise Reduction

Some Moodle devdocs pages use MDX wrappers, component tags, imports, and editorial comments that are useful for site rendering but poor retrieval targets.

The current parser and chunker therefore apply conservative cleanup:

- remove front-matter-adjacent MDX import/export lines from indexed section text
- strip standalone wrapper/component lines and markdownlint/editorial comments
- suppress very small low-signal chunks that are effectively wrapper residue

This is intentionally conservative. The goal is to reduce obvious retrieval noise without trying to fully render MDX or discard real prose.

## Testing

Run the test suite with:

```bash
pytest
```

Coverage currently includes:

- markdown discovery and section extraction
- front matter title extraction
- tokenizer behavior
- token-aware chunking
- low-signal MDX wrapper suppression
- metadata preservation
- SQLite indexing and retrieval basics
- query normalization and context bundles
- canonical-doc preference
- strict eval file loading and strong/weak/miss scoring
- bundle usefulness grading and reporting
- field-aware reranking and explanation output
- weak-pass diagnostics
- CLI ingest/query/eval sanity

## Current limitations

- Retrieval is lexical FTS-only in v1.
- The markdown parser extracts section structure generically and does not yet add deeper Moodle-specific enrichment.
- Anthropic support is designed for via the tokenizer abstraction, but only the OpenAI/tiktoken adapter is implemented.
- Evaluation matching is still substring-based and intentionally explicit; it is stricter than before, but still not semantic understanding.
- Context bundle usefulness is judged by explicit heuristics, not model-based quality judgments.
- Context bundles are compact retrieval packages, not yet full prompt construction.
- Canonical-doc preference and MDX cleanup are conservative heuristics rather than full corpus normalization.
- Field-aware ranking is hand-tuned and inspectable, not learned; expect more iteration as weak passes change.

## Future roadmap

Likely next steps after v1:

- embeddings and hybrid retrieval
- Moodle-specific enrichment for common doc types
- result formatting tuned for downstream agent prompts
- incremental reindexing instead of full rebuilds
- better diagnostics for near-duplicate and low-signal chunks

## Notes for extension

When extending this tool, prefer preserving the current design principles:

- explicit schema over hidden abstractions
- traceability over cleverness
- compact chunks over broad stuffing
- inspectability over opaque pipelines
