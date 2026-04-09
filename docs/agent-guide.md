# Agent And Maintainer Guide

This guide is aimed at future human maintainers and AI coding agents working inside the repository.

The goal is not to restate the README. The goal is to explain how to change the code safely.

## First Principles

`agentic-docs` is intentionally:

- CLI-first
- local-first
- provenance-aware
- explicit in its heuristics
- strict in its evaluation

Before changing behavior, assume that:

- explainability is a feature
- determinism is a feature
- compactness is a feature
- validation artifacts are part of the product

## How To Read The Codebase

Start in this order:

1. [README.md](/Users/mattp/projects/agentic_devdocs/README.md)
2. [docs/architecture.md](/Users/mattp/projects/agentic_devdocs/docs/architecture.md)
3. `src/agentic_docs/models.py`
4. `src/agentic_docs/query_service.py`
5. `src/agentic_docs/evaluation.py`
6. `src/agentic_docs/cli.py`

That order mirrors the main contracts of the project:

- data model
- retrieval behavior
- grading and reporting
- user-facing workflow surface

## Safe Change Strategy

When making a change:

1. identify whether the change affects ingestion, retrieval, bundle assembly, reporting, or only documentation
2. find the existing tests that cover the same area
3. add or update a focused test before making a risky behavior change when practical
4. keep the logic explicit and local
5. run the full suite before finishing

If you discover a bug that once caused a false validation signal or a retrieval regression, prefer adding a regression test immediately.

## Testing Strategy Expectations

The suite mixes three styles of tests:

### Unit tests

Use these for:

- normalization helpers
- query profiling
- provenance normalization
- grading helpers
- contract models and schema rules

### Integration tests

Use these for:

- ingest + query flows
- CLI output modes
- eval end-to-end behavior on small fixture corpora
- site-ingest behavior on representative HTML/JSON fixtures

### Regression tests

Use these for:

- ranking regressions
- bundle regressions
- reporting inconsistencies
- provenance bugs
- contract drift

The standard for a good regression test here is simple:

- it reproduces the previously broken behavior with minimal fixture setup
- it asserts the intended invariant, not an incidental implementation detail

## Invariants You Should Not Break

### Retrieval and bundle evaluation

- `STRONG PASS`, `WEAK PASS`, `MISS` semantics stay strict
- `COMPLETE`, `PARTIAL`, `INSUFFICIENT` semantics stay strict
- bundle evaluation is an additional layer, not a replacement for retrieval grading

### Validation semantics

- current-run quality is separate from baseline comparison
- "warning" state is not the same as "regressed"
- baseline comparison only happens when an explicit baseline artifact is supplied

### Runtime contract

- `query --json-contract` must output only contract JSON
- required fields must stay present
- list fields must stay present as lists
- nullable fields must stay present as `null`
- ids must remain deterministic

### Multi-source provenance

- devdocs and design-system sources must remain distinguishable
- source-aware evaluation must preserve expected-source semantics
- bundles should not mix sources casually

## Safe Extension Patterns

### Adding a new concept family

If you need a new reranking concept family:

1. add the smallest explicit detection rule needed in `query_service.py`
2. make sure the rule is visible in ranking diagnostics
3. add focused tests for:
   - query profiling
   - ranking behavior
   - any bundle consequence if relevant
4. update docs if the behavior becomes part of the supported retrieval policy

Avoid hidden one-off query hacks.

### Extending the eval fixture

When broadening a fixture:

- keep targets explicit
- encode genuine ambiguity with acceptable targets rather than relaxing the benchmark
- prefer moderate, representative additions over giant fixture growth

### Extending the runtime contract

Treat runtime contract changes as external API changes.

Before changing it:

- ask whether the new field is necessary for downstream composition
- decide whether it is required, nullable, or always-present-empty
- update the schema artifact
- update contract tests
- update README examples

Do not add internal scoring details just because they are available.

## Known Weak Spots

These are not necessarily bugs, but they are areas that deserve care:

- `query_service.py` contains a lot of explicit heuristic logic and is easy to bloat
- `evaluation.py` has a wide surface because it owns grading, aggregation, rendering, and baseline comparison
- `site_ingest.py` depends on the current external site structure and may need maintenance if the site changes
- runtime contract mapping is close to CLI output paths, so drift can happen if not tested

If you refactor any of these, keep boundaries readable and avoid abstracting away the important rules.

## Conservative Design Choices

These are intentional and should not be removed casually:

- explicit reranking instead of embeddings
- local SQLite store instead of a service
- bounded design-system scraper instead of a crawler platform
- fixture-based grading instead of LLM judging
- machine-readable validation artifacts

If a future change proposes replacing one of these, it should be done explicitly as a product decision, not as incidental cleanup.

## Practical Commands

Run the full suite:

```bash
PYTHONPATH=./.vendor:./src pytest -q
```

Run a focused area:

```bash
PYTHONPATH=./.vendor:./src pytest -q tests/test_query_service.py
PYTHONPATH=./.vendor:./src pytest -q tests/test_evaluation.py
PYTHONPATH=./.vendor:./src pytest -q tests/test_cli.py -k json_contract
```

Generate a trustworthy validation run:

```bash
agentic-docs verify-devdocs \
  --repo-url https://github.com/moodle/devdocs/ \
  --local-path ./_smoke_test/devdocs \
  --db-path ./_smoke_test/agentic-docs.db \
  --eval-file ./evals/moodle_devdocs_eval.yaml
```

## What Good Maintenance Looks Like Here

Good changes in this repository usually have these traits:

- the docs tell a future reader why the rule exists
- the tests name the behavior being protected
- the implementation is readable without a debugger
- the validation and contract outputs stay trustworthy

That is the standard to optimize for during cleanup and future extension work.
