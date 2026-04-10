# Runtime Contract Reference

`agentic-docs` is the reference implementation for the shared runtime-facing contract used by the related tooling family.

The reference outer envelope is:

```json
{
  "tool": "agentic_docs",
  "version": "v1",
  "query": "...",
  "normalized_query": "...",
  "intent": {
    "query_intent": "...",
    "task_intent": "...",
    "concept_families": []
  },
  "results": []
}
```

Cross-tool alignment rule:

- keep the outer envelope aligned across tools
- keep `source`, `diagnostics`, and top-level `intent` conventions aligned
- allow `results[].content` to vary by tool where the inner payload genuinely differs

## Devdocs Example

Query:

```bash
agentic-docs query "Where do Moodle plugin admin settings go?" --db-path ./_smoke_test/agentic-docs.db --json-contract
```

Representative top result:

```json
{
  "tool": "agentic_docs",
  "version": "v1",
  "query": "Where do Moodle plugin admin settings go?",
  "normalized_query": "where moodle plugin admin settings go",
  "intent": {
    "query_intent": "conceptual",
    "task_intent": "file_location",
    "concept_families": ["admin_settings"]
  },
  "results": [
    {
      "id": "ba441b8539ab8656",
      "type": "knowledge_bundle",
      "rank": 1,
      "confidence": "high",
      "source": {
        "name": "devdocs_repo",
        "type": "repo_markdown",
        "url": null,
        "canonical_url": null,
        "path": "docs/apis/subsystems/admin/index.md",
        "document_title": "Admin settings",
        "section_title": "Individual settings",
        "heading_path": ["Individual settings"]
      }
    }
  ]
}
```

## Design-System Example

Query:

```bash
agentic-docs query "What are semantic colour tokens?" --db-path ./_smoke_test/design-system.db --json-contract
```

Representative top result:

```json
{
  "tool": "agentic_docs",
  "version": "v1",
  "query": "What are semantic colour tokens?",
  "normalized_query": "semantic colour tokens",
  "intent": {
    "query_intent": "conceptual",
    "task_intent": "general",
    "concept_families": []
  },
  "results": [
    {
      "id": "1cee67fde6a10881",
      "type": "knowledge_bundle",
      "rank": 1,
      "confidence": "high",
      "source": {
        "name": "design_system",
        "type": "scraped_web",
        "url": "https://design.moodle.com/98292f05f/p/32c91c",
        "canonical_url": "https://design.moodle.com/98292f05f/p/32c91c",
        "path": "design_system/styles/colours-32c91c.site",
        "document_title": "Colours",
        "section_title": "Semantic colour tokens",
        "heading_path": ["Colours", "Tokens", "Semantic colour tokens"]
      }
    }
  ]
}
```

## Combined Multi-Source Example

Query:

```bash
agentic-docs query "How should this render in Moodle?" --db-path ./_smoke_test/multi-source.db --json-contract
```

Representative top result:

```json
{
  "tool": "agentic_docs",
  "version": "v1",
  "query": "How should this render in Moodle?",
  "normalized_query": "should this render moodle",
  "intent": {
    "query_intent": "keyword",
    "task_intent": "general",
    "concept_families": []
  },
  "results": [
    {
      "id": "75c65eb086020aa1",
      "type": "knowledge_bundle",
      "rank": 1,
      "confidence": "high",
      "source": {
        "name": "devdocs_repo",
        "type": "repo_markdown",
        "url": null,
        "canonical_url": null,
        "path": "docs/apis/subsystems/output/index.md",
        "document_title": "Output API",
        "section_title": "Renderable",
        "heading_path": ["Page Output Journey", "Renderable"]
      }
    }
  ]
}
```
