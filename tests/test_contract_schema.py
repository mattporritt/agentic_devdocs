import json
from pathlib import Path

from typer.testing import CliRunner

from agentic_docs.cli import app


runner = CliRunner()


def _resolve_ref(schema: dict, ref: str) -> dict:
    assert ref.startswith("#/")
    node = schema
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _validate_schema_node(schema: dict, node_schema: dict, value, path: str = "$") -> None:
    if "$ref" in node_schema:
        _validate_schema_node(schema, _resolve_ref(schema, node_schema["$ref"]), value, path)
        return

    if "anyOf" in node_schema:
        errors: list[AssertionError] = []
        for option in node_schema["anyOf"]:
            try:
                _validate_schema_node(schema, option, value, path)
                return
            except AssertionError as exc:  # pragma: no cover - only used on failure
                errors.append(exc)
        raise AssertionError(f"{path} failed anyOf validation: {errors}") from errors[0]

    if "enum" in node_schema:
        assert value in node_schema["enum"], f"{path}={value!r} not in enum {node_schema['enum']!r}"
        return

    schema_type = node_schema.get("type")
    if schema_type == "object":
        assert isinstance(value, dict), f"{path} should be object"
        required = node_schema.get("required", [])
        for key in required:
            assert key in value, f"{path}.{key} missing"
        if node_schema.get("additionalProperties") is False:
            allowed = set(node_schema.get("properties", {}).keys())
            extra = set(value.keys()) - allowed
            assert not extra, f"{path} has unexpected properties: {sorted(extra)!r}"
        for key, property_schema in node_schema.get("properties", {}).items():
            if key in value:
                _validate_schema_node(schema, property_schema, value[key], f"{path}.{key}")
        return

    if schema_type == "array":
        assert isinstance(value, list), f"{path} should be array"
        item_schema = node_schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_node(schema, item_schema, item, f"{path}[{index}]")
        return

    if schema_type == "string":
        assert isinstance(value, str), f"{path} should be string"
        return

    if schema_type == "integer":
        assert isinstance(value, int) and not isinstance(value, bool), f"{path} should be integer"
        return

    if schema_type == "null":
        assert value is None, f"{path} should be null"
        return

    raise AssertionError(f"Unsupported schema fragment at {path}: {node_schema!r}")


def _load_schema(name: str) -> dict:
    return json.loads(Path(name).read_text(encoding="utf-8"))


def test_runtime_contract_live_outputs_validate_against_shared_and_devdocs_schemas(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    (docs_dir / "apis" / "subsystems" / "admin").mkdir(parents=True)
    (docs_dir / "apis" / "subsystems" / "output").mkdir(parents=True)
    (docs_dir / "design_system" / "styles").mkdir(parents=True)

    (docs_dir / "apis" / "subsystems" / "admin" / "index.md").write_text(
        "---\n"
        "title: Admin settings\n"
        "---\n\n"
        "## Individual settings\n\n"
        "Use settings.php to add plugin admin settings.\n",
        encoding="utf-8",
    )
    (docs_dir / "apis" / "subsystems" / "output" / "index.md").write_text(
        "---\n"
        "title: Output API\n"
        "---\n\n"
        "## Page Output Journey\n\n"
        "### Renderable\n\n"
        "Renderables can be rendered through templates in Moodle output.\n",
        encoding="utf-8",
    )
    (docs_dir / "design_system" / "styles" / "colours-32c91c.md").write_text(
        "# Colours\n\n"
        "## Tokens\n\n"
        "### Semantic colour tokens\n\n"
        "Semantic colour tokens give colours a role in the interface.\n",
        encoding="utf-8",
    )

    db_path = tmp_path / "docs.db"
    ingest_result = runner.invoke(app, ["ingest", "--source", str(docs_dir), "--db-path", str(db_path), "--json"])
    assert ingest_result.exit_code == 0

    outer_schema = _load_schema("schemas/runtime_outer_v1.json")
    devdocs_schema = _load_schema("schemas/runtime_contract_v1.json")
    queries = [
        "Where do Moodle plugin admin settings go?",
        "What are semantic colour tokens?",
        "How should this render in Moodle?",
    ]

    for query in queries:
        result = runner.invoke(app, ["query", query, "--db-path", str(db_path), "--json-contract"])
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        _validate_schema_node(outer_schema, outer_schema, payload)
        _validate_schema_node(devdocs_schema, devdocs_schema, payload)
