# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

from pathlib import Path

import pytest
from pydantic import ValidationError

from agentic_docs.config import IngestConfig, QueryConfig


def test_ingest_config_applies_defaults_and_coerces_paths(tmp_path: Path) -> None:
    config = IngestConfig(source=str(tmp_path / "docs"), db_path=str(tmp_path / "docs.db"))

    assert config.source == tmp_path / "docs"
    assert config.db_path == tmp_path / "docs.db"
    assert config.tokenizer == "openai"
    assert config.max_tokens == 400
    assert config.overlap_tokens == 60


def test_ingest_config_validates_token_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        IngestConfig(source=tmp_path / "docs", db_path=tmp_path / "docs.db", max_tokens=40)

    with pytest.raises(ValidationError):
        IngestConfig(source=tmp_path / "docs", db_path=tmp_path / "docs.db", overlap_tokens=-1)


def test_query_config_validates_top_k_bounds(tmp_path: Path) -> None:
    config = QueryConfig(db_path=tmp_path / "docs.db", top_k=7)
    assert config.top_k == 7

    with pytest.raises(ValidationError):
        QueryConfig(db_path=tmp_path / "docs.db", top_k=0)

    with pytest.raises(ValidationError):
        QueryConfig(db_path=tmp_path / "docs.db", top_k=101)
