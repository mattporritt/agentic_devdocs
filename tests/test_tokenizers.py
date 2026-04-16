# Copyright (c) Moodle Pty Ltd. All rights reserved.
# Licensed under the Moodle Community License v1.3.
# See LICENSE.md in the repository root for full terms.
# Commercial use requires a separate written agreement with Moodle.

from agentic_docs.tokenizers import OpenAITokenizer, get_tokenizer


def test_openai_tokenizer_counts_tokens() -> None:
    tokenizer = OpenAITokenizer()

    assert tokenizer.count_tokens("Moodle developer docs") > 0
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_get_tokenizer_rejects_unknown() -> None:
    try:
        get_tokenizer("anthropic")
    except ValueError as exc:
        assert "Unsupported tokenizer" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

