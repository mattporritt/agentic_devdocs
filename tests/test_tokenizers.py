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

