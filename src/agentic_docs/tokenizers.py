"""Tokenizer abstractions and OpenAI-compatible implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tiktoken


class Tokenizer(ABC):
    """Abstract tokenizer used by the chunking pipeline."""

    @abstractmethod
    def name(self) -> str:
        """Return the stable tokenizer name."""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in the supplied text."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Return token ids for the supplied text."""

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Return text for the supplied token ids."""


class OpenAITokenizer(Tokenizer):
    """Tokenizer backed by tiktoken for OpenAI-compatible token counting."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def name(self) -> str:
        return "openai"

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


def get_tokenizer(name: str) -> Tokenizer:
    """Construct a tokenizer by its configured name."""

    normalized = name.strip().lower()
    if normalized == "openai":
        return OpenAITokenizer()
    msg = f"Unsupported tokenizer '{name}'. Supported values: openai"
    raise ValueError(msg)

