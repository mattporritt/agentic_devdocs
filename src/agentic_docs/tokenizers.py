"""Tokenizer abstractions and OpenAI-compatible implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


CL100K_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}

CL100K_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""


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
        self._encoding = self._build_encoding(encoding_name)

    def name(self) -> str:
        return "openai"

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def _build_encoding(self, encoding_name: str) -> tiktoken.Encoding:
        if encoding_name != "cl100k_base":
            return tiktoken.get_encoding(encoding_name)

        local_encoding_path = Path(__file__).resolve().parent / "data" / "cl100k_base.tiktoken"
        mergeable_ranks = load_tiktoken_bpe(
            str(local_encoding_path),
            expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        )
        return tiktoken.Encoding(
            name="cl100k_base",
            pat_str=CL100K_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=CL100K_SPECIAL_TOKENS,
        )


def get_tokenizer(name: str) -> Tokenizer:
    """Construct a tokenizer by its configured name."""

    normalized = name.strip().lower()
    if normalized == "openai":
        return OpenAITokenizer()
    msg = f"Unsupported tokenizer '{name}'. Supported values: openai"
    raise ValueError(msg)
