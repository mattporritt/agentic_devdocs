"""Utility helpers shared across the project."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_id(*parts: str) -> str:
    """Build a deterministic identifier from stable string components."""

    joined = "::".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def sha1_text(text: str) -> str:
    """Hash text content with SHA1 for stable provenance metadata."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    """Read a UTF-8 text file while surfacing decoding issues explicitly."""

    return path.read_text(encoding="utf-8")

