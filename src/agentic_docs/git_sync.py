"""Repository sync utilities for local-first ingestion."""

from __future__ import annotations

import subprocess
from pathlib import Path


def sync_repository(repo_url: str, local_path: Path) -> str:
    """Clone or update a git repository and return the current commit hash."""

    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        _run_git(["clone", repo_url, str(local_path)])
    else:
        if not (local_path / ".git").exists():
            msg = f"Path exists but is not a git repository: {local_path}"
            raise ValueError(msg)
        _run_git(["-C", str(local_path), "pull", "--ff-only"])
    return current_commit_hash(local_path)


def current_commit_hash(repo_path: Path) -> str | None:
    """Return the current git commit hash for a repository path, if available."""

    try:
        return _run_git(["-C", str(repo_path), "rev-parse", "HEAD"]).strip()
    except RuntimeError:
        return None


def git_head_commit(repo_path: Path) -> str | None:
    """Return the current HEAD commit for a git worktree."""

    try:
        return _run_git(["-C", str(repo_path), "rev-parse", "HEAD"]).strip()
    except RuntimeError:
        return None


def git_working_tree_status(repo_path: Path) -> dict[str, object]:
    """Return whether the git worktree is clean plus the raw porcelain lines."""

    try:
        output = _run_git(["-C", str(repo_path), "status", "--porcelain"])
    except RuntimeError:
        return {"is_git_repo": False, "clean": None, "status_lines": []}

    status_lines = [line for line in output.splitlines() if line.strip()]
    return {
        "is_git_repo": True,
        "clean": not status_lines,
        "status_lines": status_lines,
    }


def _run_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        msg = f"git command failed: {' '.join(args)}\n{stderr}"
        raise RuntimeError(msg)
    return completed.stdout
