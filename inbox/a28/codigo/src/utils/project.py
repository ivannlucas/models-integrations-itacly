"""Project-wide path and runtime helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def find_repo_root(start_path: Path | None = None) -> Path:
    """Locate the repository root from the current working path."""
    current = (start_path or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return current


def resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def to_repo_relative_path(value: str | Path, repo_root: Path) -> str:
    """Return a portable path string relative to the repository root when possible."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()

    try:
        relative_path = path.resolve().relative_to(repo_root.resolve())
        return relative_path.as_posix()
    except ValueError:
        return str(path)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_timestamp(*, compact: bool = False) -> str:
    """Return the current UTC timestamp."""
    current = datetime.now(timezone.utc)
    if compact:
        return current.strftime("%Y%m%dT%H%M%SZ")
    return current.isoformat()


def slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return slug or "run"


def make_run_id(prefix: str) -> str:
    """Build a deterministic run identifier prefix plus UTC timestamp."""
    return f"{slugify(prefix)}_{utc_timestamp(compact=True)}"
