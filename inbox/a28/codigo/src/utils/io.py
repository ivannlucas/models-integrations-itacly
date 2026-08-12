"""Input and output helpers for tabular data and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_tabular(path: str | Path, *, file_type: str | None = None) -> pd.DataFrame:
    """Read a supported tabular file into a DataFrame."""
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_path}")

    file_kind = (file_type or resolved_path.suffix).lower().lstrip(".")
    if file_kind == "csv":
        return pd.read_csv(resolved_path)
    if file_kind == "parquet":
        return pd.read_parquet(resolved_path)
    if file_kind in {"xlsx", "xls"}:
        return pd.read_excel(resolved_path)
    raise ValueError(f"Unsupported input file type: {file_kind}")


def write_json(path: str | Path, payload: Any) -> Path:
    """Persist a JSON payload with a stable UTF-8 encoding."""
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return resolved_path


def read_json(path: str | Path) -> Any:
    """Load a JSON payload from disk."""
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_latest_file(directory: str | Path, pattern: str) -> Path | None:
    """Return the most recently modified file matching a glob pattern."""
    resolved_dir = Path(directory)
    if not resolved_dir.exists():
        return None
    candidates = [candidate for candidate in resolved_dir.glob(pattern) if candidate.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
