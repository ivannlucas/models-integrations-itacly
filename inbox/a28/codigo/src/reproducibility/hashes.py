from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: str | Path, *, repo_root: Path | None = None) -> dict[str, object]:
    resolved = Path(path)
    payload = {
        "path": resolved.as_posix() if repo_root is None else resolved.resolve().relative_to(repo_root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    return payload


def describe_existing_files(paths: Iterable[str | Path], *, repo_root: Path | None = None) -> list[dict[str, object]]:
    descriptions: list[dict[str, object]] = []
    for path in paths:
        resolved = Path(path)
        if resolved.exists():
            descriptions.append(describe_file(resolved, repo_root=repo_root))
    return descriptions

