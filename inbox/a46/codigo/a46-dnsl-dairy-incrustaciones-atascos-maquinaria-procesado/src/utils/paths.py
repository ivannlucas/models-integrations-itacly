from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

REPO_TOP_LEVEL_MARKERS = {
    "data",
    "models",
    "config",
    "src",
    "scripts",
    "notebooks",
    "README.md",
    "requirements.txt",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_str(path_like: str | Path) -> str:
    return str(path_like).strip()


def _is_windows_absolute(path_str: str) -> bool:
    return PureWindowsPath(path_str).is_absolute()


def _is_posix_absolute(path_str: str) -> bool:
    return PurePosixPath(path_str).is_absolute()


def is_absolute_path_like(path_like: str | Path) -> bool:
    path_str = _as_str(path_like)
    if not path_str:
        return False
    return Path(path_str).is_absolute() or _is_windows_absolute(path_str) or _is_posix_absolute(path_str)


def _normalize_relative_candidate(path_str: str) -> str:
    cleaned = path_str.strip()
    if cleaned.startswith("./") or cleaned.startswith(".\\"):
        cleaned = cleaned[2:]
    return cleaned.replace("\\", "/")


def looks_like_repo_relative_path(path_like: str | Path) -> bool:
    path_str = _normalize_relative_candidate(_as_str(path_like))
    if not path_str:
        return False
    first = path_str.split("/", 1)[0]
    return first in REPO_TOP_LEVEL_MARKERS


def _relative_from_known_markers(path_str: str) -> Path | None:
    normalized = path_str.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    lowered = [part.lower() for part in parts]
    for idx, part in enumerate(lowered):
        if part in {marker.lower() for marker in REPO_TOP_LEVEL_MARKERS}:
            return Path(*parts[idx:])
    return None


def to_repo_relative_path(path_like: str | Path) -> str:
    if isinstance(path_like, Path):
        path_str = str(path_like)
        if path_like.is_absolute():
            try:
                return path_like.relative_to(repo_root()).as_posix()
            except ValueError:
                return Path(os.path.relpath(path_like, start=repo_root())).as_posix()
        path_str = _normalize_relative_candidate(path_str)
        return Path(path_str).as_posix()

    path_str = _as_str(path_like)
    if not path_str:
        return path_str

    if Path(path_str).is_absolute():
        abs_path = Path(path_str)
        try:
            return abs_path.relative_to(repo_root()).as_posix()
        except ValueError:
            return Path(os.path.relpath(abs_path, start=repo_root())).as_posix()

    if _is_windows_absolute(path_str) or _is_posix_absolute(path_str):
        rel = _relative_from_known_markers(path_str)
        if rel is not None:
            return rel.as_posix()
        return Path(_normalize_relative_candidate(path_str)).as_posix()

    return Path(_normalize_relative_candidate(path_str)).as_posix()


def resolve_repo_path(path_like: str | Path) -> Path:
    if isinstance(path_like, Path):
        return path_like if path_like.is_absolute() else (repo_root() / path_like).resolve()

    path_str = _as_str(path_like)
    if not path_str:
        return repo_root()
    if Path(path_str).is_absolute():
        return Path(path_str)
    if _is_windows_absolute(path_str) or _is_posix_absolute(path_str) or looks_like_repo_relative_path(path_str):
        return (repo_root() / to_repo_relative_path(path_str)).resolve()
    return (repo_root() / Path(_normalize_relative_candidate(path_str))).resolve()


def resolve_saved_path(path_like: str | Path) -> Path:
    return resolve_repo_path(path_like)


def relativize_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        return {key: relativize_payload(value) for key, value in payload.items()}
    if isinstance(payload, Path):
        return to_repo_relative_path(payload)
    if isinstance(payload, tuple):
        return [relativize_payload(value) for value in payload]
    if isinstance(payload, list):
        return [relativize_payload(value) for value in payload]
    if isinstance(payload, set):
        return [relativize_payload(value) for value in sorted(payload, key=str)]
    if isinstance(payload, str):
        if is_absolute_path_like(payload) or looks_like_repo_relative_path(payload):
            return to_repo_relative_path(payload)
        return payload
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return [relativize_payload(value) for value in payload]
    return payload
