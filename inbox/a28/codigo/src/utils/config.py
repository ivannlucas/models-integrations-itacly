"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .manufacturing_profiles import load_manufacturing_profile_registry
from .project import find_repo_root, resolve_repo_path


def load_config(
    config_path: str | Path,
    *,
    recipe_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the YAML configuration file and attach runtime metadata."""
    initial_path = Path(config_path)
    repo_root = find_repo_root(initial_path.resolve().parent if initial_path.exists() else Path.cwd())
    resolved_config_path = resolve_repo_path(initial_path, repo_root)
    if not resolved_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_config_path}")

    with resolved_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {resolved_config_path}")

    config.setdefault("project", {})
    config["project"]["repo_root"] = str(repo_root)
    config["project"]["config_path"] = str(resolved_config_path)
    return load_manufacturing_profile_registry(config, recipe_config_path=recipe_config_path)
