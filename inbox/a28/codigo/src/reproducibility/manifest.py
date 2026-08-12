from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.reproducibility.hashes import describe_existing_files, sha256_file
from src.reproducibility.runtime import official_paths, runtime_environment
from src.utils import ensure_directory, read_json, write_json


def _git_commit(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _collect_tree_files(path: Path, *, suffixes: tuple[str, ...] | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if suffixes:
        files = [candidate for candidate in files if candidate.suffix.lower() in suffixes]
    return files


def _file_entries(paths: list[Path], repo_root: Path) -> list[dict[str, object]]:
    return describe_existing_files(paths, repo_root=repo_root)


def build_reproducibility_manifest(
    config: dict[str, Any],
    *,
    commands_executed: list[str],
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    output_path: str | Path | None = None,
    manifest_scope: str = "end_to_end",
) -> dict[str, Any]:
    repo_root = Path(config["project"]["repo_root"])
    paths = official_paths(config)
    summary_dir = paths["summary_dir"]
    artifacts_dir = paths["artifacts_dir"]
    predictions_dir = paths["predictions_dir"]
    splits_dir = paths["splits_dir"]

    raw_files = _collect_tree_files(repo_root / "data" / "raw" / "external")
    processed_files = _collect_tree_files(repo_root / "data" / "processed")
    split_files = _collect_tree_files(splits_dir)
    model_artifacts = _collect_tree_files(artifacts_dir, suffixes=(".pkl", ".json"))
    metrics_files = _collect_tree_files(summary_dir, suffixes=(".json", ".csv"))
    prediction_files = _collect_tree_files(predictions_dir, suffixes=(".csv", ".json"))

    requirements_path = repo_root / "requirements.txt"
    config_path = Path(config["project"]["config_path"])
    target_path = Path(output_path) if output_path else paths["repro_manifest"]
    generated_outputs = {
        "raw_manifest": paths["raw_manifest"].relative_to(repo_root).as_posix(),
        "processed_root": "data/processed",
        "splits_root": splits_dir.relative_to(repo_root).as_posix(),
        "predictions_root": predictions_dir.relative_to(repo_root).as_posix(),
        "artifacts_root": artifacts_dir.relative_to(repo_root).as_posix(),
        "metrics_root": summary_dir.relative_to(repo_root).as_posix(),
        "reproducibility_manifest": target_path.relative_to(repo_root).as_posix()
        if target_path.is_relative_to(repo_root)
        else str(target_path),
    }

    manifest = {
        "scope": "mixed_context",
        "manifest_scope": manifest_scope,
        "reference_date": config.get("official_release", {}).get("reference_date"),
        "official_run": config.get("runtime", {}).get("official_run", {}),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(repo_root),
        "environment": runtime_environment(),
        "requirements_hash": sha256_file(requirements_path) if requirements_path.exists() else None,
        "config_hash": sha256_file(config_path) if config_path.exists() else None,
        "configuration": {
            "path": config_path.relative_to(repo_root).as_posix() if config_path.exists() else str(config_path),
            "sha256": sha256_file(config_path) if config_path.exists() else None,
        },
        "commands_executed": commands_executed,
        "generated_outputs": generated_outputs,
        "seeds": {
            "project_seed": config.get("project", {}).get("seed"),
            "training_seed": config.get("training", {}).get("random_seed"),
            "synthetic_seed": config.get("synthetic_data", {}).get("simulation_seed"),
            "neuroevolution_seed": config.get("neuroevolution", {}).get("random_seed"),
        },
        "raw_files": _file_entries(raw_files, repo_root),
        "processed_files": _file_entries(processed_files, repo_root),
        "split_files": _file_entries(split_files, repo_root),
        "model_artifacts": _file_entries(model_artifacts, repo_root),
        "metrics_files": _file_entries(metrics_files, repo_root),
        "prediction_files": _file_entries(prediction_files, repo_root),
        "warnings": warnings or [],
        "limitations": limitations or [],
    }

    write_json(target_path, manifest)
    return manifest


def verify_reproducibility_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    payload = read_json(manifest_file)
    repo_root = manifest_file.resolve().parent

    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0

    for section in ["raw_files", "processed_files", "split_files", "model_artifacts", "metrics_files", "prediction_files"]:
        for entry in payload.get(section, []):
            candidate = repo_root / entry["path"]
            if not candidate.exists():
                missing.append(entry["path"])
                continue
            if sha256_file(candidate) != entry["sha256"]:
                mismatched.append(entry["path"])
            checked += 1

    return {
        "valid": not missing and not mismatched,
        "checked_files": checked,
        "missing_files": missing,
        "hash_mismatches": mismatched,
        "scope": payload.get("scope"),
        "commit": payload.get("commit"),
    }
