"""Repository output cleanup stage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.utils import current_recipe_context, read_json, resolve_repo_path, to_repo_relative_path

TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}T\d{6}Z$")
TRACKED_EXTENSIONS = {".csv", ".json", ".log", ".pkl", ".yaml", ".yml", ".zip"}
TRACKED_ROOTS = [
    Path("logs"),
    Path("data/predictions"),
    Path("models/metrics"),
    Path("models/artifacts"),
]
SUMMARY_LATEST_PATTERNS = [
    "baseline_comparison_latest*.json",
    "baseline_comparison_latest*.csv",
    "neuroevolution_comparison_latest*.json",
    "neuroevolution_comparison_latest*.csv",
    "policy_simulation_latest*.json",
    "policy_simulation_latest*.csv",
    "metrics_summary*.json",
    "metrics_summary*.csv",
]
PREDICTION_LATEST_PATTERNS = ["predictions_latest*.csv"]
RECORD_PATH_KEYS = [
    "artifact_path",
    "metrics_path",
    "feature_importance_path",
    "test_predictions_path",
    "evolution_history_path",
    "training_metadata_path",
]


def _safe_repo_relative(path: Path, repo_root: Path) -> str | None:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _maybe_add_existing_path(value: Any, repo_root: Path, keep_paths: set[str]) -> None:
    if not isinstance(value, (str, Path)) or not value:
        return
    candidate = resolve_repo_path(value, repo_root)
    relative = _safe_repo_relative(candidate, repo_root)
    if relative is None or not candidate.exists() or not candidate.is_file():
        return
    keep_paths.add(relative)


def _collect_existing_paths(payload: Any, repo_root: Path, keep_paths: set[str]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_existing_paths(value, repo_root, keep_paths)
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_existing_paths(item, repo_root, keep_paths)
        return
    _maybe_add_existing_path(payload, repo_root, keep_paths)


def _collect_record_paths(record: dict[str, Any], repo_root: Path, keep_paths: set[str]) -> None:
    for key in RECORD_PATH_KEYS:
        _maybe_add_existing_path(record.get(key), repo_root, keep_paths)


def _collect_baseline_summary_paths(summary_path: Path, repo_root: Path, keep_paths: set[str]) -> None:
    payload = read_json(summary_path)
    _maybe_add_existing_path(summary_path, repo_root, keep_paths)
    _maybe_add_existing_path(summary_path.with_suffix(".csv"), repo_root, keep_paths)
    _maybe_add_existing_path(payload.get("input_dataset_path"), repo_root, keep_paths)
    _maybe_add_existing_path(payload.get("feature_export_path"), repo_root, keep_paths)
    for key in [
        "best_run",
        "best_run_global",
        "best_baseline_run",
        "baseline_reference_run",
        "best_neuroevolution_run",
    ]:
        value = payload.get(key)
        if isinstance(value, dict):
            _collect_record_paths(value, repo_root, keep_paths)
    best_by_target = payload.get("best_by_target", {})
    if isinstance(best_by_target, dict):
        for value in best_by_target.values():
            if isinstance(value, dict):
                _collect_record_paths(value, repo_root, keep_paths)
    for record_collection_key in ["all_results", "runs"]:
        records = payload.get(record_collection_key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict):
                _collect_record_paths(record, repo_root, keep_paths)


def _collect_neuro_summary_paths(summary_path: Path, repo_root: Path, keep_paths: set[str]) -> None:
    payload = read_json(summary_path)
    _maybe_add_existing_path(summary_path, repo_root, keep_paths)
    _maybe_add_existing_path(summary_path.with_suffix(".csv"), repo_root, keep_paths)
    _maybe_add_existing_path(payload.get("input_dataset_path"), repo_root, keep_paths)
    for key in ["baseline_reference_run", "best_neuroevolution_run"]:
        value = payload.get(key)
        if isinstance(value, dict):
            _collect_record_paths(value, repo_root, keep_paths)


def _collect_policy_summary_paths(summary_path: Path, repo_root: Path, keep_paths: set[str]) -> None:
    payload = read_json(summary_path)
    _maybe_add_existing_path(summary_path, repo_root, keep_paths)
    _maybe_add_existing_path(summary_path.with_suffix(".csv"), repo_root, keep_paths)
    _maybe_add_existing_path(payload.get("input_dataset_path"), repo_root, keep_paths)
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        _collect_existing_paths(artifacts, repo_root, keep_paths)
    reference_models = payload.get("reference_models", {})
    if isinstance(reference_models, dict):
        for value in reference_models.values():
            if isinstance(value, dict):
                _collect_record_paths(value, repo_root, keep_paths)


def _collect_metrics_summary_paths(summary_path: Path, repo_root: Path, keep_paths: set[str]) -> None:
    payload = read_json(summary_path)
    _maybe_add_existing_path(summary_path, repo_root, keep_paths)
    _maybe_add_existing_path(summary_path.with_suffix(".csv"), repo_root, keep_paths)
    for key in [
        "best_run",
        "best_run_global",
        "best_baseline_run",
        "baseline_reference_run",
        "best_neuroevolution_run",
    ]:
        value = payload.get(key)
        if isinstance(value, dict):
            _collect_record_paths(value, repo_root, keep_paths)
    best_by_target = payload.get("best_by_target", {})
    if isinstance(best_by_target, dict):
        for value in best_by_target.values():
            if isinstance(value, dict):
                _collect_record_paths(value, repo_root, keep_paths)


def _collect_official_artifact_paths(config: dict[str, Any], repo_root: Path, keep_paths: set[str]) -> None:
    official_dir = resolve_repo_path(config["paths"]["official_artifacts_dir"], repo_root)
    if official_dir.exists():
        for path in official_dir.rglob("*"):
            if path.is_file():
                relative = _safe_repo_relative(path, repo_root)
                if relative:
                    keep_paths.add(relative)

    manifest_path = official_dir / "official_reference_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        _collect_existing_paths(manifest, repo_root, keep_paths)


def _collect_latest_summary_paths(repo_root: Path, keep_paths: set[str]) -> None:
    summary_dir = repo_root / "models/metrics/summary"
    for pattern in SUMMARY_LATEST_PATTERNS:
        for path in summary_dir.glob(pattern):
            if path.is_file():
                keep_paths.add(path.relative_to(repo_root).as_posix())

    for path in summary_dir.glob("baseline_comparison_latest*.json"):
        _collect_baseline_summary_paths(path, repo_root, keep_paths)
    for path in summary_dir.glob("neuroevolution_comparison_latest*.json"):
        _collect_neuro_summary_paths(path, repo_root, keep_paths)
    for path in summary_dir.glob("policy_simulation_latest*.json"):
        _collect_policy_summary_paths(path, repo_root, keep_paths)
    for path in summary_dir.glob("metrics_summary*.json"):
        _collect_metrics_summary_paths(path, repo_root, keep_paths)


def _collect_latest_prediction_paths(repo_root: Path, keep_paths: set[str]) -> None:
    predictions_dir = repo_root / "data/predictions"
    for pattern in PREDICTION_LATEST_PATTERNS:
        for path in predictions_dir.glob(pattern):
            if path.is_file():
                keep_paths.add(path.relative_to(repo_root).as_posix())


def _log_group_key(path: Path) -> str:
    return TIMESTAMP_SUFFIX_RE.sub("", path.stem)


def _path_matches_scope(relative_path: Path, runtime_context: dict[str, Any]) -> bool:
    scope_token = runtime_context.get("scope_token")
    output_suffix = runtime_context.get("output_suffix")
    scope_type = runtime_context.get("scope_type")
    if not scope_token or scope_type == "all_scopes":
        return True

    relative_str = relative_path.as_posix()
    file_name = relative_path.name
    return any(
        token
        for token in [scope_token, output_suffix]
        if token and (token in file_name or token in relative_str)
    )


def _keep_latest_log_files(repo_root: Path, runtime_context: dict[str, Any], keep_paths: set[str]) -> None:
    logs_dir = repo_root / "logs"
    if not logs_dir.exists():
        return

    latest_by_group: dict[str, Path] = {}
    for path in logs_dir.glob("*.log"):
        if not _path_matches_scope(path.relative_to(repo_root), runtime_context):
            continue
        group = _log_group_key(path)
        incumbent = latest_by_group.get(group)
        if incumbent is None or path.stat().st_mtime > incumbent.stat().st_mtime:
            latest_by_group[group] = path

    for path in latest_by_group.values():
        relative = _safe_repo_relative(path, repo_root)
        if relative:
            keep_paths.add(relative)


def _is_repo_output_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TRACKED_EXTENSIONS


def _iter_cleanup_candidates(repo_root: Path, runtime_context: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for root in TRACKED_ROOTS:
        resolved_root = repo_root / root
        if not resolved_root.exists():
            continue
        for path in resolved_root.rglob("*"):
            if not _is_repo_output_file(path):
                continue
            relative = path.relative_to(repo_root)
            if relative.name == ".gitkeep":
                continue
            if not _path_matches_scope(relative, runtime_context):
                continue
            candidates.append(path)
    return sorted(candidates)


def _build_keep_set(config: dict[str, Any], repo_root: Path, runtime_context: dict[str, Any]) -> set[str]:
    keep_paths = {
        "logs/.gitkeep",
        "data/predictions/.gitkeep",
        "models/metrics/.gitkeep",
        "models/artifacts/.gitkeep",
    }
    _collect_latest_summary_paths(repo_root, keep_paths)
    _collect_latest_prediction_paths(repo_root, keep_paths)
    _collect_official_artifact_paths(config, repo_root, keep_paths)
    _keep_latest_log_files(repo_root, runtime_context, keep_paths)
    return keep_paths


def _remove_empty_directories(repo_root: Path) -> list[str]:
    removed: list[str] = []
    for root in [repo_root / "models/metrics", repo_root / "models/artifacts", repo_root / "data/predictions", repo_root / "logs"]:
        if not root.exists():
            continue
        directories = sorted([path for path in root.rglob("*") if path.is_dir()], reverse=True)
        for directory in directories:
            if any(directory.iterdir()):
                continue
            directory.rmdir()
            relative = _safe_repo_relative(directory, repo_root)
            if relative:
                removed.append(relative)
    return removed


def run_cleanup(config: dict[str, Any], logger) -> dict[str, Any]:
    """Remove obsolete generated outputs while preserving the active state."""
    repo_root = Path(config["project"]["repo_root"]).resolve()
    runtime_context = current_recipe_context(config)
    cleanup_runtime = config.get("runtime", {}).get("cleanup", {})
    dry_run = bool(cleanup_runtime.get("dry_run", False))

    selection_mode = runtime_context.get("selection_mode")
    cleanup_scope = "all_scopes" if selection_mode in {None, "unresolved", "cleanup_all"} else runtime_context.get("scope_token")
    keep_paths = _build_keep_set(config, repo_root, runtime_context)
    candidates = _iter_cleanup_candidates(repo_root, runtime_context)

    deleted_files: list[str] = []
    kept_files: list[str] = []
    for candidate in candidates:
        relative = candidate.relative_to(repo_root).as_posix()
        if relative in keep_paths:
            kept_files.append(relative)
            continue
        deleted_files.append(relative)
        if not dry_run:
            candidate.unlink(missing_ok=True)

    removed_directories: list[str] = []
    if not dry_run:
        removed_directories = _remove_empty_directories(repo_root)

    logger.info(
        "Cleanup completed scope=%s dry_run=%s candidates=%s deleted=%s kept=%s removed_directories=%s",
        cleanup_scope,
        dry_run,
        len(candidates),
        len(deleted_files),
        len(kept_files),
        len(removed_directories),
    )
    for relative in deleted_files:
        logger.info("Cleanup %s %s", "would_remove" if dry_run else "removed", relative)

    return {
        "cleanup_scope": cleanup_scope,
        "dry_run": dry_run,
        "kept_file_count": len(kept_files),
        "deleted_file_count": len(deleted_files),
        "removed_directory_count": len(removed_directories),
        "kept_files": kept_files,
        "deleted_files": deleted_files,
        "removed_directories": removed_directories,
        "retention_policy": {
            "preserved_summary_patterns": SUMMARY_LATEST_PATTERNS,
            "preserved_prediction_patterns": PREDICTION_LATEST_PATTERNS,
            "preserved_official_directory": to_repo_relative_path(
                resolve_repo_path(config["paths"]["official_artifacts_dir"], repo_root),
                repo_root,
            ),
            "log_retention": "latest_log_per_stage_scope",
        },
    }
