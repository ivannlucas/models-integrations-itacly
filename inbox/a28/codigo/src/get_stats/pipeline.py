"""Metrics aggregation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import (
    build_best_by_target,
    canonical_target_column,
    current_recipe_context,
    ensure_directory,
    ensure_runtime_context_resolved,
    read_json,
    resolve_reference_record,
    resolve_repo_path,
    select_best_record,
    select_preferred_record,
    selection_policy_description,
    to_repo_relative_path,
    write_json,
)


def _resolve_existing_path_obj(path_value: Any, repo_root: Path) -> Path | None:
    if not path_value:
        return None
    if isinstance(path_value, float) and pd.isna(path_value):
        return None
    if not isinstance(path_value, (str, Path)):
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = resolve_repo_path(path, repo_root)
    return path if path.exists() else None


def _resolve_existing_path(path_value: Any, repo_root: Path) -> str | None:
    path = _resolve_existing_path_obj(path_value, repo_root)
    return to_repo_relative_path(path, repo_root) if path else None


def _resolve_official_release_context(config: dict[str, Any], repo_root: Path) -> tuple[str | None, str | None]:
    official_cfg = config.get("official_release", {}).get("predictive_reference_model", {})
    official_release_pickle = _resolve_existing_path(official_cfg.get("release_pickle_path"), repo_root)
    manifest_path = _resolve_existing_path_obj(official_cfg.get("release_manifest_path"), repo_root)
    official_run_id = None
    if manifest_path:
        manifest_payload = read_json(manifest_path)
        official_run_id = manifest_payload.get("predictive_reference_model", {}).get("run_id")
    return official_run_id, official_release_pickle


def _resolve_declared_artifact_candidates(record: dict[str, Any], repo_root: Path) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    direct_artifact_path = record.get("artifact_path")
    if isinstance(direct_artifact_path, (str, Path)) and direct_artifact_path:
        candidates.append(("declared_artifact_path", to_repo_relative_path(resolve_repo_path(direct_artifact_path, repo_root), repo_root)))

    run_id = record.get("run_id")
    if run_id:
        candidates.append(("run_id_artifact_path", f"models/artifacts/{run_id}.pkl"))

    extra_artifact_paths = dict(record.get("extra_artifact_paths", {}))
    for artifact_key, artifact_path in extra_artifact_paths.items():
        if "artifact" not in str(artifact_key):
            continue
        if isinstance(artifact_path, (str, Path)) and artifact_path:
            candidates.append((f"extra_artifact_paths.{artifact_key}", to_repo_relative_path(resolve_repo_path(artifact_path, repo_root), repo_root)))

    deduped_candidates: list[tuple[str, str]] = []
    seen_paths: set[str] = set()
    for source_name, candidate_path in candidates:
        if candidate_path in seen_paths:
            continue
        seen_paths.add(candidate_path)
        deduped_candidates.append((source_name, candidate_path))
    return deduped_candidates


def _resolve_active_artifact_context(
    record: dict[str, Any],
    *,
    repo_root: Path,
    official_run_id: str | None,
    official_release_pickle: str | None,
) -> dict[str, Any]:
    for source_name, candidate_path in _resolve_declared_artifact_candidates(record, repo_root):
        resolved_path = _resolve_existing_path(candidate_path, repo_root)
        if resolved_path:
            return {
                "artifact_path": resolved_path,
                "artifact_expected_path": candidate_path,
                "artifact_status": "resolved_active",
                "artifact_resolution_source": source_name,
            }

    if official_release_pickle and record.get("run_id") == official_run_id:
        resolved_official_path = _resolve_existing_path(official_release_pickle, repo_root)
        if resolved_official_path:
            return {
                "artifact_path": resolved_official_path,
                "artifact_expected_path": resolved_official_path,
                "artifact_status": "resolved_official_release",
                "artifact_resolution_source": "official_release",
            }

    candidates = _resolve_declared_artifact_candidates(record, repo_root)
    if candidates:
        _, expected_path = candidates[0]
        return {
            "artifact_path": None,
            "artifact_expected_path": expected_path,
            "artifact_status": "missing_on_disk",
            "artifact_resolution_source": candidates[0][0],
        }

    return {
        "artifact_path": None,
        "artifact_expected_path": None,
        "artifact_status": "undeclared",
        "artifact_resolution_source": "unavailable",
    }


def _normalise_summary_run_record(
    record: dict[str, Any],
    *,
    repo_root: Path,
    official_run_id: str | None,
    official_release_pickle: str | None,
) -> dict[str, Any]:
    normalised = dict(record)
    runtime_context = dict(normalised.get("runtime_context", {}))
    if "recipe_profile" in normalised and "recipe_profile" not in runtime_context:
        runtime_context["recipe_profile"] = normalised.get("recipe_profile")
    if "manufacturing_context_profile" in normalised and "manufacturing_context_profile" not in runtime_context:
        runtime_context["manufacturing_context_profile"] = normalised.get("manufacturing_context_profile")
    normalised["runtime_context"] = runtime_context
    artifact_context = _resolve_active_artifact_context(
        record,
        repo_root=repo_root,
        official_run_id=official_run_id,
        official_release_pickle=official_release_pickle,
    )
    normalised.update(artifact_context)
    for path_key in [
        "metrics_path",
        "feature_importance_path",
        "test_predictions_path",
        "evolution_history_path",
        "training_metadata_path",
    ]:
        normalised[path_key] = _resolve_existing_path(normalised.get(path_key), repo_root)
    return normalised


def _flatten_metrics_record(
    record: dict[str, Any],
    *,
    repo_root: Path,
    official_run_id: str | None,
    official_release_pickle: str | None,
) -> dict[str, Any]:
    artifact_context = _resolve_active_artifact_context(
        record,
        repo_root=repo_root,
        official_run_id=official_run_id,
        official_release_pickle=official_release_pickle,
    )
    flat = {
        "run_id": record.get("run_id"),
        "comparison_run_id": record.get("comparison_run_id"),
        "trained_at_utc": record.get("trained_at_utc"),
        "target_role": record.get("target_role"),
        "model_family": record.get("model_family"),
        "model_name": record.get("model_name"),
        "run_category": record.get("run_category", "baseline"),
        "is_neuroevolution": record.get("is_neuroevolution", False),
        "feature_set": record.get("feature_set"),
        "feature_count": record.get("feature_count"),
        "target_column": record.get("target_column"),
        "recipe_profile": (record.get("runtime_context") or {}).get("recipe_profile"),
        "manufacturing_context_profile": (record.get("runtime_context") or {}).get("manufacturing_context_profile"),
        "selection_mode": (record.get("runtime_context") or {}).get("selection_mode"),
        "scope_token": (record.get("runtime_context") or {}).get("scope_token"),
        "mode_resolution": (record.get("runtime_context") or {}).get("mode_resolution"),
        **artifact_context,
        "feature_importance_path": _resolve_existing_path(record.get("feature_importance_path"), repo_root),
        "test_predictions_path": _resolve_existing_path(record.get("test_predictions_path"), repo_root),
        "runtime_context": dict(record.get("runtime_context", {})),
    }
    for split_name, split_metrics in record.get("metrics", {}).items():
        for metric_name, metric_value in split_metrics.items():
            flat[f"{split_name}_{metric_name}"] = metric_value
    for split_name, split_rows in record.get("split_rows", {}).items():
        flat[f"{split_name}_rows"] = split_rows
    for artifact_name, artifact_path in record.get("extra_artifact_paths", {}).items():
        flat[artifact_name] = _resolve_existing_path(artifact_path, repo_root)
    return flat


def _best_rows_by_target(summary_df: pd.DataFrame, primary_metric: str) -> list[dict[str, Any]]:
    return build_best_by_target(summary_df, primary_metric=primary_metric)


def _best_run_for_category(summary_df: pd.DataFrame, primary_metric: str, run_category: str) -> dict[str, Any] | None:
    return select_best_record(summary_df, primary_metric=primary_metric, run_category=run_category)


def _resolve_official_baseline_run(
    summary_df: pd.DataFrame,
    config: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any] | None:
    if summary_df.empty:
        return None

    baseline_df = summary_df[summary_df["run_category"] == "baseline"]
    if baseline_df.empty:
        return None

    reference_cfg = dict(config.get("neuroevolution", {}).get("baseline_reference", {}))
    if not reference_cfg:
        procurement_cfg = config.get("procurement_problem_definition", {})
        reference_cfg = {
            "target_column": procurement_cfg.get("target_column"),
            "feature_set_name": config.get("neuroevolution", {}).get("feature_set_name", "extended"),
            "model_family": "gradient_boosting",
        }

    filters = {
        "target_column": reference_cfg.get("target_column"),
        "feature_set": reference_cfg.get("feature_set_name", reference_cfg.get("feature_set")),
        "model_family": reference_cfg.get("model_family"),
    }
    return resolve_reference_record(
        baseline_df,
        primary_metric=primary_metric,
        canonical_target=canonical_target_column(config),
        run_category="baseline",
        filters=filters,
    )


def run_get_stats(config: dict[str, Any], logger) -> dict[str, Any]:
    """Aggregate metrics files into comparable summaries."""
    config = ensure_runtime_context_resolved(config)
    repo_root = Path(config["project"]["repo_root"])
    paths_cfg = config["paths"]
    stats_cfg = config["get_stats"]

    metrics_dir = resolve_repo_path(paths_cfg["model_metrics_dir"], repo_root)
    summary_dir = ensure_directory(resolve_repo_path(paths_cfg["stats_dir"], repo_root))
    training_cfg = config.get("training", {})
    runtime_recipe_context = current_recipe_context(config)
    selected_recipe = runtime_recipe_context.get("recipe_profile")
    selected_scope_token = runtime_recipe_context.get("scope_token")
    selected_scope_type = runtime_recipe_context.get("scope_type")
    logger.info(
        "Get stats runtime selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
        runtime_recipe_context.get("selection_mode"),
        runtime_recipe_context.get("mode_resolution"),
        selected_scope_token,
        selected_recipe,
    )
    metric_files = sorted(
        [
            file_path
            for file_path in metrics_dir.glob("*.json")
            if file_path.is_file()
        ]
    )
    if selected_scope_token:
        official_run_id, official_release_pickle = None, None
    else:
        official_run_id, official_release_pickle = _resolve_official_release_context(config, repo_root)
    if metric_files:
        records = [read_json(metric_file) for metric_file in metric_files]
        if selected_scope_token:
            records = [
                record
                for record in records
                if (record.get("runtime_context") or {}).get("scope_token") == selected_scope_token
                or (
                    selected_scope_type == "recipe"
                    and not (record.get("runtime_context") or {}).get("scope_token")
                    and (record.get("runtime_context") or {}).get("recipe_profile") == selected_recipe
                )
            ]
        elif selected_recipe:
            records = [
                record
                for record in records
                if (record.get("runtime_context") or {}).get("recipe_profile") == selected_recipe
            ]
        else:
            records = [
                record
                for record in records
                if not (record.get("runtime_context") or {}).get("recipe_profile")
            ]
        flat_records = [
            _flatten_metrics_record(
                record,
                repo_root=repo_root,
                official_run_id=official_run_id,
                official_release_pickle=official_release_pickle,
            )
            for record in records
        ]
    else:
        baseline_summary_name = training_cfg.get("comparison_summary_json_name", "baseline_comparison_latest.json")
        baseline_summary_path = summary_dir / baseline_summary_name
        if not baseline_summary_path.exists():
            raise FileNotFoundError(
                f"No metrics JSON files found in {metrics_dir} and no active comparison summary found at {baseline_summary_path}."
            )
        baseline_summary = read_json(baseline_summary_path)
        summary_runs = baseline_summary.get("runs", [])
        if not isinstance(summary_runs, list) or not summary_runs:
            raise FileNotFoundError(
                f"No metrics JSON files found in {metrics_dir} and {baseline_summary_path} does not expose active runs."
            )
        if selected_scope_token:
            summary_runs = [
                record
                for record in summary_runs
                if (record.get("runtime_context") or {}).get("scope_token") == selected_scope_token
                or (
                    selected_scope_type == "recipe"
                    and not (record.get("runtime_context") or {}).get("scope_token")
                    and (record.get("runtime_context") or {}).get("recipe_profile") == selected_recipe
                )
            ]
        elif selected_recipe:
            summary_runs = [
                record
                for record in summary_runs
                if (record.get("runtime_context") or {}).get("recipe_profile") == selected_recipe
            ]
        else:
            summary_runs = [
                record
                for record in summary_runs
                if not (record.get("runtime_context") or {}).get("recipe_profile")
            ]
        flat_records = [
            _normalise_summary_run_record(
                record,
                repo_root=repo_root,
                official_run_id=official_run_id,
                official_release_pickle=official_release_pickle,
            )
            for record in summary_runs
        ]
    if not flat_records:
        raise FileNotFoundError(
            f"No metrics were found for runtime scope='{selected_scope_token}' in {metrics_dir}."
            if selected_scope_token
            else f"No metrics were found for recipe_profile='{selected_recipe}' in {metrics_dir}."
            if selected_recipe
            else f"No metrics were found in {metrics_dir}."
        )
    summary_df = pd.DataFrame(flat_records).sort_values("trained_at_utc", ascending=False)
    summary_df = summary_df.astype(object).where(pd.notna(summary_df), None)

    primary_metric = stats_cfg.get("primary_metric", "test_rmse")
    canonical_target = canonical_target_column(config)
    selection_policy = selection_policy_description(primary_metric=primary_metric, canonical_target=canonical_target)
    best_run = select_preferred_record(
        summary_df,
        primary_metric=primary_metric,
        canonical_target=canonical_target,
    )
    best_run_global = select_best_record(summary_df, primary_metric=primary_metric)
    best_baseline_run = select_preferred_record(
        summary_df,
        primary_metric=primary_metric,
        canonical_target=canonical_target,
        run_category="baseline",
    )
    baseline_reference_run = _resolve_official_baseline_run(summary_df, config, primary_metric)
    best_neuroevolution_run = select_preferred_record(
        summary_df,
        primary_metric=primary_metric,
        canonical_target=canonical_target,
        run_category="neuroevolution",
    )
    best_by_target = _best_rows_by_target(summary_df, primary_metric)
    policy_simulation_summary = None
    policy_summary_path = None
    policy_simulation_cfg = config.get("policy_simulation", {})
    if policy_simulation_cfg.get("enabled", False):
        policy_summary_path = summary_dir / policy_simulation_cfg.get("summary_json_name", "policy_simulation_latest.json")
        if policy_summary_path.exists():
            policy_simulation_summary = read_json(policy_summary_path)

    summary_csv_path = summary_dir / stats_cfg.get("summary_csv_name", "metrics_summary.csv")
    summary_json_path = summary_dir / stats_cfg.get("summary_json_name", "metrics_summary.json")
    procurement_definition = dict(config.get("procurement_problem_definition", {}))
    summary_df.to_csv(summary_csv_path, index=False)
    write_json(
        summary_json_path,
        {
            "primary_metric": primary_metric,
            "procurement_problem_definition": procurement_definition,
            "recipe_context": runtime_recipe_context,
            "runtime_context": runtime_recipe_context,
            "selection_policy": selection_policy,
            "total_runs": int(len(summary_df)),
            "best_run": best_run,
            "best_run_global": best_run_global,
            "best_baseline_run": best_baseline_run,
            "baseline_reference_run": baseline_reference_run,
            "best_neuroevolution_run": best_neuroevolution_run,
            "best_by_target": best_by_target,
            "policy_simulation_summary_path": to_repo_relative_path(policy_summary_path, repo_root) if policy_simulation_summary else None,
            "best_policy_overall": (
                policy_simulation_summary.get("best_policy_overall") if policy_simulation_summary else None
            ),
            "best_classical_policy": (
                policy_simulation_summary.get("best_classical_policy") if policy_simulation_summary else None
            ),
            "best_neuroevolution_policy": (
                policy_simulation_summary.get("best_neuroevolution_policy") if policy_simulation_summary else None
            ),
            "policy_kpi_assessment": (
                policy_simulation_summary.get("kpi_assessment") if policy_simulation_summary else None
            ),
            "policy_functional_objective_assessment": (
                policy_simulation_summary.get("functional_objective_assessment") if policy_simulation_summary else None
            ),
            "runs": summary_df.to_dict(orient="records"),
        },
    )
    logger.info("Saved metrics summary CSV to %s", summary_csv_path)
    logger.info("Saved metrics summary JSON to %s", summary_json_path)

    return {
        "summary_csv_path": to_repo_relative_path(summary_csv_path, repo_root),
        "summary_json_path": to_repo_relative_path(summary_json_path, repo_root),
        "procurement_problem_definition": procurement_definition,
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
        "selection_policy": selection_policy,
        "total_runs": int(len(summary_df)),
        "best_run": best_run,
        "best_run_global": best_run_global,
        "best_baseline_run": best_baseline_run,
        "baseline_reference_run": baseline_reference_run,
        "best_neuroevolution_run": best_neuroevolution_run,
        "best_by_target": best_by_target,
        "policy_simulation_summary_path": to_repo_relative_path(policy_summary_path, repo_root) if policy_simulation_summary else None,
        "best_policy_overall": policy_simulation_summary.get("best_policy_overall") if policy_simulation_summary else None,
        "best_classical_policy": (
            policy_simulation_summary.get("best_classical_policy") if policy_simulation_summary else None
        ),
        "best_neuroevolution_policy": (
            policy_simulation_summary.get("best_neuroevolution_policy") if policy_simulation_summary else None
        ),
        "policy_kpi_assessment": policy_simulation_summary.get("kpi_assessment") if policy_simulation_summary else None,
        "policy_functional_objective_assessment": (
            policy_simulation_summary.get("functional_objective_assessment") if policy_simulation_summary else None
        ),
    }
