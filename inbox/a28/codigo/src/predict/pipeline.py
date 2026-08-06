"""Batch prediction stage."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_processing.pipeline import prepare_modeling_frame
from src.utils import (
    current_recipe_context,
    ensure_runtime_context_resolved,
    ensure_directory,
    filter_frame_to_recipe,
    read_json,
    read_tabular,
    resolve_repo_path,
    to_repo_relative_path,
    utc_timestamp,
)


def _portable_path(path: str | Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    return to_repo_relative_path(path, repo_root)


def _load_artifact(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


def _resolve_official_release_artifact_path(config: dict[str, Any], repo_root: Path) -> Path | None:
    official_release_cfg = config.get("official_release", {}).get("predictive_reference_model", {})
    release_pickle_path = official_release_cfg.get("release_pickle_path")
    if not release_pickle_path:
        return None
    resolved_path = resolve_repo_path(release_pickle_path, repo_root)
    if resolved_path.exists():
        return resolved_path
    return None


def _resolve_scoped_artifact_path(config: dict[str, Any], repo_root: Path, logger) -> tuple[Path, dict[str, Any]]:
    paths_cfg = config["paths"]
    training_cfg = config.get("training", {})
    runtime_context = current_recipe_context(config)
    stats_dir = resolve_repo_path(paths_cfg["stats_dir"], repo_root)
    summary_name = training_cfg.get("comparison_summary_json_name", "baseline_comparison_latest.json")
    summary_path = stats_dir / summary_name
    if not summary_path.exists():
        raise FileNotFoundError(
            "Prediction requires a runtime-scoped baseline summary but none was found. "
            f"Expected: {summary_path}. Run the training stage for scope={runtime_context.get('scope_token')} first."
        )

    summary_payload = read_json(summary_path)
    best_baseline_run = summary_payload.get("best_baseline_run")
    if not isinstance(best_baseline_run, dict) or not best_baseline_run.get("artifact_path"):
        raise FileNotFoundError(
            "Prediction requires an active baseline artifact in the runtime-scoped summary. "
            f"Summary checked: {summary_path}"
        )

    artifact_path = Path(best_baseline_run["artifact_path"])
    if not artifact_path.is_absolute():
        artifact_path = resolve_repo_path(artifact_path, repo_root)
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Prediction requires the runtime-scoped serialized artifact referenced by the baseline summary, "
            f"but it was not found: {artifact_path}"
        )

    logger.info(
        "Resolved prediction artifact from runtime-scoped baseline summary scope_token=%s summary=%s artifact=%s",
        runtime_context.get("scope_token"),
        summary_path,
        artifact_path,
    )
    return artifact_path, {
        "resolution": "runtime_scoped_baseline_summary",
        "summary_path": _portable_path(summary_path, repo_root),
        "summary_runtime_context": summary_payload.get("recipe_context") or summary_payload.get("runtime_context"),
    }


def _resolve_default_artifact_path(config: dict[str, Any], repo_root: Path, logger) -> tuple[Path, dict[str, Any]]:
    prediction_cfg = config["prediction"]
    runtime_context = current_recipe_context(config)
    scope_token = runtime_context.get("scope_token")

    artifact_selection = str(prediction_cfg.get("artifact_selection", "official_baseline_latest")).strip().lower()
    if scope_token:
        return _resolve_scoped_artifact_path(config, repo_root, logger)

    if artifact_selection == "official_baseline_latest":
        official_release_path = _resolve_official_release_artifact_path(config, repo_root)
        if official_release_path is not None:
            logger.info(
                "Resolved prediction artifact from official release pickle: %s",
                official_release_path,
            )
            return official_release_path, {
                "resolution": "official_release_pickle",
                "summary_path": None,
                "summary_runtime_context": None,
            }

    raise FileNotFoundError(
        "Prediction runtime scope is unresolved and no explicit runtime-scoped artifact could be selected. "
        "Use --recipe-profile, --mixed-context, or define manufacturing_profiles.default_recipe_profile."
    )


def _candidate_input_paths(config: dict[str, Any], artifact: dict[str, Any], repo_root: Path) -> list[Path]:
    paths_cfg = config["paths"]
    prediction_cfg = config["prediction"]
    feature_cfg = config.get("feature_selection", {})

    raw_candidates = [
        prediction_cfg.get("input_path"),
        artifact.get("input_dataset_path"),
        feature_cfg.get("prepared_dataset_path"),
        paths_cfg.get("processed_dataset_path"),
    ]

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.is_absolute():
            path = resolve_repo_path(path, repo_root)
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def _align_prediction_frame(input_df: pd.DataFrame, feature_columns: list[str], config: dict[str, Any], logger) -> tuple[pd.DataFrame | None, str]:
    if all(column in input_df.columns for column in feature_columns):
        return input_df.copy(), "input already contains engineered feature columns"

    logger.info("Prediction input does not match engineered features. Re-applying data processing transform.")
    transformed_frame, _ = prepare_modeling_frame(input_df, config, include_target=False, logger=logger)
    if all(column in transformed_frame.columns for column in feature_columns):
        return transformed_frame, "prediction input was re-aligned through data_processing transform"

    return None, "missing required trained features after alignment"


def _resolve_datetime_column(config: dict[str, Any], artifact: dict[str, Any], repo_root: Path) -> str | None:
    metadata_path_value = artifact.get("metadata_path")
    if metadata_path_value:
        metadata_path = Path(metadata_path_value)
        if not metadata_path.is_absolute():
            metadata_path = resolve_repo_path(metadata_path, repo_root)
        if metadata_path.exists():
            return read_json(metadata_path).get("datetime_column")

    training_metadata = artifact.get("training_metadata", {})
    if isinstance(training_metadata, dict):
        datetime_column = training_metadata.get("datetime_column")
        if datetime_column:
            return str(datetime_column)

    return config.get("data_processing", {}).get("dataset", {}).get("datetime_column")


def run_prediction(config: dict[str, Any], logger) -> dict[str, Any]:
    """Run batch inference using the latest or configured artifact."""
    config = ensure_runtime_context_resolved(config)
    repo_root = Path(config["project"]["repo_root"])
    paths_cfg = config["paths"]
    prediction_cfg = config["prediction"]
    runtime_recipe_context = current_recipe_context(config)
    logger.info(
        "Prediction runtime selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
        runtime_recipe_context.get("selection_mode"),
        runtime_recipe_context.get("mode_resolution"),
        runtime_recipe_context.get("scope_token"),
        runtime_recipe_context.get("recipe_profile"),
    )

    configured_artifact_path = prediction_cfg.get("artifact_path")
    if configured_artifact_path:
        artifact_path = resolve_repo_path(configured_artifact_path, repo_root)
        artifact_resolution = {
            "resolution": "configured_artifact_path",
            "summary_path": None,
            "summary_runtime_context": None,
        }
    else:
        artifact_path, artifact_resolution = _resolve_default_artifact_path(config, repo_root, logger)

    artifact = _load_artifact(artifact_path)
    artifact_runtime_context = dict(artifact.get("runtime_context", {}))
    logger.info(
        "Loaded prediction artifact run_id=%s artifact_scope_token=%s artifact_recipe_profile=%s",
        artifact.get("run_id"),
        artifact_runtime_context.get("scope_token"),
        artifact_runtime_context.get("recipe_profile"),
    )
    feature_columns = artifact["feature_columns"]
    target_column = artifact["target_column"]
    prediction_frame: pd.DataFrame | None = None
    resolved_input_path: Path | None = None
    attempted_paths: list[str] = []
    last_missing_features: list[str] = []

    for input_path in _candidate_input_paths(config, artifact, repo_root):
        if not input_path.exists():
            attempted_paths.append(f"{input_path} (missing)")
            continue

        input_df = read_tabular(input_path)
        input_df = filter_frame_to_recipe(input_df, config, stage_name="predict", logger=logger)
        logger.info("Loaded prediction input path=%s rows=%s", input_path, len(input_df))
        aligned_frame, alignment_reason = _align_prediction_frame(input_df, feature_columns, config, logger)
        if aligned_frame is None:
            last_missing_features = [column for column in feature_columns if column not in input_df.columns]
            attempted_paths.append(f"{input_path} ({alignment_reason})")
            continue

        prediction_frame = aligned_frame
        resolved_input_path = input_path
        logger.info("Using prediction input path=%s because %s.", input_path, alignment_reason)
        break

    if prediction_frame is None or resolved_input_path is None:
        raise ValueError(
            "Prediction input is missing model features after alignment. "
            f"Tried: {attempted_paths}. "
            f"Missing features from the last attempted input: {last_missing_features}"
        )

    feature_frame = prediction_frame[feature_columns].dropna().copy()
    if feature_frame.empty:
        raise ValueError("No rows remain for prediction after aligning features and dropping missing values.")

    aligned_prediction_frame = prediction_frame.loc[feature_frame.index].copy()
    predictions = artifact["model"].predict(feature_frame)
    output_df = pd.DataFrame({"prediction": predictions}, index=feature_frame.index)

    datetime_column = _resolve_datetime_column(config, artifact, repo_root)
    if datetime_column and datetime_column in aligned_prediction_frame.columns:
        output_df.insert(0, datetime_column, aligned_prediction_frame[datetime_column].values)
    if "recipe_profile" in aligned_prediction_frame.columns:
        output_df["recipe_profile"] = aligned_prediction_frame["recipe_profile"].astype(str).values
    if prediction_cfg.get("include_actual_target", True) and target_column in aligned_prediction_frame.columns:
        output_df["actual_target"] = aligned_prediction_frame[target_column].values

    output_df["artifact_run_id"] = artifact["run_id"]
    output_df["selected_recipe_profile"] = runtime_recipe_context.get("recipe_profile")
    output_df["selected_runtime_mode"] = runtime_recipe_context.get("selection_mode")
    output_df["selected_runtime_scope"] = runtime_recipe_context.get("scope_token")
    output_df["artifact_runtime_scope"] = artifact_runtime_context.get("scope_token")
    output_df["generated_at_utc"] = utc_timestamp()

    predictions_dir = ensure_directory(resolve_repo_path(paths_cfg["predictions_dir"], repo_root))
    output_filename = prediction_cfg.get("output_filename") or f"predictions_{artifact['run_id']}.csv"
    output_path = predictions_dir / output_filename
    output_df.to_csv(output_path, index=False)
    logger.info("Saved predictions to %s", output_path)

    return {
        "artifact_path": _portable_path(artifact_path, repo_root),
        "artifact_resolution": artifact_resolution,
        "artifact_runtime_context": artifact_runtime_context,
        "input_path_used": _portable_path(resolved_input_path, repo_root),
        "predictions_path": _portable_path(output_path, repo_root),
        "rows_predicted": int(len(output_df)),
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
    }
