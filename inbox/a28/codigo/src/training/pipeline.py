"""Training stage for tabular baseline and offline neuroevolution comparisons."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from src.reproducibility.mixed_context import (
    apply_feature_fill_values,
    fit_feature_fill_values,
    validate_feature_columns_for_stage,
)
from src.training.model_registry import build_model
from src.utils import (
    build_best_by_target,
    canonical_target_column,
    current_recipe_context,
    ensure_runtime_context_resolved,
    ensure_directory,
    filter_frame_to_recipe,
    make_run_id,
    resolve_reference_record,
    resolve_repo_path,
    select_best_record,
    select_preferred_record,
    selection_policy_description,
    to_repo_relative_path,
    utc_timestamp,
    write_json,
)


def _portable_path(path: str | Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    return to_repo_relative_path(path, repo_root)


def _compute_error_metrics(y_true: pd.Series, y_pred: np.ndarray, metric_names: list[str]) -> dict[str, float]:
    metric_map: dict[str, float] = {}
    valid_pairs = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid_pairs.empty:
        return metric_map

    y_true_clean = valid_pairs["y_true"]
    y_pred_clean = valid_pairs["y_pred"]

    for metric in metric_names:
        key = metric.lower()
        if key == "mae":
            metric_map[key] = float(mean_absolute_error(y_true_clean, y_pred_clean))
        elif key == "rmse":
            metric_map[key] = float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)))
        elif key == "r2":
            metric_map[key] = float(r2_score(y_true_clean, y_pred_clean))
        elif key == "mape":
            non_zero_mask = y_true_clean != 0
            if non_zero_mask.any():
                metric_map[key] = float(
                    mean_absolute_percentage_error(y_true_clean[non_zero_mask], y_pred_clean[non_zero_mask])
                )
            else:
                metric_map[key] = float("nan")
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return metric_map


def _compute_distribution_diagnostics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    valid_pairs = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid_pairs.empty:
        return {}

    actual = valid_pairs["y_true"].astype(float)
    prediction = valid_pairs["y_pred"].astype(float)
    residual = prediction - actual

    actual_mean = float(actual.mean())
    prediction_mean = float(prediction.mean())
    actual_std = float(actual.std(ddof=0))
    prediction_std = float(prediction.std(ddof=0))
    variance_ratio = float(prediction_std / actual_std) if actual_std > 0 else float("nan")

    actual_variance = float(actual.var(ddof=0))
    if actual_variance > 0:
        slope_pred_vs_actual = float(np.cov(actual, prediction, ddof=0)[0, 1] / actual_variance)
    else:
        slope_pred_vs_actual = float("nan")

    return {
        "prediction_bias": float(residual.mean()),
        "prediction_mean": prediction_mean,
        "actual_mean": actual_mean,
        "prediction_std": prediction_std,
        "actual_std": actual_std,
        "variance_ratio": variance_ratio,
        "slope_pred_vs_actual": slope_pred_vs_actual,
        "underprediction_rate": float((prediction < actual).mean()),
    }


def _compute_split_metrics(y_true: pd.Series, y_pred: np.ndarray, metric_names: list[str]) -> dict[str, float]:
    split_metrics = _compute_error_metrics(y_true, y_pred, metric_names)
    split_metrics.update(_compute_distribution_diagnostics(y_true, y_pred))
    return split_metrics


def _load_feature_selection_export(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], Path | None]:
    feature_cfg = config.get("feature_selection", {})
    configured_feature_sets = feature_cfg.get("feature_sets", {})
    if not isinstance(configured_feature_sets, dict):
        configured_feature_sets = {}
    export_path = resolve_repo_path(feature_cfg["feature_config_export_path"], repo_root)
    if export_path.exists():
        with export_path.open("r", encoding="utf-8") as handle:
            export_payload = yaml.safe_load(handle) or {}
        if not isinstance(export_payload, dict):
            raise ValueError(f"Feature export must contain a mapping: {export_path}")
        return export_payload, export_path

    fallback_payload = {
        "target": feature_cfg.get("target"),
        "selected_features_minimal": list(feature_cfg.get("selected_features_minimal", [])),
        "selected_features_extended": list(feature_cfg.get("selected_features_extended", [])),
        "selected_features": list(feature_cfg.get("selected_features", [])),
        "feature_sets": {
            str(name): [str(feature) for feature in features if isinstance(feature, str)]
            for name, features in configured_feature_sets.items()
            if isinstance(features, list)
        },
        "recommended_feature_set": str(feature_cfg.get("recommended_feature_set", "extended")),
        "review_pool_features": list(feature_cfg.get("review_pool_features", feature_cfg.get("review_features", []))),
        "excluded_features": list(feature_cfg.get("excluded_features", [])),
        "numerical_features": list(feature_cfg.get("numerical_features", [])),
        "categorical_features": list(feature_cfg.get("categorical_features", [])),
        "temporal_features": list(feature_cfg.get("temporal_features", [])),
        "derived_features": list(feature_cfg.get("derived_features", [])),
        "prepared_dataset_path": _portable_path(resolve_repo_path(feature_cfg["prepared_dataset_path"], repo_root), repo_root),
        "feature_contract_path": _portable_path(resolve_repo_path(feature_cfg["feature_contract_path"], repo_root), repo_root)
        if feature_cfg.get("feature_contract_path")
        else None,
        "input_dataset_mode": feature_cfg.get("input_dataset_mode"),
        "target_candidates": [
            feature_cfg.get("target_primary"),
            *feature_cfg.get("target_alternatives", []),
        ],
        "recipe_context": current_recipe_context(config),
    }
    return fallback_payload, None


def _resolve_modeling_dataset_path(config: dict[str, Any], feature_export: dict[str, Any], repo_root: Path) -> Path:
    preferred_value = feature_export.get("prepared_dataset_path") or config.get("feature_selection", {}).get("prepared_dataset_path")
    if not preferred_value:
        raise ValueError("No prepared_dataset_path found in feature selection export or config.")
    dataset_path = resolve_repo_path(preferred_value, repo_root)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Prepared modeling dataset not found: {dataset_path}. Run notebook 02 or export its prepared dataset first."
        )
    return dataset_path


def _expand_target_tokens(target_tokens: list[str], feature_export: dict[str, Any]) -> list[str]:
    expanded_targets: list[str] = []
    for token in target_tokens:
        if not token:
            continue
        if token == "from_feature_selection":
            for candidate in [feature_export.get("target"), feature_export.get("target_used")]:
                if candidate:
                    expanded_targets.append(candidate)
            continue
        expanded_targets.append(token)
    return expanded_targets


def _resolve_training_targets(
    training_cfg: dict[str, Any],
    feature_export: dict[str, Any],
    available_columns: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    explicit_targets = list(training_cfg.get("target_columns", []))
    if explicit_targets:
        candidate_tokens = explicit_targets
    else:
        candidate_tokens = []
        target_override = training_cfg.get("target_override")
        if target_override:
            candidate_tokens.append(target_override)
        default_target = training_cfg.get("target_default")
        if default_target:
            candidate_tokens.append(default_target)
        candidate_tokens.extend(training_cfg.get("target_alternatives", []))

    expanded_candidates = _expand_target_tokens(candidate_tokens, feature_export)
    deduped_candidates = list(dict.fromkeys(expanded_candidates))
    available_candidates = [candidate for candidate in deduped_candidates if candidate in available_columns]
    if not available_candidates:
        raise ValueError(
            "No training target column is available in the prepared dataset. "
            f"Candidates tried: {deduped_candidates}"
        )

    resolved_targets: list[dict[str, str]] = []
    for index, target_column in enumerate(available_candidates):
        if training_cfg.get("target_override") == target_column:
            reason = "Explicit training.target_override available in the prepared dataset."
        elif feature_export.get("target") == target_column:
            reason = "Using the target exported by notebook 02 feature engineering."
        elif index == 0:
            reason = "Using the first available configured target for the comparison run."
        else:
            reason = "Using an additional configured target available in the prepared dataset."
        resolved_targets.append(
            {
                "target_column": target_column,
                "target_role": "primary" if index == 0 else "alternative",
                "target_selection_reason": reason,
            }
        )
    return resolved_targets, available_candidates


def _resolve_feature_sets(training_cfg: dict[str, Any], feature_export: dict[str, Any]) -> dict[str, list[str]]:
    requested_sets = list(training_cfg.get("feature_sets", ["minimal", "extended"]))
    feature_sets: dict[str, list[str]] = {}
    export_feature_sets = feature_export.get("feature_sets", {})
    if not isinstance(export_feature_sets, dict):
        export_feature_sets = {}
    mapping = {
        "minimal": feature_export.get("selected_features_minimal", []),
        "extended": feature_export.get("selected_features_extended", feature_export.get("selected_features", [])),
    }
    for set_name, entries in export_feature_sets.items():
        if isinstance(entries, list):
            mapping[str(set_name)] = [str(entry) for entry in entries if isinstance(entry, str)]
    for set_name in requested_sets:
        resolved_features = list(mapping.get(set_name, []))
        if not resolved_features:
            raise ValueError(f"Feature set '{set_name}' is not available in the feature selection export.")
        feature_sets[set_name] = resolved_features
    return feature_sets


def _split_dataset(
    df: pd.DataFrame,
    *,
    datetime_column: str | None,
    train_size: float,
    valid_size: float,
    test_size: float,
) -> dict[str, pd.DataFrame]:
    total_share = train_size + valid_size + test_size
    if not np.isclose(total_share, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0. Current total={total_share}")

    working_df = df.copy()
    if datetime_column and datetime_column in working_df.columns:
        working_df = working_df.sort_values(datetime_column).reset_index(drop=True)

    total_rows = len(working_df)
    if total_rows == 0:
        raise ValueError("Prepared modeling dataset is empty; training cannot proceed.")

    train_end = max(1, int(total_rows * train_size))
    valid_end = train_end + int(total_rows * valid_size)
    return {
        "train": working_df.iloc[:train_end].reset_index(drop=True),
        "valid": working_df.iloc[train_end:valid_end].reset_index(drop=True),
        "test": working_df.iloc[valid_end:].reset_index(drop=True),
    }


def _feature_importance_frame(model, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model") or list(model.named_steps.values())[-1]

    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({"feature": feature_columns, "importance": getattr(estimator, "feature_importances_")}).sort_values(
            "importance",
            ascending=False,
        )
    if hasattr(estimator, "coef_"):
        coef = np.ravel(getattr(estimator, "coef_"))
        return pd.DataFrame({"feature": feature_columns, "importance": np.abs(coef), "coefficient": coef}).sort_values(
            "importance",
            ascending=False,
        )
    return None


def _normalise_hyperparameters(config: dict[str, Any], training_cfg: dict[str, Any], model_family: str) -> dict[str, Any]:
    if model_family == "neuroevolution":
        neuro_cfg = dict(config.get("neuroevolution", {}))
        hidden_options = neuro_cfg.get("hidden_layer_options", [4, 8, 12])
        return {
            "random_state": int(neuro_cfg.get("random_seed", training_cfg.get("random_seed", config.get("project", {}).get("seed", 42)))),
            "population_size": int(neuro_cfg.get("population_size", 36)),
            "generations": int(neuro_cfg.get("generations", 24)),
            "elite_size": int(neuro_cfg.get("elite_size", 4)),
            "tournament_size": int(neuro_cfg.get("tournament_size", 3)),
            "hidden_layer_options": list(hidden_options),
            "mutation_rate": float(neuro_cfg.get("mutation_rate", 0.18)),
            "mutation_scale": float(neuro_cfg.get("mutation_scale", 0.18)),
            "crossover_rate": float(neuro_cfg.get("crossover_rate", 0.75)),
            "architecture_mutation_rate": float(neuro_cfg.get("architecture_mutation_rate", 0.1)),
            "train_fitness_weight": float(neuro_cfg.get("train_fitness_weight", 0.7)),
            "validation_fitness_weight": float(neuro_cfg.get("validation_fitness_weight", 0.3)),
            "complexity_penalty_weight": float(neuro_cfg.get("complexity_penalty_weight", 0.01)),
            "input_clip_sigma": float(neuro_cfg.get("input_clip_sigma", 4.0)),
            "stagnation_patience": int(neuro_cfg.get("stagnation_patience", 8)),
            "log_every_generations": int(neuro_cfg.get("log_every_generations", 5)),
        }

    hyperparameters = dict(training_cfg.get("hyperparameters", {}).get(model_family, {}))
    seed = int(training_cfg.get("random_seed", config.get("project", {}).get("seed", 42)))
    if model_family in {"random_forest", "gradient_boosting", "xgboost", "lightgbm", "ridge"}:
        hyperparameters.setdefault("random_state", seed)
    if model_family == "dummy":
        hyperparameters.setdefault("strategy", training_cfg.get("dummy_strategy", "mean"))
    return hyperparameters


def _fit_model(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    logger,
) -> None:
    try:
        if getattr(model, "requires_validation_data", False):
            model.fit(x_train, y_train, validation_data=(x_valid, y_valid), logger=logger)
        else:
            model.fit(x_train, y_train)
    except PermissionError:
        if hasattr(model, "n_jobs") and getattr(model, "n_jobs", None) not in (None, 1):
            logger.warning(
                "Parallel training is not available in the current environment. Retrying with n_jobs=1."
            )
            model.set_params(n_jobs=1)
            model.fit(x_train, y_train)
        else:
            raise


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def _prepare_feature_columns(feature_columns: list[str], target_column: str, logger) -> list[str]:
    prepared_features = [column for column in feature_columns if column != target_column]
    if len(prepared_features) != len(feature_columns):
        logger.warning(
            "Dropped target column '%s' from predictors to avoid direct leakage in the current run.",
            target_column,
        )
    if not prepared_features:
        raise ValueError(f"Feature set becomes empty after removing the current target column '{target_column}'.")
    return prepared_features


def _build_metrics_payload(
    *,
    run_id: str,
    comparison_run_id: str,
    model_family: str,
    model_name: str,
    feature_set_name: str,
    feature_columns: list[str],
    target_column: str,
    target_role: str,
    repo_root: Path,
    dataset_path: Path,
    metrics: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    artifact_path: Path,
    feature_importance_path: Path | None,
    validation_predictions_path: Path,
    test_predictions_path: Path,
    run_category: str,
    is_neuroevolution: bool,
    training_metadata: dict[str, Any] | None = None,
    extra_artifact_paths: dict[str, str] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "comparison_run_id": comparison_run_id,
        "trained_at_utc": utc_timestamp(),
        "model_family": model_family,
        "model_name": model_name,
        "feature_set": feature_set_name,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_role": target_role,
        "run_category": run_category,
        "is_neuroevolution": is_neuroevolution,
        "input_dataset_path": _portable_path(dataset_path, repo_root),
        "artifact_path": _portable_path(artifact_path, repo_root),
        "metrics": metrics,
        "split_rows": {split_name: int(len(split_df)) for split_name, split_df in split_frames.items()},
        "feature_importance_path": _portable_path(feature_importance_path, repo_root) if feature_importance_path else None,
        "prediction_paths": {
            "validation": _portable_path(validation_predictions_path, repo_root),
            "test": _portable_path(test_predictions_path, repo_root),
        },
        "test_predictions_path": _portable_path(test_predictions_path, repo_root) if test_predictions_path.exists() else None,
        "training_metadata": training_metadata or {},
        "extra_artifact_paths": extra_artifact_paths or {},
        "runtime_context": runtime_context or {},
    }


def _comparison_improvements(summary_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if summary_df.empty or metric_name not in summary_df.columns:
        return summary_df

    enriched = summary_df.copy()
    enriched["improvement_vs_dummy_pct"] = np.nan
    enriched["beats_dummy"] = False

    dummy_lookup = (
        enriched[enriched["model_family"] == "dummy"][["target_column", "feature_set", metric_name]]
        .rename(columns={metric_name: "dummy_metric"})
        .drop_duplicates(subset=["target_column", "feature_set"])
    )
    if dummy_lookup.empty:
        return enriched

    enriched = enriched.merge(dummy_lookup, on=["target_column", "feature_set"], how="left")
    improvement_mask = enriched["dummy_metric"].notna() & enriched[metric_name].notna() & (enriched["model_family"] != "dummy")
    enriched.loc[improvement_mask, "improvement_vs_dummy_pct"] = (
        (enriched.loc[improvement_mask, "dummy_metric"] - enriched.loc[improvement_mask, metric_name])
        / enriched.loc[improvement_mask, "dummy_metric"]
        * 100.0
    )
    enriched["beats_dummy"] = enriched["improvement_vs_dummy_pct"].fillna(0.0) > 0.0
    return enriched.drop(columns=["dummy_metric"], errors="ignore")


def _best_rows_by_target(summary_df: pd.DataFrame, primary_metric: str) -> list[dict[str, Any]]:
    return build_best_by_target(summary_df, primary_metric=primary_metric)


def _resolve_model_families(config: dict[str, Any], training_cfg: dict[str, Any]) -> list[str]:
    model_families = list(training_cfg.get("model_families", ["dummy", "random_forest", "gradient_boosting"]))
    neuro_cfg = config.get("neuroevolution", {})
    if neuro_cfg.get("enabled", False) and "neuroevolution" not in model_families:
        model_families.append("neuroevolution")
    return model_families


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


def _resolve_neuro_reference_run(
    summary_df: pd.DataFrame,
    config: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any] | None:
    if summary_df.empty:
        return None

    baseline_df = summary_df[summary_df["run_category"] == "baseline"]
    if baseline_df.empty:
        return None

    neuro_cfg = config.get("neuroevolution", {})
    reference_cfg = neuro_cfg.get("baseline_reference", {})
    filters = {
        "target_column": reference_cfg.get("target_column"),
        "feature_set": reference_cfg.get("feature_set_name"),
        "model_family": reference_cfg.get("model_family"),
    }
    target_column = neuro_cfg.get("target_column") or canonical_target_column(config)
    reference_run = resolve_reference_record(
        baseline_df,
        primary_metric=primary_metric,
        canonical_target=target_column,
        run_category="baseline",
        filters=filters,
    )
    if reference_run is not None:
        return reference_run

    feature_set_name = neuro_cfg.get("feature_set_name")
    fallback_filters = {"feature_set": feature_set_name} if feature_set_name else None
    return resolve_reference_record(
        baseline_df,
        primary_metric=primary_metric,
        canonical_target=target_column,
        run_category="baseline",
        filters=fallback_filters,
    )


def _metric_delta_value(metric_name: str, baseline_value: Any, neuro_value: Any) -> float | None:
    try:
        baseline = float(baseline_value)
        neuro = float(neuro_value)
    except (TypeError, ValueError):
        return None
    if np.isnan(baseline) or np.isnan(neuro):
        return None
    return float(neuro - baseline)


def _build_neuroevolution_comparison(
    *,
    summary_df: pd.DataFrame,
    comparison_run_id: str,
    primary_metric: str,
    repo_root: Path,
    dataset_path: Path,
    stats_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    best_neuro_run = select_preferred_record(
        summary_df,
        primary_metric=primary_metric,
        canonical_target=config.get("neuroevolution", {}).get("target_column") or canonical_target_column(config),
        run_category="neuroevolution",
    )
    if best_neuro_run is None:
        return None

    reference_run = _resolve_neuro_reference_run(summary_df, config, primary_metric)
    if reference_run is None:
        return None

    tracked_metrics = [
        "test_mae",
        "test_rmse",
        "test_r2",
        "test_mape",
        "test_prediction_bias",
        "test_prediction_std",
        "test_actual_std",
        "test_variance_ratio",
        "test_slope_pred_vs_actual",
        "test_underprediction_rate",
    ]
    metric_deltas = {
        metric_name: _metric_delta_value(metric_name, reference_run.get(metric_name), best_neuro_run.get(metric_name))
        for metric_name in tracked_metrics
    }
    rmse_baseline = reference_run.get("test_rmse")
    rmse_neuro = best_neuro_run.get("test_rmse")
    if rmse_baseline not in (None, 0) and rmse_neuro is not None:
        metric_deltas["test_rmse_pct"] = float((float(rmse_neuro) - float(rmse_baseline)) / float(rmse_baseline) * 100.0)
    else:
        metric_deltas["test_rmse_pct"] = None

    same_conditions_validation = {
        "same_target": reference_run.get("target_column") == best_neuro_run.get("target_column"),
        "same_feature_set": reference_run.get("feature_set") == best_neuro_run.get("feature_set"),
        "same_feature_count": reference_run.get("feature_count") == best_neuro_run.get("feature_count"),
        "same_input_dataset": True,
        "same_split_rows": (
            reference_run.get("train_rows") == best_neuro_run.get("train_rows")
            and reference_run.get("validation_rows") == best_neuro_run.get("validation_rows")
            and reference_run.get("test_rows") == best_neuro_run.get("test_rows")
        ),
        "same_metric_schema": True,
    }

    lower_is_better_metrics = {
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_prediction_bias",
        "test_underprediction_rate",
    }
    winner_by_metric: dict[str, str] = {}
    for metric_name in tracked_metrics:
        baseline_value = reference_run.get(metric_name)
        neuro_value = best_neuro_run.get(metric_name)
        if baseline_value is None or neuro_value is None:
            winner_by_metric[metric_name] = "unavailable"
            continue
        if metric_name == "test_prediction_bias":
            winner_by_metric[metric_name] = (
                "neuroevolution" if abs(float(neuro_value)) < abs(float(baseline_value)) else "baseline"
            )
        elif metric_name in {"test_variance_ratio", "test_slope_pred_vs_actual"}:
            winner_by_metric[metric_name] = (
                "neuroevolution"
                if abs(float(neuro_value) - 1.0) < abs(float(baseline_value) - 1.0)
                else "baseline"
            )
        elif metric_name in lower_is_better_metrics:
            winner_by_metric[metric_name] = "neuroevolution" if float(neuro_value) < float(baseline_value) else "baseline"
        else:
            winner_by_metric[metric_name] = "neuroevolution" if float(neuro_value) > float(baseline_value) else "baseline"

    recommendation = (
        "neuroevolution_promising"
        if winner_by_metric.get(primary_metric, "baseline") == "neuroevolution"
        else "baseline_retained"
    )
    comparison_payload = {
        "comparison_run_id": comparison_run_id,
        "created_at_utc": utc_timestamp(),
        "input_dataset_path": _portable_path(dataset_path, repo_root),
        "primary_metric": primary_metric,
        "recipe_context": current_recipe_context(config),
        "target_column": best_neuro_run.get("target_column"),
        "feature_set": best_neuro_run.get("feature_set"),
        "feature_count": best_neuro_run.get("feature_count"),
        "baseline_reference_run": reference_run,
        "best_neuroevolution_run": best_neuro_run,
        "same_conditions_validation": same_conditions_validation,
        "metric_deltas_neuro_minus_baseline": metric_deltas,
        "winner_by_metric": winner_by_metric,
        "recommendation": recommendation,
    }

    neuro_cfg = config.get("neuroevolution", {})
    summary_json_name = neuro_cfg.get("comparison_summary_json_name", "neuroevolution_comparison_latest.json")
    summary_csv_name = neuro_cfg.get("comparison_summary_csv_name", "neuroevolution_comparison_latest.csv")
    comparison_json_path = stats_dir / summary_json_name
    comparison_csv_path = stats_dir / summary_csv_name
    comparison_df = pd.DataFrame([reference_run, best_neuro_run])
    comparison_df.to_csv(comparison_csv_path, index=False)
    write_json(comparison_json_path, comparison_payload)
    comparison_payload["summary_json_path"] = _portable_path(comparison_json_path, repo_root)
    comparison_payload["summary_csv_path"] = _portable_path(comparison_csv_path, repo_root)
    return comparison_payload


def run_training(config: dict[str, Any], logger) -> dict[str, Any]:
    """Train and compare configured regressors across targets and feature sets."""
    config = ensure_runtime_context_resolved(config)
    repo_root = Path(config["project"]["repo_root"])
    paths_cfg = config["paths"]
    training_cfg = config["training"]
    runtime_recipe_context = current_recipe_context(config)
    logger.info(
        "Training runtime selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
        runtime_recipe_context.get("selection_mode"),
        runtime_recipe_context.get("mode_resolution"),
        runtime_recipe_context.get("scope_token"),
        runtime_recipe_context.get("recipe_profile"),
    )
    feature_export, feature_export_path = _load_feature_selection_export(config, repo_root)

    dataset_path = _resolve_modeling_dataset_path(config, feature_export, repo_root)
    modeling_df = pd.read_csv(dataset_path)
    modeling_df = filter_frame_to_recipe(modeling_df, config, stage_name="train", logger=logger)
    logger.info("Loaded modeling dataset path=%s rows=%s cols=%s", dataset_path, len(modeling_df), len(modeling_df.columns))

    target_specs, available_targets = _resolve_training_targets(training_cfg, feature_export, modeling_df.columns.tolist())
    feature_sets = _resolve_feature_sets(training_cfg, feature_export)
    datetime_column = config["data_processing"]["dataset"].get("datetime_column")

    split_cfg = config["data_processing"]["split"]
    split_frames = _split_dataset(
        modeling_df,
        datetime_column=datetime_column if datetime_column in modeling_df.columns else None,
        train_size=float(split_cfg.get("train_size", 0.7)),
        valid_size=float(split_cfg.get("valid_size", 0.15)),
        test_size=float(split_cfg.get("test_size", 0.15)),
    )
    logger.info(
        "Training split rows train=%s valid=%s test=%s",
        len(split_frames["train"]),
        len(split_frames["valid"]),
        len(split_frames["test"]),
    )

    artifact_dir = ensure_directory(resolve_repo_path(paths_cfg["model_artifacts_dir"], repo_root))
    metrics_dir = ensure_directory(resolve_repo_path(paths_cfg["model_metrics_dir"], repo_root))
    stats_dir = ensure_directory(resolve_repo_path(paths_cfg["stats_dir"], repo_root))

    comparison_name = training_cfg.get("comparison_name", training_cfg.get("model_name", "baseline_comparison"))
    official_run_id = str(
        config.get("runtime", {}).get("official_run", {}).get("id", "")
    ).strip()
    official_run_suffix = (
        official_run_id.removeprefix("mixed_context_")
        if comparison_name.endswith("_mixed_context")
        else official_run_id
    )
    comparison_run_id = (
        f"{comparison_name}_{official_run_suffix}"
        if official_run_id
        else make_run_id(comparison_name)
    )
    model_families = _resolve_model_families(config, training_cfg)
    metrics_cfg = list(training_cfg.get("metrics", ["mae", "rmse", "r2", "mape"]))
    primary_metric = training_cfg.get("primary_metric", "test_rmse")
    neuro_cfg = config.get("neuroevolution", {})
    neuro_target_column = neuro_cfg.get("target_column")
    neuro_feature_set_name = neuro_cfg.get("feature_set_name")

    summary_rows: list[dict[str, Any]] = []
    individual_metric_paths: list[str] = []

    for target_spec in target_specs:
        target_column = target_spec["target_column"]
        target_role = target_spec["target_role"]
        target_reason = target_spec["target_selection_reason"]

        for model_family in model_families:
            if model_family == "neuroevolution" and neuro_target_column and target_column != neuro_target_column:
                continue
            hyperparameters = _normalise_hyperparameters(config, training_cfg, model_family)
            for feature_set_name, raw_feature_columns in feature_sets.items():
                if model_family == "neuroevolution" and neuro_feature_set_name and feature_set_name != neuro_feature_set_name:
                    continue
                feature_columns = _prepare_feature_columns(raw_feature_columns, target_column, logger)
                missing_features = [column for column in feature_columns if column not in modeling_df.columns]
                if missing_features:
                    raise ValueError(
                        f"Feature set '{feature_set_name}' references columns missing from the modeling dataset: {missing_features}"
                    )
                if config.get("runtime", {}).get("reproducibility_scope") == "mixed_context":
                    validate_feature_columns_for_stage(feature_columns, stage="upstream")

                model = build_model(model_family, hyperparameters)
                fill_values = fit_feature_fill_values(split_frames["train"], feature_columns)
                x_train = apply_feature_fill_values(split_frames["train"], feature_columns, fill_values)
                y_train = split_frames["train"][target_column]
                x_valid = apply_feature_fill_values(split_frames["valid"], feature_columns, fill_values)
                y_valid = split_frames["valid"][target_column]
                x_test = apply_feature_fill_values(split_frames["test"], feature_columns, fill_values)
                y_test = split_frames["test"][target_column]

                logger.info(
                    "Training model family=%s feature_set=%s target=%s rows=%s features=%s",
                    model_family,
                    feature_set_name,
                    target_column,
                    len(x_train),
                    len(feature_columns),
                )
                _fit_model(model, x_train, y_train, x_valid, y_valid, logger)

                validation_predictions = model.predict(x_valid) if not x_valid.empty else np.array([])
                test_predictions = model.predict(x_test) if not x_test.empty else np.array([])
                model_name = f"{model_family}_{feature_set_name}"
                metrics = {
                    "validation": _compute_split_metrics(y_valid, validation_predictions, metrics_cfg) if not x_valid.empty else {},
                    "test": _compute_split_metrics(y_test, test_predictions, metrics_cfg) if not x_test.empty else {},
                }
                for split_name, actual in (("validation", y_valid), ("test", y_test)):
                    if metrics[split_name]:
                        metrics[split_name].update(
                            {
                                "split": split_name,
                                "model": model_name,
                                "target": target_column,
                                "n_samples": int(len(actual)),
                                "rows": int(len(actual)),
                            }
                        )

                target_slug = _slugify(target_column)
                run_id = f"{comparison_run_id}__{target_slug}__{model_family}__{feature_set_name}"
                artifact_path = artifact_dir / f"{run_id}.pkl"
                metrics_path = metrics_dir / f"{run_id}.json"
                feature_importance_path = metrics_dir / f"{run_id}_feature_importance.csv"
                validation_predictions_path = metrics_dir / f"{run_id}_validation_predictions.csv"
                test_predictions_path = metrics_dir / f"{run_id}_test_predictions.csv"
                evolution_history_path = metrics_dir / f"{run_id}_evolution_history.csv"
                training_metadata_path = metrics_dir / f"{run_id}_training_metadata.json"
                extra_artifact_paths: dict[str, str] = {}
                training_metadata = getattr(model, "training_metadata_", None)

                if hasattr(model, "evolution_history_"):
                    pd.DataFrame(getattr(model, "evolution_history_")).to_csv(evolution_history_path, index=False)
                    extra_artifact_paths["evolution_history_path"] = _portable_path(evolution_history_path, repo_root)
                if training_metadata:
                    write_json(training_metadata_path, training_metadata)
                    extra_artifact_paths["training_metadata_path"] = _portable_path(training_metadata_path, repo_root)

                artifact_payload = {
                    "run_id": run_id,
                    "comparison_run_id": comparison_run_id,
                    "trained_at_utc": utc_timestamp(),
                    "model_family": model_family,
                    "model_name": model_name,
                    "feature_set": feature_set_name,
                    "target_column": target_column,
                    "target_role": target_role,
                    "target_selection_reason": target_reason,
                    "model": model,
                    "feature_columns": feature_columns,
                    "fill_values": fill_values,
                    "feature_imputation": "train_median_with_neutral_fallback_for_all_missing_columns",
                    "input_dataset_path": _portable_path(dataset_path, repo_root),
                    "feature_export_path": _portable_path(feature_export_path, repo_root) if feature_export_path else None,
                    "feature_contract_path": feature_export.get("feature_contract_path"),
                    "feature_export_snapshot": feature_export,
                    "training_hyperparameters": hyperparameters,
                    "run_category": "neuroevolution" if model_family == "neuroevolution" else "baseline",
                    "is_neuroevolution": model_family == "neuroevolution",
                    "training_metadata": training_metadata or {},
                    "extra_artifact_paths": extra_artifact_paths,
                    "runtime_context": runtime_recipe_context,
                }
                with artifact_path.open("wb") as handle:
                    pickle.dump(artifact_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

                importance_df = _feature_importance_frame(model, feature_columns)
                if importance_df is not None:
                    importance_df.to_csv(feature_importance_path, index=False)
                else:
                    feature_importance_path = None

                for split_name, actual, predictions, prediction_path in (
                    ("valid", y_valid, validation_predictions, validation_predictions_path),
                    ("test", y_test, test_predictions, test_predictions_path),
                ):
                    prediction_df = pd.DataFrame({"actual": actual, "prediction": predictions})
                    prediction_df["residual"] = prediction_df["prediction"] - prediction_df["actual"]
                    prediction_df["absolute_error"] = prediction_df["residual"].abs()
                    if datetime_column and datetime_column in split_frames[split_name].columns:
                        prediction_df.insert(0, datetime_column, split_frames[split_name][datetime_column])
                    prediction_df.to_csv(prediction_path, index=False)

                metrics_payload = _build_metrics_payload(
                    run_id=run_id,
                    comparison_run_id=comparison_run_id,
                    model_family=model_family,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    target_role=target_role,
                    repo_root=repo_root,
                    dataset_path=dataset_path,
                    metrics=metrics,
                    split_frames=split_frames,
                    artifact_path=artifact_path,
                    feature_importance_path=feature_importance_path,
                    validation_predictions_path=validation_predictions_path,
                    test_predictions_path=test_predictions_path,
                    run_category="neuroevolution" if model_family == "neuroevolution" else "baseline",
                    is_neuroevolution=model_family == "neuroevolution",
                    training_metadata=training_metadata,
                    extra_artifact_paths=extra_artifact_paths,
                    runtime_context=runtime_recipe_context,
                )
                write_json(metrics_path, metrics_payload)
                individual_metric_paths.append(_portable_path(metrics_path, repo_root))

                summary_rows.append(
                    {
                        "comparison_run_id": comparison_run_id,
                        "run_id": run_id,
                        "trained_at_utc": metrics_payload["trained_at_utc"],
                        "target_column": target_column,
                        "target_role": target_role,
                        "target_selection_reason": target_reason,
                        "model_family": model_family,
                        "model_name": metrics_payload["model_name"],
                        "run_category": metrics_payload["run_category"],
                        "is_neuroevolution": metrics_payload["is_neuroevolution"],
                        "feature_set": feature_set_name,
                        "feature_count": len(feature_columns),
                        "artifact_path": _portable_path(artifact_path, repo_root),
                        "metrics_path": _portable_path(metrics_path, repo_root),
                        "feature_importance_path": _portable_path(feature_importance_path, repo_root) if feature_importance_path else None,
                        "test_predictions_path": _portable_path(test_predictions_path, repo_root),
                        "evolution_history_path": extra_artifact_paths.get("evolution_history_path"),
                        "training_metadata_path": extra_artifact_paths.get("training_metadata_path"),
                        "runtime_context": runtime_recipe_context,
                        "recipe_profile": runtime_recipe_context.get("recipe_profile"),
                        "manufacturing_context_profile": runtime_recipe_context.get("manufacturing_context_profile"),
                        "train_rows": len(split_frames["train"]),
                        "validation_rows": len(split_frames["valid"]),
                        "test_rows": len(split_frames["test"]),
                        **{f"validation_{metric_name}": metric_value for metric_name, metric_value in metrics["validation"].items()},
                        **{f"test_{metric_name}": metric_value for metric_name, metric_value in metrics["test"].items()},
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("No training runs were produced.")

    summary_df = _comparison_improvements(summary_df, primary_metric)
    summary_df = summary_df.sort_values(["target_column", "trained_at_utc", "run_id"], ascending=[True, False, True]).reset_index(
        drop=True
    )

    canonical_target = canonical_target_column(config)
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
    neuroevolution_comparison = _build_neuroevolution_comparison(
        summary_df=summary_df,
        comparison_run_id=comparison_run_id,
        primary_metric=primary_metric,
        repo_root=repo_root,
        dataset_path=dataset_path,
        stats_dir=stats_dir,
        config=config,
    )
    selection_policy = selection_policy_description(primary_metric=primary_metric, canonical_target=canonical_target)

    comparison_payload = {
        "comparison_run_id": comparison_run_id,
        "trained_at_utc": utc_timestamp(),
        "input_dataset_path": _portable_path(dataset_path, repo_root),
        "feature_export_path": _portable_path(feature_export_path, repo_root) if feature_export_path else None,
        "targets_compared": target_specs,
        "available_target_candidates": available_targets,
        "feature_sets": {name: features for name, features in feature_sets.items()},
        "model_families": model_families,
        "split_rows": {name: int(len(frame)) for name, frame in split_frames.items()},
        "primary_metric": primary_metric,
        "recipe_context": runtime_recipe_context,
        "selection_policy": selection_policy,
        "best_run": best_run,
        "best_run_global": best_run_global,
        "best_baseline_run": best_baseline_run,
        "baseline_reference_run": baseline_reference_run,
        "best_neuroevolution_run": best_neuroevolution_run,
        "best_by_target": best_by_target,
        "neuroevolution_comparison": neuroevolution_comparison,
        "runs": summary_df.to_dict(orient="records"),
    }

    summary_csv_name = training_cfg.get("comparison_summary_csv_name", "baseline_comparison_latest.csv")
    summary_json_name = training_cfg.get("comparison_summary_json_name", "baseline_comparison_latest.json")
    summary_csv_path = stats_dir / summary_csv_name
    summary_json_path = stats_dir / summary_json_name
    summary_df.to_csv(summary_csv_path, index=False)
    write_json(summary_json_path, comparison_payload)

    logger.info("Saved comparison summary CSV to %s", summary_csv_path)
    logger.info("Saved comparison summary JSON to %s", summary_json_path)

    return {
        "comparison_run_id": comparison_run_id,
        "summary_csv_path": str(summary_csv_path.resolve()),
        "summary_json_path": str(summary_json_path.resolve()),
        "targets_compared": target_specs,
        "available_target_candidates": available_targets,
        "feature_sets": {name: len(features) for name, features in feature_sets.items()},
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
        "selection_policy": selection_policy,
        "best_run": best_run,
        "best_run_global": best_run_global,
        "best_baseline_run": best_baseline_run,
        "baseline_reference_run": baseline_reference_run,
        "best_neuroevolution_run": best_neuroevolution_run,
        "best_by_target": best_by_target,
        "neuroevolution_comparison": neuroevolution_comparison,
        "metrics_paths": individual_metric_paths,
    }
