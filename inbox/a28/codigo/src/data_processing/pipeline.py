"""Configurable data processing stage for the baseline pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data_processing.cu04_external_ingestion import run_external_context_pipeline
from src.data_processing.feature_report import (
    build_feature_contract_from_config,
    build_feature_roles_metadata_from_config,
)
from src.data_processing.ine_ingestion import run_ine_ingestion_pipeline
from src.data_processing.synthetic_plant import build_synthetic_plant_dataset, write_synthetic_plant_outputs
from src.utils import (
    current_recipe_context,
    ensure_runtime_context_resolved,
    ensure_directory,
    filter_frame_to_recipe,
    read_tabular,
    resolve_repo_path,
    to_repo_relative_path,
    utc_timestamp,
    write_json,
)


def _portable_path(path: str | Path, repo_root: Path) -> str:
    return to_repo_relative_path(path, repo_root)


def _bootstrap_missing_path(target_path: Path, *, search_root: Path, logger) -> Path:
    if target_path.exists():
        return target_path

    candidate_paths = sorted(
        [
            path
            for path in search_root.rglob(f"*{target_path.name}")
            if path.is_file() and path.resolve() != target_path.resolve()
        ],
        key=lambda path: (len(path.parts), str(path)),
    )
    if not candidate_paths:
        return target_path

    source_path = candidate_paths[0]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    logger.info("Bootstrapped missing input from %s to %s", source_path, target_path)
    return target_path


def _bootstrap_context_proxy_inputs(config: dict[str, Any], repo_root: Path, logger) -> None:
    context_cfg = config.get("context_proxy", {})
    search_root = repo_root / "data" / "processed" / "external"
    path_keys = [
        "weekly_context_path",
        "wide_weekly_path",
        "wide_monthly_path",
        "metadata_path",
        "diagnostics_path",
        "limitations_path",
    ]
    for key in path_keys:
        path_value = context_cfg.get(key)
        if not path_value:
            continue
        resolved_path = resolve_repo_path(path_value, repo_root)
        _bootstrap_missing_path(resolved_path, search_root=search_root, logger=logger)


def _validate_required_columns(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def _sort_frame(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    if not sort_columns:
        return df
    return df.sort_values(sort_columns).reset_index(drop=True)


def _apply_outlier_filters(df: pd.DataFrame, filters: dict[str, dict[str, float]], logger) -> tuple[pd.DataFrame, int]:
    if not filters:
        return df, 0

    filtered = df.copy()
    removed_total = 0
    for column, spec in filters.items():
        if column not in filtered.columns:
            logger.warning("Outlier filter skipped because column is missing: %s", column)
            continue

        before_rows = len(filtered)
        numeric = pd.to_numeric(filtered[column], errors="coerce")
        min_value = spec.get("min")
        max_value = spec.get("max")

        if min_value is not None:
            filtered = filtered[numeric.isna() | (numeric >= float(min_value))]
            numeric = pd.to_numeric(filtered[column], errors="coerce")
        if max_value is not None:
            filtered = filtered[numeric.isna() | (numeric <= float(max_value))]

        removed_total += before_rows - len(filtered)

    return filtered.reset_index(drop=True), removed_total


def _fill_categorical(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "constant":
        return series.fillna("missing")
    if strategy == "most_frequent":
        if series.dropna().empty:
            return series.fillna("missing")
        return series.fillna(series.dropna().mode().iloc[0])
    if strategy == "drop_rows":
        return series
    raise ValueError(f"Unsupported categorical null strategy: {strategy}")


def _fill_numeric(series: pd.Series, strategy: str) -> pd.Series:
    if strategy == "ffill":
        return series.ffill()
    if strategy == "ffill_bfill":
        return series.ffill().bfill()
    if strategy == "median":
        if series.dropna().empty:
            return series
        return series.fillna(series.median())
    if strategy == "mean":
        if series.dropna().empty:
            return series
        return series.fillna(series.mean())
    if strategy == "zero":
        return series.fillna(0.0)
    if strategy == "drop_rows":
        return series
    raise ValueError(f"Unsupported numeric null strategy: {strategy}")


def _apply_null_handling(
    df: pd.DataFrame,
    *,
    protected_columns: list[str],
    numeric_strategy: str,
    categorical_strategy: str,
) -> pd.DataFrame:
    transformed = df.copy()
    protected = set(protected_columns)
    for column in transformed.columns:
        if column in protected:
            continue
        if pd.api.types.is_numeric_dtype(transformed[column]):
            transformed[column] = _fill_numeric(transformed[column], numeric_strategy)
        else:
            transformed[column] = _fill_categorical(transformed[column], categorical_strategy)
    return transformed


def _add_date_features(df: pd.DataFrame, datetime_column: str, config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("enabled", False):
        return df

    enriched = df.copy()
    dates = pd.to_datetime(enriched[datetime_column], errors="coerce")
    if config.get("add_year", True):
        enriched["date_year"] = dates.dt.year
    if config.get("add_month", True):
        enriched["date_month"] = dates.dt.month
    if config.get("add_quarter", True):
        enriched["date_quarter"] = dates.dt.quarter
    if config.get("add_week_of_year", True):
        enriched["date_week_of_year"] = dates.dt.isocalendar().week.astype("Int64")
    return enriched


def _add_lag_features(df: pd.DataFrame, config: dict[str, Any], logger) -> pd.DataFrame:
    lag_config = config.get("lag_features", {})
    if not lag_config.get("enabled", False):
        return df

    lags = lag_config.get("lags", [])
    columns = lag_config.get("columns", [])
    enriched = df.copy()
    for column in columns:
        if column not in enriched.columns:
            logger.warning("Lag feature skipped because column is missing: %s", column)
            continue
        for lag in lags:
            enriched[f"{column}_lag_{lag}"] = pd.to_numeric(enriched[column], errors="coerce").shift(int(lag))
    return enriched


def _add_rolling_features(df: pd.DataFrame, config: dict[str, Any], logger) -> pd.DataFrame:
    rolling_config = config.get("rolling_features", {})
    if not rolling_config.get("enabled", False):
        return df

    statistics = rolling_config.get("statistics", ["mean"])
    windows = rolling_config.get("windows", [])
    columns = rolling_config.get("columns", [])
    enriched = df.copy()
    for column in columns:
        if column not in enriched.columns:
            logger.warning("Rolling feature skipped because column is missing: %s", column)
            continue
        numeric = pd.to_numeric(enriched[column], errors="coerce")
        for window in windows:
            roll = numeric.rolling(window=int(window), min_periods=1)
            if "mean" in statistics:
                enriched[f"{column}_roll_mean_{window}"] = roll.mean()
            if "std" in statistics:
                enriched[f"{column}_roll_std_{window}"] = roll.std()
    return enriched


def _add_interaction_features(df: pd.DataFrame, config: dict[str, Any], logger) -> pd.DataFrame:
    interaction_config = config.get("interaction_features", {})
    if not interaction_config.get("enabled", False):
        return df

    enriched = df.copy()
    for definition in interaction_config.get("definitions", []):
        name = definition.get("name")
        operation = definition.get("operation")
        if not name or not operation:
            logger.warning("Skipping malformed interaction feature definition: %s", definition)
            continue

        if operation == "difference":
            left = definition.get("left")
            right = definition.get("right")
            if left not in enriched.columns or right not in enriched.columns:
                logger.warning("Skipping difference feature %s because inputs are missing.", name)
                continue
            enriched[name] = pd.to_numeric(enriched[left], errors="coerce") - pd.to_numeric(enriched[right], errors="coerce")
        elif operation == "ratio":
            numerator = definition.get("numerator")
            denominator = definition.get("denominator")
            if numerator not in enriched.columns or denominator not in enriched.columns:
                logger.warning("Skipping ratio feature %s because inputs are missing.", name)
                continue
            denominator_series = pd.to_numeric(enriched[denominator], errors="coerce").replace(0, np.nan)
            enriched[name] = pd.to_numeric(enriched[numerator], errors="coerce") / denominator_series
        else:
            logger.warning("Skipping unsupported interaction operation=%s for feature=%s", operation, name)
    return enriched


def _encode_categorical_features(
    df: pd.DataFrame,
    *,
    protected_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    protected = set(protected_columns)
    categorical_columns = [
        column
        for column in df.columns
        if column not in protected
        and (
            pd.api.types.is_object_dtype(df[column])
            or pd.api.types.is_string_dtype(df[column])
            or pd.api.types.is_categorical_dtype(df[column])
            or pd.api.types.is_bool_dtype(df[column])
        )
    ]
    if not categorical_columns:
        return df, []

    encoded = pd.get_dummies(
        df[categorical_columns],
        prefix_sep="__",
        dtype=float,
    )
    encoded = pd.concat([df, encoded], axis=1)
    return encoded, categorical_columns


def prepare_modeling_frame(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    include_target: bool,
    logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform a raw tabular dataset into a modeling-ready frame."""
    data_cfg = config["data_processing"]
    dataset_cfg = data_cfg["dataset"]
    preprocessing_cfg = data_cfg.get("preprocessing", {})
    engineering_cfg = data_cfg.get("feature_engineering", {})

    datetime_column = dataset_cfg["datetime_column"]
    feature_columns = list(dataset_cfg.get("feature_columns", []))
    drop_columns = list(dataset_cfg.get("drop_columns", []))
    target_column = dataset_cfg["target_column"]
    target_horizon = int(dataset_cfg.get("target_horizon", 0))
    target_output_column = dataset_cfg.get(
        "target_output_column",
        target_column if target_horizon == 0 else f"{target_column}_t_plus_{target_horizon}",
    )
    sort_columns = list(dataset_cfg.get("sort_columns", [datetime_column]))

    required_columns = [datetime_column, target_column, *feature_columns]
    _validate_required_columns(df, required_columns, context="data_processing")

    frame = df.copy()
    frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns], errors="ignore")
    frame[datetime_column] = pd.to_datetime(frame[datetime_column], errors="coerce")
    if preprocessing_cfg.get("remove_rows_missing_datetime", True):
        frame = frame[frame[datetime_column].notna()].copy()

    if preprocessing_cfg.get("drop_duplicates", True):
        frame = frame.drop_duplicates(subset=sort_columns, keep="last")

    frame = _sort_frame(frame, sort_columns)
    frame, removed_outliers = _apply_outlier_filters(frame, preprocessing_cfg.get("outlier_filters", {}), logger)

    working_columns = [datetime_column, *sorted(set(feature_columns + [target_column]))]
    frame = frame[working_columns].copy()
    frame = _apply_null_handling(
        frame,
        protected_columns=[datetime_column],
        numeric_strategy=preprocessing_cfg.get("numeric_null_strategy", "ffill"),
        categorical_strategy=preprocessing_cfg.get("categorical_null_strategy", "most_frequent"),
    )

    engineered = _add_date_features(frame, datetime_column, engineering_cfg.get("date_features", {}))
    engineered = _add_lag_features(engineered, engineering_cfg, logger)
    engineered = _add_rolling_features(engineered, engineering_cfg, logger)
    engineered = _add_interaction_features(engineered, engineering_cfg, logger)

    if include_target:
        if target_horizon > 0:
            engineered[target_output_column] = pd.to_numeric(engineered[target_column], errors="coerce").shift(-target_horizon)
        else:
            engineered[target_output_column] = pd.to_numeric(engineered[target_column], errors="coerce")

    protected_columns = [datetime_column, target_column]
    if include_target:
        protected_columns.append(target_output_column)
    engineered, encoded_categorical_columns = _encode_categorical_features(
        engineered,
        protected_columns=list(dict.fromkeys(protected_columns)),
    )

    candidate_feature_columns = [
        column
        for column in engineered.columns
        if column not in {datetime_column, target_output_column}
    ]

    if preprocessing_cfg.get("drop_rows_with_missing_features", True):
        allowed_missing_features = set(preprocessing_cfg.get("allowed_missing_feature_columns", []))
        drop_subset = [column for column in candidate_feature_columns if column not in allowed_missing_features]
        if include_target and preprocessing_cfg.get("remove_rows_missing_target", True):
            drop_subset.append(target_output_column)
        engineered = engineered.dropna(subset=drop_subset).reset_index(drop=True)
    elif include_target and preprocessing_cfg.get("remove_rows_missing_target", True):
        engineered = engineered[engineered[target_output_column].notna()].reset_index(drop=True)

    summary = {
        "rows": int(len(engineered)),
        "feature_columns": candidate_feature_columns,
        "target_output_column": target_output_column,
        "target_column": target_column,
        "datetime_column": datetime_column,
        "target_horizon": target_horizon,
        "removed_outliers": removed_outliers,
        "raw_feature_columns": feature_columns,
        "encoded_categorical_columns": encoded_categorical_columns,
        "allowed_missing_feature_columns": list(preprocessing_cfg.get("allowed_missing_feature_columns", [])),
    }
    return engineered, summary


def _split_dataset(df: pd.DataFrame, config: dict[str, Any], logger) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    split_cfg = config["data_processing"]["split"]
    strategy = split_cfg.get("strategy", "chronological")
    train_size = float(split_cfg.get("train_size", 0.7))
    valid_size = float(split_cfg.get("valid_size", 0.15))
    test_size = float(split_cfg.get("test_size", 0.15))
    total_share = train_size + valid_size + test_size
    if not np.isclose(total_share, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0. Current total={total_share}")

    if df.empty:
        raise ValueError("Processed dataset is empty after cleaning and feature engineering.")

    if strategy == "chronological":
        total_rows = len(df)
        train_end = max(1, int(total_rows * train_size))
        valid_end = train_end + int(total_rows * valid_size)

        train_df = df.iloc[:train_end].copy()
        valid_df = df.iloc[train_end:valid_end].copy()
        test_df = df.iloc[valid_end:].copy()
    elif strategy == "random":
        seed = int(config.get("project", {}).get("seed", 42))
        train_df, holdout_df = train_test_split(df, train_size=train_size, random_state=seed, shuffle=True)
        if valid_size == 0:
            valid_df = df.iloc[0:0].copy()
            test_df = holdout_df.copy()
        else:
            valid_share_within_holdout = valid_size / (valid_size + test_size)
            valid_df, test_df = train_test_split(
                holdout_df,
                train_size=valid_share_within_holdout,
                random_state=seed,
                shuffle=True,
            )
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    for split_name, split_df in {"train": train_df, "valid": valid_df, "test": test_df}.items():
        logger.info("Split %s rows=%s", split_name, len(split_df))

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def _run_optional_existing_ingestion(config: dict[str, Any], repo_root: Path, logger) -> None:
    refresh_cfg = config["data_processing"].get("source_refresh", {})
    if refresh_cfg.get("run_ine_ingestion", False):
        logger.info("Refreshing INE processed sources before modeling.")
        run_ine_ingestion_pipeline(
            start_date=refresh_cfg.get("start_date"),
            end_date=refresh_cfg.get("end_date"),
            repo_root=repo_root,
            force_download=bool(refresh_cfg.get("force_download", False)),
            logger=logger,
        )
    run_external_context = bool(
        refresh_cfg.get(
            "run_external_context_pipeline",
            refresh_cfg.get("run_cu04_external_pipeline", False),
        )
    )
    if run_external_context:
        logger.info("Refreshing external context before modeling.")
        context_cfg = config.get("context_proxy", {})
        weekly_context_path = resolve_repo_path(
            context_cfg.get("weekly_context_path", "data/processed/external/context/context_weekly_for_simulation.csv"),
            repo_root,
        )
        run_external_context_pipeline(
            start_date=refresh_cfg.get("start_date", "2004-01-01"),
            end_date=refresh_cfg.get("end_date", "today"),
            force_download=bool(refresh_cfg.get("force_download", False)),
            raw_dir=repo_root / "data" / "raw" / "external",
            proc_dir=weekly_context_path.parent,
            logger=logger,
        )


def _ensure_external_context_inputs(config: dict[str, Any], repo_root: Path, logger) -> None:
    refresh_cfg = config.get("data_processing", {}).get("source_refresh", {})
    paths_to_check = [
        config.get("paths", {}).get("input_data_path"),
        config.get("context_proxy", {}).get("weekly_context_path"),
        config.get("context_proxy", {}).get("wide_weekly_path"),
        config.get("context_proxy", {}).get("wide_monthly_path"),
        config.get("synthetic_data", {}).get("input_dataset_path"),
        config.get("feature_selection", {}).get("fallback_input_dataset_path"),
    ]
    resolved_paths = [
        resolve_repo_path(path_value, repo_root)
        for path_value in paths_to_check
        if path_value
    ]
    missing_paths = [path for path in resolved_paths if not path.exists()]
    if not missing_paths:
        return

    raw_external_dir = repo_root / "data" / "raw" / "external"
    if not raw_external_dir.exists():
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Processed external context inputs are missing and the raw external cache is unavailable. "
            f"Missing paths: {missing_display}"
        )

    weekly_context_path = resolve_repo_path(
        config.get("context_proxy", {}).get("weekly_context_path", "data/processed/external/context/context_weekly_for_simulation.csv"),
        repo_root,
    )
    logger.info(
        "Processed external context is incomplete; rebuilding active context inputs. Missing=%s",
        [str(path) for path in missing_paths],
    )
    run_external_context_pipeline(
        start_date=refresh_cfg.get("start_date", "2004-01-01"),
        end_date=refresh_cfg.get("end_date", "today"),
        force_download=bool(refresh_cfg.get("force_download", False)),
        raw_dir=raw_external_dir,
        proc_dir=weekly_context_path.parent,
        logger=logger,
    )


def _feature_set_mapping_from_config(
    feature_cfg: dict[str, Any],
    *,
    available_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    feature_sets: dict[str, list[str]] = {}
    configured_sets = feature_cfg.get("feature_sets", {})
    if isinstance(configured_sets, dict):
        for set_name, entries in configured_sets.items():
            if not isinstance(entries, list):
                continue
            feature_sets[str(set_name)] = [str(entry) for entry in entries if isinstance(entry, str)]

    minimal_features = [str(feature) for feature in feature_cfg.get("selected_features_minimal", []) if isinstance(feature, str)]
    extended_features = [
        str(feature)
        for feature in feature_cfg.get("selected_features_extended", feature_cfg.get("selected_features", []))
        if isinstance(feature, str)
    ]
    if minimal_features:
        feature_sets.setdefault("minimal", minimal_features)
    if extended_features:
        feature_sets.setdefault("extended", extended_features)
    if available_columns is not None:
        available = set(available_columns)
        feature_sets = {
            name: [feature for feature in features if feature in available]
            for name, features in feature_sets.items()
        }
    return {name: list(dict.fromkeys(features)) for name, features in feature_sets.items() if features}


def _feature_tokens_from_config(feature_cfg: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    keys = [
        "selected_features_minimal",
        "selected_features_extended",
        "selected_features",
        "review_features",
        "review_pool_features",
    ]
    for key in keys:
        for entry in feature_cfg.get(key, []):
            if isinstance(entry, str):
                tokens.append(entry)
            elif isinstance(entry, dict):
                feature_name = entry.get("feature_name") or entry.get("name")
                if feature_name:
                    tokens.append(str(feature_name))
    for feature_list in _feature_set_mapping_from_config(feature_cfg).values():
        tokens.extend(feature_list)
    return list(dict.fromkeys(tokens))


def _collect_engineering_bases(feature_tokens: list[str]) -> tuple[list[str], list[str]]:
    lag_columns: list[str] = []
    rolling_columns: list[str] = []
    for token in feature_tokens:
        if "_lag_" in token:
            lag_columns.append(token.split("_lag_", 1)[0])
        if "_roll_" in token:
            rolling_columns.append(token.split("_roll_", 1)[0])
    return list(dict.fromkeys(lag_columns)), list(dict.fromkeys(rolling_columns))


def _build_prepared_feature_config(
    config: dict[str, Any],
    source_df: pd.DataFrame,
) -> dict[str, Any]:
    prepared_config = deepcopy(config)
    feature_cfg = dict(prepared_config.get("feature_selection", {}))
    dataset_cfg = prepared_config["data_processing"]["dataset"]
    engineering_cfg = prepared_config["data_processing"].get("feature_engineering", {})

    feature_tokens = _feature_tokens_from_config(feature_cfg)
    lag_columns, rolling_columns = _collect_engineering_bases(feature_tokens)

    target_column = str(
        feature_cfg.get("target_primary")
        or feature_cfg.get("target")
        or prepared_config.get("procurement_problem_definition", {}).get("target_column")
        or "synthetic_procurement_need"
    )
    datetime_column = str(dataset_cfg.get("datetime_column", "date"))
    dataset_cfg["feature_columns"] = [column for column in source_df.columns.tolist() if column != datetime_column]
    dataset_cfg["target_column"] = target_column
    dataset_cfg["target_horizon"] = 0
    dataset_cfg["target_output_column"] = target_column
    dataset_cfg["sort_columns"] = [datetime_column]

    lag_cfg = engineering_cfg.setdefault("lag_features", {})
    lag_cfg["enabled"] = bool(lag_columns)
    lag_cfg["columns"] = [column for column in lag_columns if column in source_df.columns]

    rolling_cfg = engineering_cfg.setdefault("rolling_features", {})
    rolling_cfg["enabled"] = bool(rolling_columns)
    rolling_cfg["columns"] = [column for column in rolling_columns if column in source_df.columns]

    preprocessing_cfg = prepared_config["data_processing"].setdefault("preprocessing", {})
    existing_allowed_missing = list(preprocessing_cfg.get("allowed_missing_feature_columns", []))
    supply_missing_features = ["supply_index", "demand_supply_gap", "demand_supply_ratio"]
    for lag in lag_cfg.get("lags", []):
        supply_missing_features.append(f"supply_index_lag_{lag}")
    for window in rolling_cfg.get("windows", []):
        if "mean" in rolling_cfg.get("statistics", ["mean"]):
            supply_missing_features.append(f"supply_index_roll_mean_{window}")
        if "std" in rolling_cfg.get("statistics", ["mean"]):
            supply_missing_features.append(f"supply_index_roll_std_{window}")
    preprocessing_cfg["allowed_missing_feature_columns"] = list(
        dict.fromkeys([*existing_allowed_missing, *supply_missing_features])
    )

    return prepared_config


def _resolve_feature_source_frame(
    config: dict[str, Any],
    *,
    repo_root: Path,
    synthetic_output_df: pd.DataFrame | None,
    raw_input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    feature_cfg = config.get("feature_selection", {})
    if feature_cfg.get("prefer_synthetic_input", False):
        if synthetic_output_df is not None:
            synthetic_input_path = feature_cfg.get("input_dataset_path")
            if synthetic_input_path:
                return synthetic_output_df.copy(), _portable_path(resolve_repo_path(synthetic_input_path, repo_root), repo_root)
            return synthetic_output_df.copy(), "synthetic_output_in_memory"
        synthetic_input_path = feature_cfg.get("input_dataset_path")
        if synthetic_input_path:
            resolved_path = resolve_repo_path(synthetic_input_path, repo_root)
            if resolved_path.exists():
                return pd.read_csv(resolved_path), _portable_path(resolved_path, repo_root)

    fallback_path_value = feature_cfg.get("fallback_input_dataset_path")
    if fallback_path_value:
        fallback_path = resolve_repo_path(fallback_path_value, repo_root)
        if fallback_path.exists():
            return pd.read_csv(fallback_path), _portable_path(fallback_path, repo_root)
    return raw_input_df.copy(), "raw_input_fallback"


def _export_prepared_feature_artifacts(
    config: dict[str, Any],
    *,
    repo_root: Path,
    source_df: pd.DataFrame,
    source_path_label: str,
    logger,
) -> dict[str, Any]:
    feature_cfg = dict(config.get("feature_selection", {}))
    if not feature_cfg:
        return {}
    runtime_recipe_context = current_recipe_context(config)

    prepared_config = _build_prepared_feature_config(config, source_df)
    prepared_df, prepared_summary = prepare_modeling_frame(
        source_df,
        prepared_config,
        include_target=False,
        logger=logger,
    )

    prepared_dataset_path = resolve_repo_path(feature_cfg["prepared_dataset_path"], repo_root)
    feature_contract_path = resolve_repo_path(feature_cfg["feature_contract_path"], repo_root)
    feature_roles_metadata_path = resolve_repo_path(feature_cfg["feature_roles_metadata_path"], repo_root)
    feature_config_export_path = resolve_repo_path(feature_cfg["feature_config_export_path"], repo_root)
    feature_catalog_path_value = feature_cfg.get("feature_catalog_path")
    feature_catalog_path = resolve_repo_path(feature_catalog_path_value, repo_root) if feature_catalog_path_value else None

    prepared_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(prepared_dataset_path, index=False)

    feature_contract_df = build_feature_contract_from_config(
        config,
        modeling_df=prepared_df,
    )
    feature_contract_df.to_csv(feature_contract_path, index=False)

    feature_roles_metadata = build_feature_roles_metadata_from_config(
        config,
        feature_contract_df,
        dataset_path=_portable_path(prepared_dataset_path, repo_root),
        available_columns=prepared_df.columns.tolist(),
    )
    feature_roles_metadata["source_dataset_path"] = source_path_label
    feature_roles_metadata["prepared_rows"] = int(len(prepared_df))
    feature_roles_metadata["prepared_columns"] = int(len(prepared_df.columns))
    feature_roles_metadata["datetime_column"] = prepared_summary["datetime_column"]
    feature_roles_metadata["recipe_context"] = runtime_recipe_context
    write_json(feature_roles_metadata_path, feature_roles_metadata)

    feature_set_mapping = _feature_set_mapping_from_config(
        feature_cfg,
        available_columns=prepared_df.columns.tolist(),
    )
    feature_export_payload = {
        "target": feature_roles_metadata["target_column"],
        "target_primary": feature_roles_metadata["target_column"],
        "target_alternatives": feature_roles_metadata["target_alternatives"],
        "selected_features_minimal": feature_roles_metadata["official_minimal_inputs"],
        "selected_features_extended": feature_roles_metadata["official_extended_inputs"],
        "selected_features": feature_roles_metadata["official_extended_inputs"],
        "feature_sets": feature_set_mapping,
        "recommended_feature_set": str(feature_cfg.get("recommended_feature_set", "extended")),
        "review_features": feature_roles_metadata["review_pool_features"],
        "review_pool_features": feature_roles_metadata["review_pool_features"],
        "excluded_features": feature_contract_df[
            feature_contract_df["official_baseline_status"].astype(str).eq("drop")
        ]["feature_name"].tolist(),
        "prepared_dataset_path": _portable_path(prepared_dataset_path, repo_root),
        "feature_contract_path": _portable_path(feature_contract_path, repo_root),
        "feature_roles_metadata_path": _portable_path(feature_roles_metadata_path, repo_root),
        "input_dataset_mode": source_path_label,
        "input_dataset_path": source_path_label,
        "recipe_context": runtime_recipe_context,
    }
    feature_config_export_path.parent.mkdir(parents=True, exist_ok=True)
    feature_config_export_path.write_text(
        yaml.safe_dump(feature_export_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    if feature_catalog_path is not None:
        feature_catalog_path.parent.mkdir(parents=True, exist_ok=True)
        feature_contract_df.to_csv(feature_catalog_path, index=False)

    logger.info("Saved prepared modeling dataset to %s", prepared_dataset_path)
    logger.info("Saved feature contract to %s", feature_contract_path)
    logger.info("Saved feature roles metadata to %s", feature_roles_metadata_path)
    logger.info("Saved feature selection export to %s", feature_config_export_path)

    return {
        "prepared_dataset_path": _portable_path(prepared_dataset_path, repo_root),
        "feature_contract_path": _portable_path(feature_contract_path, repo_root),
        "feature_roles_metadata_path": _portable_path(feature_roles_metadata_path, repo_root),
        "feature_config_export_path": _portable_path(feature_config_export_path, repo_root),
        "feature_catalog_path": _portable_path(feature_catalog_path, repo_root) if feature_catalog_path is not None else None,
        "prepared_rows": int(len(prepared_df)),
        "prepared_columns": int(len(prepared_df.columns)),
        "recipe_context": runtime_recipe_context,
    }


def run_data_processing(config: dict[str, Any], logger) -> dict[str, Any]:
    """Execute the data processing stage and persist processed outputs."""
    config = ensure_runtime_context_resolved(config)
    repo_root = Path(config["project"]["repo_root"])
    paths_cfg = config["paths"]
    _run_optional_existing_ingestion(config, repo_root, logger)
    _ensure_external_context_inputs(config, repo_root, logger)
    _bootstrap_context_proxy_inputs(config, repo_root, logger)

    input_path = resolve_repo_path(paths_cfg["input_data_path"], repo_root)
    processed_dataset_path = resolve_repo_path(paths_cfg["processed_dataset_path"], repo_root)
    processed_metadata_path = resolve_repo_path(paths_cfg["processed_metadata_path"], repo_root)
    splits_dir = ensure_directory(resolve_repo_path(paths_cfg["splits_dir"], repo_root))

    input_file_type = config["data_processing"].get("io", {}).get("input_file_type")
    runtime_recipe_context = current_recipe_context(config)
    logger.info(
        "Data processing runtime selection_mode=%s mode_resolution=%s scope_token=%s recipe_profile=%s",
        runtime_recipe_context.get("selection_mode"),
        runtime_recipe_context.get("mode_resolution"),
        runtime_recipe_context.get("scope_token"),
        runtime_recipe_context.get("recipe_profile"),
    )
    raw_df = read_tabular(input_path, file_type=input_file_type)
    logger.info("Loaded input data path=%s rows=%s cols=%s", input_path, len(raw_df), len(raw_df.columns))

    processed_df, summary = prepare_modeling_frame(raw_df, config, include_target=True, logger=logger)
    logger.info("Processed modeling dataset rows=%s features=%s", len(processed_df), len(summary["feature_columns"]))

    processed_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_dataset_path, index=False)

    split_frames = _split_dataset(processed_df, config, logger)
    split_paths: dict[str, str] = {}
    for split_name, split_df in split_frames.items():
        split_path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths[split_name] = _portable_path(split_path, repo_root)

    metadata = {
        "created_at_utc": utc_timestamp(),
        "input_data_path": _portable_path(input_path, repo_root),
        "processed_dataset_path": _portable_path(processed_dataset_path, repo_root),
        "processed_rows": int(len(processed_df)),
        "split_paths": split_paths,
        "split_rows": {name: int(len(frame)) for name, frame in split_frames.items()},
        "recipe_context": runtime_recipe_context,
        **summary,
    }
    write_json(processed_metadata_path, metadata)
    logger.info("Saved processed dataset to %s", processed_dataset_path)
    logger.info("Saved processing metadata to %s", processed_metadata_path)

    synthetic_output_df: pd.DataFrame | None = None
    synthetic_output_paths: dict[str, str] = {}
    synthetic_cfg = dict(config.get("synthetic_data", {}))
    if synthetic_cfg.get("enabled", False):
        synthetic_input_path_value = synthetic_cfg.get("input_dataset_path")
        if synthetic_input_path_value:
            synthetic_input_path = resolve_repo_path(synthetic_input_path_value, repo_root)
            synthetic_input_df = read_tabular(synthetic_input_path, file_type=input_file_type)
        else:
            synthetic_input_path = input_path
            synthetic_input_df = raw_df.copy()
        logger.info(
            "Building synthetic plant dataset from path=%s rows=%s cols=%s",
            synthetic_input_path,
            len(synthetic_input_df),
            len(synthetic_input_df.columns),
        )
        synthetic_output_df, lineage_df, assumptions_payload = build_synthetic_plant_dataset(
            synthetic_input_df,
            synthetic_cfg,
            recipe_runtime_context=runtime_recipe_context,
            recipe_registry_path=config.get("manufacturing_profiles", {}).get("registry_path"),
        )
        synthetic_output_paths = write_synthetic_plant_outputs(
            synthetic_output_df,
            lineage_df,
            assumptions_payload,
            synthetic_cfg,
            repo_root,
        )
        logger.info("Saved synthetic plant outputs: %s", synthetic_output_paths)

    feature_source_df, feature_source_label = _resolve_feature_source_frame(
        config,
        repo_root=repo_root,
        synthetic_output_df=synthetic_output_df,
        raw_input_df=raw_df,
    )
    feature_source_df = filter_frame_to_recipe(
        feature_source_df,
        config,
        stage_name="data_processing.prepared_features",
        logger=logger,
    )
    prepared_feature_outputs = _export_prepared_feature_artifacts(
        config,
        repo_root=repo_root,
        source_df=feature_source_df,
        source_path_label=feature_source_label,
        logger=logger,
    )

    return {
        "processed_dataset_path": _portable_path(processed_dataset_path, repo_root),
        "processed_metadata_path": _portable_path(processed_metadata_path, repo_root),
        "split_paths": split_paths,
        "processed_rows": int(len(processed_df)),
        "feature_columns": summary["feature_columns"],
        "target_output_column": summary["target_output_column"],
        "synthetic_outputs": synthetic_output_paths,
        "prepared_feature_outputs": prepared_feature_outputs,
        "recipe_context": runtime_recipe_context,
        "runtime_context": runtime_recipe_context,
    }
