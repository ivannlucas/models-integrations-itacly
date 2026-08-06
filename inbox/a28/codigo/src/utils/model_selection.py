"""Deterministic model-selection helpers shared by training and stats stages."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def canonical_target_column(config: dict[str, Any]) -> str | None:
    """Return the canonical target column used for operational selection."""
    procurement_cfg = dict(config.get("procurement_problem_definition", {}))
    if procurement_cfg.get("target_column"):
        return str(procurement_cfg["target_column"])

    target_roles = dict(config.get("synthetic_data", {}).get("target_roles", {}))
    if target_roles.get("canonical_target_column"):
        return str(target_roles["canonical_target_column"])
    return None


def selection_policy_description(*, primary_metric: str, canonical_target: str | None) -> dict[str, Any]:
    """Describe the selection semantics exposed in training and stats summaries."""
    return {
        "canonical_target_column": canonical_target,
        "primary_metric": primary_metric,
        "best_run": (
            "Best run within the active scope after prioritising the canonical target column when it is available."
        ),
        "best_run_global": "Best run within the active scope across all targets.",
        "best_baseline_run": "Best baseline-category run for the canonical target within the active scope.",
        "best_neuroevolution_run": "Best neuroevolution-category run for the canonical target within the active scope.",
        "baseline_reference_run": (
            "Configured baseline reference used for neuroevolution comparison and downstream policy simulation."
        ),
        "best_by_target": "Per-target winners resolved with the same deterministic ranking rules.",
        "sort_order": [
            {
                "field": primary_metric,
                "direction": "desc" if str(primary_metric).endswith("_r2") else "asc",
            },
            {"field": "test_r2", "direction": "desc"},
            {"field": "abs(test_prediction_bias)", "direction": "asc"},
            {"field": "test_underprediction_rate", "direction": "asc"},
            {"field": "trained_at_utc", "direction": "desc"},
            {"field": "run_id", "direction": "asc"},
        ],
    }


def _prepare_selection_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    prepared = summary_df.copy()
    if "test_prediction_bias" in prepared.columns:
        prepared["_selection_abs_test_prediction_bias"] = pd.to_numeric(
            prepared["test_prediction_bias"],
            errors="coerce",
        ).abs()
    else:
        prepared["_selection_abs_test_prediction_bias"] = np.nan

    if "test_underprediction_rate" in prepared.columns:
        prepared["test_underprediction_rate"] = pd.to_numeric(
            prepared["test_underprediction_rate"],
            errors="coerce",
        )
    if "test_r2" in prepared.columns:
        prepared["test_r2"] = pd.to_numeric(prepared["test_r2"], errors="coerce")
    if "test_rmse" in prepared.columns:
        prepared["test_rmse"] = pd.to_numeric(prepared["test_rmse"], errors="coerce")

    prepared["_selection_trained_at_utc"] = prepared.get("trained_at_utc", "").fillna("").astype(str)
    prepared["_selection_run_id"] = prepared.get("run_id", "").fillna("").astype(str)
    return prepared


def _selection_sort_columns(summary_df: pd.DataFrame, primary_metric: str) -> tuple[list[str], list[bool]]:
    sort_columns: list[str] = []
    ascending: list[bool] = []

    def add(column_name: str, is_ascending: bool) -> None:
        if column_name in summary_df.columns and column_name not in sort_columns:
            sort_columns.append(column_name)
            ascending.append(is_ascending)

    add(primary_metric, not str(primary_metric).endswith("_r2"))
    if primary_metric != "test_rmse":
        add("test_rmse", True)
    if primary_metric != "test_r2":
        add("test_r2", False)
    add("_selection_abs_test_prediction_bias", True)
    add("test_underprediction_rate", True)
    add("_selection_trained_at_utc", False)
    add("_selection_run_id", True)
    return sort_columns, ascending


def _ordered_selection_frame(summary_df: pd.DataFrame, primary_metric: str) -> pd.DataFrame:
    prepared = _prepare_selection_frame(summary_df)
    sort_columns, ascending = _selection_sort_columns(prepared, primary_metric)
    if not sort_columns:
        return prepared.reset_index(drop=True)
    return prepared.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)


def _drop_selection_helpers(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    for helper_key in ["_selection_abs_test_prediction_bias", "_selection_trained_at_utc", "_selection_run_id"]:
        cleaned.pop(helper_key, None)
    return cleaned


def select_best_record(
    summary_df: pd.DataFrame,
    *,
    primary_metric: str,
    target_column: str | None = None,
    run_category: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if summary_df.empty:
        return None

    filtered = summary_df.copy()
    if target_column and "target_column" in filtered.columns:
        filtered = filtered[filtered["target_column"] == target_column]
    if run_category and "run_category" in filtered.columns:
        filtered = filtered[filtered["run_category"] == run_category]
    for column_name, value in (filters or {}).items():
        if value is not None and column_name in filtered.columns:
            filtered = filtered[filtered[column_name] == value]
    if filtered.empty:
        return None

    ordered = _ordered_selection_frame(filtered, primary_metric)
    if ordered.empty:
        return None
    return _drop_selection_helpers(ordered.iloc[0].to_dict())


def select_preferred_record(
    summary_df: pd.DataFrame,
    *,
    primary_metric: str,
    canonical_target: str | None,
    run_category: str | None = None,
) -> dict[str, Any] | None:
    preferred = select_best_record(
        summary_df,
        primary_metric=primary_metric,
        target_column=canonical_target,
        run_category=run_category,
    )
    if preferred is not None:
        return preferred
    return select_best_record(summary_df, primary_metric=primary_metric, run_category=run_category)


def resolve_reference_record(
    summary_df: pd.DataFrame,
    *,
    primary_metric: str,
    canonical_target: str | None,
    run_category: str | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    filtered_reference = select_best_record(
        summary_df,
        primary_metric=primary_metric,
        target_column=canonical_target,
        run_category=run_category,
        filters=filters,
    )
    if filtered_reference is not None:
        return filtered_reference

    filtered_reference = select_best_record(
        summary_df,
        primary_metric=primary_metric,
        run_category=run_category,
        filters=filters,
    )
    if filtered_reference is not None:
        return filtered_reference

    return select_preferred_record(
        summary_df,
        primary_metric=primary_metric,
        canonical_target=canonical_target,
        run_category=run_category,
    )


def build_best_by_target(summary_df: pd.DataFrame, *, primary_metric: str) -> list[dict[str, Any]]:
    if summary_df.empty or "target_column" not in summary_df.columns:
        return []

    target_values = [
        target_column
        for target_column in summary_df["target_column"].tolist()
        if target_column is not None and not (isinstance(target_column, float) and pd.isna(target_column))
    ]
    ordered_targets = list(dict.fromkeys(target_values))

    best_rows: list[dict[str, Any]] = []
    for target_column in ordered_targets:
        target_df = summary_df[summary_df["target_column"] == target_column]
        best_run = select_best_record(target_df, primary_metric=primary_metric)
        if best_run is None:
            continue
        best_rows.append(
            {
                "target_column": target_column,
                "target_role": best_run.get("target_role"),
                "best_run": best_run,
                "best_baseline_run": select_best_record(
                    target_df,
                    primary_metric=primary_metric,
                    run_category="baseline",
                ),
                "best_neuroevolution_run": select_best_record(
                    target_df,
                    primary_metric=primary_metric,
                    run_category="neuroevolution",
                ),
            }
        )
    return best_rows
