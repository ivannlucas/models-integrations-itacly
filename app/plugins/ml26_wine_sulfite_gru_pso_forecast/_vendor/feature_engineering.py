"""Feature engineering for the a26 sulfite model (vendored from
src/data_processing/preprocessing/feature_engineering.py — logic unchanged).

Builds the 22 model features (derived lab freshness, SO2 addition traceability, observed stage
progress, one-hot categoricals) and, for training, the two 72 h targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.plugins.ml26_wine_sulfite_gru_pso_forecast._vendor.enology import (
    estimate_underprotection_risk,
    target_free_sulfite,
)

INTERNAL_COLUMNS = ["lot_id", "timestamp", "timestamp_index"]

OBSERVABLE_NUMERIC_COLUMNS = [
    "volume_l",
    "ambient_temp_c",
    "stage_progress",
    "hours_since_last_lab",
    "last_so2_addition_mg_l",
    "hours_since_last_addition",
    "temperature_c",
    "dissolved_oxygen_mg_l",
    "density_g_ml",
    "co2_g_l",
    "ph_lab",
    "free_sulfite_lab",
    "total_sulfite_lab",
]

AUXILIARY_NUMERIC_COLUMNS = [
    "target_free_sulfite",
]

OBSERVABLE_FLAG_COLUMNS = [
    "lab_sample",
]

DEFAULT_CATEGORICAL_COLUMNS = ["wine_type", "stage"]
DEFAULT_CATEGORY_LEVELS = {
    "wine_type": ["red", "rose", "white"],
    "stage": ["fermentation", "must", "pre_bottling", "stabilization", "storage"],
}


def target_columns_for_horizon(horizon_hours: float | int) -> list[str]:
    """Return the two target column names for a decision horizon in hours."""
    horizon = int(round(float(horizon_hours)))
    return [f"future_free_sulfite_{horizon}h", f"underprotection_risk_{horizon}h"]


def target_columns_for_config(config: dict[str, Any]) -> list[str]:
    """Return the target column names implied by config.preprocessing.decision_horizon_hours."""
    horizon_hours = float(config.get("preprocessing", {}).get("decision_horizon_hours", 24))
    return target_columns_for_horizon(horizon_hours)


def _ensure_required_columns(  # pylint: disable=too-many-branches  # vendored as-is
    df: pd.DataFrame, categorical_columns: list[str], target_columns: list[str]
) -> pd.DataFrame:
    frame = df.copy()
    if "actual_dose_mg_l" not in frame.columns and "dose_mg_l" in frame.columns:
        frame["actual_dose_mg_l"] = frame["dose_mg_l"]
    if "lot_id" not in frame.columns:
        frame["lot_id"] = "inference_lot"
    if "timestamp" not in frame.columns:
        frame["timestamp"] = pd.RangeIndex(start=0, stop=len(frame), step=1)
    if "timestamp_index" not in frame.columns:
        frame["timestamp_index"] = np.arange(len(frame), dtype=int)
    if "target_only" not in frame.columns:
        frame["target_only"] = False
    frame["target_only"] = frame["target_only"].fillna(False).astype(bool)

    for column in ("free_sulfite_mg_l", "oxidation_risk", "actual_dose_mg_l"):
        if column not in frame.columns:
            frame[column] = np.nan

    for column in target_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    for column in OBSERVABLE_NUMERIC_COLUMNS + AUXILIARY_NUMERIC_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    if "lab_sample" not in frame.columns:
        lab_columns = [
            column
            for column in ("ph_lab", "free_sulfite_lab", "total_sulfite_lab")
            if column in frame.columns
        ]
        frame["lab_sample"] = (
            frame[lab_columns].notna().any(axis=1).astype(int) if lab_columns else 0
        )
    for column in OBSERVABLE_FLAG_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0

    for column in categorical_columns:
        if column not in frame.columns:
            frame[column] = "unknown"

    return frame


def _add_observed_stage_progress(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True).copy()
    if "stage_progress" in ordered.columns and ordered["stage_progress"].notna().any():
        ordered["stage_progress"] = ordered.groupby(["lot_id", "stage"], sort=False)[
            "stage_progress"
        ].transform(lambda series: series.ffill().bfill())
        return ordered

    # Approximation used when the winery does not provide stage_progress (manifest known_issues
    # KI-01):
    # relative position of each reading within the OBSERVED history of its stage.
    stage_position = ordered.groupby(["lot_id", "stage"], sort=False).cumcount()
    stage_count = ordered.groupby(["lot_id", "stage"], sort=False)["timestamp_index"].transform(
        "count"
    )
    ordered["stage_progress"] = (stage_position / (stage_count - 1).clip(lower=1)).astype(float)
    return ordered


def _add_lab_freshness(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True).copy()
    if "hours_since_last_lab" in ordered.columns and ordered["hours_since_last_lab"].notna().any():
        ordered["hours_since_last_lab"] = ordered.groupby("lot_id", sort=False)[
            "hours_since_last_lab"
        ].transform(lambda series: series.ffill().bfill())
        return ordered

    lab_columns = [
        column
        for column in ("ph_lab", "free_sulfite_lab", "total_sulfite_lab")
        if column in ordered.columns
    ]
    observed_lab = (
        ordered[lab_columns].notna().any(axis=1)
        if lab_columns
        else pd.Series(False, index=ordered.index)
    )
    ordered["lab_sample"] = ordered.get("lab_sample", 0).fillna(0).astype(int)
    ordered.loc[observed_lab, "lab_sample"] = 1

    last_lab_step = ordered["timestamp_index"].where(ordered["lab_sample"].astype(bool))
    last_lab_step = last_lab_step.groupby(ordered["lot_id"], sort=False).transform(
        lambda series: series.ffill()
    )
    first_step = ordered.groupby("lot_id", sort=False)["timestamp_index"].transform("min")
    ordered["hours_since_last_lab"] = (
        (ordered["timestamp_index"] - last_lab_step.fillna(first_step)) * 2.0
    ).clip(lower=0.0)
    return ordered


def _add_operational_features(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = _add_lab_freshness(_add_observed_stage_progress(frame))
    shifted_dose = ordered.groupby("lot_id", sort=False)["actual_dose_mg_l"].shift(1).fillna(0.0)
    shifted_step = ordered.groupby("lot_id", sort=False)["timestamp_index"].shift(1)
    last_addition = shifted_dose.where(shifted_dose > 0.0)
    last_addition_step = shifted_step.where(shifted_dose > 0.0)
    ordered["last_so2_addition_mg_l"] = (
        last_addition.groupby(ordered["lot_id"], sort=False)
        .transform(lambda series: series.ffill())
        .fillna(0.0)
    )
    last_seen_step = last_addition_step.groupby(ordered["lot_id"], sort=False).transform(
        lambda series: series.ffill()
    )
    ordered["hours_since_last_addition"] = (
        (ordered["timestamp_index"] - last_seen_step).fillna(ordered["timestamp_index"]) * 2.0
    ).astype(float)
    ordered["target_free_sulfite"] = ordered.apply(
        lambda row: target_free_sulfite(
            str(row["wine_type"]),
            str(row["stage"]),
            float(row["ph_lab"]) if pd.notna(row["ph_lab"]) else 3.3,
        ),
        axis=1,
    )
    return ordered


def _future_risk(row: pd.Series) -> float:
    if (
        pd.notna(row["future_free_sulfite"])
        and pd.notna(row["stage"])
        and pd.notna(row["hours_since_last_lab"])
    ):
        return estimate_underprotection_risk(
            wine_type=str(row["wine_type"]),
            stage=str(row["stage"]),
            ph_lab=float(row["ph_lab"]) if pd.notna(row["ph_lab"]) else 3.3,
            future_free_sulfite=float(row["future_free_sulfite"]),
            dissolved_oxygen_mg_l=(
                float(row["dissolved_oxygen_mg_l"])
                if pd.notna(row["dissolved_oxygen_mg_l"])
                else 1.6
            ),
            temperature_c=float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else 20.0,
            hours_since_last_lab=float(row["hours_since_last_lab"]),
        )
    return np.nan


def _add_prediction_targets(
    frame: pd.DataFrame, config: dict[str, Any], target_columns: list[str]
) -> pd.DataFrame:
    horizon_hours = float(config["preprocessing"].get("decision_horizon_hours", 24))
    horizon_steps = max(1, int(round(horizon_hours / 2.0)))
    ordered = frame.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True).copy()
    grouped = ordered.groupby("lot_id", sort=False)
    future_free = grouped["free_sulfite_mg_l"].shift(-horizon_steps)
    future_inputs = pd.DataFrame(
        {
            "wine_type": ordered["wine_type"],
            "stage": grouped["stage"].shift(-horizon_steps),
            "ph_lab": grouped["ph_lab"].shift(-horizon_steps),
            "future_free_sulfite": future_free,
            "temperature_c": grouped["temperature_c"].shift(-horizon_steps),
            "dissolved_oxygen_mg_l": grouped["dissolved_oxygen_mg_l"].shift(-horizon_steps),
            "hours_since_last_lab": grouped["hours_since_last_lab"].shift(-horizon_steps),
        }
    )
    ordered[target_columns[0]] = future_free.astype(float)
    ordered[target_columns[1]] = future_inputs.apply(_future_risk, axis=1)
    return ordered


def _impute_observable_columns(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["lot_id", "timestamp_index"]).reset_index(drop=True).copy()

    forward_fill_columns = [
        "temperature_c",
        "dissolved_oxygen_mg_l",
        "density_g_ml",
        "co2_g_l",
        "ph_lab",
        "free_sulfite_lab",
        "total_sulfite_lab",
    ]
    for column in forward_fill_columns:
        ordered[column] = ordered.groupby("lot_id", sort=False)[column].transform(
            lambda series: series.ffill().bfill()
        )

    # NOTE (manifest KI-09): residual NaNs use the median of the frame received, not train
    # statistics.
    for column in OBSERVABLE_NUMERIC_COLUMNS + AUXILIARY_NUMERIC_COLUMNS:
        median = ordered[column].median()
        ordered[column] = ordered[column].fillna(0.0 if pd.isna(median) else float(median))

    for column in OBSERVABLE_FLAG_COLUMNS:
        ordered[column] = ordered[column].fillna(0).astype(int)

    return ordered


def _category_levels(config: dict[str, Any], column: str) -> list[str]:
    configured = config.get("preprocessing", {}).get("categorical_levels", {}).get(column)
    if configured:
        return [str(value) for value in configured]
    return DEFAULT_CATEGORY_LEVELS.get(column, [])


def _encode_categoricals(
    frame: pd.DataFrame, categorical_columns: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    active_columns = [column for column in categorical_columns if column in frame.columns]
    if not active_columns:
        return frame
    categorical_frame = frame[active_columns].copy()
    for column in active_columns:
        levels = _category_levels(config, column)
        if levels:
            categorical_frame[column] = pd.Categorical(
                categorical_frame[column].fillna("unknown"), categories=levels
            )
        else:
            categorical_frame[column] = categorical_frame[column].fillna("unknown")
    encoded = pd.get_dummies(categorical_frame, prefix=active_columns, dtype=float)
    return pd.concat([frame.drop(columns=active_columns), encoded], axis=1)


def build_feature_frame(
    sequential_df: pd.DataFrame,
    config: dict[str, Any],
    include_targets: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (feature_frame, feature_columns, target_columns) exactly as the AI team's pipeline
    does.
    """
    categorical_columns = list(
        config["preprocessing"].get("categorical_columns", DEFAULT_CATEGORICAL_COLUMNS)
    )
    target_columns = target_columns_for_config(config)
    prepared = _ensure_required_columns(sequential_df, categorical_columns, target_columns)
    enriched = _add_operational_features(prepared)
    if include_targets:
        enriched = _add_prediction_targets(enriched, config, target_columns)
    enriched = enriched.loc[~enriched["target_only"]].copy()
    selected_columns = (
        INTERNAL_COLUMNS
        + target_columns
        + OBSERVABLE_NUMERIC_COLUMNS
        + AUXILIARY_NUMERIC_COLUMNS
        + OBSERVABLE_FLAG_COLUMNS
        + categorical_columns
    )
    frame = enriched[selected_columns].copy()
    if include_targets:
        frame = frame.dropna(subset=target_columns).reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    frame = _impute_observable_columns(frame)
    frame = _encode_categoricals(frame, categorical_columns, config)
    excluded = INTERNAL_COLUMNS + target_columns + AUXILIARY_NUMERIC_COLUMNS
    feature_columns = [column for column in frame.columns if column not in excluded]
    return frame, feature_columns, target_columns
