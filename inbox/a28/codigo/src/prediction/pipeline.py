from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.prediction.predict_trigger import predict_trigger
from src.prediction.predict_upstream import predict_upstream
from src.reproducibility.mixed_context import apply_feature_fill_values, validate_feature_columns_for_stage
from src.reproducibility.runtime import official_paths


def _load_prediction_input(config: dict[str, Any], input_path: str | Path | None = None) -> pd.DataFrame:
    paths = official_paths(config)
    candidate = Path(input_path) if input_path else paths["splits_dir"] / "test.csv"
    frame = pd.read_csv(candidate)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def run_reproducibility_prediction(config: dict[str, Any], logger, input_path: str | Path | None = None) -> dict[str, Any]:
    paths = official_paths(config)
    df = _load_prediction_input(config, input_path=input_path).copy()

    df["synthetic_procurement_need_pred"] = predict_upstream(df, config)
    trigger_proba, trigger_flag = predict_trigger(df, config)
    df["purchase_trigger_proba"] = trigger_proba
    df["purchase_trigger_flag"] = trigger_flag

    with paths["quantity_optimizer_artifact"].open("rb") as handle:
        quantity_artifact = pickle.load(handle)
    quantity_features = list(quantity_artifact["feature_columns"])
    validate_feature_columns_for_stage(quantity_features, stage="quantity_optimizer")
    fill_values = dict(quantity_artifact.get("fill_values", {}))
    x_quantity = apply_feature_fill_values(df, quantity_features, fill_values)
    df["quantity_optimizer_recommendation_tons"] = pd.Series(
        quantity_artifact["model"].predict(x_quantity),
        index=df.index,
    ).clip(lower=0.0)
    df["order_quantity_tons"] = df["quantity_optimizer_recommendation_tons"].where(df["purchase_trigger_flag"] == 1, 0.0)

    if "baseline_order_quantity_tons" not in df.columns:
        df["baseline_order_quantity_tons"] = df["replenishment_gap_tons"].clip(lower=0.0) * 1.12
    df["scenario_name"] = "test_split"

    output_columns = [
        "date",
        "raw_material_id",
        "destination_profile",
        "scenario_name",
        "current_inventory_tons",
        "expected_requirement_tons",
        "lead_time_days",
        "safety_coverage_days",
        "expected_yield_rate",
        "expected_waste_rate",
        "synthetic_procurement_need",
        "synthetic_procurement_need_pred",
        "purchase_trigger_proba",
        "purchase_trigger_flag",
        "quantity_optimizer_recommendation_tons",
        "baseline_order_quantity_tons",
        "order_quantity_tons",
        "replenishment_gap_tons",
        "unit_purchase_cost",
        "shelf_life_days",
    ]
    available_columns = [column for column in output_columns if column in df.columns]
    predictions_df = df[available_columns].copy()
    paths["predictions_latest"].parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(paths["predictions_latest"], index=False)
    logger.info("Saved reproducible predictions to %s", paths["predictions_latest"])
    return {
        "prediction_path": str(paths["predictions_latest"]),
        "row_count": int(len(predictions_df)),
        "columns": predictions_df.columns.tolist(),
    }
