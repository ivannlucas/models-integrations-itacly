"""Orchestrates the vendored rules engine — mirrors inbox/a28/codigo/src/cli/platform_run.py's
run_platform_pipeline(), minus the local outputs/ file writes (the plugin returns the frame and
summary directly instead).
"""
from __future__ import annotations

import pandas as pd

from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.features import (
    build_platform_features,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.input_validation import (
    validate_input_dataframe,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.optimizer import (
    run_quantity_optimizer,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.output_writer import (
    RECOMMENDATION_COLUMNS,
    _ensure_explainability_columns,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.simulation import (
    run_policy_simulation,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction._vendor.trigger import (
    run_purchase_trigger,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.model_loader import (
    load_platform_config,
)


def run_recommendation_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate *df* against the input contract and run the full recommendation pipeline.

    Returns (recommendations_frame with RECOMMENDATION_COLUMNS, summary_metrics dict).
    Raises ValueError with the validation report's errors if the input contract is violated.
    """
    validation_report = validate_input_dataframe(df)
    if not validation_report["valid"]:
        raise ValueError(
            "Input contract validation failed: " + " ".join(validation_report["errors"])
        )

    platform_config = load_platform_config()
    feature_frame = build_platform_features(df, config=platform_config)
    trigger_frame = run_purchase_trigger(feature_frame, config=platform_config)
    optimized_frame = run_quantity_optimizer(trigger_frame, config=platform_config)
    simulation_frame, summary_metrics = run_policy_simulation(optimized_frame, config=platform_config)

    recommendations_frame = _ensure_explainability_columns(simulation_frame)
    recommendations_frame = recommendations_frame.copy()
    recommendations_frame["date"] = recommendations_frame["date"].dt.strftime("%Y-%m-%d")

    return recommendations_frame[RECOMMENDATION_COLUMNS], summary_metrics
