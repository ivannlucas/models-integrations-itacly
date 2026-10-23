"""Vendored from the AI team's src/training/train.py (a24) -- stage helpers, verbatim.

Copied without changes: _filter_feature_pool_by_coverage, _select_feature_blocks,
_evaluate_stage and _score_train_stage. They only depend on DataFrames and on the
``config["training"]`` / ``horizon_cfg`` dicts, which the plugin builds from constants.py
(literal copy of config/config.yaml). The path-bound parts of train.py (_load_panel_for_horizon,
_prepare_frame, _export_split_datasets, _train_final_artifact, train_model) are ported in
app/plugins/ml24_dairy_spatial_price_forecast/training.py.
"""
# pylint: skip-file
from __future__ import annotations

from typing import Any

import pandas as pd

from app.plugins.ml24_dairy_spatial_price_forecast._vendor.spatial_price_model import (
    apply_missing_value_strategy,
    baseline_predict,
    build_model,
    choose_ml_blend_alpha,
    evaluate_predictions,
    materialize_selected_feature_views,
    select_stable_exogenous_features,
)


def _filter_feature_pool_by_coverage(
    train_df: pd.DataFrame,
    columns: list[str],
    min_non_missing_ratio: float,
    min_months_observed: int,
) -> list[str]:
    if not columns:
        return []
    filtered: list[str] = []
    n_rows = max(len(train_df), 1)
    for column in columns:
        if column not in train_df.columns:
            continue
        non_missing = int(train_df[column].notna().sum())
        if (non_missing / n_rows) < min_non_missing_ratio:
            continue
        months_observed = int(train_df.loc[train_df[column].notna(), "target_period_dt"].nunique())
        if months_observed < min_months_observed:
            continue
        filtered.append(column)
    return filtered


def _select_feature_blocks(
    train_df: pd.DataFrame,
    config: dict[str, Any],
    horizon_cfg: dict[str, Any],
    input_exog_lag_cols: list[str],
    anchor_feature_pool: list[str],
    context_feature_pool: list[str],
) -> tuple[list[str], list[str], list[str]]:
    training_cfg = config["training"]
    min_non_missing_ratio = float(horizon_cfg.get("exog_min_non_missing_ratio", 0.0))
    min_months_observed = int(horizon_cfg.get("exog_min_months_observed", 0))
    selected_exog: list[str] = []
    selected_anchor: list[str] = []
    selected_context: list[str] = []

    input_exog_lag_cols = _filter_feature_pool_by_coverage(
        train_df=train_df,
        columns=input_exog_lag_cols,
        min_non_missing_ratio=min_non_missing_ratio,
        min_months_observed=min_months_observed,
    )

    if horizon_cfg.get("use_input_exog", False):
        selected_exog = select_stable_exogenous_features(
            train_df=train_df,
            exog_lag_cols=input_exog_lag_cols,
            target_col=training_cfg["target_col"],
            top_k=int(horizon_cfg["exog_top_k"]),
            stability_windows=int(horizon_cfg["exog_stability_windows"]),
            min_support=int(horizon_cfg["exog_min_support"]),
            training_cfg=training_cfg,
            entity_col=training_cfg["entity_col"],
        )
    if horizon_cfg.get("use_anchor_extra_metrics", False):
        selected_anchor = select_stable_exogenous_features(
            train_df=train_df,
            exog_lag_cols=anchor_feature_pool,
            target_col=training_cfg["target_col"],
            top_k=int(horizon_cfg["anchor_metrics_top_k"]),
            stability_windows=int(horizon_cfg["panel_feature_stability_windows"]),
            min_support=int(horizon_cfg["panel_feature_min_support"]),
            training_cfg=training_cfg,
            entity_col=training_cfg["entity_col"],
        )
    if horizon_cfg.get("use_context_features", False):
        selected_context = select_stable_exogenous_features(
            train_df=train_df,
            exog_lag_cols=context_feature_pool,
            target_col=training_cfg["target_col"],
            top_k=int(horizon_cfg["context_top_k"]),
            stability_windows=int(horizon_cfg["panel_feature_stability_windows"]),
            min_support=int(horizon_cfg["panel_feature_min_support"]),
            training_cfg=training_cfg,
            entity_col=training_cfg["entity_col"],
        )
    return selected_anchor, selected_context, selected_exog


def _evaluate_stage(
    fit_df: pd.DataFrame,
    score_df: pd.DataFrame,
    split_name: str,
    base_feature_cols: list[str],
    categorical_cols: list[str],
    selected_anchor: list[str],
    selected_context: list[str],
    selected_exog: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    training_cfg = config["training"]
    (fit_df_view, score_df_view), feature_cols, _ = materialize_selected_feature_views(
        dfs=[fit_df, score_df],
        base_feature_cols=base_feature_cols,
        selected_anchor_features=selected_anchor,
        selected_context_features=selected_context,
        selected_exog=selected_exog,
        add_selected_exog_missing_indicators=bool(
            training_cfg.get("add_selected_exog_missing_indicators", False)
        ),
    )
    (fit_df_view, score_df_view), imputation_report = apply_missing_value_strategy(
        dfs=[fit_df_view, score_df_view],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        training_cfg=training_cfg,
        entity_col=training_cfg["entity_col"],
    )

    y_baseline, w1, w12, calib_mae, calibration_start = baseline_predict(
        train_df=fit_df_view,
        target_df=score_df_view,
        training_cfg=training_cfg,
    )
    ml_alpha = 1.0
    ml_alpha_calib_mae = None
    ml_alpha_calibration_start = ""
    if training_cfg.get("calibrate_ml_blend", False):
        ml_alpha, ml_alpha_calib_mae, ml_alpha_calibration_start = choose_ml_blend_alpha(
            train_df=fit_df_view,
            feature_cols=feature_cols,
            categorical_cols=categorical_cols,
            training_cfg=training_cfg,
        )

    model = build_model(training_cfg)
    model.fit(fit_df_view[feature_cols], fit_df_view["y_t_plus_h"], cat_features=categorical_cols)
    y_ml = model.predict(score_df_view[feature_cols])
    y_pred = y_baseline + ml_alpha * (y_ml - y_baseline)

    metrics_df, preds_df = evaluate_predictions(
        eval_df=score_df_view,
        entity_col=training_cfg["entity_col"],
        y_pred=y_pred,
        y_baseline=y_baseline,
        split_name=split_name,
    )
    metrics_df["baseline_w_persist_1"] = w1
    metrics_df["baseline_w_persist_12"] = w12
    metrics_df["ml_alpha"] = ml_alpha
    metrics_df["selected_anchor_features"] = "|".join(selected_anchor)
    metrics_df["selected_context_features"] = "|".join(selected_context)
    metrics_df["selected_exog"] = "|".join(selected_exog)
    metrics_df["missing_value_strategy"] = imputation_report["strategy"]

    preds_df["baseline_w_persist_1"] = w1
    preds_df["baseline_w_persist_12"] = w12
    preds_df["ml_alpha"] = ml_alpha
    preds_df["selected_anchor_features"] = "|".join(selected_anchor)
    preds_df["selected_context_features"] = "|".join(selected_context)
    preds_df["selected_exog"] = "|".join(selected_exog)
    preds_df["missing_value_strategy"] = imputation_report["strategy"]

    stage_meta = {
        "split_name": split_name,
        "n_rows_fit": int(len(fit_df_view)),
        "n_rows_score": int(len(score_df_view)),
        "baseline_w_persist_1": w1,
        "baseline_w_persist_12": w12,
        "baseline_calib_mae": calib_mae,
        "baseline_calibration_start": calibration_start,
        "ml_alpha": ml_alpha,
        "ml_alpha_calib_mae": None if pd.isna(ml_alpha_calib_mae) else float(ml_alpha_calib_mae),
        "ml_alpha_calibration_start": ml_alpha_calibration_start,
        "selected_anchor_features": selected_anchor,
        "selected_context_features": selected_context,
        "selected_exog": selected_exog,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "missing_value_strategy": imputation_report["strategy"],
        "missing_value_report": imputation_report,
        "model": model,
    }
    return metrics_df, preds_df, stage_meta


def _score_train_stage(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    model,
    selected_anchor: list[str],
    selected_context: list[str],
    selected_exog: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_cfg = config["training"]
    (train_df,), imputation_report = apply_missing_value_strategy(
        dfs=[train_df],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        training_cfg=training_cfg,
        entity_col=training_cfg["entity_col"],
    )
    y_baseline, w1, w12, _, _ = baseline_predict(train_df, train_df, training_cfg)
    y_ml = model.predict(train_df[feature_cols])
    metrics_df, preds_df = evaluate_predictions(
        eval_df=train_df,
        entity_col=training_cfg["entity_col"],
        y_pred=y_ml,
        y_baseline=y_baseline,
        split_name="train",
    )
    metrics_df["baseline_w_persist_1"] = w1
    metrics_df["baseline_w_persist_12"] = w12
    metrics_df["ml_alpha"] = 1.0
    metrics_df["selected_anchor_features"] = "|".join(selected_anchor)
    metrics_df["selected_context_features"] = "|".join(selected_context)
    metrics_df["selected_exog"] = "|".join(selected_exog)
    metrics_df["missing_value_strategy"] = imputation_report["strategy"]

    preds_df["baseline_w_persist_1"] = w1
    preds_df["baseline_w_persist_12"] = w12
    preds_df["ml_alpha"] = 1.0
    preds_df["selected_anchor_features"] = "|".join(selected_anchor)
    preds_df["selected_context_features"] = "|".join(selected_context)
    preds_df["selected_exog"] = "|".join(selected_exog)
    preds_df["missing_value_strategy"] = imputation_report["strategy"]
    return metrics_df, preds_df
