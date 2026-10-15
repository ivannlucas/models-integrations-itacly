"""ml24 /train — port of the AI team's training procedure (src/training/train.py::train_model).

Same steps, same order, same hyperparameters (constants.TRAINING_CFG / HORIZONS_CFG, literal copy
of config/config.yaml), with the stage helpers vendored verbatim in _vendor/train.py:

1. ensure_release_inputs: anchor panel (1999..max_year) and context panel (2000..max_year) built
   from the long-format series; monthly exogenous block.
2. Per horizon h (1, 3, 6), _prepare_frame: build_supervised_training_frame with
   publication_lag_months=12, add_compact_features, monthly exogenous lags filtered by
   input_exog_patterns, anchor/context feature pools.
3. Temporal split by target_period_dt: train < validation year, validation year, test year.
4. Feature selection on train -> train-stage refit (train metrics) -> validation stage;
   selection on train+validation -> test stage (fit on train+validation, score test).
5. Final selection on the whole frame and final artifact: baseline weights, ml_alpha and a
   CatBoostRegressor (MAE, 300 it., depth 2, lr 0.02, l2 100, bootstrap No, seed 24) on every row.

Plugin adaptations, all deliberate:

* The series comes from /train's ``data_path`` (same schema as series_consumo_lacteos_ccaa.csv);
  the spatial matrix and the monthly exogenous block are the base artifacts (the original reads
  them from fixed config paths).
* ``validation_year``/``test_year`` default to the delivered config (2019/2020, panels capped at
  the test year). They are exposed so a retraining can use data after 2020; with the defaults the
  procedure is exactly the original one.
* The rolling summary (_run_rolling_summary) is not recomputed: it is a diagnostic that does not
  change the artifact (11 extra fits per horizon).
* ``seed_everything`` (np.random.seed) is not called: nothing in the pipeline draws from NumPy's
  global RNG (CatBoost receives random_seed=24 explicitly) and the server is shared.
* task_type GPU like the original when CatBoost sees a GPU, otherwise CPU (the memoria accepts
  CPU with small numerical differences). The resolved value is stored in the metadata.
* Artifacts are written to a temp dir that the caller uploads to MLflow, never to artifacts/.
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.plugins.ml24_dairy_spatial_price_forecast import preprocessing
from app.plugins.ml24_dairy_spatial_price_forecast._vendor import train as vendor_train
from app.plugins.ml24_dairy_spatial_price_forecast._vendor.spatial_price_model import (
    add_compact_features,
    add_monthly_exog_lags,
    apply_missing_value_strategy,
    baseline_predict,
    build_model,
    build_product_panel,
    build_supervised_training_frame,
    choose_ml_blend_alpha,
    filter_anchor_extra_feature_cols,
    filter_context_feature_cols,
    filter_input_exog_lag_cols,
    materialize_selected_feature_views,
)
from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
    ANCHOR_PANEL_MIN_YEAR,
    ANCHOR_PRODUCT,
    CONTEXT_PANEL_MIN_YEAR,
    CONTEXT_PRODUCTS,
    HORIZONS_CFG,
    INCLUDE_SPAIN,
    MODEL_MANIFEST_FILENAME,
    ORIGINAL_MODEL_NAME,
    TRAIN_TEST_YEAR,
    TRAIN_VALIDATION_YEAR,
    TRAINING_CFG,
    metadata_filename,
    model_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """Final artifact of one horizon plus its stage metrics."""

    horizon: int
    model: Any
    metadata: dict
    metrics: dict
    stage_summary: dict


@dataclass
class TrainingResult:
    """The three retrained horizons and the flat metrics returned by /train."""

    horizons: list[HorizonResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    task_type: str = "CPU"
    training_time_s: float = 0.0


def resolve_task_type() -> str:
    """GPU like config.yaml when CatBoost sees a CUDA device, else CPU."""
    try:
        # pylint: disable=import-outside-toplevel
        from catboost.utils import get_gpu_device_count
        return "GPU" if get_gpu_device_count() > 0 else "CPU"
    except Exception:  # pylint: disable=broad-exception-caught  # no CUDA runtime at all
        return "CPU"


def build_training_config(task_type: str, validation_year: int, test_year: int) -> dict:
    """config dict with the shape train.py expects (config["training"])."""
    training_cfg = dict(TRAINING_CFG)
    training_cfg["task_type"] = task_type
    training_cfg["validation_start"] = f"{validation_year}-01-01"
    training_cfg["validation_end"] = f"{validation_year}-12-01"
    training_cfg["test_start"] = f"{test_year}-01-01"
    training_cfg["test_end"] = f"{test_year}-12-01"
    return {"training": training_cfg}


def build_release_panels(series: pd.DataFrame, max_year: int) -> dict[str, pd.DataFrame]:
    """spatial_price_model.ensure_release_inputs, without writing the panels to disk.

    The original writes the panels to CSV and train.py reads them back (period_dt parsed again);
    the round trip does not change any value, so the frames are passed directly.
    """
    anchor = build_product_panel(
        series=series, anchor_product=ANCHOR_PRODUCT, context_products=[], include_spain=INCLUDE_SPAIN,
        min_year=ANCHOR_PANEL_MIN_YEAR, max_year=max_year, add_context_derivatives=False,
    )
    context = build_product_panel(
        series=series, anchor_product=ANCHOR_PRODUCT, context_products=list(CONTEXT_PRODUCTS),
        include_spain=INCLUDE_SPAIN, min_year=CONTEXT_PANEL_MIN_YEAR, max_year=max_year,
        add_context_derivatives=True,
    )
    return {"anchor": anchor, "context": context}


def prepare_frame(config, horizon_cfg, panels, spatial_weights, monthly_exog):
    """train.py::_prepare_frame with the panel/exogenous frames passed in instead of read from disk."""
    training_cfg = config["training"]
    panel = panels[horizon_cfg["panel_key"]].copy()
    panel["period_dt"] = pd.to_datetime(panel["period_dt"])
    frame, full_feature_cols, categorical_cols, _ = build_supervised_training_frame(
        panel=panel,
        entity_col=training_cfg["entity_col"],
        target_col=training_cfg["target_col"],
        horizon_months=int(horizon_cfg["horizon_months"]),
        spatial_weights_csv=spatial_weights,
        publication_lag_months=int(training_cfg.get("publication_lag_months", 0)),
    )
    frame, base_feature_cols, categorical_cols = add_compact_features(
        frame,
        training_cfg["target_col"],
        include_target_history=bool(training_cfg.get("include_target_history", True)),
    )
    input_exog_lag_cols: list[str] = []
    if horizon_cfg.get("use_input_exog", False):
        frame, exog_lag_cols = add_monthly_exog_lags(
            frame=frame,
            monthly_exog=monthly_exog,
            entity_col=training_cfg["entity_col"],
            lags=tuple(int(lag) for lag in horizon_cfg.get("exog_lags", [1, 2, 3, 6])),
        )
        input_exog_lag_cols = filter_input_exog_lag_cols(
            exog_lag_cols,
            patterns=tuple(horizon_cfg.get("input_exog_patterns", [])) or None,
        )
    anchor_feature_pool: list[str] = []
    if horizon_cfg.get("use_anchor_extra_metrics", False):
        anchor_feature_pool = filter_anchor_extra_feature_cols(full_feature_cols, training_cfg["target_col"])
    context_feature_pool: list[str] = []
    if horizon_cfg.get("use_context_features", False):
        context_feature_pool = filter_context_feature_cols(
            full_feature_cols=full_feature_cols,
            panel_columns=panel.columns.tolist(),
            target_col=training_cfg["target_col"],
        )
    return frame, base_feature_cols, categorical_cols, input_exog_lag_cols, anchor_feature_pool, context_feature_pool


def _split(frame: pd.DataFrame, training_cfg: dict) -> dict[str, pd.DataFrame]:
    valid_start = pd.Timestamp(training_cfg["validation_start"])
    valid_end = pd.Timestamp(training_cfg["validation_end"])
    test_start = pd.Timestamp(training_cfg["test_start"])
    test_end = pd.Timestamp(training_cfg["test_end"])
    target = frame["target_period_dt"]
    return {
        "train": frame[target < valid_start].copy(),
        "validation": frame[(target >= valid_start) & (target <= valid_end)].copy(),
        "train_valid": frame[target <= valid_end].copy(),
        "test": frame[(target >= test_start) & (target <= test_end)].copy(),
    }


def _check_splits(horizon: int, parts: dict[str, pd.DataFrame], training_cfg: dict) -> None:
    """Fail with a clear 400 before CatBoost/pandas fail with an obscure message."""
    empty = [name for name in ("train", "validation", "test") if parts[name].empty]
    if empty:
        raise ValueError(
            f"h={horizon}: particiones vacías {empty}. El CSV debe cubrir las ventanas de validación "
            f"({training_cfg['validation_start']}..{training_cfg['validation_end']}) y test "
            f"({training_cfg['test_start']}..{training_cfg['test_end']}) por fecha objetivo, además del "
            "histórico previo con el retraso de publicación de 12 meses."
        )


def _stage_metrics(metrics_df: pd.DataFrame) -> dict:
    row = metrics_df[metrics_df["scope"] == "global"].iloc[0]
    return {
        "mae": float(row["mae"]),
        "smape": float(row["smape"]),
        "mae_baseline": float(row["mae_baseline_calibrated"]),
        "smape_baseline": float(row["smape_baseline_calibrated"]),
        "accuracy_pct": float(100.0 - row["smape"]),
        "n": int(row["n_obs"]),
    }


def train_final_artifact(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    frame, base_feature_cols, categorical_cols, selection, horizon_cfg, config,
):
    """train.py::_train_final_artifact, returning the model and metadata instead of saving them."""
    selected_anchor, selected_context, selected_exog = selection
    training_cfg = config["training"]
    horizon = int(horizon_cfg["horizon_months"])
    (full_train_view,), feature_cols, _ = materialize_selected_feature_views(
        dfs=[frame.copy()],
        base_feature_cols=base_feature_cols,
        selected_anchor_features=selected_anchor,
        selected_context_features=selected_context,
        selected_exog=selected_exog,
        add_selected_exog_missing_indicators=bool(training_cfg.get("add_selected_exog_missing_indicators", False)),
    )
    (full_train_view,), imputation_report = apply_missing_value_strategy(
        dfs=[full_train_view],
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        training_cfg=training_cfg,
        entity_col=training_cfg["entity_col"],
    )
    _, w1, w12, baseline_calib_mae, baseline_calibration_start = baseline_predict(
        full_train_view,
        full_train_view.tail(min(len(full_train_view), 12)).copy(),
        training_cfg,
    )
    ml_alpha, ml_alpha_calib_mae, ml_alpha_calibration_start = 1.0, None, ""
    if training_cfg.get("calibrate_ml_blend", False):
        ml_alpha, ml_alpha_calib_mae, ml_alpha_calibration_start = choose_ml_blend_alpha(
            full_train_view, feature_cols, categorical_cols, training_cfg,
        )
    model = build_model(training_cfg)
    model.fit(full_train_view[feature_cols], full_train_view["y_t_plus_h"], cat_features=categorical_cols)
    metadata = {
        "model_id": ORIGINAL_MODEL_NAME,
        "model_name": ORIGINAL_MODEL_NAME,
        "horizon_months": horizon,
        "panel_key": horizon_cfg["panel_key"],
        "target_col": training_cfg["target_col"],
        "entity_col": training_cfg["entity_col"],
        "feature_cols": feature_cols,
        "feature_cols_base": base_feature_cols,
        "categorical_cols": categorical_cols,
        "selected_anchor_features": selected_anchor,
        "selected_context_features": selected_context,
        "selected_exog": selected_exog,
        "selected_exog_missing_indicators": bool(training_cfg.get("add_selected_exog_missing_indicators", False)),
        "exog_lags": [int(lag) for lag in horizon_cfg.get("exog_lags", [1, 2, 3, 6])],
        "publication_lag_months": int(training_cfg.get("publication_lag_months", 0)),
        "include_target_history": bool(training_cfg.get("include_target_history", True)),
        "input_exog_patterns": list(horizon_cfg.get("input_exog_patterns", [])) or None,
        "baseline_w_persist_1": w1,
        "baseline_w_persist_12": w12,
        "baseline_calib_mae": baseline_calib_mae,
        "baseline_calibration_start": baseline_calibration_start,
        "ml_alpha": ml_alpha,
        "ml_alpha_calib_mae": None if ml_alpha_calib_mae is None or pd.isna(ml_alpha_calib_mae)
        else float(ml_alpha_calib_mae),
        "ml_alpha_calibration_start": ml_alpha_calibration_start,
        "training_rows": int(len(full_train_view)),
        "missing_value_strategy": imputation_report["strategy"],
        "missing_value_report": imputation_report,
        "task_type": training_cfg["task_type"],
        "devices": training_cfg.get("devices", "") if training_cfg["task_type"] == "GPU" else "",
        "random_state": int(training_cfg["random_state"]),
        "numpy_seed": int(training_cfg.get("numpy_seed", training_cfg["random_state"])),
        "model_path": model_filename(horizon),
    }
    return model, metadata


def train_horizon(  # pylint: disable=too-many-locals  # faithful port of train_model's loop
    horizon_cfg, config, panels, spatial_weights, monthly_exog,
) -> HorizonResult:
    """One iteration of train.py::train_model's horizon loop (without the rolling summary)."""
    training_cfg = config["training"]
    horizon = int(horizon_cfg["horizon_months"])
    frame, base_cols, cat_cols, exog_pool, anchor_pool, context_pool = prepare_frame(
        config, horizon_cfg, panels, spatial_weights, monthly_exog,
    )
    parts = _split(frame, training_cfg)
    _check_splits(horizon, parts, training_cfg)

    sel_val = vendor_train._select_feature_blocks(  # pylint: disable=protected-access
        parts["train"], config, horizon_cfg, exog_pool, anchor_pool, context_pool,
    )
    # Refit explicitly for train metrics using the selected validation-stage feature set.
    (train_view,), train_feature_cols, _ = materialize_selected_feature_views(
        dfs=[parts["train"]], base_feature_cols=base_cols, selected_anchor_features=sel_val[0],
        selected_context_features=sel_val[1], selected_exog=sel_val[2],
        add_selected_exog_missing_indicators=bool(training_cfg.get("add_selected_exog_missing_indicators", False)),
    )
    train_model_stage = build_model(training_cfg)
    (train_view,), _ = apply_missing_value_strategy(
        dfs=[train_view], feature_cols=train_feature_cols, categorical_cols=cat_cols,
        training_cfg=training_cfg, entity_col=training_cfg["entity_col"],
    )
    train_model_stage.fit(train_view[train_feature_cols], train_view["y_t_plus_h"], cat_features=cat_cols)
    train_metrics, _ = vendor_train._score_train_stage(  # pylint: disable=protected-access
        train_df=train_view, feature_cols=train_feature_cols, categorical_cols=cat_cols,
        model=train_model_stage, selected_anchor=sel_val[0], selected_context=sel_val[1],
        selected_exog=sel_val[2], config=config,
    )
    valid_metrics, _, valid_meta = vendor_train._evaluate_stage(  # pylint: disable=protected-access
        fit_df=parts["train"], score_df=parts["validation"], split_name="validation",
        base_feature_cols=base_cols, categorical_cols=cat_cols, selected_anchor=sel_val[0],
        selected_context=sel_val[1], selected_exog=sel_val[2], config=config,
    )
    sel_test = vendor_train._select_feature_blocks(  # pylint: disable=protected-access
        parts["train_valid"], config, horizon_cfg, exog_pool, anchor_pool, context_pool,
    )
    test_metrics, _, test_meta = vendor_train._evaluate_stage(  # pylint: disable=protected-access
        fit_df=parts["train_valid"], score_df=parts["test"], split_name="test",
        base_feature_cols=base_cols, categorical_cols=cat_cols, selected_anchor=sel_test[0],
        selected_context=sel_test[1], selected_exog=sel_test[2], config=config,
    )
    sel_final = vendor_train._select_feature_blocks(  # pylint: disable=protected-access
        frame, config, horizon_cfg, exog_pool, anchor_pool, context_pool,
    )
    model, metadata = train_final_artifact(frame, base_cols, cat_cols, sel_final, horizon_cfg, config)
    valid_meta.pop("model")
    test_meta.pop("model")
    return HorizonResult(
        horizon=horizon,
        model=model,
        metadata=metadata,
        metrics={
            "train": _stage_metrics(train_metrics),
            "validation": _stage_metrics(valid_metrics),
            "test": _stage_metrics(test_metrics),
        },
        stage_summary={"validation_stage": valid_meta, "test_stage": test_meta},
    )


def train_spatial_price_model(  # pylint: disable=too-many-locals
    series: pd.DataFrame,
    reference: dict,
    *,
    validation_year: int = TRAIN_VALIDATION_YEAR,
    test_year: int = TRAIN_TEST_YEAR,
    task_type: str | None = None,
) -> TrainingResult:
    """Retrain the three horizons from scratch with the AI team's procedure."""
    if test_year != validation_year + 1:
        raise ValueError(
            f"test_year ({test_year}) debe ser validation_year + 1 ({validation_year + 1}), como en "
            "config.yaml (validación 2019, test 2020)."
        )
    series = preprocessing.validate_series(series)
    start = time.perf_counter()
    resolved_task_type = task_type or resolve_task_type()
    config = build_training_config(resolved_task_type, validation_year, test_year)
    panels = build_release_panels(series, max_year=test_year)
    result = TrainingResult(task_type=resolved_task_type)
    for horizon_cfg in HORIZONS_CFG:
        logger.info("ml24 /train: horizonte h=%s (%s)", horizon_cfg["horizon_months"], resolved_task_type)
        result.horizons.append(
            train_horizon(horizon_cfg, config, panels, reference["spatial_weights"], reference["exog"])
        )
    result.training_time_s = round(time.perf_counter() - start, 2)
    flat: dict[str, Any] = {}
    for item in result.horizons:
        for split, values in item.metrics.items():
            for key, value in values.items():
                flat[f"h{item.horizon}_{split}_{key}"] = value
    result.metrics = flat
    return result


def save_training_artifacts(directory: Path, result: TrainingResult) -> None:
    """Write the same layout as the base artifacts (flat): manifest, .cbm and metadata per horizon."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_horizons = []
    for item in result.horizons:
        item.model.save_model(str(directory / model_filename(item.horizon)))
        with open(directory / metadata_filename(item.horizon), "w", encoding="utf-8") as fh:
            json.dump(item.metadata, fh, indent=2, ensure_ascii=True)
        with open(directory / f"stage_summary_h{item.horizon}.json", "w", encoding="utf-8") as fh:
            json.dump(item.stage_summary, fh, indent=2, ensure_ascii=True, default=str)
        manifest_horizons.append({
            "horizon_months": item.horizon,
            "artifact_model_path": model_filename(item.horizon),
            "artifact_metadata_path": metadata_filename(item.horizon),
            "final_selected_anchor_features": item.metadata["selected_anchor_features"],
            "final_selected_context_features": item.metadata["selected_context_features"],
            "final_selected_exog": item.metadata["selected_exog"],
        })
    with open(directory / MODEL_MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
        json.dump({
            "model_id": ORIGINAL_MODEL_NAME,
            "model_name": ORIGINAL_MODEL_NAME,
            "horizons": manifest_horizons,
            "task_type": result.task_type,
            "random_state": int(TRAINING_CFG["random_state"]),
        }, fh, indent=2, ensure_ascii=True)
    with open(directory / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(result.metrics, fh, indent=2)
