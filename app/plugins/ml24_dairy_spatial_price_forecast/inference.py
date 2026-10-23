"""Inference for ml24 — port of the AI team's src/predict/predictor.py (run_prediction).

Same steps, same order, same vendored functions:

1. Rebuild the anchor and context panels from the long-format series (build_product_panel,
   min_year 1999/2000, no max_year, include_spain=false).
2. Per horizon (manifest order 1, 3, 6): build_inference_feature_frame -> add_compact_features ->
   add_monthly_exog_lags (if the horizon selected exogenous features) ->
   materialize_selected_feature_views -> apply_missing_value_strategy (median of the inference
   frame itself, as the original does) -> last row per CCAA -> y_baseline (w1*y_persist_1 +
   w12*y_persist_12) and y_pred = y_baseline + ml_alpha * (y_ml - y_baseline).
3. Data-freshness diagnostics and the max_validated_lag_months warning (_compute_data_freshness).

Plugin differences, all deliberate and documented in the manifest:

* Data are passed as DataFrames (the original re-reads CSV paths per run).
* --refresh-source (MAPA/EU scrapers) is not ported: a pod must not scrape at request time. The
  data the plugin uses are the base artifacts or the CSV sent to /predict batch.
* CCAA whose last row lacks 13 consecutive anchor prices are omitted with a warning instead of
  being predicted from median-imputed values (the original predicts them; the memoria 8.4 says
  they are omitted). On the delivered data no CCAA is affected.
* The CCAA/horizon filters are applied after predicting, exactly like --ccaa in the original
  (spatial features need every CCAA).
"""
# pylint: disable=line-too-long  # repo limit is 120 (.flake8), enforced by flake8
from __future__ import annotations

from datetime import datetime

import pandas as pd

from app.plugins.ml24_dairy_spatial_price_forecast import preprocessing
from app.plugins.ml24_dairy_spatial_price_forecast._vendor.spatial_price_model import (
    add_compact_features,
    add_monthly_exog_lags,
    apply_missing_value_strategy,
    build_inference_feature_frame,
    build_product_panel,
    materialize_selected_feature_views,
)
from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
    ANCHOR_PANEL_MIN_YEAR,
    ANCHOR_PRODUCT,
    CCAA_CODE_TO_NAME,
    CONTEXT_PANEL_MIN_YEAR,
    CONTEXT_PRODUCTS,
    INCLUDE_SPAIN,
    MAX_VALIDATED_LAG_MONTHS,
    ORIGINAL_MODEL_NAME,
    TARGET_COL,
    TRAINING_CFG,
)
from app.plugins.ml24_dairy_spatial_price_forecast.exceptions import (
    InsufficientPriceHistoryError,
    SeriesContractError,
)

_NAME_TO_CODE = {name: code for code, name in CCAA_CODE_TO_NAME.items()}


def _panel(series: pd.DataFrame, context_products: list[str], min_year: int, derivatives: bool) -> pd.DataFrame:
    try:
        return build_product_panel(
            series=series,
            anchor_product=ANCHOR_PRODUCT,
            context_products=context_products,
            include_spain=INCLUDE_SPAIN,
            min_year=min_year,
            max_year=None,
            add_context_derivatives=derivatives,
        )
    except ValueError as exc:   # e.g. "No rows left after filtering product panel" for a too-early cut
        raise SeriesContractError(str(exc)) from exc


def build_anchor_panel(series: pd.DataFrame) -> pd.DataFrame:
    """Anchor half of predictor._build_runtime_panels (validation already done in preprocessing)."""
    return _panel(series, [], ANCHOR_PANEL_MIN_YEAR, False)


def build_context_panel(series: pd.DataFrame) -> pd.DataFrame:
    """Context half of predictor._build_runtime_panels."""
    return _panel(series, list(CONTEXT_PRODUCTS), CONTEXT_PANEL_MIN_YEAR, True)


def predict_horizon(
    horizon: dict,
    runtime_panels: dict[str, pd.DataFrame],
    monthly_exog: pd.DataFrame,
    spatial_weights: pd.DataFrame,
) -> pd.DataFrame:
    """predictor._predict_horizon, with the model/metadata already loaded."""
    metadata = horizon["metadata"]
    panel = runtime_panels[metadata["panel_key"]]
    inference_frame, _, _ = build_inference_feature_frame(
        panel=panel,
        entity_col=metadata["entity_col"],
        target_col=metadata["target_col"],
        horizon_months=int(metadata["horizon_months"]),
        spatial_weights_csv=spatial_weights,
    )
    inference_frame, base_feature_cols, _ = add_compact_features(
        inference_frame,
        metadata["target_col"],
    )

    if metadata["selected_exog"]:
        inference_frame, _ = add_monthly_exog_lags(
            frame=inference_frame,
            monthly_exog=monthly_exog,
            entity_col=metadata["entity_col"],
            lags=tuple(int(lag) for lag in metadata.get("exog_lags", [1, 2, 3, 6])),
        )

    (inference_view,), _, _ = materialize_selected_feature_views(
        dfs=[inference_frame],
        base_feature_cols=base_feature_cols,
        selected_anchor_features=list(metadata["selected_anchor_features"]),
        selected_context_features=list(metadata["selected_context_features"]),
        selected_exog=list(metadata["selected_exog"]),
        add_selected_exog_missing_indicators=bool(metadata.get("selected_exog_missing_indicators", False)),
    )
    (inference_view,), _ = apply_missing_value_strategy(
        dfs=[inference_view],
        feature_cols=list(metadata["feature_cols"]),
        categorical_cols=list(metadata["categorical_cols"]),
        training_cfg=TRAINING_CFG,
        entity_col=metadata["entity_col"],
    )
    inference_view = (
        inference_view.sort_values([metadata["entity_col"], "period_dt"])
        .groupby(metadata["entity_col"], as_index=False)
        .tail(1)
        .copy()
    )

    model = horizon["model"]
    y_baseline = (
        float(metadata["baseline_w_persist_1"]) * inference_view["y_persist_1"].to_numpy()
        + float(metadata["baseline_w_persist_12"]) * inference_view["y_persist_12"].to_numpy()
    )
    y_ml = model.predict(inference_view[metadata["feature_cols"]])
    y_pred = y_baseline + float(metadata["ml_alpha"]) * (y_ml - y_baseline)

    output = inference_view[
        ["anchor_product", "context_products", "period", "period_dt", "target_period", "target_period_dt",
         metadata["entity_col"]]
    ].copy()
    output = output.rename(
        columns={
            "period": "feature_period",
            "period_dt": "feature_period_dt",
            "target_period": "period",
            "target_period_dt": "period_dt",
        }
    )
    output["model_id"] = metadata["model_id"]
    output["model_name"] = ORIGINAL_MODEL_NAME
    output["horizon_months"] = int(metadata["horizon_months"])
    output["y_pred"] = y_pred
    output["y_baseline_calibrated"] = y_baseline
    output["y_persist_1"] = inference_view["y_persist_1"].to_numpy()
    output["y_persist_12"] = inference_view["y_persist_12"].to_numpy()
    output["selected_anchor_features"] = "|".join(metadata["selected_anchor_features"])
    output["selected_context_features"] = "|".join(metadata["selected_context_features"])
    output["selected_exog"] = "|".join(metadata["selected_exog"])
    return output


def compute_data_freshness(
    runtime_panels: dict[str, pd.DataFrame],
    monthly_exog: pd.DataFrame,
    target_col: str,
    as_of: datetime,
) -> dict:
    """predictor._compute_data_freshness, verbatim."""
    anchor_panel = runtime_panels["anchor"]
    main_last_period_dt = pd.to_datetime(anchor_panel.loc[anchor_panel[target_col].notna(), "period_dt"]).max()
    exog_last_period_dt = pd.to_datetime(monthly_exog["period_dt"]).max()
    as_of_month = pd.Timestamp(as_of.year, as_of.month, 1)

    def months_between(older: pd.Timestamp, newer: pd.Timestamp) -> int:
        return (newer.year - older.year) * 12 + (newer.month - older.month)

    return {
        "main_source_last_period": main_last_period_dt.strftime("%Y-%m"),
        "main_source_lag_months": int(months_between(main_last_period_dt, as_of_month)),
        "exog_last_period": exog_last_period_dt.strftime("%Y-%m"),
        "exog_lag_months": int(months_between(exog_last_period_dt, as_of_month)),
    }


def _to_records(predictions: pd.DataFrame) -> list[dict]:
    out = predictions.copy()
    out.insert(out.columns.get_loc("ccaa") + 1, "ccaa_code", out["ccaa"].map(_NAME_TO_CODE))
    for column in ("feature_period_dt", "period_dt"):
        out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
    out["context_products"] = out["context_products"].fillna("")
    records = out.to_dict(orient="records")
    for record in records:
        for key in ("y_pred", "y_baseline_calibrated", "y_persist_1", "y_persist_12"):
            record[key] = float(record[key])
        record["horizon_months"] = int(record["horizon_months"])
    return records


def run_inference(  # pylint: disable=too-many-locals
    bundle: dict,
    reference: dict,
    series: pd.DataFrame,
    *,
    ccaa: list[str] | None = None,
    horizons: list[int] | None = None,
) -> tuple[list[dict], dict, list[str]]:
    """Return (prediction rows, data freshness, warnings) for every CCAA x horizon requested.

    ``bundle`` holds the models (base or a user's MLflow run); ``reference`` the spatial weights
    and monthly exogenous of the base artifacts; ``series`` the long-format series to predict
    from (already validated, possibly truncated at as_of_period).
    """
    anchor_panel = build_anchor_panel(series)
    monthly_exog = reference["exog"]
    warnings: list[str] = []

    insufficient = preprocessing.ccaa_with_insufficient_history(anchor_panel, TARGET_COL)
    present = sorted(anchor_panel["ccaa"].astype(str).unique())
    requested = ccaa or present
    for name in requested:
        if name not in present:
            insufficient[name] = "no aparece en la serie de entrada"
    served = [name for name in requested if name not in insufficient]
    for name in requested:
        if name in insufficient:
            warnings.append(f"{name} omitida: {insufficient[name]} (mínimo de la memoria 8.4).")
    if not served:
        raise InsufficientPriceHistoryError(
            "Ninguna CCAA solicitada tiene los 13 meses consecutivos de precio del producto ancla que "
            f"necesitan los rezagos estacionales (lag_12): {insufficient}."
        )
    runtime_panels = {"anchor": anchor_panel, "context": build_context_panel(series)}

    wanted_horizons = horizons or list(bundle["horizons"])
    outputs = [
        predict_horizon(bundle["horizons"][h], runtime_panels, monthly_exog, reference["spatial_weights"])
        for h in bundle["horizons"] if h in wanted_horizons
    ]
    predictions = pd.concat(outputs, ignore_index=True)
    predictions = predictions[predictions["ccaa"].isin(served)]

    freshness = compute_data_freshness(runtime_panels, monthly_exog, TARGET_COL, as_of=datetime.now())
    if freshness["main_source_lag_months"] > MAX_VALIDATED_LAG_MONTHS:
        warnings.append(
            f"AVISO: el desfase de la fuente principal (MAPA) es de {freshness['main_source_lag_months']} "
            f"meses, por encima de los {MAX_VALIDATED_LAG_MONTHS} meses para los que hay un barrido de "
            "robustez validado (training.publication_lag_months). El rendimiento del modelo es "
            "susceptible de empeorar; la predicción se genera igualmente."
        )
    return _to_records(predictions), freshness, warnings
