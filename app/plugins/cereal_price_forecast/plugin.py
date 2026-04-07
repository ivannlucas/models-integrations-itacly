import logging
from datetime import datetime, timezone
from typing import Any

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import UnsupportedProductError
from app.plugins.cereal_price_forecast.model_loader import PRODUCT_TO_FILE, load_artifacts
from app.plugins.cereal_price_forecast.postprocessing import run_lgbm_predict
from app.plugins.cereal_price_forecast.predict_dto import VALID_PRODUCTS, PredictInlineRequest
from app.plugins.cereal_price_forecast.preprocessing import prepare_features

logger = logging.getLogger(__name__)

MODEL_NAME = "cereal-price-forecast"
MODEL_VERSION = "1.0.0"


class CerealPriceForecastPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._feature_cols: list[str] = []
        self._feature_medians: dict[str, float] = {}
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None
        self._pipeline: Any | None = None  # Optional FeaturePipeline (not injected in central app)

    def load(self) -> None:
        self._models, self._feature_cols, self._feature_medians = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_batch(self, *, data_path: str) -> dict:
        import pandas as pd

        df = pd.read_csv(data_path)
        predictions = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            try:
                req = PredictInlineRequest(mode="inline", **row_dict)
                if req.product_name not in VALID_PRODUCTS:
                    predictions.append({"row": int(idx), "error": f"Unsupported product '{req.product_name}'"})
                    continue
                model = self._models[req.product_name]
                X = prepare_features(req, self._feature_cols, self._feature_medians)
                predicted_price = run_lgbm_predict(model, X)
                predictions.append({
                    "row": int(idx),
                    "product_name": req.product_name,
                    "market_name": req.market_name,
                    "week_begin_date": req.week_begin_date,
                    "predicted_price_eur_ton": predicted_price,
                })
            except Exception as exc:
                predictions.append({"row": int(idx), "error": str(exc)})

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d predictions count=%d", len(predictions), self._predict_count)

        return {"model_id": MODEL_NAME, "predictions": predictions, "output_path": None}

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        import pandas as pd

        if self._pipeline is not None:
            product_name = features.get("product_name")
            market_name = features.get("market_name")
            week_begin_date = features.get("week_begin_date")
            if (
                product_name
                and market_name not in (None, "Unknown")
                and week_begin_date not in (None, "Unknown")
            ):
                try:
                    computed = self._pipeline.get_features(product_name, market_name, week_begin_date)
                    metadata_keys = {"product_name", "market_name", "week_begin_date", "mode", "model_key", "threshold"}
                    user_features = {k: v for k, v in features.items() if k not in metadata_keys and v is not None}
                    features = {**features, **computed, **user_features}
                except Exception as exc:
                    logger.warning("Pipeline failed (%s) — falling back to median imputation", exc)

        req = PredictInlineRequest(mode="inline", **features)

        if req.product_name not in VALID_PRODUCTS:
            raise UnsupportedProductError(
                f"No trained model for product '{req.product_name}'. "
                f"Valid products: {sorted(VALID_PRODUCTS)}"
            )

        model = self._models[req.product_name]
        X = prepare_features(req, self._feature_cols, self._feature_medians)
        predicted_price = run_lgbm_predict(model, X)

        _XAI_FEATURES = [
            "price_lag_1w", "price_lag_2w", "price_lag_4w", "price_lag_8w",
            "price_rolling_mean_4w", "price_rolling_mean_12w", "price_rolling_std_4w",
            "prec", "tmed", "tmin", "tmax", "Fertilizers_index", "Seeds_index",
            "month_sin", "month_cos", "week_sin", "week_cos",
        ]
        _API_TO_MODEL = {"Fertilizers_index": "Fertilizers index", "Seeds_index": "Seeds index"}
        row = X.iloc[0]
        xai_feature_values: dict[str, float] = {}
        for feat in _XAI_FEATURES:
            col = _API_TO_MODEL.get(feat, feat)
            val = row.get(col) if col in row.index else None
            if val is not None and not pd.isna(float(val)):
                xai_feature_values[feat] = float(val)

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — product='%s' price=%.2f EUR/t count=%d",
            req.product_name, predicted_price, self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": threshold,
            "prediction": predicted_price,
            "confidence": None,
            "features_used": self._feature_cols,
            "product_name": req.product_name,
            "market_name": req.market_name,
            "week_begin_date": req.week_begin_date,
            "model_version": "1.0",
            "xai_feature_values": xai_feature_values or None,
        }

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type="LGBMRegressor (one model per product, 5 total)",
            framework="lightgbm",
            artifact_path="model-runtime-cereal_price_forecast/artifacts/",
            input_schema={
                "mode=inline": {
                    "product_name": f"str — one of {sorted(VALID_PRODUCTS)}",
                    "market_name": "str — Spanish market, e.g. 'Madrid'",
                    "week_begin_date": "str — ISO date (YYYY-MM-DD, Monday)",
                    "...74 pre-computed features...": "Temporal, climate, input index, autoregressive",
                },
                "mode=batch": {"data_path": "str — path to CSV with pre-computed feature rows"},
            },
            output_schema={
                "batch": {"predictions": "list[dict] — per-row price predictions"},
                "inline": {
                    "prediction": "float (EUR/tonne)",
                    "product_name": "str", "market_name": "str", "week_begin_date": "str",
                },
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
