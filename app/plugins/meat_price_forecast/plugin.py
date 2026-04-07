import logging
import math
from datetime import datetime, timezone
from typing import Any

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.plugins.meat_price_forecast.model_loader import TARGET_EXCLUDE_PREFIX, load_artifacts
from app.plugins.meat_price_forecast.postprocessing import predict_lstm, predict_rf
from app.plugins.meat_price_forecast.predict_dto import BUSINESS_TARGETS, MeatPriceRow, TargetPrediction
from app.plugins.meat_price_forecast.preprocessing import build_lagged_features

logger = logging.getLogger(__name__)

MODEL_NAME = "meat-price-forecast"
MODEL_VERSION = "1.0.0"


class MeatPriceForecastPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._bundle: dict[str, Any] = {}
        self._lstm_models: dict[str, Any] = {}
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._bundle, self._lstm_models = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def _run_inference(self, rows_data: list, include_lstm: bool) -> dict:
        models_meta = self._bundle["models"]
        required_features: list[str] = sorted({
            feat
            for target, meta in models_meta.items()
            if not target.startswith(TARGET_EXCLUDE_PREFIX)
            for feat in meta.get("features", [])
        })

        rows = [MeatPriceRow(**r) for r in rows_data]
        lagged_df = build_lagged_features(rows, required_features)
        last_row = lagged_df.iloc[[-1]]

        predictions: dict[str, TargetPrediction] = {}
        for target in BUSINESS_TARGETS:
            meta = models_meta.get(target)
            if not meta:
                continue
            features_list = meta.get("features", [])
            if not features_list:
                continue
            X = last_row[features_list]
            rf_model = meta.get("rf_model")
            rf_pred = predict_rf(rf_model, X) if rf_model is not None else float("nan")

            lstm_pred: float | None = None
            if include_lstm and target in self._lstm_models:
                lstm_pred = predict_lstm(
                    self._lstm_models[target], X, meta["scaler_x"], meta["scaler_y"]
                )
            predictions[target] = TargetPrediction(rf=rf_pred, lstm=lstm_pred)

        prediction_date = str(last_row["date"].values[0])[:10]
        return {"predictions": predictions, "prediction_date": prediction_date, "rows_used": len(lagged_df)}

    def predict_batch(self, *, data_path: str) -> dict:
        import pandas as pd

        df = pd.read_csv(data_path)
        batch_predictions = []

        for i in range(4, len(df) + 1):
            window = df.iloc[:i]
            rows_data = window.to_dict(orient="records")
            try:
                res = self._run_inference(rows_data, include_lstm=False)
                batch_predictions.append({
                    "prediction_date": res["prediction_date"],
                    "predictions": {t: {"rf": v.rf, "lstm": v.lstm} for t, v in res["predictions"].items()},
                })
            except Exception as exc:
                batch_predictions.append({"prediction_date": f"window_{i}", "error": str(exc)})
                break

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("predict_batch done — %d predictions count=%d", len(batch_predictions), self._predict_count)

        return {"model_id": MODEL_NAME, "predictions": batch_predictions, "output_path": None}

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
    ) -> dict:
        rows_data = features["rows"]
        include_lstm = features.get("include_lstm", False)
        res = self._run_inference(rows_data, include_lstm)

        models_meta = self._bundle["models"]
        required_features: list[str] = sorted({
            feat
            for target, meta in models_meta.items()
            if not target.startswith(TARGET_EXCLUDE_PREFIX)
            for feat in meta.get("features", [])
        })
        rows = [MeatPriceRow(**r) for r in rows_data]
        lagged_df = build_lagged_features(rows, required_features)
        last_row = lagged_df.iloc[-1]
        xai_feature_values: dict[str, float] = {
            col: float(last_row[col])
            for col in required_features
            if col in last_row.index and not math.isnan(float(last_row[col]))
        }

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — date=%s targets=%s include_lstm=%s count=%d",
            res["prediction_date"], list(res["predictions"].keys()), include_lstm, self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": threshold,
            "prediction": {t: {"rf": v.rf, "lstm": v.lstm} for t, v in res["predictions"].items()},
            "confidence": None,
            "features_used": list(BUSINESS_TARGETS),
            "prediction_date": res["prediction_date"],
            "rows_used": res["rows_used"],
            "xai_feature_values": xai_feature_values or None,
        }

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type="RandomForestRegressor + LSTM (Keras) — hybrid ensemble",
            framework="sklearn + tensorflow",
            artifact_path="model-runtime-meat_price_forecast/artifacts/trained_models_bundle.pkl",
            input_schema={
                "mode=inline": {
                    "rows": {"type": "array", "min_items": 4,
                             "items": {"date": "str (YYYY-MM-DD)", "bovino/porcino/ovino/ave/carne": "float (IPC)"}},
                    "include_lstm": "bool (default false)",
                },
                "mode=batch": {"data_path": "str — path to CSV"},
            },
            output_schema={
                "batch": {"predictions": "list[dict] — per-window predictions"},
                "inline": {"prediction": "dict[target -> {rf, lstm}]", "prediction_date": "str"},
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
