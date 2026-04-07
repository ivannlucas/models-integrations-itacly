import logging
import math
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.plugins.wine_price_fluctuation.model_loader import load_artifacts
from app.plugins.wine_price_fluctuation.postprocessing import apply_threshold
from app.plugins.wine_price_fluctuation.predict_dto import WeeklyPriceRecord
from app.plugins.wine_price_fluctuation.preprocessing import build_features

logger = logging.getLogger(__name__)

MODEL_NAME = "wine-price-fluctuation"
MODEL_VERSION = "1.0.0"


class WinePriceFluctuationPlugin(ModelPluginPort):
    def __init__(self) -> None:
        self._ml_model = None
        self._scaler = None
        self._feature_columns: list[str] = []
        self._model_type: str = ""
        self._loaded: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        self._ml_model, self._scaler, self._feature_columns, self._model_type = load_artifacts()
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def predict_batch(self, *, data_path: str) -> dict:
        df_csv = pd.read_csv(data_path)
        records = [WeeklyPriceRecord(**row) for row in df_csv.to_dict(orient="records")]
        df_features = build_features(records, self._feature_columns)
        X_scaled = self._scaler.transform(df_features.values)

        predictions = []
        for i, idx in enumerate(df_features.index):
            X = X_scaled[[i]]
            proba = float(self._ml_model.predict_proba(X)[:, 1][0])
            pred = apply_threshold(proba)
            predictions.append(
                {
                    "prediction_date": idx.strftime("%Y-%m-%d"),
                    "prediction": pred,
                    "pred_proba_up": proba,
                    "model_type": self._model_type,
                }
            )

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
        records = [WeeklyPriceRecord(**r) for r in features["records"]]
        df_features = build_features(records, self._feature_columns)
        X_scaled = self._scaler.transform(df_features.values)
        X_last = X_scaled[[-1]]

        pred_proba_up = float(self._ml_model.predict_proba(X_last)[:, 1][0])
        actual_threshold = threshold if threshold is not None else 0.5
        prediction = apply_threshold(pred_proba_up, actual_threshold)
        prediction_date = df_features.index[-1].strftime("%Y-%m-%d")

        last_unscaled = df_features.iloc[-1]
        xai_feature_values: dict[str, float] = {
            col: float(last_unscaled[col])
            for col in self._feature_columns
            if col in last_unscaled.index and not math.isnan(float(last_unscaled[col]))
        }

        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info(
            "predict_inline done — prediction=%d proba=%.4f date=%s count=%d",
            prediction, pred_proba_up, prediction_date, self._predict_count,
        )

        return {
            "model_id": MODEL_NAME,
            "threshold": threshold,
            "prediction": prediction,
            "confidence": pred_proba_up,
            "features_used": self._feature_columns,
            "model_type": self._model_type,
            "prediction_date": prediction_date,
            "xai_feature_values": xai_feature_values or None,
        }

    def stats(self) -> StatsResponse:
        return StatsResponse(
            model_name=MODEL_NAME,
            model_type=self._model_type if self._loaded else "unknown",
            framework="sklearn/xgboost",
            artifact_path="model-runtime-wine_price_fluctuation/artifacts/",
            input_schema={
                "mode=inline": {
                    "records": {
                        "type": "array",
                        "min_items": 22,
                        "items": {
                            "campaign": "str — viticulture campaign e.g. '2023/2024'",
                            "week": "int — ISO week number (1-52)",
                            "price_red": "float — red wine price in EUR/hl",
                        },
                    }
                },
                "mode=batch": {"data_path": "str — path to CSV with weekly price records"},
            },
            output_schema={
                "batch": {"predictions": "list[dict] — per-week predictions"},
                "inline": {
                    "prediction": "int — 0 or 1 (1 = price rises >= 2.5% in next 4 weeks)",
                    "confidence": "float — probability of class 1 (0.0 to 1.0)",
                    "model_type": "str — 'logreg' or 'xgboost'",
                    "prediction_date": "str — ISO date of the predicted week",
                },
            },
            predict_count=self._predict_count,
            last_predict_at=self._last_predict_at,
        )
