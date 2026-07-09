"""Ml17MeatMarketPriceAnalysisPlugin — Ridge pork price forecast (official_v1_4).

Predicts pork class E Spain price at t+1 (€/100 kg) from 6 exogenous features
plus auto-computed month_sin/cos derived from a reference date.
Externally trained; ``train()`` raises 501.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml17_meat_market_price_analysis.constants import (
    FEATURE_COLUMNS,
    FRAMEWORK,
    LINE,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml17_meat_market_price_analysis.model_loader import load_model
from app.plugins.ml17_meat_market_price_analysis.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)

logger = logging.getLogger(__name__)

_EMPTY_TOKENS = {"", "nan", "none", "null", "nat"}


def _is_blank(v: Any) -> bool:
    return v is None or str(v).strip().lower() in _EMPTY_TOKENS


def _parse_date(date_val: Any) -> datetime:
    """Parse ISO string or epoch (ms or s) to datetime."""
    s = str(date_val).strip()
    if not s:
        raise ValueError("empty date value")
    numeric = s.replace(".", "", 1)
    if numeric.isdigit() and "-" not in s and "/" not in s:
        ts = float(s)
        if ts > 1e11:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    return datetime.strptime(s[:10], "%Y-%m-%d")


def _iso_date(date_val: Any) -> str:
    """Normalise any date value to YYYY-MM-DD string (best-effort)."""
    try:
        return _parse_date(date_val).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError, OSError):
        return str(date_val)[:10]


def _seasonal(date_val: Any) -> tuple[float, float]:
    """Return (month_sin, month_cos) from a date value."""
    try:
        month = _parse_date(date_val).month
    except (ValueError, TypeError, OverflowError, OSError) as exc:
        raise ValueError(f"Invalid date '{date_val}': {exc}") from exc
    return math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12)


def _find_date(row: dict) -> Any:
    """Locate a usable date value from a row, tolerant of column name variants."""
    for key in ("date", "fecha", "Date", "DATE", "Fecha"):
        if key in row and not _is_blank(row[key]):
            return row[key]
    for k, v in row.items():
        if ("date" in str(k).lower() or "fecha" in str(k).lower()) and not _is_blank(v):
            return v
    return None


def _seasonal_from_row(row: dict) -> tuple[float, float]:
    """Resolve (month_sin, month_cos): precomputed columns first, then date column."""
    ms, mc = row.get("month_sin"), row.get("month_cos")
    if not _is_blank(ms) and not _is_blank(mc):
        try:
            return float(ms), float(mc)
        except (ValueError, TypeError):
            pass
    date_val = _find_date(row)
    if date_val is not None:
        return _seasonal(date_val)
    raise ValueError(
        "no usable 'date'/'month_sin'/'month_cos' column found; "
        f"available: {list(row.keys())}"
    )


class Ml17MeatMarketPriceAnalysisPlugin(ModelPluginPort):
    """Ridge regression plugin for pork price forecasting (official_v1_4)."""

    def __init__(self) -> None:
        """Initialize unloaded plugin with empty runtime counters."""
        self._model: Any = None
        self._predict_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load the Ridge pickle via ArtifactStore."""
        self._model = load_model()
        logger.info("Ml17 loaded — %s (%s)", MODEL_ID, LINE)

    def is_loaded(self) -> bool:
        """Return True when the Ridge model is ready for inference."""
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _build_frame(self, features: dict) -> pd.DataFrame:
        month_sin, month_cos = _seasonal_from_row(features)
        row = {
            "target_price_pigmeat_class_e_es": float(
                features["target_price_pigmeat_class_e_es"]
            ),
            "eurostat_pigmeat_slaughter_tonnes_es": float(
                features["eurostat_pigmeat_slaughter_tonnes_es"]
            ),
            "eurostat_pigmeat_slaughter_tonnes_eu": float(
                features["eurostat_pigmeat_slaughter_tonnes_eu"]
            ),
            "cereal_feed_barley_price_monthly": float(
                features["cereal_feed_barley_price_monthly"]
            ),
            "cereal_feed_maize_price_monthly": float(
                features["cereal_feed_maize_price_monthly"]
            ),
            "mapa_porcino_otras_razas_price_monthly": float(
                features["mapa_porcino_otras_razas_price_monthly"]
            ),
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        return pd.DataFrame([row])[FEATURE_COLUMNS]

    def _record(self, elapsed_ms: float) -> None:
        self._predict_count += 1
        self._total_latency_ms += elapsed_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict pork price t+1 from a single feature dict."""
        _ = model_key, threshold, mlflow_run_id
        self._require_loaded()

        date_str = features.get("date", "")
        t0 = time.perf_counter()
        X = self._build_frame(features)
        y_pred = float(self._model.predict(X)[0])
        self._record((time.perf_counter() - t0) * 1000)

        xai_fv = {col: float(X.iloc[0][col]) for col in FEATURE_COLUMNS}
        logger.info(
            "predict_inline done — date='%s' y_pred=%.4f count=%d",
            date_str, y_pred, self._predict_count,
        )
        return PredictInlineResponse(
            model_id=MODEL_ID,
            line=LINE,
            prediction=y_pred,
            y_pred=y_pred,
            confidence=None,
            base_date=_iso_date(date_str),
            xai_feature_values=xai_fv,
        )

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Predict pork price t+1 for every row in a CSV."""
        _ = mlflow_run_id
        self._require_loaded()

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        predictions: list[dict] = []
        t0 = time.perf_counter()
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            date_val = _find_date(row_dict)
            try:
                X = self._build_frame(row_dict)
                y_pred = float(self._model.predict(X)[0])
                predictions.append({
                    "row": int(idx),
                    "date": _iso_date(date_val) if date_val is not None else "",
                    "y_pred": y_pred,
                    "model_id": MODEL_ID,
                    "line": LINE,
                    "xai_feature_values": {col: float(X.iloc[0][col]) for col in FEATURE_COLUMNS},
                })
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Error en fila %s: %s", idx, exc)
                predictions.append({
                    "row": int(idx),
                    "date": _iso_date(date_val) if date_val is not None else "",
                    "error": str(exc),
                })
        self._record((time.perf_counter() - t0) * 1000)
        logger.info(
            "predict_batch done — %d rows count=%d", len(predictions), self._predict_count
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, line=LINE, predictions=predictions, output_path=None
        )

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> None:
        """Raise TrainingNotSupportedError — model is externally trained."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml17 usa artefactos Ridge entrenados externamente (official_v1_4). "
            "Re-entrena con el pipeline de ciencia de datos y sube el pickle a S3 "
            "bajo artifacts/fixed/ml17_meat_market_price_analysis/."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        _ = mlflow_run_id
        avg = self._total_latency_ms / self._predict_count if self._predict_count else None
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción mensual del precio de porcino clase E España (€/100 kg) a t+1 "
                "mediante regresión Ridge (sklearn). Features: Eurostat + precios pienso + MAPA. "
                f"Línea operativa: {LINE}. Benchmark OOS: MAE=4.46, RMSE=6.41, R²=0.91."
            ),
            task_type="time_series_regression",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="date",
                    type="str",
                    description="Fecha de referencia (YYYY-MM-DD) — genera month_sin/cos automáticamente",
                ),
                InputField(
                    name="target_price_pigmeat_class_e_es",
                    type="float",
                    description="Precio porcino clase E España en t (€/100 kg) — lag autorregresivo",
                ),
                InputField(
                    name="eurostat_pigmeat_slaughter_tonnes_es",
                    type="float",
                    description="Sacrificio porcino España (Eurostat, miles de toneladas)",
                ),
                InputField(
                    name="eurostat_pigmeat_slaughter_tonnes_eu",
                    type="float",
                    description="Sacrificio porcino UE (Eurostat, miles de toneladas)",
                ),
                InputField(
                    name="cereal_feed_barley_price_monthly",
                    type="float",
                    description="Precio mensual cebada pienso (€/tonelada)",
                ),
                InputField(
                    name="cereal_feed_maize_price_monthly",
                    type="float",
                    description="Precio mensual maíz pienso (€/tonelada)",
                ),
                InputField(
                    name="mapa_porcino_otras_razas_price_monthly",
                    type="float",
                    description="Precio mensual porcino otras razas MAPA (€/100 kg)",
                ),
            ],
            outputs=[
                OutputField(
                    name="y_pred",
                    type="float",
                    description="Precio predicho porcino clase E España a t+1 (€/100 kg)",
                ),
            ],
            metrics={
                "mae": 4.46,
                "rmse": 6.41,
                "r2_oos": 0.91,
                "n_test": 31,
                "benchmark": LINE,
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=round(avg, 1) if avg is not None else None,
            ),
        )
