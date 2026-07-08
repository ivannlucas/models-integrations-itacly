"""Ml23LacticMarketPriceForecastPlugin — GRU-based dairy price forecasting.

Predicts monthly lácteo prices (whole/skim/semi-skim milk) at a 6-month horizon
using a pre-trained GRU model. Externally trained; ``train()`` raises 501.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml23_lactic_market_price_forecast.constants import (
    FRAMEWORK,
    MODEL_ID,
    VERSION,
)
from app.plugins.ml23_lactic_market_price_forecast.model_loader import load_model_bundle
from app.plugins.ml23_lactic_market_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml23_lactic_market_price_forecast.rnn_models import GRUModel

logger = logging.getLogger(__name__)


class Ml23LacticMarketPriceForecastPlugin(ModelPluginPort):
    """GRU plugin for monthly dairy price forecasting at 6-month horizon."""

    def __init__(self) -> None:
        """Initialize unloaded plugin with empty runtime counters."""
        self._model: GRUModel | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._manifest: dict = {}
        self._predict_count: int = 0
        self._total_latency_ms: float = 0.0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        """Load GRU model, scaler and manifest from artifact store."""
        self._model, self._scaler_mean, self._scaler_scale, self._manifest = (
            load_model_bundle()
        )
        logger.info(
            "Ml23 loaded — input_size=%d hidden_size=%d model=%s",
            len(self._manifest.get("feature_cols", [])),
            self._manifest.get("hidden_size", 0),
            self._manifest.get("selected_model", "GRU"),
        )

    def is_loaded(self) -> bool:
        """Return True when the GRU model is ready for inference."""
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _scale(self, X: np.ndarray) -> np.ndarray:
        safe_scale = np.where(self._scaler_scale == 0, 1.0, self._scaler_scale)
        return (X - self._scaler_mean) / safe_scale

    def _infer_window(self, window: np.ndarray) -> float:
        """Run the GRU on a [seq_len, n_features] scaled array; return scalar."""
        X_t = torch.tensor(window[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            return float(self._model(X_t).cpu().numpy().reshape(-1)[0])

    def _record(self, elapsed_ms: float) -> None:
        self._predict_count += 1
        self._total_latency_ms += elapsed_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _run_on_df(self, df: pd.DataFrame) -> list[dict]:
        """Slide a seq_len window per (producto, canal) group and collect predictions."""
        feature_cols: list[str] = self._manifest["feature_cols"]
        seq_len: int = int(self._manifest["seq_len"])
        rows: list[dict] = []

        if "producto" not in df.columns or "canal" not in df.columns:
            if len(df) < seq_len:
                logger.warning("Not enough rows (%d) for seq_len=%d", len(df), seq_len)
                return rows
            df_s = df.sort_values("fecha").reset_index(drop=True) if "fecha" in df.columns else df
            X_sc = self._scale(df_s[feature_cols].values)
            for i in range(seq_len - 1, len(df_s)):
                pred = self._infer_window(X_sc[i - seq_len + 1: i + 1])
                r = df_s.iloc[i]
                rows.append({
                    "fecha": str(r.get("fecha", "")),
                    "current_price": float(r.get("current_price", float("nan"))),
                    "y_pred": round(pred, 4),
                })
            return rows

        for (producto, canal), grp in df.groupby(["producto", "canal"]):
            grp = grp.sort_values("fecha").reset_index(drop=True)
            if len(grp) < seq_len:
                logger.warning(
                    "Skipping (%s, %s): %d rows < seq_len=%d", producto, canal, len(grp), seq_len
                )
                continue
            missing = [c for c in feature_cols if c not in grp.columns]
            if missing:
                logger.warning("Missing cols for (%s, %s): %s", producto, canal, missing)
                continue
            X_sc = self._scale(grp[feature_cols].values)
            for i in range(seq_len - 1, len(grp)):
                pred = self._infer_window(X_sc[i - seq_len + 1: i + 1])
                r = grp.iloc[i]
                rows.append({
                    "fecha": str(r.get("fecha", "")),
                    "producto": str(producto),
                    "canal": str(canal),
                    "current_price": float(r.get("current_price", float("nan"))),
                    "y_pred": round(pred, 4),
                })
        return rows

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Predict from a CSV of historical features; one y_pred per sliding window."""
        _ = mlflow_run_id
        self._require_loaded()

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        t0 = time.perf_counter()
        predictions = self._run_on_df(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record(elapsed_ms)

        feature_cols: list[str] = self._manifest["feature_cols"]
        horizon: int = self._manifest.get("horizon", 6)
        xai_fv: dict[str, float] = {}
        if not df.empty:
            last = df.iloc[-1]
            for col in feature_cols:
                if col in last.index:
                    try:
                        v = float(last[col])
                        if not np.isnan(v):
                            xai_fv[col] = v
                    except (TypeError, ValueError):
                        pass

        for i, p in enumerate(predictions):
            p["model_id"] = MODEL_ID
            p["horizon"] = horizon
            if i == 0:
                p["xai_feature_values"] = xai_fv

        logger.info(
            "predict_batch done — %d preds in %.1fms count=%d",
            len(predictions), elapsed_ms, self._predict_count,
        )
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict from a single feature dict (tiled seq_len times to build the sequence)."""
        _ = model_key, threshold, mlflow_run_id
        self._require_loaded()

        feature_cols: list[str] = self._manifest["feature_cols"]
        seq_len: int = int(self._manifest["seq_len"])

        row_values = [float(features.get(col) or 0.0) for col in feature_cols]
        X_seq = np.tile(np.array(row_values, dtype=np.float32), (seq_len, 1))
        X_sc = self._scale(X_seq)

        t0 = time.perf_counter()
        pred = self._infer_window(X_sc)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._record(elapsed_ms)

        xai_fv: dict[str, float] = {}
        for col in feature_cols:
            val = features.get(col)
            if val is not None:
                try:
                    fv = float(val)
                    if not np.isnan(fv):
                        xai_fv[col] = fv
                except (TypeError, ValueError):
                    pass

        logger.info("predict_inline done — y_pred=%.4f count=%d", pred, self._predict_count)
        return PredictInlineResponse(
            model_id=MODEL_ID,
            prediction=round(pred, 4),
            confidence=None,
            horizon=self._manifest.get("horizon", 6),
            features_used=feature_cols,
            model_version=VERSION,
            xai_feature_values=xai_fv or None,
        )

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> None:
        """Raise TrainingNotSupportedError — model is externally trained."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml23 usa artefactos GRU entrenados externamente. "
            "Re-entrena con el script src/training/compare_models.py del repo "
            "a23-rnn-dairy-prediccion y sube los artefactos a S3 bajo "
            "artifacts/fixed/ml23_lactic_market_price_forecast/."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        _ = mlflow_run_id
        feature_cols: list[str] = self._manifest.get("feature_cols", [])
        avg = self._total_latency_ms / self._predict_count if self._predict_count else None
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción mensual del precio de productos lácteos (leche entera, "
                "desnatada, semidesnatada) a horizonte de 6 meses mediante GRU (PyTorch). "
                "Sectores: DISCOUNTS, HIPERMERCADOS, SUPER+AUTOS, T.ESPAÑA."
            ),
            task_type="time_series_regression",
            framework=FRAMEWORK,
            inputs=[
                InputField(
                    name="data_path",
                    type="str",
                    description=(
                        f"CSV con {len(feature_cols)} features del dataset lácteo "
                        f"(columnas: {', '.join(feature_cols[:4])}…)"
                    ),
                ),
            ],
            outputs=[
                OutputField(
                    name="y_pred",
                    type="float",
                    description="Precio predicho (€/litro) a 6 meses vista",
                ),
            ],
            metrics={
                "horizon_months": self._manifest.get("horizon", 6),
                "seq_len": self._manifest.get("seq_len", 6),
                "hidden_size": self._manifest.get("hidden_size", 64),
                "num_features": len(feature_cols),
                "selected_model": self._manifest.get("selected_model", "GRU"),
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=round(avg, 1) if avg is not None else None,
            ),
        )
