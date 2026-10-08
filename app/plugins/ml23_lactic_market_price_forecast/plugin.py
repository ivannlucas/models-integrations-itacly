"""Ml23LacticMarketPriceForecastPlugin — GRU-based dairy price forecasting.

Predicts monthly lácteo prices (whole/skim/semi-skim milk) at a 6-month horizon
using a pre-trained GRU model. Real fine-tuning is supported (train()) — it refits the
already-selected GRU architecture (src/training/compare_models.py from inbox/a23/codigo/)
on new data, persisting the refit model to MLflow only, never overwriting the served
artifact.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, ModelPersistenceError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml23_lactic_market_price_forecast.constants import (
    FRAMEWORK,
    MODEL_ID,
    TRAIN_BATCH_SIZE,
    TRAIN_CV_FOLDS,
    TRAIN_CV_VAL_SIZE,
    TRAIN_EPOCHS,
    TRAIN_HIDDEN_SIZE,
    TRAIN_HORIZON,
    TRAIN_LR,
    TRAIN_N_SEEDS,
    TRAIN_PATIENCE,
    TRAIN_SEED_BASE,
    TRAIN_SEQ_LEN,
    TRAIN_TEST_RATIO,
    TRAIN_VAL_RATIO,
    VERSION,
)
from app.plugins.ml23_lactic_market_price_forecast.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml23_lactic_market_price_forecast.model_loader import load_model_bundle
from app.plugins.ml23_lactic_market_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml23_lactic_market_price_forecast.rnn_models import GRUModel
from app.plugins.ml23_lactic_market_price_forecast.train_dto import TrainResponse
from app.plugins.ml23_lactic_market_price_forecast.training import (
    get_feature_cols,
    make_refit_split,
    prepare_horizon_dataset,
    prepare_rnn_split_with_test,
    resolve_cv_val_window,
    split_dev_test,
    train_rnn_multi_seed,
)

logger = logging.getLogger(__name__)


def _describe_feature(column: str) -> str:
    """Classify a feature_cols entry by role, mirroring a23's own
    src/get_stats/column_info.py::_describe_column() (the reference repo's
    authoritative column-role classification, never reproduced here before —
    stats() used to just name the CSV and truncate the first 4 columns).
    'current_price' is the one addition: it never appears in that repo's own
    dataset_forecast_ready.csv (column_info.py is run against that file), so
    the original classifier never had to describe it — it's only assembled at
    train/inference time from target_precio_medio (see compare_models.py and
    predictor.py::_prepare_input_df(), mirrored in predict_batch() above)."""
    if column == "current_price":
        return "Precio actual observado en t — ancla junto al resto de features para el pronóstico a horizonte"
    if column.startswith("precio_lag_"):
        return "Retardo del precio objetivo (autoregressive)"
    if column.startswith("media_movil_"):
        return "Media móvil histórica basada en valores pasados (trend)"
    if column == "variacion_mensual":
        return "Cambio porcentual mensual desplazado un periodo (trend)"
    if column in {"year", "mes", "trimestre", "es_verano", "es_navidad"}:
        return "Variable de calendario derivada de la fecha (calendar)"
    if column.endswith("_lag1"):
        return "Variable exógena desplazada un periodo para evitar leakage (exogenous)"
    return "Columna auxiliar"


def _xai_values_from_row(row, feature_cols: list[str]) -> dict[str, float]:
    """Build xai_feature_values from a features dict (inline) or DataFrame row (batch)."""
    xai: dict[str, float] = {}
    for col in feature_cols:
        val = row.get(col)
        if val is not None:
            try:
                fv = float(val)
                if not np.isnan(fv):
                    xai[col] = fv
            except (TypeError, ValueError):
                pass
    return xai


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

    def _resolve_for_predict(self, mlflow_run_id: str):
        """Return (model, mean, scale, manifest, temp_dir_to_cleanup) — the user's MLflow
        bundle if mlflow_run_id is given and downloadable, otherwise the served base
        bundle. Never mutates self.*, so concurrent requests without mlflow_run_id are
        unaffected by one that uses a retrained bundle."""
        self._require_loaded()
        if not mlflow_run_id:
            return self._model, self._scaler_mean, self._scaler_scale, self._manifest, None
        logger.info("Using user-trained GRU bundle from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        model, mean, scale, manifest, temp_dir = loaded
        return model, mean, scale, manifest, temp_dir

    @staticmethod
    def _scale(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        safe_scale = np.where(scale == 0, 1.0, scale)
        return (X - mean) / safe_scale

    @staticmethod
    def _infer_window(model: GRUModel, window: np.ndarray) -> float:
        """Run the GRU on a [seq_len, n_features] scaled array; return scalar."""
        X_t = torch.tensor(window[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            return float(model(X_t).cpu().numpy().reshape(-1)[0])

    def _record(self, elapsed_ms: float) -> None:
        self._predict_count += 1
        self._total_latency_ms += elapsed_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _run_on_df(self, df: pd.DataFrame, model: GRUModel, mean, scale, manifest: dict) -> list[dict]:
        """Slide a seq_len window per (producto, canal) group and collect predictions."""
        feature_cols: list[str] = manifest["feature_cols"]
        seq_len: int = int(manifest["seq_len"])
        rows: list[dict] = []

        if "producto" not in df.columns or "canal" not in df.columns:
            if len(df) < seq_len:
                logger.warning("Not enough rows (%d) for seq_len=%d", len(df), seq_len)
                return rows
            df_s = df.sort_values("fecha").reset_index(drop=True) if "fecha" in df.columns else df
            X_sc = self._scale(df_s[feature_cols].values, mean, scale)
            for i in range(seq_len - 1, len(df_s)):
                pred = self._infer_window(model, X_sc[i - seq_len + 1: i + 1])
                r = df_s.iloc[i]
                rows.append({
                    "fecha": str(r.get("fecha", "")),
                    "current_price": float(r.get("current_price", float("nan"))),
                    "y_pred": round(pred, 4),
                    "xai_feature_values": _xai_values_from_row(r, feature_cols),
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
            X_sc = self._scale(grp[feature_cols].values, mean, scale)
            for i in range(seq_len - 1, len(grp)):
                pred = self._infer_window(model, X_sc[i - seq_len + 1: i + 1])
                r = grp.iloc[i]
                rows.append({
                    "fecha": str(r.get("fecha", "")),
                    "producto": str(producto),
                    "canal": str(canal),
                    "current_price": float(r.get("current_price", float("nan"))),
                    "y_pred": round(pred, 4),
                    "xai_feature_values": _xai_values_from_row(r, feature_cols),
                })
        return rows

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Predict from a CSV of historical features; one y_pred per sliding window."""
        model, mean, scale, manifest, temp_dir = self._resolve_for_predict(mlflow_run_id)
        try:
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            if "fecha" in df.columns:
                df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            # dataset_forecast_ready.csv (el propio dataset de entrenamiento/inferencia del
            # modelo) trae 'target_precio_medio', no 'current_price' -- misma derivación que
            # el código original (predictor.py::_prepare_input_df()). Sin ella, _run_on_df()
            # descarta todas las filas de todos los grupos por falta de 'current_price' y
            # predict_batch devuelve predictions=[] en silencio (ver inbox/a23/manifest.yaml
            # known_issues).
            if "current_price" not in df.columns and "target_precio_medio" in df.columns:
                df["current_price"] = df["target_precio_medio"]

            t0 = time.perf_counter()
            predictions = self._run_on_df(df, model, mean, scale, manifest)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record(elapsed_ms)

            horizon: int = manifest.get("horizon", 6)
            for p in predictions:
                p["model_id"] = MODEL_ID
                p["horizon"] = horizon

            logger.info(
                "predict_batch done — %d preds in %.1fms count=%d",
                len(predictions), elapsed_ms, self._predict_count,
            )
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict from a single feature dict (tiled seq_len times to build the sequence)."""
        _ = model_key, threshold
        model, mean, scale, manifest, temp_dir = self._resolve_for_predict(mlflow_run_id)
        try:
            feature_cols: list[str] = manifest["feature_cols"]
            seq_len: int = int(manifest["seq_len"])

            row_values = [float(features.get(col) or 0.0) for col in feature_cols]
            X_seq = np.tile(np.array(row_values, dtype=np.float32), (seq_len, 1))
            X_sc = self._scale(X_seq, mean, scale)

            t0 = time.perf_counter()
            pred = self._infer_window(model, X_sc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record(elapsed_ms)

            logger.info("predict_inline done — y_pred=%.4f count=%d", pred, self._predict_count)
            return PredictInlineResponse(
                model_id=MODEL_ID,
                prediction=round(pred, 4),
                confidence=None,
                horizon=manifest.get("horizon", 6),
                features_used=feature_cols,
                model_version=VERSION,
                xai_feature_values=_xai_values_from_row(features, feature_cols) or None,
            )
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def train(  # pylint: disable=too-many-locals
        self, *, data_path: str, mlflow_run_id: str
    ) -> TrainResponse:
        """Refit the GRU architecture already selected and deployed, faithfully porting
        src/training/compare_models.py's refit step (prepare_horizon_dataset ->
        split_dev_test -> make_refit_split -> prepare_rnn_split_with_test ->
        train_rnn_multi_seed) from inbox/a23/codigo/.

        Does not re-run the full model search (Naive/Drift/XGBoost/LSTM/GRU compared
        across CV folds) that originally selected GRU — see training.py module docstring.
        The refit checkpoint is written only to a temp dir and uploaded to MLflow; the
        served artifacts/.../gru_model.pt is never overwritten.
        """
        upload_dir = Path(tempfile.mkdtemp(prefix="ml23_train_upload_"))
        upload_warning = None
        try:
            with local_file_path(data_path) as local_path:
                raw_df = pd.read_csv(local_path, parse_dates=["fecha"])
            raw_df = raw_df.sort_values(["producto", "canal", "fecha"]).reset_index(drop=True)

            df = prepare_horizon_dataset(raw_df, horizon=TRAIN_HORIZON)
            feature_cols = get_feature_cols(df)

            dev_df, test_df = split_dev_test(df, test_ratio=TRAIN_TEST_RATIO)
            cv_val_window = resolve_cv_val_window(
                n_dev_dates=dev_df["fecha"].nunique(),
                n_splits=TRAIN_CV_FOLDS,
                seq_len=TRAIN_SEQ_LEN,
                cv_val_size=TRAIN_CV_VAL_SIZE,
                val_ratio=TRAIN_VAL_RATIO,
            )
            refit_train_df, refit_val_df = make_refit_split(
                dev_df, val_window=cv_val_window, min_train_dates=max(TRAIN_SEQ_LEN, 1),
            )
            train_X, train_y, val_X, val_y, test_X, test_y, test_cp, scaler = (
                prepare_rnn_split_with_test(
                    refit_train_df, refit_val_df, test_df, feature_cols, TRAIN_SEQ_LEN,
                )
            )

            device = torch.device("cpu")
            test_metrics, elapsed_s, best_model = train_rnn_multi_seed(
                GRUModel, train_X.shape[2], TRAIN_HIDDEN_SIZE,
                train_X, train_y, val_X, val_y, test_X, test_y, test_cp,
                n_seeds=TRAIN_N_SEEDS, seed_base=TRAIN_SEED_BASE, device=device,
                epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=TRAIN_LR,
                patience=TRAIN_PATIENCE,
            )

            torch.save(best_model.state_dict(), upload_dir / "gru_model.pt")
            np.savez(upload_dir / "rnn_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
            manifest_out = {
                "selected_model": "GRU",
                "horizon": TRAIN_HORIZON,
                "seq_len": TRAIN_SEQ_LEN,
                "hidden_size": TRAIN_HIDDEN_SIZE,
                "feature_cols": feature_cols,
                "group_cols": ["producto", "canal"],
                "time_col": "fecha",
                "target_col": "target_h",
                "current_price_col": "current_price",
            }
            with open(upload_dir / "manifest.json", "w", encoding="utf-8") as fh:
                json.dump(manifest_out, fh, indent=2, ensure_ascii=False)

            n_train = int(len(train_X))
            n_test = int(len(test_X))

            if not mlflow_run_id:
                raise ModelPersistenceError("Sin run de MLflow: el modelo reentrenado no se ha guardado.")
            else:
                try:
                    upload_artifacts_to_mlflow(
                        str(upload_dir), mlflow_run_id,
                        metrics={
                            "mae": test_metrics["MAE"], "rmse": test_metrics["RMSE"],
                            "mape_pct": test_metrics["MAPE_pct"], "r2": test_metrics["R2"],
                            "direction_acc_pct": test_metrics["direction_acc_pct"],
                            "n_train": n_train, "n_test": n_test,
                        },
                    )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.error("MLflow artifact upload failed: %s", exc)
                    raise ModelPersistenceError(f"El modelo reentrenado no se ha podido guardar en MLflow: {exc}") from exc

            logger.info(
                "train() done — mae=%.4f rmse=%.4f r2=%.4f n_train=%d n_test=%d time=%.1fs",
                test_metrics["MAE"], test_metrics["RMSE"], test_metrics["R2"],
                n_train, n_test, elapsed_s,
            )
            return TrainResponse(
                detail="Reentrenamiento completado",
                mae=round(test_metrics["MAE"], 4),
                rmse=round(test_metrics["RMSE"], 4),
                mape_pct=test_metrics["MAPE_pct"],
                r2=round(test_metrics["R2"], 4),
                direction_acc_pct=test_metrics["direction_acc_pct"],
                n_train=n_train,
                n_test=n_test,
                training_time_s=round(elapsed_s, 1),
                upload_warning=upload_warning,
            )
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        feature_cols: list[str] = self._manifest.get("feature_cols", [])
        avg = self._total_latency_ms / self._predict_count if self._predict_count else None
        metrics = {
            "horizon_months": self._manifest.get("horizon", 6),
            "seq_len": self._manifest.get("seq_len", 6),
            "hidden_size": self._manifest.get("hidden_size", 64),
            "num_features": len(feature_cols),
            "selected_model": self._manifest.get("selected_model", "GRU"),
        }
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning(
                    "Could not fetch MLflow metrics for run_id=%s: %s", mlflow_run_id, exc
                )
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
                InputField(name=col, type="float", description=_describe_feature(col))
                for col in feature_cols
            ],
            outputs=[
                OutputField(
                    name="y_pred",
                    type="float",
                    description="Precio predicho (€/litro) a 6 meses vista",
                ),
            ],
            metrics=metrics,
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=round(avg, 1) if avg is not None else None,
            ),
        )
