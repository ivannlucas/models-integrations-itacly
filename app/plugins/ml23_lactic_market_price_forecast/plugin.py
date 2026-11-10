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
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.mlflow_artifacts import add_mlflow_run_to_stats, removing_dir
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.domain.services.xai_values import xai_feature_values
from app.infrastructure.artifact_store import local_file_path
from app.infrastructure.base_model_publisher import persist_retrained_bundle
from app.plugins.ml23_lactic_market_price_forecast.constants import (
    ARTIFACT_FOLDER_NAME,
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


_FEATURE_ROLES: tuple[tuple[Callable[[str], bool], str], ...] = (
    (lambda c: c == "current_price", (
        "Precio actual observado en t — ancla junto al resto de features para el pronóstico "
        "a horizonte"
    )),
    (lambda c: c.startswith("precio_lag_"), "Retardo del precio objetivo (autoregressive)"),
    (lambda c: c.startswith("media_movil_"),
     "Media móvil histórica basada en valores pasados (trend)"),
    (lambda c: c == "variacion_mensual",
     "Cambio porcentual mensual desplazado un periodo (trend)"),
    (lambda c: c in {"year", "mes", "trimestre", "es_verano", "es_navidad"},
     "Variable de calendario derivada de la fecha (calendar)"),
    (lambda c: c.endswith("_lag1"),
     "Variable exógena desplazada un periodo para evitar leakage (exogenous)"),
)


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
    for matches, description in _FEATURE_ROLES:
        if matches(column):
            return description
    return "Columna auxiliar"


class _RefitSplit(NamedTuple):
    """prepare_rnn_split_with_test's result, by name."""

    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_cp: np.ndarray
    scaler: Any


def _fit_gru(split: _RefitSplit) -> tuple[dict, float, Any]:
    """train_rnn_multi_seed with the deployed GRU config: (test_metrics, elapsed_s, model)."""
    device = torch.device("cpu")
    return train_rnn_multi_seed(
        GRUModel, split.train_x.shape[2], TRAIN_HIDDEN_SIZE,
        split.train_x, split.train_y, split.val_x, split.val_y,
        split.test_x, split.test_y, split.test_cp,
        n_seeds=TRAIN_N_SEEDS, seed_base=TRAIN_SEED_BASE, device=device,
        epochs=TRAIN_EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=TRAIN_LR,
        patience=TRAIN_PATIENCE,
    )


def _refit_split(data_path: str) -> tuple[list[str], _RefitSplit]:
    """Read the training CSV and build the refit train/val/test sequences (compare_models.py).

    Returns (feature_cols, prepare_rnn_split_with_test(...)).
    """
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
    return feature_cols, _RefitSplit(*prepare_rnn_split_with_test(
        refit_train_df, refit_val_df, test_df, feature_cols, TRAIN_SEQ_LEN,
    ))


def _write_bundle(upload_dir: Path, best_model, scaler, feature_cols: list[str]) -> None:
    """gru_model.pt + rnn_scaler.npz + manifest.json — the files the loader reads."""
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


class Ml23LacticMarketPriceForecastPlugin(PluginRuntimeMixin, ModelPluginPort):
    """GRU plugin for monthly dairy price forecasting at 6-month horizon."""

    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model

    def __init__(self) -> None:
        """Initialize unloaded plugin (runtime counters start at the mixin defaults)."""
        self._model: GRUModel | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._manifest: dict = {}

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
    def _scale(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        safe_scale = np.where(scale == 0, 1.0, scale)
        return (x - mean) / safe_scale

    @staticmethod
    def _infer_window(model: GRUModel, window: np.ndarray) -> float:
        """Run the GRU on a [seq_len, n_features] scaled array; return scalar."""
        x_t = torch.tensor(window[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            return float(model(x_t).cpu().numpy().reshape(-1)[0])

    def _slide(self, frame: pd.DataFrame, bundle: tuple, manifest: dict, extra: dict) -> list[dict]:
        """One prediction per seq_len window of an ordered frame (bundle = model, mean, scale);
        extra (producto/canal) goes right after "fecha" in each row."""
        model, mean, scale = bundle
        feature_cols: list[str] = manifest["feature_cols"]
        seq_len: int = int(manifest["seq_len"])
        rows: list[dict] = []
        x_sc = self._scale(frame[feature_cols].values, mean, scale)
        for i in range(seq_len - 1, len(frame)):
            pred = self._infer_window(model, x_sc[i - seq_len + 1: i + 1])
            r = frame.iloc[i]
            rows.append({
                "fecha": str(r.get("fecha", "")),
                **extra,
                "current_price": float(r.get("current_price", float("nan"))),
                "y_pred": round(pred, 4),
                "xai_feature_values": xai_feature_values(r, feature_cols),
            })
        return rows

    def _run_on_df(
        self, df: pd.DataFrame, model: GRUModel, mean, scale, manifest: dict
    ) -> list[dict]:
        """Slide a seq_len window per (producto, canal) group and collect predictions."""
        feature_cols: list[str] = manifest["feature_cols"]
        seq_len: int = int(manifest["seq_len"])
        rows: list[dict] = []

        if "producto" not in df.columns or "canal" not in df.columns:
            if len(df) < seq_len:
                logger.warning("Not enough rows (%d) for seq_len=%d", len(df), seq_len)
                return rows
            df_s = df.sort_values("fecha").reset_index(drop=True) if "fecha" in df.columns else df
            return self._slide(df_s, (model, mean, scale), manifest, {})

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
            rows.extend(self._slide(
                grp, (model, mean, scale), manifest,
                {"producto": str(producto), "canal": str(canal)},
            ))
        return rows

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Predict from a CSV of historical features; one y_pred per sliding window."""
        model, mean, scale, manifest, temp_dir = self._resolve_for_predict(mlflow_run_id)
        with removing_dir(temp_dir):
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
            self._record_latency(elapsed_ms)

            horizon: int = manifest.get("horizon", 6)
            for p in predictions:
                p["model_id"] = MODEL_ID
                p["horizon"] = horizon

            logger.info(
                "predict_batch done — %d preds in %.1fms count=%d",
                len(predictions), elapsed_ms, self._predict_count,
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None
            )

    def _inline_window(self, features: dict, mean, scale, manifest: dict) -> np.ndarray:
        """The feature dict tiled seq_len times (missing/None -> 0.0), scaled."""
        feature_cols: list[str] = manifest["feature_cols"]
        seq_len: int = int(manifest["seq_len"])

        row_values = [float(features.get(col) or 0.0) for col in feature_cols]
        x_seq = np.tile(np.array(row_values, dtype=np.float32), (seq_len, 1))
        return self._scale(x_seq, mean, scale)

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
        with removing_dir(temp_dir):
            feature_cols: list[str] = manifest["feature_cols"]
            x_sc = self._inline_window(features, mean, scale, manifest)

            t0 = time.perf_counter()
            pred = self._infer_window(model, x_sc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._record_latency(elapsed_ms)

            logger.info("predict_inline done — y_pred=%.4f count=%d", pred, self._predict_count)
            return PredictInlineResponse(
                model_id=MODEL_ID,
                prediction=round(pred, 4),
                confidence=None,
                horizon=manifest.get("horizon", 6),
                features_used=feature_cols,
                model_version=VERSION,
                xai_feature_values=xai_feature_values(features, feature_cols) or None,
            )

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Refit the GRU architecture already selected and deployed, faithfully porting
        src/training/compare_models.py's refit step (prepare_horizon_dataset ->
        split_dev_test -> make_refit_split -> prepare_rnn_split_with_test ->
        train_rnn_multi_seed) from inbox/a23/codigo/.

        Does not re-run the full model search (Naive/Drift/XGBoost/LSTM/GRU compared
        across CV folds) that originally selected GRU — see training.py module docstring.
        The refit bundle is published as the new base model version (never overwriting a
        previous one) and, if mlflow_run_id is given, also uploaded to that run.
        """
        upload_dir = Path(tempfile.mkdtemp(prefix="ml23_train_upload_"))
        with removing_dir(str(upload_dir)):
            feature_cols, split = _refit_split(data_path)
            test_metrics, elapsed_s, best_model = _fit_gru(split)

            _write_bundle(upload_dir, best_model, split.scaler, feature_cols)

            n_train = int(len(split.train_x))
            n_test = int(len(split.test_x))

            metrics = {
                "mae": test_metrics["MAE"], "rmse": test_metrics["RMSE"],
                "mape_pct": test_metrics["MAPE_pct"], "r2": test_metrics["R2"],
                "direction_acc_pct": test_metrics["direction_acc_pct"],
                "n_train": n_train, "n_test": n_test,
            }
            persisted = persist_retrained_bundle(
                str(upload_dir),
                artifact_folder=ARTIFACT_FOLDER_NAME,
                data_path=data_path,
                metrics=metrics,
                mlflow_run_id=mlflow_run_id,
                upload_to_mlflow=lambda: upload_artifacts_to_mlflow(
                    str(upload_dir), mlflow_run_id, metrics=metrics
                ),
            )

            logger.info(
                "train() done — mae=%.4f rmse=%.4f r2=%.4f n_train=%d n_test=%d time=%.1fs",
                test_metrics["MAE"], test_metrics["RMSE"], test_metrics["R2"],
                n_train, n_test, elapsed_s,
            )
            return TrainResponse(
                detail=(
                    "Reentrenamiento completado — modelo base "
                    f"{persisted.base_version['version']}."
                ),
                mae=round(test_metrics["MAE"], 4),
                rmse=round(test_metrics["RMSE"], 4),
                mape_pct=test_metrics["MAPE_pct"],
                r2=round(test_metrics["R2"], 4),
                direction_acc_pct=test_metrics["direction_acc_pct"],
                n_train=n_train,
                n_test=n_test,
                training_time_s=round(elapsed_s, 1),
                upload_warning=persisted.upload_warning,
                base_version=persisted.base_version,
            )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        feature_cols: list[str] = self._manifest.get("feature_cols", [])
        metrics = {
            "horizon_months": self._manifest.get("horizon", 6),
            "seq_len": self._manifest.get("seq_len", 6),
            "hidden_size": self._manifest.get("hidden_size", 64),
            "num_features": len(feature_cols),
            "selected_model": self._manifest.get("selected_model", "GRU"),
        }
        return add_mlflow_run_to_stats(StatsResponse(
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
                avg_latency_ms=self._avg_latency_ms(1),
            ),
        ), mlflow_run_id)
