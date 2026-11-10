"""Ml17MeatMarketPriceAnalysisPlugin — Ridge pork price forecast (official_v1_4).

Predicts pork class E Spain price at t+1 (€/100 kg) from 6 exogenous features
plus auto-computed month_sin/cos derived from a reference date. ``train()`` refits the
Ridge pipeline and publishes it as the new base model (see train()).
"""
from __future__ import annotations

import logging
import math
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.mlflow_artifacts import add_mlflow_run_to_stats, removing_dir
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.artifact_store import local_file_path
from app.infrastructure.base_model_publisher import persist_retrained_bundle
from app.plugins.ml17_meat_market_price_analysis.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    FRAMEWORK,
    IMPUTER_STRATEGY,
    LINE,
    MODEL_FILENAME,
    MODEL_ID,
    RIDGE_ALPHA,
    VERSION,
)
from app.plugins.ml17_meat_market_price_analysis.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml17_meat_market_price_analysis.model_loader import load_model
from app.plugins.ml17_meat_market_price_analysis.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml17_meat_market_price_analysis.train_dto import TrainResponse

logger = logging.getLogger(__name__)

TARGET_COLUMN = "target_price_pigmeat_class_e_es"
DATE_COLUMN = "date"
TRAIN_REQUIRED_COLUMNS = [DATE_COLUMN, *FEATURE_COLUMNS]

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


def _read_train_frame(data_path: str) -> pd.DataFrame:
    """Read the monthly training CSV, check it and sort it by date."""
    with local_file_path(data_path) as local_path:
        df = pd.read_csv(local_path)

    missing = [c for c in TRAIN_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV falta columnas requeridas: {missing}")
    if len(df) < 2:
        raise ValueError(
            "Se requieren al menos 2 filas mensuales para entrenar (supervisión "
            "one-step-ahead t -> t+1)."
        )

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def _build_ridge_pipeline():
    """ridge.py::build_ridge_pipeline: SimpleImputer(median) -> StandardScaler -> Ridge."""
    # pylint: disable=import-outside-toplevel
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # pylint: enable=import-outside-toplevel

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=IMPUTER_STRATEGY)),
            ("scaler", StandardScaler()),
        ]
    )
    return Pipeline(steps=[
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[("numeric", numeric_preprocessor, list(FEATURE_COLUMNS))],
                remainder="drop",
            ),
        ),
        ("model", Ridge(alpha=RIDGE_ALPHA)),
    ])


def _train_metrics(
    y_train: np.ndarray, y_pred: np.ndarray, *,
    previous_actual: np.ndarray, train_series: np.ndarray,
) -> dict:
    """In-sample MAE/RMSE/n_samples plus MASE, R² and directional accuracy when defined."""
    mae = float(np.mean(np.abs(y_train - y_pred)))
    rmse = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))

    diff_scale = float(np.mean(np.abs(np.diff(train_series)))) if len(train_series) > 1 else 0.0
    mase = (mae / diff_scale) if not np.isclose(diff_scale, 0.0) else None

    denom = float(np.sum((y_train - np.mean(y_train)) ** 2))
    r2_train = (
        (1.0 - float(np.sum((y_train - y_pred) ** 2)) / denom)
        if not np.isclose(denom, 0.0)
        else None
    )

    actual_direction = np.sign(y_train - previous_actual)
    predicted_direction = np.sign(y_pred - previous_actual)
    valid_mask = (actual_direction != 0) & (predicted_direction != 0)
    directional_accuracy = (
        float(np.mean(actual_direction[valid_mask] == predicted_direction[valid_mask]))
        if np.any(valid_mask) else None
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "n_samples": len(y_train),
        **({"mase": mase} if mase is not None else {}),
        **({"r2_train": r2_train} if r2_train is not None else {}),
        **(
            {"directional_accuracy": directional_accuracy}
            if directional_accuracy is not None
            else {}
        ),
    }


def _round4(value: float | None) -> float | None:
    """Round an optional metric to 4 decimals for the response."""
    return round(value, 4) if value is not None else None


class Ml17MeatMarketPriceAnalysisPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Ridge regression plugin for pork price forecasting (official_v1_4)."""

    BASE_ARTIFACT_FOLDER = ARTIFACT_FOLDER_NAME   # every /train publishes the new base model

    def __init__(self) -> None:
        """Initialize unloaded plugin (runtime counters start at the mixin defaults)."""
        self._model: Any = None

    def load(self) -> None:
        """Load the Ridge pickle via ArtifactStore."""
        self._model = load_model()
        logger.info("Ml17 loaded — %s (%s)", MODEL_ID, LINE)

    def is_loaded(self) -> bool:
        """Return True when the Ridge model is ready for inference."""
        return self._model is not None

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

    def _resolve_model(self, mlflow_run_id: str) -> tuple[Any, str | None]:
        """Return (model, temp_dir) to serve this request.

        The user's retrained model (if mlflow_run_id) is returned, never assigned to self:
        the plugin instance is shared by concurrent requests, so swapping it into self leaked
        a user's model to other requests — and, with two overlapping runs, left it in place
        as the "base" model. temp_dir is None for the base model.
        """
        if not mlflow_run_id:
            self._require_loaded()
            return self._model, None
        return download_user_model_from_mlflow(mlflow_run_id)

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict pork price t+1 from a single feature dict."""
        _ = model_key, threshold
        model, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            date_str = features.get("date", "")
            t0 = time.perf_counter()
            x = self._build_frame(features)
            y_pred = float(model.predict(x)[0])
            self._record_latency((time.perf_counter() - t0) * 1000)

            xai_fv = {col: float(x.iloc[0][col]) for col in FEATURE_COLUMNS}
            logger.info(
                "predict_inline done — date='%s' y_pred=%.4f count=%d mlflow=%s",
                date_str, y_pred, self._predict_count, bool(mlflow_run_id),
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

    def _predict_batch_row(self, model: Any, idx: int, row_dict: dict, rows: list[dict]) -> None:
        """Append one predict_batch row: its prediction, or its error (a bad row never fails the
        batch)."""
        date_val = _find_date(row_dict)
        date = _iso_date(date_val) if date_val is not None else ""
        with row_error_capture(rows, {"row": idx, "date": date}, what=f"fila {idx}"):
            x = self._build_frame(row_dict)
            y_pred = float(model.predict(x)[0])
            rows.append({
                "row": idx,
                "date": date,
                "y_pred": y_pred,
                "model_id": MODEL_ID,
                "line": LINE,
                "xai_feature_values": {
                    col: float(x.iloc[0][col]) for col in FEATURE_COLUMNS
                },
            })

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = ""
    ) -> PredictBatchResponse:
        """Predict pork price t+1 for every row in a CSV."""
        model, user_temp_dir = self._resolve_model(mlflow_run_id)
        with removing_dir(user_temp_dir):
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            predictions: list[dict] = []
            t0 = time.perf_counter()
            for idx, row in df.iterrows():
                self._predict_batch_row(model, int(idx), row.to_dict(), predictions)
            self._record_latency((time.perf_counter() - t0) * 1000)
            logger.info(
                "predict_batch done — %d rows count=%d mlflow=%s",
                len(predictions), self._predict_count, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID, line=LINE, predictions=predictions, output_path=None
            )

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Refit the Ridge pipeline on a user CSV (one-step-ahead t -> t+1 supervision).

        Mirrors the original training procedure (src/cu05/training/service.py::run_training +
        src/cu05/models/ridge.py::build_ridge_pipeline from the delivered code): a fresh
        ColumnTransformer(SimpleImputer(median) -> StandardScaler) + Ridge(alpha=RIDGE_ALPHA)
        pipeline fit on X=features[:-1], y=target[1:]. The refit pipeline is a brand-new
        object — self._model is never mutated. It is published as the new base model version
        (persist_retrained_bundle) and, if mlflow_run_id is given, also saved to that run; the
        container then serves the new base.
        """
        tracker = BaseMLflowTracker(mlflow_run_id)

        df = _read_train_frame(data_path)
        x_train = df.iloc[:-1][FEATURE_COLUMNS].copy()
        y_train = df.iloc[1:][TARGET_COLUMN].to_numpy(dtype=float)

        new_model = _build_ridge_pipeline()
        new_model.fit(x_train, y_train)

        metrics = _train_metrics(
            y_train, new_model.predict(x_train),
            previous_actual=x_train[TARGET_COLUMN].to_numpy(dtype=float),
            train_series=df[TARGET_COLUMN].to_numpy(dtype=float),
        )
        bundle_dir = tempfile.mkdtemp(prefix="ml17_train_")
        with removing_dir(bundle_dir):
            joblib.dump(new_model, f"{bundle_dir}/{MODEL_FILENAME}")

            def _upload_to_mlflow() -> None:
                tracker.log_params({
                    "alpha": RIDGE_ALPHA,
                    "imputer_strategy": IMPUTER_STRATEGY,
                    "scaler": "StandardScaler",
                })
                tracker.log_metrics(metrics)
                tracker.upload_artifacts(bundle_dir, artifact_path="model")

            persisted = persist_retrained_bundle(
                bundle_dir, artifact_folder=ARTIFACT_FOLDER_NAME, data_path=data_path,
                metrics=metrics, mlflow_run_id=mlflow_run_id, upload_to_mlflow=_upload_to_mlflow,
            )

        logger.info(
            "train() done — mae=%.4f rmse=%.4f n=%d mlflow_run_id=%s",
            metrics["mae"], metrics["rmse"], len(x_train), mlflow_run_id,
        )
        return TrainResponse(
            detail=f"Reentrenamiento completado — modelo base {persisted.base_version['version']}.",
            mae=round(metrics["mae"], 4),
            rmse=round(metrics["rmse"], 4),
            mase=_round4(metrics.get("mase")),
            r2_train=_round4(metrics.get("r2_train")),
            directional_accuracy=_round4(metrics.get("directional_accuracy")),
            n_samples=int(len(x_train)),
            upload_warning=persisted.upload_warning,
            base_version=persisted.base_version,
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata and runtime statistics."""
        base = StatsResponse(
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
                    description=(
                        "Fecha de referencia (YYYY-MM-DD) — genera month_sin/cos automáticamente"
                    ),
                ),
                InputField(
                    name="target_price_pigmeat_class_e_es",
                    type="float",
                    description=(
                        "Precio porcino clase E España en t (€/100 kg) — lag autorregresivo"
                    ),
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
                avg_latency_ms=self._avg_latency_ms(1),
            ),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
