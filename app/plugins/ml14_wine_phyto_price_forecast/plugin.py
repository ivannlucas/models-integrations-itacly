"""Ml14WinePhytoPriceForecastPlugin — LSTM wine phytosanitary price forecast, t+16 weeks.

Predicts the Spanish national phytosanitary price index (MAPA, sector vitivinícola, base
2020=100) 16 weeks ahead via an LSTM trained on a drift-residual decomposition (PyTorch,
input_size=39, hidden_size=64, num_layers=2). See inbox/a14/manifest.yaml for the full
input/output contract, the golden-dataset verification and known issues.

IMPORTANT — model_status (see manifest): this is the real, audited 2.0.0 LSTM artifact
(selected by validation RMSE), not the retracted GRU numbers from the memoria/README. It does
NOT beat the Drift baseline in the final, correctly-audited test (RMSE 3.0168 vs 2.5406) — it
is served as-is because it is the genuine artifact the AI team's own audit selected, pending a
human decision on production exposure before the PR opens.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml14_wine_phyto_price_forecast import model_loader, preprocessing
from app.plugins.ml14_wine_phyto_price_forecast.constants import (
    FRAMEWORK,
    HORIZON_WEEKS,
    METRICS_REPORTED,
    MIN_HISTORY_ROWS,
    MODEL_ID,
    MODEL_NAME,
    MODEL_VERSION,
    RAW_VALUE_COLS,
    VERSION,
)
from app.plugins.ml14_wine_phyto_price_forecast.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)

logger = logging.getLogger(__name__)


class Ml14WinePhytoPriceForecastPlugin(ModelPluginPort):
    """LSTM plugin for the national phytosanitary price index (t+16 weeks)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed LSTM artifact (model + scalers + feature contract) via ArtifactStore."""
        self._bundle = model_loader.load_artifact_bundle()
        logger.info("ml14 plugin loaded: %s (model=%s)", MODEL_ID, MODEL_NAME)

    def is_loaded(self) -> bool:
        """Return True once the artifact bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict the national phytosanitary price index at t+16 weeks from a raw history."""
        _ = model_key, threshold, mlflow_run_id  # mlflow_run_id no aplica — ver mlflow_utils.py
        self._require_loaded()
        rows: list[dict] = features["rows"]
        r = preprocessing.run_inference(self._bundle, rows)
        self._record_prediction()
        logger.info(
            "predict_inline done — last_observed_date=%s predicted_price=%.4f count=%d",
            r["last_observed_date"], r["predicted_price"], self._predict_count,
        )
        return PredictInlineResponse(
            model_id=MODEL_ID,
            predicted_price=r["predicted_price"],
            current_price=r["current_price"],
            drift_baseline=r["drift_baseline"],
            horizon_weeks=r["horizon_weeks"],
            last_observed_date=r["last_observed_date"],
            prediction_date=r["prediction_date"],
            model_used=MODEL_NAME,
            gap_warning=r["gap_warning"],
            n_rows_used=r["n_rows_used"],
            xai_feature_values=r["xai_feature_values"],
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Predict from a CSV with a raw weekly history — one prediction anchored at its last row.

        Mirrors the delivered CLI exactly: the whole CSV is treated as one continuous history,
        not a sliding-window multi-prediction batch (the delivered code has no such mode — see
        inbox/a14/manifest.yaml known_issues).
        """
        _ = mlflow_run_id  # no aplica — ver mlflow_utils.py
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        rows: list[dict[str, Any]] = df.to_dict(orient="records")

        r = preprocessing.run_inference(self._bundle, rows)
        r["model_id"] = MODEL_ID
        r["model_used"] = MODEL_NAME
        self._record_prediction()
        logger.info(
            "predict_batch done — last_observed_date=%s predicted_price=%.4f count=%d",
            r["last_observed_date"], r["predicted_price"], self._predict_count,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=[r], n_predictions=1, output_path=None,
        )

    # ── train (no soportado — ver inbox/a14/manifest.yaml::training) ──────────

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> None:
        """Raise TrainingNotSupportedError — the delivered training procedure has no
        client-data path (scripts/train.py always re-reads the fixed bundled dataset and only
        exposes hyperparameters, never a --data/--input CSV — see manifest.training)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml14 no soporta reentrenamiento por usuario: el procedimiento de entrenamiento "
            "entregado (src/training/compare_models.py) siempre reentrena sobre el mismo "
            "histórico fijo bundled — no acepta ningún CSV de datos de cliente, solo "
            "hiperparámetros. Reentrenar requiere el repo original "
            "a14-rnn-vitivinicola-precios-mercado-fitosanitarios y volver a subir el artefacto "
            "a S3 bajo artifacts/fixed/ml14_wine_phyto_price_forecast/."
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        _ = mlflow_run_id  # no aplica — ver mlflow_utils.py
        inputs = [
            InputField(
                name="rows", type="list[dict]",
                description=(
                    f"Histórico semanal (W-SUN) continuo, mínimo {MIN_HISTORY_ROWS} filas, sin "
                    "huecos ni fechas duplicadas. Cada fila: date (YYYY-MM-DD) + "
                    f"{', '.join(RAW_VALUE_COLS)}."
                ),
            ),
        ]
        outputs = [
            OutputField(name="predicted_price", type="float", description="PROTECCION_FITO predicho a horizon_weeks vista (índice base 2020=100)."),
            OutputField(name="current_price", type="float", description="PROTECCION_FITO de la última fila del histórico aportado."),
            OutputField(name="drift_baseline", type="float", description="Baseline lineal de drift — referencia frente a la que el LSTM no consigue mejorar en test final."),
            OutputField(name="horizon_weeks", type="int", description=f"Siempre {HORIZON_WEEKS}."),
            OutputField(name="prediction_date", type="date", description="last_observed_date + horizon_weeks."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Predicción semanal del índice MAPA de precios de protección fitosanitaria "
                "(sector vitivinícola, base 2020=100) a horizonte de 16 semanas mediante LSTM "
                "(PyTorch) sobre un residual respecto a un baseline lineal de drift, con 39 "
                f"features autorregresivas/exógenas/estacionales. Versión de modelo {MODEL_VERSION}. "
                "Ver inbox/a14/manifest.yaml para el contrato completo y las métricas de test "
                "final frente al baseline."
            ),
            task_type="regression_timeseries",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=dict(METRICS_REPORTED),
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,
            ),
        )
