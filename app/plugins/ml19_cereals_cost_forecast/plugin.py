"""Ml19CerealsCostForecastPlugin — DATAGIA-19 multi-horizon cereal price signal.

Predicts, for h=1/2/3 months, the expected return (%) and the up/down direction of the average
price of bulk cereals (trigo, cebada, maiz) in Spain, combined into an ensemble LONG/SHORT/FLAT
signal with ALTA/BAJA confidence. Production models: Ridge (h1 regression), XGBoost (h2/h3
regression), LogReg (h1 classification), XGBoost (h2/h3 classification).

See inbox/a19/manifest.yaml for the full contract and the "DECISION DE ARQUITECTURA" note: this
plugin answers against a frozen reference dataset bundled with the model artifacts (the
delivered src/predict/predict.py never takes raw client data either — see manifest for why).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import DataContractError, ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml19_cereals_cost_forecast import inference, model_loader
from app.plugins.ml19_cereals_cost_forecast.constants import FRAMEWORK, METRICS_REPORTED, MODEL_ID, VERSION
from app.plugins.ml19_cereals_cost_forecast.predict_dto import (
    BatchPrediction,
    ClfResult,
    HorizonResult,
    PredictBatchResponse,
    PredictInlineResponse,
    RegResult,
)

logger = logging.getLogger(__name__)


def _build_horizons(result: dict[int, dict], date_label: str) -> list[HorizonResult]:
    """Convert inference.predict_next_month()'s dict[int, dict] into typed HorizonResult rows."""
    horizons: list[HorizonResult] = []
    for h in sorted(result):
        r = result[h]
        horizons.append(
            HorizonResult(
                horizon=h,
                predicted_for=inference.shift_month_label(date_label, h),
                reg=RegResult(**r["reg"]) if "reg" in r else None,
                clf=ClfResult(**r["clf"]) if "clf" in r else None,
                ensemble_signal=r.get("ensemble_signal"),
                ensemble_str=r.get("ensemble_str"),
                confidence=r.get("confidence"),
            )
        )
    return horizons


class Ml19CerealsCostForecastPlugin(ModelPluginPort):
    """Multi-horizon cereal price signal (regression + classification ensemble)."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the fixed artifact bundle (6 model+scaler pairs + the frozen reference dataset)."""
        self._bundle = model_loader.load_artifact_bundle()
        logger.info(
            "ml19 plugin loaded: %s (reg=%s, clf=%s)",
            MODEL_ID, self._bundle["best_reg"], self._bundle["best_clf"],
        )

    def is_loaded(self) -> bool:
        """Return True once the fixed bundle is loaded."""
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
        """Score one month (the latest available, or 'date' if given) of the bundled dataset."""
        _ = model_key, threshold, mlflow_run_id
        self._require_loaded()
        date_str = features.get("date") if features else None

        row, date_label = inference.get_row_from_dataset(
            self._bundle["dataset"], self._bundle["features"], date_str,
        )
        result = inference.predict_next_month(row, self._bundle)
        self._record_prediction()
        logger.info("predict_inline done — date_label=%s count=%d", date_label, self._predict_count)

        return PredictInlineResponse(
            model_id=MODEL_ID,
            date_label=date_label,
            mapa_month_used=inference.mapa_month_used(date_label),
            train_cutoff=self._bundle["meta"]["train_end"],
            horizons=_build_horizons(result, date_label),
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every month listed in a CSV's 'date' column (YYYY-MM) against the bundled
        reference dataset. Unknown months are reported per-row as an error, not silently
        dropped — one bad month in a batch must not hide the others nor pass as a null result."""
        _ = mlflow_run_id
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path, dtype=str)
        if "date" not in df.columns:
            raise DataContractError("El CSV de entrada debe traer una columna 'date' (YYYY-MM).")

        predictions: list[BatchPrediction] = []
        for date_str in df["date"]:
            try:
                row, date_label = inference.get_row_from_dataset(
                    self._bundle["dataset"], self._bundle["features"], date_str,
                )
                result = inference.predict_next_month(row, self._bundle)
                predictions.append(
                    BatchPrediction(
                        date_requested=date_str,
                        date_label=date_label,
                        mapa_month_used=inference.mapa_month_used(date_label),
                        horizons=_build_horizons(result, date_label),
                    )
                )
            except DataContractError as exc:
                predictions.append(BatchPrediction(date_requested=date_str, error=str(exc)))

        n_predictions = sum(1 for p in predictions if p.error is None)
        self._record_prediction()
        logger.info(
            "predict_batch done — %d filas, %d predicciones, %d errores",
            len(predictions), n_predictions, len(predictions) - n_predictions,
        )
        return PredictBatchResponse(
            model_id=MODEL_ID,
            train_cutoff=self._bundle["meta"]["train_end"],
            predictions=predictions,
            n_rows=len(predictions),
            n_predictions=n_predictions,
            output_path=None,
        )

    # ── train (no soportado — ver inbox/a19/manifest.yaml::training) ──────────

    def train(self, *, data_path: str = "", mlflow_run_id: str = "") -> None:
        """Raise TrainingNotSupportedError — the delivered training procedure (src/training/
        train.py) always reads the fixed data/processed/auto/dataset_v7_fe.csv, never a
        client-supplied CSV. Regenerating that dataset requires the full external ETL pipeline
        (Google Earth Engine auth, manual MAPA/ESYRCE downloads, Yahoo Finance) — out of scope
        for a plugin retraining endpoint. See manifest.training.reason."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml19 no soporta reentrenamiento por usuario: el procedimiento de entrenamiento "
            "entregado (src/training/train.py) siempre reentrena sobre el dataset fijo "
            "data/processed/auto/dataset_v7_fe.csv, generado por un pipeline ETL externo "
            "(Google Earth Engine, descargas manuales MAPA/ESYRCE, Yahoo Finance) que no acepta "
            "ningún CSV de cliente. Reentrenar requiere el repo original "
            "a19-rnn-cereals-predictivo-coste-materias-primas-redes y volver a subir los "
            "artefactos a S3 bajo artifacts/fixed/ml19_cereals_cost_forecast/."
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        _ = mlflow_run_id
        inputs = [
            InputField(
                name="date", type="str | null",
                description=(
                    "Mes a consultar dentro del dataset de referencia empaquetado (YYYY-MM). "
                    "Si se omite, se usa el mes mas reciente disponible."
                ),
            ),
        ]
        outputs = [
            OutputField(
                name="horizons", type="list[dict]",
                description=(
                    "Una entrada por horizonte (1, 2, 3 meses): retorno esperado (regresion), "
                    "probabilidad de subida (clasificacion) y senal ensemble LONG/SHORT/FLAT "
                    "con confianza ALTA/BAJA."
                ),
            ),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Senal de trading mensual (magnitud de retorno + direccion) para el precio "
                "medio de cereales a granel en España (trigo, cebada, maiz), a 1/2/3 meses "
                "vista, mediante un ensemble de Ridge/XGBoost (regresion) y LogReg/XGBoost "
                "(clasificacion). Responde contra un dataset de referencia empaquetado y "
                "congelado a la fecha de entrega -- ver inbox/a19/manifest.yaml."
            ),
            task_type="regression_classification_ensemble",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=dict(METRICS_REPORTED),
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
