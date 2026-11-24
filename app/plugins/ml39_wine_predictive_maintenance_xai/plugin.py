"""Ml39WinePredictiveMaintenanceXaiPlugin — Integrated Gradients + SHAP explanations for the
exact model ml37-wine-press-predictive-maintenance serves (see inbox/a39/manifest.yaml).

a39 is NOT a new predictive model: it ships its own copy of ml37's artifacts (best_model.keras/
scaler.json/class_mappings.json, MD5-confirmed identical) and adds per-cycle explainability on
top — Integrated Gradients always, SHAP opt-in (``include_shap``, costly in CPU). It never
trains anything of its own (``train()`` raises ``TrainingNotSupportedError``) and never resolves
a user-retrained model (there is nothing to retrain — see mlflow_utils.py).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    DataContractError,
    ModelNotLoadedError,
    TrainingNotSupportedError,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml39_wine_predictive_maintenance_xai import inference, model_loader, preprocessing
from app.plugins.ml39_wine_predictive_maintenance_xai.constants import (
    FRAMEWORK,
    METRICS_REPORTED,
    MODEL_ID,
    PHYSICAL_SENSORS,
    TARGET_NAMES,
    VERSION,
)
from app.plugins.ml39_wine_predictive_maintenance_xai.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml39_wine_predictive_maintenance_xai.predict_dto import (
    BatchCycleExplanation,
    PredictBatchResponse,
    PredictInlineResponse,
)

logger = logging.getLogger(__name__)

REQUIRED_BATCH_COLUMNS = ("cycle_id", "t", *PHYSICAL_SENSORS)


class Ml39WinePredictiveMaintenanceXaiPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Integrated Gradients (always) + SHAP (opt-in) explanations of ml37's predictions."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin (runtime counters start at the mixin defaults)."""
        self._bundle: dict | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the base model bundle (own copy of ml37's artifacts) + SHAP background."""
        self._bundle = model_loader.load_model_bundle()
        logger.info("ml39 plugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the bundle is loaded."""
        return self._bundle is not None

    def _require_loaded(self) -> dict:
        if self._bundle is None:
            raise ModelNotLoadedError("El modelo no está cargado.")
        return self._bundle

    def _resolve_bundle(self, mlflow_run_id: str) -> dict:
        """a39 has nothing to retrain — always explains the base bundle. ``mlflow_run_id`` is
        accepted for contract consistency with every other plugin, logged, and otherwise
        ignored (see mlflow_utils.py)."""
        if mlflow_run_id:
            download_user_model_from_mlflow(mlflow_run_id)
        return self._require_loaded()

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Explain one cycle's prediction: Integrated Gradients (+ SHAP if requested)."""
        _ = model_key, threshold
        bundle = self._resolve_bundle(mlflow_run_id)
        readings = features.get("readings") or []
        include_shap = bool(features.get("include_shap", False))
        result = _explain_cycle(bundle, readings, include_shap=include_shap)
        self._record()
        logger.info(
            "predict_inline done — count=%d include_shap=%s", self.predict_count, include_shap
        )
        return PredictInlineResponse(model_id=MODEL_ID, **result)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Explain every cycle of a long-format CSV/parquet (cycle_id, t, 14 sensores).

        A cycle with a contract error (missing sensor, non-finite value, duplicate timestamps,
        or a SHAP request without a persisted background) falls to that cycle's `error` field
        alone — one bad cycle must never abort the batch.
        """
        bundle = self._resolve_bundle(mlflow_run_id)
        df = _read_batch_table(data_path)
        include_shap = False  # batch explanation defaults to IG-only (SHAP is per-cycle costly)
        predictions = []
        for cycle_id, group in df.groupby("cycle_id", sort=False):
            predictions.append(_explain_batch_cycle(bundle, str(cycle_id), group, include_shap))

        n_predictions = sum(1 for p in predictions if p.error is None)
        self._record()
        logger.info(
            "predict_batch done — %d ciclos, %d explicaciones, mlflow=%s",
            len(predictions), n_predictions, bool(mlflow_run_id),
        )
        return PredictBatchResponse(
            model_id=MODEL_ID, predictions=predictions,
            n_rows=len(predictions), n_predictions=n_predictions, output_path=None,
        )

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """a39 has no training of its own: it explains an already-trained model (ml37's), it
        does not train one. Reentrenamiento del modelo base es competencia exclusiva de
        ml37-wine-press-predictive-maintenance (ver inbox/a39/manifest.yaml::training)."""
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml39 no soporta reentrenamiento: es la capa de explicabilidad (Integrated "
            "Gradients + SHAP) sobre el modelo ya entrenado de ml37-wine-press-predictive-"
            "maintenance, no un modelo con parámetros propios que ajustar. El reentrenamiento "
            "del modelo base sigue siendo competencia exclusiva de ese otro plugin."
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real audited metrics."""
        _ = mlflow_run_id
        metrics = dict(METRICS_REPORTED)

        inputs = [
            InputField(name="readings", type="list[dict[str, float]]",
                       description=f"Serie temporal de un ciclo (t + {len(PHYSICAL_SENSORS)} "
                       "sensores físicos) — mismo contrato que ml37, remuestreada internamente "
                       "a 600 pasos."),
            InputField(name="include_shap", type="bool",
                       description="Añade importancia SHAP por sensor y por head (coste alto "
                       "en CPU); por defecto false."),
        ]
        outputs = [
            OutputField(name="predicted_class", type="dict[str, int]",
                        description="Código real UCI que se explica por head — la misma "
                        "predicción que devolvería ml37 para este ciclo."),
            OutputField(name="ig_importance", type="dict[str, dict[str, float]]",
                        description="head -> sensor -> |IG| medio (perfil full, steps=50)."),
            OutputField(name="shap_importance", type="dict[str, dict[str, float]] | null",
                        description="head -> sensor -> |SHAP| medio, solo si include_shap=true."),
        ]
        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Capa de explicabilidad (XAI) de ml37-wine-press-predictive-maintenance: "
                "Integrated Gradients (siempre) y SHAP (opcional) sobre el mismo modelo LSTM "
                "multitarea, mismo dataset y mismo run_id (no_virtual_v1__rand_scool). No "
                "entrena ni evalúa un modelo propio. Ver inbox/a39/manifest.yaml: el dataset es "
                "un proxy técnico (UCI Hydraulic Systems), no telemetría real de bodega."
            ),
            task_type="explainability_layer (Integrated Gradients + SHAP)",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
            runtime_stats=self._runtime_stats(),
        )


def _explain_cycle(bundle: dict, readings: list[dict], *, include_shap: bool) -> dict:
    """Run the full pipeline (resample -> schema tensor -> scale -> predict -> IG [-> SHAP])
    for one cycle's readings — shared by predict_inline and predict_batch's per-cycle loop."""
    schema_tensor = preprocessing.cycle_from_readings(readings)
    x_scaled = preprocessing.apply_scaler(schema_tensor[None, ...], bundle["scaler"])

    model = bundle["model"]
    output_names = list(getattr(model, "output_names", TARGET_NAMES))
    pos = {name: i for i, name in enumerate(output_names)}
    probs_all = model.predict(x_scaled, verbose=0)
    predicted_idx = {t: int(np.argmax(probs_all[pos[t]][0])) for t in TARGET_NAMES}

    i2v = bundle["class_mappings"]["mappings_index_to_value"]
    predicted_class = {t: int(i2v[t][str(predicted_idx[t])]) for t in TARGET_NAMES}

    ig_importance = inference.compute_ig_importance(model, x_scaled)

    shap_importance = None
    if include_shap:
        if bundle["shap_background"] is None:
            raise DataContractError(
                "SHAP no disponible: falta el artefacto shap_background.npy del plugin."
            )
        shap_importance = inference.compute_shap_importance(
            model, x_scaled, bundle["shap_background"]
        )

    return {
        "predicted_class": predicted_class,
        "ig_importance": ig_importance,
        "shap_importance": shap_importance,
    }


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _read_batch_table(data_path: str) -> pd.DataFrame:
    with local_file_path(data_path) as local_path:
        df = _read_table(local_path)
    missing = [c for c in REQUIRED_BATCH_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(f"Faltan columnas obligatorias en el fichero de entrada: {missing}")
    return df


def _explain_batch_cycle(
    bundle: dict, cycle_id: str, group: pd.DataFrame, include_shap: bool,
) -> BatchCycleExplanation:
    """Explain one cycle's group of rows; a contract error becomes that cycle's ``error``."""
    try:
        readings = group.sort_values("t").to_dict("records")
        result = _explain_cycle(bundle, readings, include_shap=include_shap)
        return BatchCycleExplanation(cycle_id=cycle_id, **result)
    except DataContractError as exc:
        return BatchCycleExplanation(cycle_id=cycle_id, error=str(exc))
