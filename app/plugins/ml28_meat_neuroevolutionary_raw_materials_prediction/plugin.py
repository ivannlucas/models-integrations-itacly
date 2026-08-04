from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import ModelNotLoadedError, TrainingNotSupportedError
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.constants import (
    DESTINATION_PROFILES,
    FRAMEWORK,
    MODEL_ID,
    REQUIRED_COLUMNS,
    VERSION,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.pipeline import (
    run_recommendation_pipeline,
)
from app.plugins.ml28_meat_neuroevolutionary_raw_materials_prediction.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)

logger = logging.getLogger(__name__)


class Ml28MeatNeuroevolutionaryRawMaterialsPredictionPlugin(ModelPluginPort):
    """Deterministic decision-support rules engine for meat raw-material procurement (CU28).

    NOTE on the package name: "neuroevolutionary" reflects the requested plugin name, not what
    is served. This plugin ports src/cli/platform_run.py from the original delivery verbatim —
    a documented rules engine (sigmoid purchase trigger + gap/rate quantity optimizer + gating +
    baseline policy simulation), with no trained model of any kind (neither the neuroevolutionary
    comparison nor the official LinearRegression/Ridge baseline). See inbox/a28/manifest.yaml for
    the full rationale — this was an explicit, confirmed product decision, not an oversight.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        # No artifact to download — the "model" is the vendored rules engine itself, driven by
        # fixed constants in constants.py (mirroring config/platform_config.yaml).
        self._loaded = True
        logger.info("ml28 plugin loaded: %s (rules engine, no trained artifact)", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._loaded

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key, threshold
        self._require_loaded()
        if mlflow_run_id:
            logger.warning(
                "mlflow_run_id=%s ignored — ml28 has no trained artifact (training.supported=false)",
                mlflow_run_id,
            )

        row_df = pd.DataFrame([features])
        recommendations, _summary = run_recommendation_pipeline(row_df)
        row = recommendations.iloc[0].to_dict()

        self._record()
        return PredictInlineResponse(model_id=MODEL_ID, **row)

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        self._require_loaded()
        if mlflow_run_id:
            logger.warning(
                "mlflow_run_id=%s ignored — ml28 has no trained artifact (training.supported=false)",
                mlflow_run_id,
            )

        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)

        recommendations, summary = run_recommendation_pipeline(df)

        self._record()
        return PredictBatchResponse(
            model_id=MODEL_ID,
            predictions=recommendations.to_dict(orient="records"),
            summary=summary,
            output_path=None,
        )

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        _ = data_path, mlflow_run_id
        raise TrainingNotSupportedError(
            "ml28 no soporta reentrenamiento: el motor servido es una función determinista de "
            "config/platform_config.yaml, sin pesos que ajustar. Ver inbox/a28/manifest.yaml::"
            "training para el detalle (el pipeline ML real del repo entregado opera sobre "
            "columnas objetivo sintéticas que no forman parte del contrato de inferencia del "
            "cliente y entrena artefactos que este plugin no usa)."
        )

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        _ = mlflow_run_id
        inputs = [
            InputField(name=col, type="string" if col in ("date", "raw_material_id", "destination_profile") else "float", description=f"Ver docs/input_contract.md — {col}")
            for col in REQUIRED_COLUMNS
        ]
        outputs = [
            OutputField(name="purchase_trigger_flag", type="int", description="1=BUY, 0=DO_NOT_BUY"),
            OutputField(name="order_quantity_tons", type="float", description="Cantidad final recomendada (0.0 si trigger=0)"),
            OutputField(name="risk_level", type="string", description="LOW | MEDIUM | HIGH"),
        ]

        return StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Motor de reglas deterministas de soporte a la decisión para el aprovisionamiento "
                "de materia prima cárnica (CU28/NEUROCARN-OPT): separa la decisión de 'si comprar' "
                "(purchase trigger) de 'cuánto comprar' (quantity optimizer), con gating y "
                "simulación de política frente a una línea base. No usa ningún modelo entrenado — "
                "ver inbox/a28/manifest.yaml known_issues sobre el nombre del paquete."
            ),
            task_type="decision_support_rules_engine",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                # policy_simulation oficial (reports/official/cu28_metrics_official__mixed_context.md,
                # 175 periodos de test) — es la MISMA lógica que este plugin ejecuta (run_policy_simulation),
                # no la de los modelos ML desconectados (upstream_predictor/purchase_trigger/quantity_optimizer .pkl).
                "aggregate_excess_reduction_pct": 21.7108,
                "stockout_guardrail_pass": True,
                "n_periods_evaluated": 175,
                "destination_profiles_catalog": DESTINATION_PROFILES,
            },
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
