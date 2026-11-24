"""Ml38WineMachineryFaultDetectionPlugin — hierarchical fault diagnosis (a38, DNLS-VINO).

RandomForest → fault family; per-family Keras MLP (physics-guided loss) → binary severity.
Inference, preprocessing and training are ports of the AI team's code (inbox/a38/codigo/); see
inference.py, preprocessing.py and training.py for the line-by-line mapping, and
inbox/a38/manifest.yaml for the known issues.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import joblib
import numpy as np

from app.application.dto.stats_dto import InputField, OutputField, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.batch_rows import row_error_capture
from app.domain.services.mlflow_artifacts import (
    add_mlflow_run_to_stats,
    removing_dir,
    require_train_tracker,
    upload_model_to_mlflow,
)
from app.domain.services.plugin_runtime import PluginRuntimeMixin
from app.domain.services.xai_values import xai_feature_values
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    FAMILY_MODEL_FILENAME,
    FAMILY_SCALER_FILENAME,
    FEATURE_NAMES,
    FRAMEWORK,
    MODEL_ID,
    REPORTED_METRICS,
    SEVERITY_MODELS_FILENAME,
    SEVERITY_SCALERS_FILENAME,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LAMBDA_PHYS,
    TRAIN_LEARNING_RATE,
    TRAIN_N_SPLITS,
    TRAIN_RANDOM_STATE,
    TRAIN_RF_MIN_SAMPLES_LEAF,
    TRAIN_RF_N_ESTIMATORS,
    VERSION,
)
from app.plugins.ml38_wine_machinery_fault_detection.inference import predict_matrix
from app.plugins.ml38_wine_machinery_fault_detection.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml38_wine_machinery_fault_detection.model_loader import (
    ModelBundle,
    load_artifacts,
)
from app.plugins.ml38_wine_machinery_fault_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml38_wine_machinery_fault_detection.preprocessing import (
    build_feature_vector,
    channel_warning,
    check_operating_configuration,
    check_signal_lengths,
    feature_frame_from_table,
    read_signal,
    read_table,
)
from app.plugins.ml38_wine_machinery_fault_detection.train_dto import (
    SeverityCvMetrics,
    TrainResponse,
)
from app.plugins.ml38_wine_machinery_fault_detection.training import train_bundle

logger = logging.getLogger(__name__)

RAW_KEYS = ("vibration_csv", "current_csv", "motor", "speed", "window_index")


def _inline_feature_vector(features: dict) -> np.ndarray:
    """The 24 inline features as a float vector in contract order (ValueError if any is bad)."""
    missing = [name for name in FEATURE_NAMES if features.get(name) is None]
    if missing:
        raise ValueError(f"Faltan características obligatorias: {missing}")
    vector = np.array([float(features[name]) for name in FEATURE_NAMES])
    if not np.all(np.isfinite(vector)):
        bad = [n for n, v in zip(FEATURE_NAMES, vector) if not np.isfinite(v)]
        raise ValueError(f"Características no finitas: {bad}")
    return vector


def _raw_feature_vector(features: dict) -> tuple[np.ndarray, list[str]]:
    """Compute the 24 features from the raw CSV pair (scripts/predict_raw.py) + domain warnings.

    Stricter than the original script, which accepts any channel and length: a file named with a
    channel other than ch1, or a signal that is not exactly one training window long, is rejected
    (FeatureContractError → 422; manifest KI-05/KI-06, checked on the raw NLN-EMP data).
    """
    motor, speed = str(features["motor"]), int(features["speed"])
    check_operating_configuration(motor, speed)
    vib_csv, ele_csv = str(features["vibration_csv"]), str(features["current_csv"])
    warnings = [w for w in (channel_warning(vib_csv, "vib"), channel_warning(ele_csv, "ele")) if w]
    window_index = int(features.get("window_index") or 0)
    with local_file_path(vib_csv) as vib_path:
        vib_signal = read_signal(vib_path, window_index)
    with local_file_path(ele_csv) as ele_path:
        ele_signal = read_signal(ele_path, window_index)
    check_signal_lengths(len(vib_signal), len(ele_signal))
    return build_feature_vector(vib_signal, ele_signal, motor, speed), warnings


class Ml38WineMachineryFaultDetectionPlugin(PluginRuntimeMixin, ModelPluginPort):
    """Fault family (RandomForest) + conditional binary severity (per-family Keras MLP)."""

    def __init__(self) -> None:
        self._bundle: ModelBundle | None = None

    def load(self) -> None:
        self._bundle = load_artifacts()
        logger.info("Ml38WineMachineryFaultDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._bundle is not None

    def _resolve_bundle(self, mlflow_run_id: str) -> tuple[ModelBundle, str | None]:
        """(bundle, temp_dir) for this request; the user's model is returned, never stored."""
        if not mlflow_run_id:
            self._require_loaded()
            return self._bundle, None
        return download_user_model_from_mlflow(mlflow_run_id)

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Predict one window, from its 24 features or (raw request) from raw signals."""
        _ = threshold  # no decision threshold in this model (contract compatibility)
        # The request's own fields decide the mode, not model_key: an inline request can carry
        # model_key="raw" and has no signal paths (it used to fail with KeyError → 500).
        _ = model_key
        if "vibration_csv" in features:
            vector, warnings = _raw_feature_vector(features)
        else:
            vector, warnings = _inline_feature_vector(features), []
        bundle, user_temp_dir = self._resolve_bundle(mlflow_run_id)
        with removing_dir(user_temp_dir):
            pred = predict_matrix(bundle, vector.reshape(1, -1))[0]
        self._record()
        return PredictInlineResponse(
            model_id=MODEL_ID,
            input_warnings=warnings,
            xai_feature_values=xai_feature_values(dict(zip(FEATURE_NAMES, vector)), FEATURE_NAMES)
            or None,
            **pred,
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Predict every row of a CSV/parquet feature table (run_inference_pipeline)."""
        with local_file_path(data_path) as local_path:
            table = feature_frame_from_table(read_table(local_path))
        bundle, user_temp_dir = self._resolve_bundle(mlflow_run_id)
        predictions: list[dict[str, Any]] = []
        with removing_dir(user_temp_dir):
            values = table.to_numpy(float)
            valid = np.all(np.isfinite(values), axis=1)
            preds = iter(predict_matrix(bundle, values[valid]) if valid.any() else [])
            for row_idx, (idx, ok) in enumerate(zip(table.index, valid)):
                with row_error_capture(predictions, {"row": int(row_idx)}, what=f"fila {idx}"):
                    if not ok:
                        bad = [c for c in FEATURE_NAMES if not np.isfinite(table.at[idx, c])]
                        raise ValueError(f"Valores vacíos o no numéricos: {bad}")
                    predictions.append({
                        "row": int(row_idx),
                        **next(preds),
                        "xai_feature_values": xai_feature_values(table.loc[idx], FEATURE_NAMES)
                        or None,
                    })
        self._record()
        logger.info("predict_batch done — %d rows, mlflow=%s",
                    len(predictions), bool(mlflow_run_id))
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Retrain RF + severity MLPs from scratch like scripts.train; save only to MLflow.

        A RandomForest cannot be fine-tuned, and the AI team's procedure always trains both
        levels from scratch on the given table, so that is what is ported (training.py). The
        served base artifacts are never touched; the new bundle lives only in the MLflow run.
        """
        tracker = require_train_tracker(mlflow_run_id)
        with local_file_path(data_path) as local_path:
            table = read_table(local_path)
        start = time.perf_counter()
        bundle, metrics = train_bundle(table)
        elapsed = time.perf_counter() - start

        family_cv = metrics["family_cv"]
        tracker.log_params({
            "rf_n_estimators": TRAIN_RF_N_ESTIMATORS,
            "rf_min_samples_leaf": TRAIN_RF_MIN_SAMPLES_LEAF,
            "n_splits": TRAIN_N_SPLITS,
            "random_state": TRAIN_RANDOM_STATE,
            "mlp_epochs": TRAIN_EPOCHS,
            "mlp_batch_size": TRAIN_BATCH_SIZE,
            "mlp_learning_rate": TRAIN_LEARNING_RATE,
            "lambda_phys": TRAIN_LAMBDA_PHYS,
        })
        run_metrics = {f"family_cv_{k}": v for k, v in family_cv.items()}
        for fam, fam_metrics in metrics["severity_cv"].items():
            run_metrics.update({f"severity_cv_{fam}_{k}": v for k, v in fam_metrics.items()})
        run_metrics.update({"n_samples": metrics["n_samples"], "n_groups": metrics["n_groups"],
                            "training_time_s": elapsed})
        tracker.log_metrics(run_metrics)

        def _write(model_dir: str) -> None:
            joblib.dump(bundle.family_model, os.path.join(model_dir, FAMILY_MODEL_FILENAME))
            joblib.dump(bundle.family_scaler, os.path.join(model_dir, FAMILY_SCALER_FILENAME))
            joblib.dump(bundle.severity_models, os.path.join(model_dir, SEVERITY_MODELS_FILENAME))
            joblib.dump(bundle.severity_scalers,
                        os.path.join(model_dir, SEVERITY_SCALERS_FILENAME))

        upload_model_to_mlflow(tracker, _write, prefix="ml38_mlflow_")
        logger.info("ml38 train() done — n=%d, %.1f s, macro_f1_cv=%.4f",
                    metrics["n_samples"], elapsed, family_cv["macro_f1_mean"])
        return TrainResponse(
            detail="Reentrenamiento completado (familia + severidad)",
            family_cv_accuracy_mean=round(family_cv["accuracy_mean"], 4),
            family_cv_accuracy_std=round(family_cv["accuracy_std"], 4),
            family_cv_macro_f1_mean=round(family_cv["macro_f1_mean"], 4),
            family_cv_macro_f1_std=round(family_cv["macro_f1_std"], 4),
            severity_cv={
                fam: SeverityCvMetrics(**{k: round(v, 4) for k, v in fam_metrics.items()})
                for fam, fam_metrics in metrics["severity_cv"].items()
            },
            severity_skipped=metrics["severity_skipped"],
            families=metrics["families"],
            n_samples=metrics["n_samples"],
            n_groups=metrics["n_groups"],
            training_time_s=round(elapsed, 2),
            mlflow_run_id=mlflow_run_id,
            upload_warning=None,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        inputs = [
            InputField(name=name, type="float",
                       description=f"Característica {name} de la ventana (ver memoria Tabla 4)")
            for name in FEATURE_NAMES
        ] + [
            InputField(name="vibration_csv", type="str",
                       description="[Modo raw] CSV de vibración (time + señal, g). Solo canal 1 "
                                   "y ventanas de exactamente 240000 muestras (12 s a 20 kHz); "
                                   "otro canal (-chN.csv) u otra longitud → 422"),
            InputField(name="current_csv", type="str",
                       description="[Modo raw] CSV eléctrico del mismo ensayo. Solo corriente de "
                                   "fase 1 (ch1) y ventanas de exactamente 300000 muestras (15 s); "
                                   "otro canal u otra longitud → 422"),
            InputField(name="motor", type="str", format=["Motor-2", "Motor-4"],
                       description="[Modo raw] Configuración operativa NLN-EMP"),
            InputField(name="speed", type="int", format=["50", "75", "100", "70"],
                       description="[Modo raw] Velocidad %: Motor-2 → 50/75/100, Motor-4 → 70"),
            InputField(name="window_index", type="int", default=0,
                       description="[Modo raw] Columna de señal posterior a time"),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Diagnóstico jerárquico de fallos en un sistema motor-bomba a partir de "
                "características de vibración y corriente: un Random Forest predice la familia "
                "de fallo (alignment, bearing, cavitation, electrical, healthy, mechanical_other, "
                "unbalance) y, si no es healthy, un MLP específico de la familia (pérdida con "
                "término físico sobre la banda baja de vibración) estima la severidad binaria. "
                "Entrenado con el banco de ensayo público NLN-EMP, no con maquinaria de bodega: "
                "requiere validación con datos reales antes de desplegar."
            ),
            task_type="hierarchical_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=[
                OutputField(name="family", type="str", description="Familia de fallo predicha"),
                OutputField(name="severity", type="int",
                            description="0 = leve (nivel 1), 1 = significativo (nivel > 1); "
                                        "null si healthy o sin modelo de severidad"),
                OutputField(name="severity_label", type="str",
                            description="leve, significativo, sin fallo o modelo no disponible"),
                OutputField(name="severity_confidence", type="float",
                            description="Probabilidad de severidad significativa"),
                OutputField(name="family_probabilities", type="dict",
                            description="Probabilidad del Random Forest por familia"),
                OutputField(name="input_warnings", type="list",
                            description="Avisos de dominio de la entrada (modo raw: canal no "
                                        "comprobable por el nombre del fichero)"),
            ],
            metrics=dict(REPORTED_METRICS),
            runtime_stats=self._runtime_stats(),
        )
        return add_mlflow_run_to_stats(base, mlflow_run_id)
