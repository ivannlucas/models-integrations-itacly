"""Ml43CerealsDnslAnomalyFaultDetectionPlugin — Deep Neuro-Fuzzy anomaly/fault detection for
cereal dryers (CU43) with integrated XAI + corrective-action generation (CU44).

Detects anomalies/faults in cereal drying cycles from 180-row sensor windows (13 sensors),
combining a BiLSTM branch with an ANFIS-style fuzzy-rule branch (late fusion) plus SHAP +
fuzzy-rule explanations. Vendored model code is kept in sync with
inbox/a43/codigo/ (a43-44-neurofuzzy-anomalias-fallas, the training repo this runtime serves) —
see inbox/a43/manifest.yaml for the full input/output contract and the discrepancy between the
delivered memoria (v1.4) and the actually-shipped checkpoint (known_issues).
"""
from __future__ import annotations

import logging
import pickle
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
from app.domain.services.exceptions import ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection import (
    model_loader,
    postprocessing,
    preprocessing,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection._vendor.preprocess import stats_windows
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.constants import (
    DECISION_THRESHOLD,
    DEFAULT_MODEL_CFG,
    FRAMEWORK,
    MODEL_FILENAME,
    MODEL_ID,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    SEQ_LENGTH,
    STATS_CREATION,
    TARGET_COLUMN,
    TEST_METRICS,
    VERSION,
    XAI_BACKGROUND_FILENAME,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.mlflow_utils import (
    download_user_model_from_mlflow,
    upload_artifacts_to_mlflow,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml43_cereals_dnsl_anomaly_fault_detection.train_dto import TrainResponse

logger = logging.getLogger(__name__)


class Ml43CerealsDnslAnomalyFaultDetectionPlugin(ModelPluginPort):
    """Deep Neuro-Fuzzy (BiLSTM + fuzzy rules) anomaly/fault detector for cereal dryers, CU43+CU44."""

    def __init__(self) -> None:
        self._model = None
        self._model_cfg: dict = DEFAULT_MODEL_CFG.copy()
        self._scaler_x = None
        self._scaler_num = None
        self._xai_background = None
        self._explainer = None
        self._threshold: float = DECISION_THRESHOLD
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    def load(self) -> None:
        (
            self._model,
            self._model_cfg,
            self._scaler_x,
            self._scaler_num,
            self._xai_background,
            self._explainer,
        ) = model_loader.load_artifacts()
        logger.info("Ml43CerealsDnslAnomalyFaultDetectionPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _load_model_for_predict(self, mlflow_run_id: str) -> dict | None:
        """Resolve the model/scalers/explainer to use — user-trained (MLflow) or served (S3)."""
        if not mlflow_run_id:
            self._require_loaded()
            return None
        logger.info("Using user-trained model from MLflow run_id=%s", mlflow_run_id)
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if loaded is None:
            logger.warning("MLflow download failed for %s, falling back to served model", mlflow_run_id)
            self._require_loaded()
            return None
        model, model_cfg, scaler_x, scaler_num, xai_background, explainer, temp_dir = loaded
        return {
            "model": model, "model_cfg": model_cfg, "scaler_x": scaler_x, "scaler_num": scaler_num,
            "xai_background": xai_background, "explainer": explainer, "temp_dir": temp_dir,
        }

    def _served_ctx(self) -> dict:
        return {
            "model": self._model, "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x, "scaler_num": self._scaler_num,
            "xai_background": self._xai_background, "explainer": self._explainer, "temp_dir": None,
        }

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        """Score a single sensor snapshot via a synthetic steady-state 180-row window.

        See manifest known_issues: there is no natural "one row = one prediction" mode for
        this model family (DNSL) — the minimum inference unit is a temporal window.
        """
        _ = model_key
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or self._served_ctx()
        try:
            t0 = time.perf_counter()
            effective_threshold = threshold if threshold is not None else self._threshold

            x_arr = preprocessing.build_inline_window(features)
            x_scaled = postprocessing.scale_sequences(x_arr, ctx["scaler_x"])
            stats_scaled = postprocessing.compute_scaled_stats(x_arr, ctx["scaler_num"])

            scores = postprocessing.run_model(ctx["model"], x_scaled, stats_scaled)
            score = float(scores[0])

            self._record()
            logger.info(
                "predict_inline done — prob=%.4f threshold=%.2f latency_ms=%.1f count=%d",
                score, effective_threshold, (time.perf_counter() - t0) * 1000, self._predict_count,
            )

            corrective_actions, xai_error = postprocessing.run_xai(
                ctx["explainer"], ctx["xai_background"], x_scaled[0], stats_scaled[0], effective_threshold,
            )
            dto = postprocessing.format_inline_response(
                score, effective_threshold, features, corrective_actions, xai_error,
            )
            return PredictInlineResponse(**dto)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid 180-row window in a raw sensor CSV (one or more cycles)."""
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or self._served_ctx()
        try:
            t0 = time.perf_counter()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)

            x_arr, _, cycle_ids = preprocessing.prepare_batch_sequences(df)
            if x_arr.shape[0] == 0:
                logger.warning("predict_batch — no valid windows found in %s", data_path)
                self._record()
                return PredictBatchResponse(model_id=MODEL_ID, predictions=[], output_path=None)

            x_scaled = postprocessing.scale_sequences(x_arr, ctx["scaler_x"])
            stats_scaled = postprocessing.compute_scaled_stats(x_arr, ctx["scaler_num"])
            scores = postprocessing.run_model(ctx["model"], x_scaled, stats_scaled)

            predictions = postprocessing.format_batch_predictions(
                x_arr, scores, cycle_ids, self._threshold,
                ctx["explainer"], ctx["xai_background"], x_scaled, stats_scaled,
            )

            self._record()
            logger.info(
                "predict_batch done — %d windows, %d failures (threshold=%.2f), latency_ms=%.1f count=%d",
                len(predictions),
                sum(1 for p in predictions if p["predicted_anomaly_label"] == "Fallo"),
                self._threshold, (time.perf_counter() - t0) * 1000, self._predict_count,
            )
            return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train a fresh model from a labeled CSV and upload it to MLflow.

        Does not replace the served checkpoint — pass the returned MLflow run_id back to
        /predict or /stats to use the newly trained model. Uses the real optimizer
        (Adam, manifest.training.hyperparams) and a simplified single-term BCE loss
        (the original DNFLoss's structural regularization terms — rule diversity/entropy/
        balance — are not reproduced here; see manifest.training for the full procedure).
        """
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.preprocessing import StandardScaler

        logger.info("Starting ml43 training from %s", data_path)
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        df.columns = df.columns.str.lower()

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"CSV falta columna objetivo '{TARGET_COLUMN}'. Requerida para entrenamiento.")

        x_arr, y_arr, _ = preprocessing.prepare_batch_sequences(df)
        if x_arr.shape[0] < 10:
            raise ValueError(
                f"Muy pocas secuencias para entrenar: {x_arr.shape[0]}. "
                f"Se necesitan al menos 10 ventanas de {SEQ_LENGTH} filas."
            )

        n_windows, _, n_feat = x_arr.shape
        split_idx = int(n_windows * 0.8)
        x_train_raw, x_test_raw = x_arr[:split_idx], x_arr[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train_raw.reshape(-1, n_feat)).reshape(x_train_raw.shape).astype(np.float32)
        x_test = scaler_x.transform(x_test_raw.reshape(-1, n_feat)).reshape(x_test_raw.shape).astype(np.float32)

        df_stats_train = stats_windows(x_train_raw, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        df_stats_test = stats_windows(x_test_raw, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        scaler_num = StandardScaler()
        s_train = scaler_num.fit_transform(df_stats_train.values).astype(np.float32)
        s_test = scaler_num.transform(df_stats_test.values).astype(np.float32)

        model_cfg = DEFAULT_MODEL_CFG.copy()
        model = model_loader.build_model(model_cfg)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00175, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()

        x_t = torch.from_numpy(x_train)
        s_t = torch.from_numpy(s_train)
        y_t = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)

        batch_size = 128
        n_epochs = 50
        best_f1 = 0.0
        best_state = None
        best_threshold = DECISION_THRESHOLD

        for epoch in range(n_epochs):
            model.train()
            idx = np.random.permutation(split_idx)
            for start in range(0, split_idx, batch_size):
                b_idx = idx[start:start + batch_size]
                optimizer.zero_grad()
                out = model(x_t[b_idx], s_t[b_idx])
                loss = criterion(out["anomaly_score"], y_t[b_idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out_v = model(torch.from_numpy(x_test), torch.from_numpy(s_test))
                    probs = torch.sigmoid(out_v["anomaly_score"]).squeeze(-1).numpy()
                    epoch_best_f1, epoch_best_thr = _best_f1_threshold(probs, y_test)
                    logger.info("Epoch %d/%d — val_f1=%.4f (thr=%.2f)", epoch + 1, n_epochs, epoch_best_f1, epoch_best_thr)
                    if epoch_best_f1 > best_f1:
                        best_f1 = epoch_best_f1
                        best_threshold = epoch_best_thr
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            out_v = model(torch.from_numpy(x_test), torch.from_numpy(s_test))
            probs = torch.sigmoid(out_v["anomaly_score"]).squeeze(-1).numpy()
        preds = (probs >= best_threshold).astype(int)

        accuracy = float((preds == y_test).mean())
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, labels=[0, 1], zero_division=0,
        )
        macro_precision, macro_recall, macro_f1 = float(precision.mean()), float(recall.mean()), float(f1.mean())
        fallo_precision, fallo_recall, fallo_f1 = float(precision[1]), float(recall[1]), float(f1[1])

        temp_dir = Path(tempfile.mkdtemp(prefix="ml43_train_"))
        upload_warning = None
        try:
            checkpoint = {"model_state_dict": best_state, "model_cfg": model_cfg}
            torch.save(checkpoint, temp_dir / MODEL_FILENAME)
            with open(temp_dir / SCALER_FILENAME, "wb") as f:
                pickle.dump({"scaler_x": scaler_x, "scaler_num": scaler_num}, f)
            if self._xai_background is not None:
                np.save(temp_dir / XAI_BACKGROUND_FILENAME, self._xai_background)

            try:
                new_run_id = upload_artifacts_to_mlflow(
                    str(temp_dir), mlflow_run_id=mlflow_run_id,
                    metrics={
                        "accuracy": accuracy, "macro_f1": macro_f1, "macro_precision": macro_precision,
                        "macro_recall": macro_recall, "fallo_f1": fallo_f1,
                        "fallo_precision": fallo_precision, "fallo_recall": fallo_recall,
                        "best_threshold": best_threshold, "n_train": split_idx, "n_test": n_windows - split_idx,
                    },
                )
                logger.info("Training complete. MLflow run_id=%s", new_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed (model trained but not persisted): %s", exc)
                upload_warning = f"Entrenamiento completado, pero falló la subida a MLflow: {exc}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return TrainResponse(
            detail="Entrenamiento completado",
            accuracy=round(accuracy, 4),
            macro_f1=round(macro_f1, 4),
            macro_precision=round(macro_precision, 4),
            macro_recall=round(macro_recall, 4),
            fallo_f1=round(fallo_f1, 4),
            fallo_precision=round(fallo_precision, 4),
            fallo_recall=round(fallo_recall, 4),
            n_train=int(split_idx),
            n_test=int(n_windows - split_idx),
            n_windows_total=int(n_windows),
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and real test-split metrics."""
        inputs = [
            InputField(name=name, type="float", description=desc)
            for name, desc in [
                ("temp_zona1", "Temperatura zona 1 del horno (°C)"),
                ("temp_zona2", "Temperatura zona 2 del horno (°C)"),
                ("temp_zona3", "Temperatura zona 3 del horno (°C)"),
                ("temp_salida_gases", "Temperatura de salida de gases (°C)"),
                ("presion_camara", "Presión interna de la cámara (mbar)"),
                ("presion_ventilacion", "Presión del sistema de ventilación (mbar)"),
                ("potencia_kw", "Potencia eléctrica consumida (kW)"),
                ("flujo_gas", "Flujo de gas del combustible (m3/h)"),
                ("humedad_relativa", "Humedad relativa interior (%)"),
                ("temp_ambiente", "Temperatura ambiente exterior (°C)"),
                ("setpoint_temp", "Temperatura objetivo programada (°C)"),
                ("posicion_valvula", "Posición de la válvula de gas (%)"),
                ("velocidad_ventilador", "Velocidad del ventilador de circulación (RPM)"),
            ]
        ] + [
            InputField(name="timestamp", type="string", description="Marca temporal de la medida (obligatoria)"),
            InputField(name="cycle_id", type="string", description="Identificador del ciclo de secado (opcional)"),
        ]
        outputs = [
            OutputField(name="predicted_anomaly_label", type="string", description="'Fallo' o 'No Fallo'"),
            OutputField(name="anomaly_probability", type="float", description="Probabilidad de anomalía [0, 1]"),
            OutputField(name="decision_threshold", type="float", description="Umbral de decisión utilizado (0.41)"),
            OutputField(name="cycle_id", type="string", description="Identificador del ciclo asociado a la ventana"),
            OutputField(name="window_index", type="int", description="Índice de ventana temporal"),
            OutputField(name="corrective_actions", type="object", description="Reporte XAI (CU44): estado interpretativo, bloques de acción y recomendaciones — solo en ventanas anómalas"),
        ]

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de anomalías y fallos en secadoras de cereales mediante modelo híbrido "
                "Deep Neuro-Fuzzy (BiLSTM + reglas fuzzy), con capa de explicabilidad XAI (SHAP + "
                f"reglas fuzzy) que genera acciones correctivas (CU43+CU44). Clasifica ventanas "
                f"temporales de {SEQ_LENGTH} medidas como Normal o Fallo. Dataset de entrenamiento "
                "100% sintético (3000 ciclos) — ver manifest known_issues."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                **TEST_METRICS,
                "decision_threshold": self._threshold,
                "dataset": "test_split",
                "synthetic_data_warning": (
                    "Dataset y métricas provienen de un generador sintético propio, sin validar "
                    "contra datos reales de hornos industriales — ver inbox/a43/manifest.yaml known_issues."
                ),
            },
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {"params": tracker.get_params(), "metrics": tracker.get_metrics()}
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base


def _best_f1_threshold(probs: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    """Scan candidate thresholds and return (best_f1, best_threshold) for the anomaly class.

    Mirrors the original pipeline's threshold selection criterion (manifest.yaml: "el umbral se
    selecciona... como el valor que maximiza el F1-score de la clase Fallo").
    """
    best_f1, best_thr = 0.0, DECISION_THRESHOLD
    for thr in np.arange(0.05, 0.96, 0.01):
        preds = (probs >= thr).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_f1, best_thr
