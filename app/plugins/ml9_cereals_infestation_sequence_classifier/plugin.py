"""Ml9CerealsInfestationSequenceClassifierPlugin — GRU classifier over 48 h windows of stored-cereal telemetry.

Classifies each temporal window of a monitored grain series as sano / insectos / moho_critico from
CO2, temperature, ambient RH and grain humidity (a09 / INF-CER). See inbox/a09/manifest.yaml for the
full input/output contract, the "Ensemble learning" title vs. LSTM-vs-GRU implementation
discrepancy, and the synthetic-data caveat that applies to every metric reported here.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InsufficientSequenceHistoryError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml9_cereals_infestation_sequence_classifier import model_loader, postprocessing, preprocessing
from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.preprocess import split_sample_ids_stratified
from app.plugins.ml9_cereals_infestation_sequence_classifier._vendor.sequential import (
    build_class_weights,
    evaluate_sequence_model,
    save_checkpoint,
    save_pickle,
    set_global_seed,
    train_sequence_model,
    transform_sequences,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.constants import (
    ACCEPTANCE_THRESHOLDS,
    CLASS_LABELS,
    FRAMEWORK,
    GROUP_COLUMN,
    HOLDOUT_METRICS,
    LABEL_MODE,
    MODEL_ID,
    N_FEATURES,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    STRIDE,
    SYNTHETIC_DATA_WARNING,
    TARGET_COLUMN,
    TASK_TYPE,
    TIMESTAMP_COLUMN,
    TRAIN_HARD_REQUIRED_COLUMNS,
    TRAIN_RECOMMENDED_COLUMNS,
    USER_BUNDLE_FILENAME,
    USER_MODEL_FILENAME,
    USER_SCALER_FILENAME,
    VERSION,
    WINDOW_SIZE,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.mlflow_utils import download_user_model_from_mlflow
from app.plugins.ml9_cereals_infestation_sequence_classifier.model_loader import _store
from app.plugins.ml9_cereals_infestation_sequence_classifier.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml9_cereals_infestation_sequence_classifier.train_dto import TrainResponse

logger = logging.getLogger(__name__)

_METRIC_KEYS = ("accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "log_loss")


class Ml9CerealsInfestationSequenceClassifierPlugin(ModelPluginPort):
    """Recurrent (GRU) 3-class classifier over 48-hour sliding windows per monitored series."""

    def __init__(self) -> None:
        """Initialize an unloaded plugin with empty runtime counters."""
        self._checkpoint: dict | None = None
        self._scaler: Any = None
        self._bundle: dict | None = None
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the served GRU checkpoint plus its fitted scaler and training bundle."""
        self._checkpoint, self._scaler, self._bundle = model_loader.load_artifacts()
        logger.info("Ml9CerealsInfestationSequenceClassifierPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        """Return True once the checkpoint, scaler and bundle are loaded."""
        return self._checkpoint is not None and self._scaler is not None and self._bundle is not None

    def _require_loaded(self) -> None:
        if not self.is_loaded():
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record_prediction(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _resolve_user_model(self, mlflow_run_id: str) -> str | None:
        """Swap in a user fine-tuned model from MLflow. Returns the temp dir to clean up, if any."""
        loaded = download_user_model_from_mlflow(mlflow_run_id)
        if not loaded:
            logger.warning(
                "No se pudo recuperar el modelo de MLflow run_id=%s; se usa el artefacto fijo.",
                mlflow_run_id,
            )
            return None
        self._checkpoint, self._scaler, self._bundle, user_tmp = loaded
        return user_tmp

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid 48-hour window in a raw hourly-telemetry CSV (one or more series)."""
        user_tmp = None
        saved = (self._checkpoint, self._scaler, self._bundle)
        if mlflow_run_id:
            logger.info("predict_batch — using user model from MLflow run_id=%s", mlflow_run_id)
            user_tmp = self._resolve_user_model(mlflow_run_id)
        try:
            self._require_loaded()
            with local_file_path(data_path) as local_path:
                raw_df = pd.read_csv(local_path)

            missing = preprocessing.missing_required_columns(raw_df)
            if missing:
                raise ValueError(f"El CSV no trae las columnas obligatorias: {missing}")

            payload, _, has_target = preprocessing.build_windows(raw_df, self._bundle)
            if len(payload["X_seq"]) == 0:
                raise InsufficientSequenceHistoryError(
                    f"Ninguna serie del CSV alcanza las {WINDOW_SIZE} observaciones horarias "
                    f"consecutivas que necesita una ventana (pad_short_sequences=false). Series "
                    f"recibidas: {raw_df[GROUP_COLUMN].nunique()}, filas: {len(raw_df)}."
                )

            proba = postprocessing.run_inference(self._checkpoint, self._scaler, payload["X_seq"])
            pred_df = postprocessing.build_predictions_frame(
                payload["window_meta"], proba, payload["y_seq"], has_target=has_target,
                x_seq=payload["X_seq"], feature_columns=payload["feature_columns"],
            )

            evaluated = None
            if has_target and np.size(payload["y_seq"]):
                evaluated = self._metrics_from_windows(payload["X_seq"], payload["y_seq"])

            self._record_prediction()
            logger.info(
                "predict_batch done — %d ventanas, %d series, target=%s, mlflow=%s",
                len(pred_df), int(pred_df[GROUP_COLUMN].nunique()), has_target, bool(mlflow_run_id),
            )
            return PredictBatchResponse(
                model_id=MODEL_ID,
                n_windows=len(pred_df),
                n_series=int(pred_df[GROUP_COLUMN].nunique()),
                predictions=postprocessing.serialize_records(pred_df),
                class_distribution=postprocessing.class_distribution(pred_df),
                evaluated_metrics=evaluated,
                output_path=None,
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
                self._checkpoint, self._scaler, self._bundle = saved

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(  # pylint: disable=too-many-locals
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:  # pylint: disable=too-many-locals
        """Score the most recent 48-hour window of the submitted series history."""
        _ = model_key
        user_tmp = None
        saved = (self._checkpoint, self._scaler, self._bundle)
        if mlflow_run_id:
            logger.info("predict_inline — using user model from MLflow run_id=%s", mlflow_run_id)
            user_tmp = self._resolve_user_model(mlflow_run_id)
        try:
            self._require_loaded()
            rows: list[dict] = features["rows"]
            raw_df = preprocessing.build_raw_dataframe(rows)

            missing = preprocessing.missing_required_columns(raw_df)
            if missing:
                raise ValueError(f"Las filas recibidas no traen las columnas obligatorias: {missing}")

            payload, _, has_target = preprocessing.build_windows(raw_df, self._bundle)
            idx = preprocessing.last_window_index(payload)
            if idx == -1:
                raise InsufficientSequenceHistoryError(
                    f"El histórico recibido no genera ninguna ventana: se necesitan al menos "
                    f"{WINDOW_SIZE} observaciones horarias consecutivas del mismo sample_id "
                    f"(pad_short_sequences=false). Filas recibidas: {len(raw_df)}, series: "
                    f"{raw_df[GROUP_COLUMN].nunique()}."
                )

            proba = postprocessing.run_inference(self._checkpoint, self._scaler, payload["X_seq"])
            pred_df = postprocessing.build_predictions_frame(
                payload["window_meta"], proba, payload["y_seq"], has_target=has_target,
            )
            # build_predictions_frame sorts by (sample_id, timestamp_end); re-find the latest window.
            row = pred_df.loc[pd.to_datetime(pred_df["timestamp_end"]).idxmax()]

            confidence = float(row["confidence"])
            low_confidence = bool(threshold is not None and confidence < float(threshold))
            y_true = int(row["y_true"]) if "y_true" in pred_df.columns and pd.notna(row.get("y_true")) else None

            self._record_prediction()
            logger.info(
                "predict_inline done — sample_id=%s clase=%s confianza=%.4f ventanas=%d count=%d",
                row[GROUP_COLUMN], row["pred_label"], confidence, len(pred_df), self._predict_count,
            )
            return PredictInlineResponse(
                model_id=MODEL_ID,
                sample_id=str(row[GROUP_COLUMN]),
                window_index=int(row["window_index"]),
                timestamp_start=str(postprocessing.clean_scalar(row["timestamp_start"])),
                timestamp_end=str(postprocessing.clean_scalar(row["timestamp_end"])),
                pred_class=int(row["pred_class"]),
                pred_label=str(row["pred_label"]),
                proba_sano=float(row["proba_sano"]),
                proba_insectos=float(row["proba_insectos"]),
                proba_moho_critico=float(row["proba_moho_critico"]),
                confidence=confidence,
                low_confidence=low_confidence,
                n_rows_used=len(raw_df),
                n_windows_available=len(pred_df),
                y_true=y_true,
                model_name=MODEL_ID,
                # Las 65 variables derivadas que vio el modelo en el último paso de la ventana
                # puntuada — es lo que el servicio de explicabilidad de la plataforma necesita
                # para atribuir la decisión (las 4 señales crudas por sí solas no lo permiten).
                xai_feature_values=postprocessing.window_feature_values(
                    payload["X_seq"], payload["feature_columns"], idx,
                ),
            )
        finally:
            if user_tmp:
                shutil.rmtree(user_tmp, ignore_errors=True)
                self._checkpoint, self._scaler, self._bundle = saved

    # ── train (fine-tuning) ───────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:  # pylint: disable=too-many-locals,too-many-statements
        """Fine-tune the served GRU checkpoint on the caller's own labelled CSV.

        Follows the delivered training procedure (src/training/trainer.py) for everything that
        affects semantics — stratified split by sample_id, sliding windows, class-weighted
        CrossEntropy, AdamW with the original lr/weight_decay/batch_size, early stopping on
        validation f1_macro — with two deliberate differences, both documented in
        inbox/a09/manifest.yaml:

        1. It fine-tunes the served GRU (weights cloned into a fresh instance, so a concurrent
           /predict never sees a half-trained model) instead of re-running the AI team's 8+8
           hyperparameter search from scratch. The winning architecture is not re-selected.
        2. The fitted StandardScaler is reused as-is, never refitted: the 65-feature contract of the
           bundle must keep holding, and a small user CSV is not a sound basis to re-derive scaling.
        """
        self._require_loaded()
        with local_file_path(data_path) as local_path:
            raw_df = pd.read_csv(local_path)

        missing = [c for c in TRAIN_HARD_REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing:
            raise ValueError(f"El CSV de entrenamiento no trae las columnas requeridas: {missing}")

        payload, runtime_cfg, _ = preprocessing.build_windows(raw_df, self._bundle, require_target=True)
        if len(payload["X_seq"]) == 0:
            raise ValueError(
                f"El CSV no genera ninguna ventana de entrenamiento: se necesitan al menos "
                f"{WINDOW_SIZE} observaciones horarias consecutivas por sample_id."
            )

        try:
            split_ids = split_sample_ids_stratified(raw_df, runtime_cfg)
        except ValueError as exc:
            raise ValueError(
                f"No se puede partir el CSV en train/validation/test sin fuga entre series: {exc} "
                "Se necesitan al menos 3 sample_id por clase."
            ) from exc

        group_seq = np.asarray(payload["group_seq"])
        subsets = {}
        for key in ("train_sample_ids", "validation_sample_ids", "test_sample_ids"):
            mask = np.isin(group_seq, np.asarray(split_ids[key]))
            subsets[key] = (payload["X_seq"][mask], payload["y_seq"][mask])
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = (
            subsets["train_sample_ids"], subsets["validation_sample_ids"], subsets["test_sample_ids"],
        )
        if min(len(x_train), len(x_val), len(x_test)) == 0:
            raise ValueError(
                "Alguna de las particiones (train/validation/test) se ha quedado sin ventanas. "
                f"Ventanas por partición: train={len(x_train)}, validation={len(x_val)}, "
                f"test={len(x_test)}. Aporta más series o series más largas."
            )

        model_config = dict(self._checkpoint["model_config"])
        num_classes = int(model_config["num_classes"])
        params = {
            "hidden_size": model_config["hidden_size"],
            "num_layers": model_config["num_layers"],
            "dropout": model_config["dropout"],
            "bidirectional": model_config["bidirectional"],
            **{k: v for k, v in self._checkpoint.get("training_params", {}).items()
               if k in ("batch_size", "learning_rate", "weight_decay")},
        }
        set_global_seed(int(runtime_cfg.get("training", {}).get("random_state", 42)))
        class_weights = build_class_weights(y_train, num_classes=num_classes)

        baseline = self._metrics_from_windows(x_test, y_test)

        tracker: BaseMLflowTracker | None = BaseMLflowTracker(mlflow_run_id) if mlflow_run_id else None
        if tracker:
            tracker.log_params({
                "modo": "fine_tuning",
                "model_type": self._checkpoint["model_type"],
                "optimizer": "AdamW",
                "loss": "CrossEntropyLoss(class_weights)",
                "window_size": WINDOW_SIZE,
                "stride": STRIDE,
                "label_mode": LABEL_MODE,
                **params,
            })

        result = train_sequence_model(
            str(self._checkpoint["model_type"]),
            x_train, y_train, x_val, y_val,
            params=params,
            cfg=runtime_cfg,
            scaler=self._scaler,          # reused, never refitted
            num_classes=num_classes,
            class_weights=class_weights,
            device="cpu",
            init_state_dict=self._checkpoint["state_dict"],
        )

        test_eval = evaluate_sequence_model(
            result.model,
            transform_sequences(self._scaler, x_test),
            y_test,
            batch_size=int(params.get("batch_size", 64)),
            device="cpu",
        )
        metrics = {k: float(test_eval["metrics"][k]) for k in _METRIC_KEYS}

        artifact_path, upload_warning = self._persist_retrained(result, metrics, tracker)

        logger.info(
            "ml9 train() done — ventanas train/val/test=%d/%d/%d f1_macro=%.4f (baseline %.4f) mlflow=%s",
            len(x_train), len(x_val), len(x_test), metrics["f1_macro"],
            baseline["f1_macro"], bool(mlflow_run_id),
        )
        return TrainResponse(
            detail=(
                "Fine-tuning completado sobre el checkpoint servido "
                f"({self._checkpoint['model_type'].upper()}), reutilizando el escalador original. "
                "Métricas medidas sobre el hold-out por sample_id del CSV aportado."
            ),
            n_series_train=len(split_ids["train_sample_ids"]),
            n_series_validation=len(split_ids["validation_sample_ids"]),
            n_series_test=len(split_ids["test_sample_ids"]),
            n_windows_train=len(x_train),
            n_windows_validation=len(x_val),
            n_windows_test=len(x_test),
            epochs_run=int(result.checkpoint["best_epoch"]),
            **metrics,
            validation_f1_macro=float(result.metrics["f1_macro"]),
            baseline_f1_macro=baseline["f1_macro"],
            artifact_path=artifact_path,
            upload_warning=upload_warning,
        )

    def _metrics_from_windows(self, x_seq: np.ndarray, y_seq: np.ndarray) -> dict[str, float]:
        """Evaluate the currently served model on already-built windows."""
        evaluation = evaluate_sequence_model(
            self._checkpoint["model"],
            transform_sequences(self._scaler, x_seq),
            np.asarray(y_seq).astype(int),
            batch_size=int(self._checkpoint.get("training_params", {}).get("batch_size", 128)),
            device="cpu",
        )
        return {k: float(evaluation["metrics"][k]) for k in _METRIC_KEYS}

    def _persist_retrained(
        self,
        result: Any,
        metrics: dict[str, float],
        tracker: BaseMLflowTracker | None,
    ) -> tuple[str, str | None]:
        """Save the fine-tuned bundle locally under user_* names and, if asked, upload to MLflow.

        The fixed S3 artifacts are never overwritten: a user retrain lands on user_*.pt/json/pkl.
        """
        _store.local_dir.mkdir(parents=True, exist_ok=True)
        local_model_path = _store.local_dir / USER_MODEL_FILENAME
        save_checkpoint(local_model_path, result.checkpoint)

        user_bundle = dict(self._bundle or {})
        user_bundle["model_files"] = {"winner": USER_MODEL_FILENAME}
        user_bundle["scaler_file"] = USER_SCALER_FILENAME
        user_bundle["best_model_name"] = str(result.model_type)
        user_bundle["best_model_metric"] = float(result.metrics["f1_macro"])
        user_bundle["retrained_by_user"] = True
        with open(_store.local_dir / USER_BUNDLE_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(user_bundle, fh, ensure_ascii=False, indent=2)
        save_pickle(self._scaler, _store.local_dir / USER_SCALER_FILENAME)

        upload_warning = None
        if tracker:
            try:
                tracker.log_metrics(metrics)
                mlflow_tmp = tempfile.mkdtemp(prefix="ml9_mlflow_")
                try:
                    torch.save(result.checkpoint, Path(mlflow_tmp) / USER_MODEL_FILENAME)
                    shutil.copy2(_store.local_dir / USER_BUNDLE_FILENAME, Path(mlflow_tmp) / USER_BUNDLE_FILENAME)
                    shutil.copy2(_store.path(SCALER_FILENAME), Path(mlflow_tmp) / USER_SCALER_FILENAME)
                    tracker.upload_artifacts(mlflow_tmp, artifact_path="model")
                finally:
                    shutil.rmtree(mlflow_tmp, ignore_errors=True)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed: %s", exc)
                upload_warning = (
                    f"El fine-tuning se completó y el modelo se guardó en local ({local_model_path}), "
                    f"pero falló la subida a MLflow: {exc}"
                )
        return str(local_model_path), upload_warning

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and the real hold-out metrics."""
        inputs = [
            InputField(
                name=GROUP_COLUMN, type="string",
                description="Identificador de la serie monitorizada (silo/lote). Agrupa las observaciones.",
            ),
            InputField(
                name=TIMESTAMP_COLUMN, type="datetime",
                description="Marca temporal horaria de la observación. Debe parsear y no tener nulos.",
            ),
        ] + [
            InputField(
                name=name, type="float",
                description=desc,
            )
            for name, desc in (
                ("co2_ppm", "Concentración de CO2 en ppm (rango observado en el dataset: 350–2543)."),
                ("temp_c", "Temperatura en °C (rango observado: 7,8–37,2)."),
                ("ambient_rh_pct", "Humedad relativa ambiental en % (rango observado: 24,6–89,4)."),
                ("humidity_grain_pct", "Humedad del grano en % (rango observado: 11,3–17,2)."),
            )
        ] + [
            InputField(
                name=TARGET_COLUMN, type="int", default=None,
                description=(
                    "Etiqueta real por observación (0=sano, 1=insectos, 2=moho_critico). Obligatoria "
                    "en /train; opcional en /predict (si viene, se devuelve y_true y las métricas)."
                ),
            ),
            InputField(
                name="target_global", type="int", default=None,
                description="Clase global de la serie; solo se usa para estratificar la partición en /train.",
            ),
        ]
        outputs = [
            OutputField(name="pred_class", type="int", description="Clase predicha de la ventana: 0, 1 o 2"),
            OutputField(name="pred_label", type="str", description="sano / insectos / moho_critico"),
            OutputField(name="proba_sano", type="float", description="Probabilidad de la clase sano"),
            OutputField(name="proba_insectos", type="float", description="Probabilidad de la clase insectos"),
            OutputField(name="proba_moho_critico", type="float", description="Probabilidad de la clase moho crítico"),
            OutputField(name="confidence", type="float", description="Probabilidad de la clase predicha"),
            OutputField(name="window_index", type="int", description="Índice de la ventana dentro de la serie"),
            OutputField(name="timestamp_end", type="datetime", description="Instante al que se refiere la clase"),
        ]
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Clasificador secuencial recurrente (GRU) que asigna a cada ventana de 48 horas de "
                "una serie de cereal almacenado uno de tres estados: sano, insectos o moho crítico. "
                "Entradas: CO2, temperatura, humedad ambiental y humedad del grano, muestreadas cada "
                "hora. La unidad de decisión es la ventana temporal, no la observación aislada ni la "
                "serie completa. El entregable se titula 'Ensemble learning' por continuidad "
                "administrativa, pero la solución validada compara LSTM y GRU y sirve solo la "
                "ganadora (GRU) — ver inbox/a09/manifest.yaml."
            ),
            task_type=TASK_TYPE,
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={
                **HOLDOUT_METRICS,
                "clases": CLASS_LABELS,
                "ventana": {
                    "window_size": WINDOW_SIZE,
                    "stride": STRIDE,
                    "label_mode": LABEL_MODE,
                    "n_features_derivadas": N_FEATURES,
                    "pad_short_sequences": False,
                    "min_observaciones_por_serie": WINDOW_SIZE,
                },
                "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
                "sensores": SENSOR_COLUMNS,
                "synthetic_data_warning": SYNTHETIC_DATA_WARNING,
                "train_columns": {
                    "required": TRAIN_HARD_REQUIRED_COLUMNS,
                    "recommended": TRAIN_RECOMMENDED_COLUMNS,
                },
            },
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=None,  # a single call scores many windows; per-window latency is not comparable
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
                logger.info("Stats enriched with MLflow data for run_id=%s", mlflow_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
