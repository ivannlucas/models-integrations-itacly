"""Modelo43CerealesPlugin — Deep Neuro-Fuzzy anomaly detection for cereal ovens.

Detects anomalies/faults in cereal baking cycles from 180-row sensor windows
(13 sensors), combining a BiLSTM branch with an ANFIS-style fuzzy-rule branch
(late fusion) plus SHAP + fuzzy-rule explanations (XAI). Vendored model code
is kept in sync with a43-44-neurofuzzy-anomalias-fallas/src/ (the training
repo this runtime serves).
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import InsufficientSensorWindowError, ModelNotLoadedError
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.modelo43_cereales import model_loader, postprocessing, preprocessing
from app.plugins.modelo43_cereales._vendor.dnf_loss import DNFLoss
from app.plugins.modelo43_cereales._vendor.preprocess import split_train_val_test_by_id, stats_windows
from app.plugins.modelo43_cereales._vendor.train_metrics import (
    alpha_entropy_mean,
    compute_class_weights,
    optimize_threshold_by_f1,
    rule_corr_mean,
)
from app.plugins.modelo43_cereales.constants import (
    DECISION_THRESHOLD,
    DEFAULT_MODEL_CFG,
    DNF_LOSS_KWARGS,
    FRAMEWORK,
    ID_COLUMN,
    MODEL_FILENAME,
    MODEL_ID,
    NORMAL_TOKENS,
    SCALER_FILENAME,
    SENSOR_COLUMNS,
    SEQ_LENGTH,
    STATS_CREATION,
    TARGET_COLUMN,
    TEST_METRICS,
    TRAIN_BATCH_SIZE,
    TRAIN_EXTERNAL_TEST_PCT,
    TRAIN_EXTERNAL_VAL_PCT,
    TRAIN_GRAD_CLIP,
    TRAIN_LR,
    TRAIN_MIN_DELTA,
    TRAIN_NUM_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_SCHEDULER_ETA_MIN,
    TRAIN_SCHEDULER_T_MAX,
    TRAIN_THRESHOLD_SEARCH_POINTS,
    TRAIN_WARMUP_EPOCHS,
    TRAIN_WEIGHT_DECAY,
    UNIFIED_METRIC_KEYS,
    VERSION,
    XAI_BACKGROUND_FILENAME,
    monitor_score,
)
from app.plugins.modelo43_cereales.mlflow_utils import (
    download_user_model_from_mlflow,
    get_calibrated_threshold,
    upload_artifacts_to_mlflow,
)
from app.plugins.modelo43_cereales.predict_dto import PredictBatchResponse
from app.plugins.modelo43_cereales.train_dto import TrainResponse

logger = logging.getLogger(__name__)


def _validate_saved_artifact(path: Path, loader) -> None:
    """Raise ValueError if a just-saved training artifact is missing, empty, or unreadable.

    Feedback 3 (modelo 43-44 audit, point 8 — previously investigated but left
    unimplemented pending confirmation): torch.save/pickle.dump/np.save have no return
    value to check, so a process killed or a disk filled mid-write can silently leave a
    0-byte or truncated file on disk. upload_artifacts_to_mlflow's own broad except only
    guards the *upload* step — a corrupt local file never reaches it, gets uploaded as-is,
    and the run is still reported as "Entrenamiento completado" with no warning. Catching
    this immediately after each save, before the upload is even attempted, turns that into
    a clear, immediate error instead.
    """
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' se guardó vacío (0 bytes) o no "
            "se generó — entrenamiento abortado, no se subió nada a MLflow."
        )
    try:
        loader(path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(
            f"El artefacto de entrenamiento '{path.name}' quedó corrupto tras guardarse "
            f"({exc}) — entrenamiento abortado, no se subió nada a MLflow."
        ) from exc


class Modelo43CerealesPlugin(ModelPluginPort):
    """Deep Neuro-Fuzzy (BiLSTM + fuzzy rules) anomaly detector for cereal ovens."""

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
        self._total_latency_ms: float = 0.0

    def load(self) -> None:
        (
            self._model,
            self._model_cfg,
            self._scaler_x,
            self._scaler_num,
            self._xai_background,
            self._explainer,
        ) = model_loader.load_artifacts()
        logger.info("Modelo43CerealesPlugin loaded: %s", MODEL_ID)

    def is_loaded(self) -> bool:
        return self._model is not None

    def _require_loaded(self) -> None:
        if self._model is None:
            raise ModelNotLoadedError("El modelo no está cargado.")

    def _record(self, latency_ms: float = 0.0) -> None:
        # modelo 43-44 audit: latency was already measured with perf_counter() (see
        # predict_batch) but only ever logged, never accumulated — stats() hardcoded
        # avg_latency_ms=None. Same _total_latency_ms/_predict_count pattern documented
        # in IMPLEMENTATION.md and already used by other plugins (e.g. ml25_wine_sulphites).
        self._predict_count += 1
        self._total_latency_ms += latency_ms
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    def _load_model_for_predict(self, mlflow_run_id: str) -> dict | None:
        """Resolve the model/scalers/explainer to use — user-trained (MLflow) or served."""
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
        # The trained weights and the trained decision_threshold must travel together — a
        # run's probabilities are only meaningful against the cutoff it was calibrated
        # with. Without this, predict_batch used to classify Fallo/No Fallo with the
        # served model's fixed 0.41 constant even when genuinely scoring with different,
        # user-trained weights.
        threshold = get_calibrated_threshold(mlflow_run_id, default=self._threshold)
        return {
            "model": model, "model_cfg": model_cfg, "scaler_x": scaler_x, "scaler_num": scaler_num,
            "xai_background": xai_background, "explainer": explainer, "temp_dir": temp_dir,
            "threshold": threshold,
        }

    def _served_ctx(self) -> dict:
        return {
            "model": self._model, "model_cfg": self._model_cfg,
            "scaler_x": self._scaler_x, "scaler_num": self._scaler_num,
            "xai_background": self._xai_background, "explainer": self._explainer, "temp_dir": None,
            "threshold": self._threshold,
        }

    # ── predict_inline ────────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictBatchResponse:
        """Not supported — modelo43-cereales requires a real 180-row sensor window.

        There is no meaningful single-snapshot prediction for this model (a
        synthetic repeated-snapshot window used to exist here but was removed:
        it never reflected real transient dynamics and could not be compared to
        genuine batch/CSV predictions). ``ModelPluginPort.predict_inline`` is
        abstract so this method must exist, but in practice it is unreachable
        through the HTTP API — ``PredictRequest`` only accepts ``mode="batch"``
        bodies, so FastAPI/Pydantic rejects anything else before this runs.
        """
        _ = features, model_key, threshold, mlflow_run_id
        raise InsufficientSensorWindowError(
            "modelo43-cereales no soporta predicción inline (una sola lectura). "
            "Usa el modo batch con un CSV de al menos 180 registros consecutivos."
        )

    # ── predict_batch ─────────────────────────────────────────────────────────

    def predict_batch(self, *, data_path: str, mlflow_run_id: str = "") -> PredictBatchResponse:
        """Score every valid 180-row window in a raw sensor CSV (one or more cycles)."""
        mlflow_ctx = self._load_model_for_predict(mlflow_run_id)
        ctx = mlflow_ctx or self._served_ctx()
        try:
            t0 = time.perf_counter()
            with local_file_path(data_path) as local_path:
                df = pd.read_csv(local_path)
            has_cycle_id_column = ID_COLUMN in df.columns.str.lower().tolist()

            x_arr, _, cycle_ids, window_timestamps = preprocessing.prepare_batch_sequences(df)
            if x_arr.shape[0] == 0:
                logger.warning("predict_batch — no valid windows found in %s", data_path)
                self._record((time.perf_counter() - t0) * 1000)
                return PredictBatchResponse(
                    model_id=MODEL_ID, predictions=[], output_path=None,
                    warning=(
                        f"No se encontró ninguna ventana válida de {SEQ_LENGTH} registros consecutivos. "
                        + ("El CSV no tiene columna 'cycle_id': se trató como un único ciclo continuo — "
                           "si contiene varios ciclos reales, añade 'cycle_id' para separarlos. "
                           if not has_cycle_id_column else "")
                        + f"Se requieren al menos {SEQ_LENGTH} filas por ciclo."
                    ),
                )

            x_scaled = postprocessing.scale_sequences(x_arr, ctx["scaler_x"])
            stats_scaled = postprocessing.compute_scaled_stats(x_arr, ctx["scaler_num"])
            scores = postprocessing.run_model(ctx["model"], x_scaled, stats_scaled)

            predictions = postprocessing.format_batch_predictions(
                x_arr, scores, cycle_ids, ctx["threshold"],
                ctx["explainer"], ctx["xai_background"], x_scaled, stats_scaled,
                window_timestamps=window_timestamps,
            )

            latency_ms = (time.perf_counter() - t0) * 1000
            self._record(latency_ms)
            logger.info(
                "predict_batch done — %d windows, %d failures (threshold=%.2f), latency_ms=%.1f count=%d",
                len(predictions),
                sum(1 for p in predictions if p["predicted_anomaly_label"] == "Fallo"),
                ctx["threshold"], latency_ms, self._predict_count,
            )
            warning = None
            if not has_cycle_id_column:
                warning = (
                    "El CSV no tiene columna 'cycle_id': todas las filas se trataron como un único "
                    "ciclo continuo. Si el archivo realmente contiene varios ciclos del horno, añade "
                    "'cycle_id' para que las ventanas no mezclen datos de ciclos distintos."
                )
            return PredictBatchResponse(
                model_id=MODEL_ID, predictions=predictions, output_path=None, warning=warning,
            )
        finally:
            if mlflow_ctx and mlflow_ctx["temp_dir"]:
                shutil.rmtree(mlflow_ctx["temp_dir"], ignore_errors=True)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str) -> TrainResponse:
        """Train a fresh model from a labeled CSV and upload it to MLflow.

        Does not replace the served checkpoint — pass the returned MLflow
        run_id back to /predict or /stats to use the newly trained model.

        Faithfully ports a43-44-neurofuzzy-anomalias-fallas's real training pipeline
        (config.yaml's training:/data_processing.external_data_split: sections) —
        modelo 43-44 audit, Fase 5. Earlier versions of this method used a much
        simplified stand-in (plain BCE loss, no class weighting, fixed 50 epochs, no
        LR schedule, a 70/15/15 window-count split) that trained a materially
        different, less-converged model — its F1-optimal decision_threshold was not
        comparable to the one the same CSV produces in the real repo (confirmed:
        0.77 here vs 0.49 there on an identical dataset). This version reproduces the
        real repo's split-by-cycle-id, DNFLoss, Adam+CosineAnnealingLR, warmup +
        early-stopping-on-a-composite-monitor-score, and 101-point threshold search,
        so results trained via this service are comparable to results trained
        locally on the same data.
        """
        from sklearn.preprocessing import StandardScaler

        logger.info("Starting modelo43-cereales training from %s", data_path)
        with local_file_path(data_path) as local_path:
            df = pd.read_csv(local_path)
        df.columns = df.columns.str.lower()

        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"CSV falta columna objetivo '{TARGET_COLUMN}'. Requerida para entrenamiento.")

        # Split RAW rows by whole cycle_id (never by row/window count) — see
        # _vendor/preprocess.py::split_train_val_test_by_id. A window is a sliding
        # SEQ_LENGTH-row slice within a single cycle, so splitting whole cycles first
        # and windowing each split independently guarantees no window straddles two
        # splits, and reproduces the real repo's train_pct/val_pct/test_pct exactly.
        raw_splits, _ = split_train_val_test_by_id(
            df, id_col=ID_COLUMN, target_column=TARGET_COLUMN,
            val_size=TRAIN_EXTERNAL_VAL_PCT / 100.0, test_size=TRAIN_EXTERNAL_TEST_PCT / 100.0,
            normal_tokens=NORMAL_TOKENS,
        )

        x_train_raw, y_train, _, _ = preprocessing.prepare_batch_sequences(raw_splits["train"])
        x_val_raw, y_val, _, _ = preprocessing.prepare_batch_sequences(raw_splits["val"])
        x_test_raw, y_test, _, _ = preprocessing.prepare_batch_sequences(raw_splits["test"])
        n_train, n_val, n_test = x_train_raw.shape[0], x_val_raw.shape[0], x_test_raw.shape[0]
        if n_train + n_val + n_test < 10:
            raise ValueError(
                f"Muy pocas secuencias para entrenar: {n_train + n_val + n_test}. "
                f"Se necesitan al menos 10 ventanas de {SEQ_LENGTH} filas en total."
            )
        if n_train == 0 or n_val == 0 or n_test == 0:
            raise ValueError(
                f"El split train/val/test por cycle_id dejó una partición vacía "
                f"(train={n_train}, val={n_val}, test={n_test}) — sube un CSV con más "
                f"ciclos distintos."
            )
        n_windows = n_train + n_val + n_test
        n_feat = x_train_raw.shape[-1]

        scaler_x = StandardScaler()
        x_train = scaler_x.fit_transform(x_train_raw.reshape(-1, n_feat)).reshape(x_train_raw.shape).astype(np.float32)
        x_val = scaler_x.transform(x_val_raw.reshape(-1, n_feat)).reshape(x_val_raw.shape).astype(np.float32)
        x_test = scaler_x.transform(x_test_raw.reshape(-1, n_feat)).reshape(x_test_raw.shape).astype(np.float32)

        df_stats_train = stats_windows(x_train_raw, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        df_stats_val = stats_windows(x_val_raw, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        df_stats_test = stats_windows(x_test_raw, feature_names=SENSOR_COLUMNS, stats_creation=STATS_CREATION)
        scaler_num = StandardScaler()
        s_train = scaler_num.fit_transform(df_stats_train.values).astype(np.float32)
        s_val = scaler_num.transform(df_stats_val.values).astype(np.float32)
        s_test = scaler_num.transform(df_stats_test.values).astype(np.float32)

        model_cfg = DEFAULT_MODEL_CFG.copy()
        model = model_loader.build_model(model_cfg)

        pos_weight = compute_class_weights(y_train)
        criterion = DNFLoss(anomaly_pos_weight=pos_weight, **DNF_LOSS_KWARGS)
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=TRAIN_SCHEDULER_T_MAX, eta_min=TRAIN_SCHEDULER_ETA_MIN,
        )

        x_t = torch.from_numpy(x_train)
        s_t = torch.from_numpy(s_train)
        y_t = torch.from_numpy(y_train.astype(np.float32))
        x_val_t = torch.from_numpy(x_val)
        s_val_t = torch.from_numpy(s_val)
        y_val_arr = y_val.astype(int)

        def _forward_step(x_batch, s_batch, y_batch):
            out = model(x_batch, s_batch)
            anomaly_score = torch.nan_to_num(out["anomaly_score"], nan=0.0, posinf=20.0, neginf=-20.0)
            rule_activations = out["rule_activations"]
            loss, _ = criterion(
                anomaly_score=anomaly_score,
                logit_anomaly_dl=out["logit_anomaly_dl"],
                logit_anomaly_fuzzy=out["logit_anomaly_fuzzy"],
                rule_activations=rule_activations,
                alpha_probs=out["alpha_probs"],
                y_anomaly=y_batch,
            )
            return loss

        def _evaluate_val():
            """Full forward pass over the validation split — returns the rich metric
            dict monitor_score() needs (anomaly_f1, fuzzy_f1, dead_rules_ratio,
            alpha_entropy_mean, rule_corr_mean), plus raw probabilities."""
            model.eval()
            with torch.no_grad():
                out_v = model(x_val_t, s_val_t)
                probs = torch.sigmoid(out_v["anomaly_score"]).view(-1).numpy()
                preds = (probs >= 0.5).astype(int)
                fuzzy_probs = torch.sigmoid(out_v["logit_anomaly_fuzzy"]).view(-1).numpy()
                fuzzy_preds = (fuzzy_probs >= 0.5).astype(int)
                rule_activations = out_v["rule_activations"]
                rule_activation_mean = rule_activations.mean(dim=0).numpy()
                dead_rules = int((rule_activation_mean < 1e-4).sum())
                dead_rules_ratio = float(dead_rules / max(len(rule_activation_mean), 1))
                val_out = {
                    "anomaly_f1": f1_score(y_val_arr, preds, zero_division=0),
                    "fuzzy_f1": f1_score(y_val_arr, fuzzy_preds, zero_division=0),
                    "dead_rules_ratio": dead_rules_ratio,
                    "alpha_entropy_mean": alpha_entropy_mean(out_v["alpha_probs"]),
                    "rule_corr_mean": rule_corr_mean(rule_activations),
                }
            return val_out, probs

        best_score = -np.inf
        best_state = None
        epochs_without_improve = 0

        for epoch in range(TRAIN_NUM_EPOCHS):
            model.train()
            # shuffle_train=False in the real config — batches iterate in the same
            # fixed (unshuffled) order every epoch.
            for start in range(0, n_train, TRAIN_BATCH_SIZE):
                end = start + TRAIN_BATCH_SIZE
                optimizer.zero_grad(set_to_none=True)
                loss = _forward_step(x_t[start:end], s_t[start:end], y_t[start:end])
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_GRAD_CLIP)
                optimizer.step()

            val_out, _ = _evaluate_val()
            score = monitor_score(val_out)
            scheduler.step()

            logger.info(
                "Epoch %d/%d — val_anomaly_f1=%.4f val_fuzzy_f1=%.4f monitor_score=%.4f lr=%.2e",
                epoch + 1, TRAIN_NUM_EPOCHS, val_out["anomaly_f1"], val_out["fuzzy_f1"],
                score, optimizer.param_groups[0]["lr"],
            )

            improved = score > (best_score + TRAIN_MIN_DELTA)
            if epoch < TRAIN_WARMUP_EPOCHS:
                if improved:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                if improved:
                    best_score = score
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                if epochs_without_improve >= TRAIN_PATIENCE:
                    logger.info("Early stopping en epoch %d (patience=%d).", epoch + 1, TRAIN_PATIENCE)
                    break

        if best_state is None:
            best_state = model.state_dict()
        model.load_state_dict(best_state)

        # Threshold calibration — 101-point uniform grid search over [0, 1] on the
        # VALIDATION split (never test), matching the real repo's
        # optimize_threshold_by_f1 exactly so a threshold trained here is comparable
        # to one trained locally on the same data.
        _, val_probs = _evaluate_val()
        threshold_metrics = optimize_threshold_by_f1(
            y_true=y_val_arr, prob_anomaly=val_probs, n_points=TRAIN_THRESHOLD_SEARCH_POINTS,
        )
        calibrated_threshold = float(threshold_metrics["best_threshold"])
        logger.info("Calibrated decision_threshold=%.4f (val F1-optimal, 101-point grid)", calibrated_threshold)

        # Final test evaluation — the ONLY time the test split is used, on the
        # checkpoint and threshold already selected above (never used to choose either).
        model.eval()
        with torch.no_grad():
            out_t = model(torch.from_numpy(x_test), torch.from_numpy(s_test))
            test_probs = torch.sigmoid(out_t["anomaly_score"]).view(-1).numpy()
        test_preds = (test_probs >= calibrated_threshold).astype(int)
        y_test_arr = y_test.astype(int)
        test_f1 = float(f1_score(y_test_arr, test_preds, zero_division=0))

        # F1 (Macro) / Recall (Macro) — modelo 43-44 audit, Fase 2. Unweighted mean across
        # BOTH classes (Normal=0, Fallo=1), computed on the same held-out test predictions
        # above. labels=[0, 1] is explicit so a class absent from a (small) test fold still
        # counts as 0 for that class instead of silently being dropped from the average.
        test_f1_macro = float(f1_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0))
        test_recall_macro = float(recall_score(y_test_arr, test_preds, labels=[0, 1], average="macro", zero_division=0))

        # accuracy/auc/precision/recall (binary, "Fallo" positive class) — modelo 43-44
        # audit, Fase 5 follow-up: stats(mlflow_run_id=...) has always reported these
        # under the anomaly_acc/anomaly_f1/anomaly_auc/anomaly_precision/anomaly_recall
        # keys, but train() never computed nor logged them, so a retrain's /stats could
        # never surface real values under those exact tiles — only decision_threshold
        # happened to work, by naming coincidence.
        test_accuracy = float(accuracy_score(y_test_arr, test_preds))
        test_precision = float(precision_score(y_test_arr, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test_arr, test_preds, zero_division=0))
        test_auc = (
            float(roc_auc_score(y_test_arr, test_probs)) if len(np.unique(y_test_arr)) > 1 else float("nan")
        )

        logger.info(
            "Final test evaluation — test_f1=%.4f test_f1_macro=%.4f test_recall_macro=%.4f "
            "(n_train=%d, n_val=%d, n_test=%d)",
            test_f1, test_f1_macro, test_recall_macro, n_train, n_val, n_test,
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="modelo43_train_"))
        upload_warning = None
        new_run_id: str | None = None
        try:
            checkpoint = {"model_state_dict": best_state, "model_cfg": model_cfg}
            torch.save(checkpoint, temp_dir / MODEL_FILENAME)
            _validate_saved_artifact(
                temp_dir / MODEL_FILENAME, lambda p: torch.load(p, weights_only=False)
            )

            with open(temp_dir / SCALER_FILENAME, "wb") as f:
                pickle.dump({"scaler_x": scaler_x, "scaler_num": scaler_num}, f)
            _validate_saved_artifact(
                temp_dir / SCALER_FILENAME, lambda p: pickle.load(open(p, "rb"))
            )

            if self._xai_background is not None:
                np.save(temp_dir / XAI_BACKGROUND_FILENAME, self._xai_background)
                _validate_saved_artifact(temp_dir / XAI_BACKGROUND_FILENAME, np.load)

            try:
                # Key names unified with ml45_cereals_dnsl_critical_point_detection and with
                # both real training repos' own results.json (modelo 43-45 audit, metrics
                # unification) — these ARE the final display names now, so stats() no longer
                # needs a separate legacy-key alias map: it just copies these straight through.
                loggable_metrics = {
                    "accuracy": round(test_accuracy, 4),
                    "fallo_auc": round(test_auc, 4) if test_auc == test_auc else test_auc,
                    "fallo_precision": round(test_precision, 4),
                    "fallo_recall": round(test_recall, 4),
                    "macro_f1": round(test_f1_macro, 4),
                    "macro_recall": round(test_recall_macro, 4),
                    "decision_threshold": round(calibrated_threshold, 4),
                    "n_windows_train": n_train, "n_windows_val": n_val,
                    "n_windows_test": n_test,
                }
                # MLflow rejects NaN/Inf metric values (auc is NaN when the test split
                # happens to contain only one class) — log only finite numbers so a
                # small/unbalanced retrain CSV can't turn "training succeeded, upload
                # failed" into an opaque MLflow client error.
                loggable_metrics = {
                    k: v for k, v in loggable_metrics.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)
                }
                new_run_id = upload_artifacts_to_mlflow(
                    str(temp_dir), mlflow_run_id=mlflow_run_id, metrics=loggable_metrics,
                )
                logger.info("Training complete. MLflow run_id=%s", new_run_id)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("MLflow artifact upload failed (model trained but not persisted): %s", exc)
                upload_warning = f"Entrenamiento completado, pero falló la subida a MLflow: {exc}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        split_val_pct = TRAIN_EXTERNAL_VAL_PCT
        split_test_pct = TRAIN_EXTERNAL_TEST_PCT
        split_train_pct = 100.0 - split_val_pct - split_test_pct

        return TrainResponse(
            detail="Entrenamiento completado",
            accuracy=round(test_accuracy, 4),
            fallo_auc=round(test_auc, 4) if test_auc == test_auc else test_auc,
            fallo_precision=round(test_precision, 4),
            fallo_recall=round(test_recall, 4),
            macro_f1=round(test_f1_macro, 4),
            macro_recall=round(test_recall_macro, 4),
            decision_threshold=round(calibrated_threshold, 4),
            n_windows_train=int(n_train),
            n_windows_val=int(n_val),
            n_windows_test=int(n_test),
            n_windows_total=int(n_windows),
            split_train_pct=round(split_train_pct, 1),
            split_val_pct=round(split_val_pct, 1),
            split_test_pct=round(split_test_pct, 1),
            mlflow_run_id=new_run_id,
            upload_warning=upload_warning,
        )

    # ── stats ─────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        """Return model metadata, the input/output contract and test-split metrics."""
        inputs = [
            InputField(name=name, type="float", description=desc)
            for name, desc in [
                ("temp_zona1", "Temperatura zona 1 (°C)"),
                ("temp_zona2", "Temperatura zona 2 (°C)"),
                ("temp_zona3", "Temperatura zona 3 (°C)"),
                ("temp_salida_gases", "Temperatura salida de gases (°C)"),
                ("presion_camara", "Presión de cámara (Pa)"),
                ("presion_ventilacion", "Presión de ventilación (Pa)"),
                ("potencia_kw", "Potencia eléctrica (kW)"),
                ("flujo_gas", "Flujo de gas (m³/h)"),
                ("humedad_relativa", "Humedad relativa (%)"),
                ("temp_ambiente", "Temperatura ambiente (°C)"),
                ("setpoint_temp", "Temperatura de consigna (°C)"),
                ("posicion_valvula", "Posición de válvula (%)"),
                ("velocidad_ventilador", "Velocidad del ventilador (RPM)"),
            ]
        ] + [
            InputField(name="timestamp", type="datetime", description="Marca temporal de la medida"),
            InputField(name="cycle_id", type="string", description="Identificador del ciclo de horneado"),
        ]
        outputs = [
            OutputField(name="predicted_anomaly_label", type="string", description="'Fallo' o 'No Fallo'"),
            OutputField(name="anomaly_probability", type="float", description="Probabilidad de anomalía [0, 1]"),
            OutputField(name="decision_threshold", type="float", description="Umbral de decisión utilizado"),
            OutputField(name="cycle_id", type="string", description="Identificador del ciclo asociado a la ventana"),
            OutputField(name="window_index", type="int", description="Índice de ventana temporal (1..N)"),
            OutputField(name="timestamp_init", type="string", description="Timestamp de la primera lectura de la ventana"),
            OutputField(name="timestamp_end", type="string", description="Timestamp de la última lectura de la ventana"),
            OutputField(name="Estado del sistema", type="string", description="Estado interpretativo global de la ventana (Normal / Normal con señales / Alerta no confirmada / Anomalía confirmada)"),
            OutputField(name="Subsistemas con alteraciones", type="string", description="Subsistemas del horno identificados con comportamiento alterado"),
            OutputField(name="Variables con alteraciones", type="string", description="Variables/sensores concretos que motivan el estado interpretativo"),
            OutputField(name="Acciones correctivas recomendadas", type="string", description="Acciones sugeridas para corregir la alteración detectada"),
        ]

        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de fallos en hornos cerealistas y generación de acciones "
                "correctivas (XAI). Modelo híbrido Deep Neuro-Fuzzy (BiLSTM + reglas "
                f"fuzzy) que clasifica ventanas temporales de {SEQ_LENGTH} medidas del "
                "horno como Normal o Fallo, con explicabilidad SHAP + reglas fuzzy y "
                "generación de acciones correctivas."
            ),
            task_type="binary_classification",
            framework=FRAMEWORK,
            inputs=inputs,
            outputs=outputs,
            metrics={**TEST_METRICS, "decision_threshold": self._threshold},
            runtime_stats=RuntimeStats(
                total_predictions=self._predict_count,
                avg_latency_ms=(
                    round(self._total_latency_ms / self._predict_count, 1)
                    if self._predict_count > 0 else None
                ),
            ),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                mlflow_metrics = tracker.get_metrics()
                # train() now logs metrics to MLflow under exactly the unified display key
                # names (accuracy, fallo_auc, fallo_precision, fallo_recall, macro_f1,
                # macro_recall, decision_threshold, n_windows_train/val/test) — modelo 43-45
                # audit, metrics unification. So this OVERWRITES base.metrics key-by-key with
                # the real per-run values instead of adding differently-named keys alongside
                # the base/reference ones (which used to leave both an old and a new key
                # showing the same number under two different labels). Only known keys are
                # copied — unlike the previous generic "copy every numeric key" loop, this
                # can't be polluted by unrelated metrics a future MLflow run might log.
                for key in UNIFIED_METRIC_KEYS:
                    value = mlflow_metrics.get(key)
                    if isinstance(value, (int, float)):
                        base.metrics[key] = value
                for key in ("n_windows_train", "n_windows_val", "n_windows_test"):
                    value = mlflow_metrics.get(key)
                    if isinstance(value, (int, float)):
                        base.metrics[key] = value
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
