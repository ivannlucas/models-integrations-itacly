"""Ml41MeatCuringMachineryAcousticAnomalyPlugin — Audio-MAE (ViT-Tiny) acoustic anomaly
detection for curing-chamber machinery (fan/pump/slider/valve), MIMII-based.

See inbox/a41/manifest.yaml for the full provenance of every design decision referenced
in comments below (architecture per machine, thresholds, golden cases, known_issues).
"""
from __future__ import annotations

import csv
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from app.application.dto.stats_dto import InputField, OutputField, RuntimeStats, StatsResponse
from app.domain.ports.model_plugin_port import ModelPluginPort
from app.domain.services.exceptions import (
    InvalidAudioError, ModelNotLoadedError, UnsupportedMachineConfigurationError,
)
from app.domain.services.mlflow_tracker import BaseMLflowTracker
from app.infrastructure.artifact_store import local_file_path
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.constants import (
    ARTIFACT_FOLDER_NAME,
    ARCHITECTURE_BY_MACHINE,
    CHECKPOINT_FILENAME,
    MACHINE_IDS,
    MACHINES,
    MAHA_STATS_FILENAME,
    METRICS_REPORTED,
    MODEL_ID,
    SNRS,
    TRAINING_HYPERPARAMS_BY_MACHINE,
    VERSION,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.inference import (
    fit_mahalanobis,
    get_cls_embedding_single,
    mahalanobis_score,
    run_inference_single,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.mlflow_utils import (
    download_user_model_from_mlflow,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.model_loader import (
    CheckpointCache,
    LoadedCombination,
    artifacts_available,
    build_model,
    combination_dir,
    ensure_artifacts_downloaded,
    _safe_device,
    _store,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.postprocessing import build_inline_result
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.predict_dto import (
    PredictBatchResponse,
    PredictInlineResponse,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.preprocessing import (
    audio_base64_to_logmel,
    normalize_logmel,
    wav_to_logmel,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.train_dto import (
    CombinationTrainMetrics,
    TrainResponse,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.trainer import (
    evaluate_against_abnormal,
    fine_tune,
)

logger = logging.getLogger(__name__)


class Ml41MeatCuringMachineryAcousticAnomalyPlugin(ModelPluginPort):

    def __init__(self) -> None:
        self._device: torch.device | None = None
        self._cache: CheckpointCache | None = None
        self._ready: bool = False
        self._predict_count: int = 0
        self._last_predict_at: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        ensure_artifacts_downloaded()
        self._device = _safe_device()
        self._cache = CheckpointCache()
        self._ready = artifacts_available()
        if not self._ready:
            logger.warning(
                "Ml41 plugin loaded but no combination checkpoints were found under %s",
                ARTIFACT_FOLDER_NAME,
            )
        logger.info("Ml41MeatCuringMachineryAcousticAnomalyPlugin loaded: ready=%s device=%s",
                    self._ready, self._device)

    def is_loaded(self) -> bool:
        return self._ready

    def _require_loaded(self) -> None:
        if not self._ready:
            raise ModelNotLoadedError("El modelo no está cargado (no hay checkpoints disponibles).")

    def _record(self) -> None:
        self._predict_count += 1
        self._last_predict_at = datetime.now(tz=timezone.utc).isoformat()

    # ── scoring ──────────────────────────────────────────────────────────────

    @staticmethod
    def _as_dict(loaded) -> dict:
        """Normalize a LoadedCombination (local/cached) or plain dict (from MLflow) to a dict."""
        if isinstance(loaded, LoadedCombination):
            return {
                "model": loaded.model, "norm_mean": loaded.norm_mean, "norm_std": loaded.norm_std,
                "maha_mean": loaded.maha_mean, "maha_inv_cov": loaded.maha_inv_cov,
                "maha_pca": loaded.maha_pca, "device": loaded.device,
                "threshold": loaded.threshold,
            }
        return loaded

    def _prediction_threshold(self, loaded, override: float | None = None) -> float:
        """Use the selected model's calibration, or an explicit inline override."""
        value = override if override is not None else self._as_dict(loaded)["threshold"]
        if value is None:
            raise UnsupportedMachineConfigurationError(
                "This checkpoint has no calibrated decision threshold. Train with normal "
                "and abnormal reference audio, or supply an explicit inline threshold."
            )
        if not np.isfinite(value) or value < 0:
            raise UnsupportedMachineConfigurationError("The decision threshold must be finite and non-negative.")
        return float(value)

    def _score_spectrogram(self, loaded, spec_raw: np.ndarray) -> tuple[float, float]:
        """Given a raw (unnormalized) log-Mel spectrogram, return (mse_score, maha_score)."""
        ctx = self._as_dict(loaded)
        spec_norm = normalize_logmel(spec_raw, ctx["norm_mean"], ctx["norm_std"])
        tensor = torch.from_numpy(spec_norm).float().unsqueeze(0)  # (1, 1, n_mels, target_frames)

        mse_score = run_inference_single(ctx["model"], tensor, ctx["device"])
        embedding = get_cls_embedding_single(ctx["model"], tensor, ctx["device"])
        maha = mahalanobis_score(embedding, ctx["maha_mean"], ctx["maha_inv_cov"], ctx["maha_pca"])
        return mse_score, maha

    def _resolve_combination(self, machine: str, machine_id: str, snr: str, mlflow_run_id: str):
        """Return (loaded_combination, temp_dir_or_None). loaded_combination is either a
        LoadedCombination (local/cached) or a plain dict (from MLflow)."""
        if mlflow_run_id:
            downloaded = download_user_model_from_mlflow(mlflow_run_id)
            if downloaded:
                combinations, tmp = downloaded
                key = (machine, machine_id, snr)
                if key in combinations:
                    return combinations[key], tmp
                logger.warning(
                    "MLflow run_id=%s has no fine-tuned checkpoint for %s — falling back "
                    "to the standard artifact.", mlflow_run_id, key,
                )
                shutil.rmtree(tmp, ignore_errors=True)
        self._require_loaded()
        loaded = self._cache.get(machine, machine_id, snr, self._device)
        return loaded, None

    # ── predict_inline ───────────────────────────────────────────────────────

    def predict_inline(
        self,
        *,
        features: dict,
        model_key: str | None = None,
        threshold: float | None = None,
        mlflow_run_id: str = "",
    ) -> PredictInlineResponse:
        _ = model_key  # not used: (machine, machine_id, snr) already select the combination
        machine, machine_id, snr = features["machine"], features["machine_id"], features["snr"]
        temp_dir = None
        try:
            loaded, temp_dir = self._resolve_combination(machine, machine_id, snr, mlflow_run_id)
            threshold_used = self._prediction_threshold(loaded, threshold)
            spec_raw = audio_base64_to_logmel(features["audio_base64"])
            mse_score, maha = self._score_spectrogram(loaded, spec_raw)

            self._record()
            return PredictInlineResponse(**build_inline_result(
                model_id=MODEL_ID, machine=machine, machine_id=machine_id, snr=snr,
                mse_score=mse_score, maha_score=maha, threshold=threshold_used,
            ))
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    # ── predict_batch ────────────────────────────────────────────────────────

    def predict_batch(
        self, *, data_path: str, mlflow_run_id: str = "", threshold: float | None = None,
    ) -> PredictBatchResponse:
        predictions: list[dict] = []
        with local_file_path(data_path) as local_zip:
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(local_zip, "r") as zf:
                    zf.extractall(tmp_dir)

                manifest_path = next(Path(tmp_dir).rglob("manifest.csv"), None)
                if manifest_path is None:
                    raise ValueError("El ZIP debe contener un manifest.csv (filename,machine,machine_id,snr)")

                with open(manifest_path, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))

                for row in rows:
                    filename = row["filename"]
                    machine, machine_id, snr = row["machine"], row["machine_id"], row["snr"]
                    temp_dir = None
                    try:
                        wav_path = manifest_path.parent / filename
                        if not wav_path.exists():
                            raise InvalidAudioError(f"Referenced WAV not found in ZIP: {filename}")
                        loaded, temp_dir = self._resolve_combination(machine, machine_id, snr, mlflow_run_id)
                        threshold_used = self._prediction_threshold(loaded, threshold)
                        spec_raw = wav_to_logmel(str(wav_path))
                        mse_score, maha = self._score_spectrogram(loaded, spec_raw)
                        result = build_inline_result(
                            model_id=MODEL_ID, machine=machine, machine_id=machine_id, snr=snr,
                            mse_score=mse_score, maha_score=maha, threshold=threshold_used,
                        )
                        result["filename"] = filename
                        predictions.append(result)
                    except Exception as exc:
                        logger.warning("Error processing %s: %s", filename, exc)
                        predictions.append({"filename": filename, "error": str(exc)})
                    finally:
                        if temp_dir:
                            shutil.rmtree(temp_dir, ignore_errors=True)

        self._record()
        return PredictBatchResponse(model_id=MODEL_ID, predictions=predictions, output_path=None)

    # ── train ────────────────────────────────────────────────────────────────

    def train(self, *, data_path: str, mlflow_run_id: str = "") -> TrainResponse:
        """Fine-tune (or train from scratch) every (machine, machine_id, snr) combination
        found under data_path's {snr}_{machine}/{machine_id}/{normal,abnormal}/*.wav layout.

        'normal' is required per combination trained; 'abnormal' is optional (if present,
        full auc/fnr/fpr/recall/threshold metrics are computed for that combination — see
        train_dto.py).
        """
        tracker = BaseMLflowTracker(mlflow_run_id) if mlflow_run_id else None
        per_combination: list[CombinationTrainMetrics] = []
        upload_warning: str | None = None

        with local_file_path(data_path) as local_zip:
            extract_dir = Path(tempfile.mkdtemp(prefix="ml41_train_"))
            mlflow_upload_dir = Path(tempfile.mkdtemp(prefix="ml41_mlflow_")) if tracker else None
            try:
                with zipfile.ZipFile(local_zip, "r") as zf:
                    zf.extractall(extract_dir)

                combos_found = self._discover_combinations(extract_dir)
                if not combos_found:
                    raise ValueError(
                        "No se encontraron combinaciones válidas. Estructura esperada: "
                        "{snr}_{machine}/{machine_id}/normal/*.wav"
                    )

                for machine, machine_id, snr, combo_root in combos_found:
                    metrics = self._train_one_combination(
                        machine, machine_id, snr, combo_root, tracker, mlflow_upload_dir,
                    )
                    per_combination.append(metrics)

                if tracker and mlflow_upload_dir:
                    try:
                        tracker.upload_artifacts(str(mlflow_upload_dir), artifact_path="model")
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        logger.error("MLflow artifact upload failed: %s", exc)
                        upload_warning = f"Local save OK, MLflow upload failed: {exc}"
            finally:
                shutil.rmtree(extract_dir, ignore_errors=True)
                if mlflow_upload_dir:
                    shutil.rmtree(mlflow_upload_dir, ignore_errors=True)

        # Invalidate old in-memory models without re-downloading fixed S3 artifacts
        # over the checkpoints that were just trained and saved locally.
        self._cache = CheckpointCache()
        self._device = self._device or _safe_device()
        self._ready = artifacts_available()
        return TrainResponse(
            detail=f"Entrenamiento completado para {len(per_combination)} combinación(es)",
            per_combination=per_combination,
            upload_warning=upload_warning,
        )

    @staticmethod
    def _discover_combinations(root: Path) -> list[tuple[str, str, str, Path]]:
        """Find {snr}_{machine}/{machine_id}/normal/*.wav under root."""
        found = []
        for snr_machine_dir in root.rglob("*"):
            if not snr_machine_dir.is_dir():
                continue
            name = snr_machine_dir.name
            matched_machine = next((m for m in MACHINES if name.endswith(f"_{m}")), None)
            if matched_machine is None:
                continue
            snr_part = name[: -(len(matched_machine) + 1)]
            if snr_part not in SNRS:
                continue
            for machine_id_dir in snr_machine_dir.iterdir():
                if machine_id_dir.name in MACHINE_IDS and (machine_id_dir / "normal").is_dir():
                    found.append((matched_machine, machine_id_dir.name, snr_part, machine_id_dir))
        return found

    def _train_one_combination(
        self, machine: str, machine_id: str, snr: str, combo_root: Path,
        tracker: BaseMLflowTracker | None, mlflow_upload_dir: Path | None,
    ) -> CombinationTrainMetrics:
        normal_wavs = sorted((combo_root / "normal").glob("*.wav"))
        if not normal_wavs:
            raise ValueError(f"No .wav files under {combo_root / 'normal'}")

        specs = np.stack([wav_to_logmel(str(p)) for p in normal_wavs], axis=0)  # (N,1,mels,frames)

        existing_dir = _store.local_dir / combination_dir(machine, machine_id, snr)
        existing_ckpt = existing_dir / CHECKPOINT_FILENAME
        device = self._device or _safe_device()

        # Split BEFORE computing normalization stats (matches the original pipeline: stats
        # are always fit on the train split only, applied to val — no leakage).
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(specs))
        split_idx = max(1, int(len(perm) * 0.8))
        train_idx = perm[:split_idx]
        val_idx = perm[split_idx:] if len(perm) > split_idx else perm[:1]
        train_raw, val_raw = specs[train_idx], specs[val_idx]

        if existing_ckpt.exists():
            checkpoint = torch.load(existing_ckpt, map_location=device, weights_only=False)
            norm_mean, norm_std = float(checkpoint["norm_mean"]), float(checkpoint["norm_std"])
            model = build_model(machine, device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Fine-tuning existing checkpoint for %s/%s/%s", machine, machine_id, snr)
        else:
            norm_mean, norm_std = float(train_raw.mean()), float(train_raw.std())
            model = build_model(machine, device)
            logger.info("Training %s/%s/%s from scratch (no existing checkpoint)", machine, machine_id, snr)

        train_data = normalize_logmel(train_raw, norm_mean, norm_std)
        val_data = normalize_logmel(val_raw, norm_mean, norm_std)

        hyperparams = TRAINING_HYPERPARAMS_BY_MACHINE[machine]
        if tracker:
            tracker.log_params({f"{machine}_{machine_id}_{snr}_{k}": v for k, v in hyperparams.items()})

        best_val_loss, _history = fine_tune(model, train_data, val_data, device, hyperparams)

        # Recalibrate Mahalanobis stats on the (fine-tuned) normal training embeddings.
        model.eval()
        embeddings = []
        for i in range(len(train_data)):
            spec = torch.from_numpy(train_data[i:i + 1]).float()
            embeddings.append(get_cls_embedding_single(model, spec, device))
        embeddings = np.stack(embeddings, axis=0)
        pca_components_n = ARCHITECTURE_BY_MACHINE[machine]["pca_components"]
        maha_mean, maha_inv_cov, maha_pca = fit_mahalanobis(embeddings, n_pca=pca_components_n)

        metrics_kwargs = dict(
            machine=machine, machine_id=machine_id, snr=snr,
            best_val_loss=round(best_val_loss, 6), n_train=len(train_data), n_val=len(val_data),
        )

        abnormal_dir = combo_root / "abnormal"
        if abnormal_dir.is_dir() and any(abnormal_dir.glob("*.wav")):
            abnormal_wavs = sorted(abnormal_dir.glob("*.wav"))
            abn_specs = np.stack([wav_to_logmel(str(p)) for p in abnormal_wavs], axis=0)
            abn_specs_norm = normalize_logmel(abn_specs, norm_mean, norm_std)

            normal_mse, normal_maha, abnormal_mse, abnormal_maha = [], [], [], []
            for i in range(len(val_data)):
                spec = torch.from_numpy(val_data[i:i + 1]).float()
                mse, maha = run_inference_single(model, spec, device), mahalanobis_score(
                    get_cls_embedding_single(model, spec, device), maha_mean, maha_inv_cov, maha_pca)
                normal_mse.append(mse)
                normal_maha.append(maha)
            for i in range(len(abn_specs_norm)):
                spec = torch.from_numpy(abn_specs_norm[i:i + 1]).float()
                mse, maha = run_inference_single(model, spec, device), mahalanobis_score(
                    get_cls_embedding_single(model, spec, device), maha_mean, maha_inv_cov, maha_pca)
                abnormal_mse.append(mse)
                abnormal_maha.append(maha)

            eval_metrics = evaluate_against_abnormal(
                np.array(normal_maha), np.array(abnormal_maha),
                np.array(normal_mse), np.array(abnormal_mse),
            )
            metrics_kwargs.update(eval_metrics)
            metrics_kwargs["n_abnormal_samples"] = len(abnormal_wavs)
            if tracker:
                tracker.log_metrics({f"{machine}_{machine_id}_{snr}_{k}": v for k, v in eval_metrics.items()})

        if tracker:
            tracker.log_metrics({f"{machine}_{machine_id}_{snr}_best_val_loss": best_val_loss})

        # Save locally (always) via ArtifactStore's local_dir.
        target_dir = _store.local_dir / combination_dir(machine, machine_id, snr)
        target_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_out = {
            "model_state_dict": model.state_dict(),
            "norm_mean": norm_mean, "norm_std": norm_std,
            "machine": machine, "machine_id": machine_id, "snr": snr,
            # Explicit None means retrained but uncalibrated. Only legacy checkpoints
            # without this key may use the original benchmark thresholds.
            "threshold": metrics_kwargs.get("threshold"),
        }
        torch.save(checkpoint_out, target_dir / CHECKPOINT_FILENAME)
        np.savez(
            target_dir / MAHA_STATS_FILENAME,
            mean=maha_mean, inv_cov=maha_inv_cov,
            pca_mean=maha_pca[0] if maha_pca else np.zeros(0, dtype=np.float32),
            pca_v=maha_pca[1] if maha_pca else np.zeros((0, 0), dtype=np.float32),
            has_pca=np.array(maha_pca is not None),
        )

        if mlflow_upload_dir:
            combo_upload_dir = mlflow_upload_dir / machine / machine_id / snr
            combo_upload_dir.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_out, combo_upload_dir / CHECKPOINT_FILENAME)
            shutil.copy(target_dir / MAHA_STATS_FILENAME, combo_upload_dir / MAHA_STATS_FILENAME)

        return CombinationTrainMetrics(**metrics_kwargs)

    # ── stats ────────────────────────────────────────────────────────────────

    def stats(self, mlflow_run_id: str = "") -> StatsResponse:
        base = StatsResponse(
            model_name=MODEL_ID,
            version=VERSION,
            description=(
                "Detección de anomalías acústicas (Audio-MAE ViT-Tiny) en maquinaria de "
                "curado de embutidos: 48 combinaciones fan/pump/slider/valve × 4 unidades × "
                "3 niveles de ruido (dataset MIMII). Score de decisión: distancia de "
                "Mahalanobis en el espacio latente CLS proyectado por PCA."
            ),
            task_type="anomaly_detection_audio",
            framework="pytorch",
            inputs=[
                InputField(name="audio_base64", type="file", format=["wav"],
                           description="Grabación WAV de la máquina (base64 inline, ZIP+manifest.csv en batch)"),
                InputField(name="machine", type="str", description="fan | pump | slider | valve"),
                InputField(name="machine_id", type="str", description="id_00 | id_02 | id_04 | id_06"),
                InputField(name="snr", type="str", description="-6_dB | 0_dB | 6_dB"),
            ],
            outputs=[
                OutputField(name="mse_score", type="float", description="Error de reconstrucción (informativo, no determinista)"),
                OutputField(name="maha_score", type="float", description="Distancia de Mahalanobis (señal de decisión, determinista)"),
                OutputField(name="predicted_label", type="int", description="0=normal, 1=anómalo"),
                OutputField(name="threshold_used", type="float", description="Umbral aplicado para esta combinación"),
            ],
            metrics=dict(METRICS_REPORTED),
            runtime_stats=RuntimeStats(total_predictions=self._predict_count, avg_latency_ms=None),
        )
        if mlflow_run_id:
            try:
                tracker = BaseMLflowTracker(mlflow_run_id)
                base.metrics["mlflow"] = {
                    "params": tracker.get_params(),
                    "metrics": tracker.get_metrics(),
                }
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Could not fetch MLflow stats for run_id=%s: %s", mlflow_run_id, exc)
        return base
