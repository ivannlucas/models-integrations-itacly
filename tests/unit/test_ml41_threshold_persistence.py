"""Regression tests for model-specific calibration, using temporary real checkpoints.

Only the expensive neural-network training/audio decoding is replaced. The real
train orchestration, threshold calculation, torch serialization, local/MLflow
loaders and both prediction methods are exercised without touching user artifacts.
"""
import shutil
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.domain.services.exceptions import UnsupportedMachineConfigurationError
from app.application.use_cases.predict_model_use_case import PredictModelUseCase
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly import (
    mlflow_utils, model_loader, plugin,
)
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.thresholds import THRESHOLDS
from app.plugins.ml41_meat_curing_machinery_acoustic_anomaly.predict_dto import PredictBatchRequest

COMBINATION = ("fan", "id_00", "0_dB")
FEATURES = dict(machine="fan", machine_id="id_00", snr="0_dB", audio_base64="test-audio")


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    store = SimpleNamespace(local_dir=artifact_root, path=lambda name: artifact_root / name)
    monkeypatch.setattr(plugin, "_store", store)
    monkeypatch.setattr(model_loader, "_store", store)
    for module in (plugin, model_loader, mlflow_utils):
        monkeypatch.setattr(module, "build_model", lambda machine, device: torch.nn.Linear(2, 2).to(device))
        monkeypatch.setattr(module, "_safe_device", lambda: torch.device("cpu"))

    combo_dir = artifact_root / model_loader.combination_dir(*COMBINATION)
    combo_dir.mkdir(parents=True)
    checkpoint = {
        "model_state_dict": torch.nn.Linear(2, 2).state_dict(),
        "norm_mean": 0.0, "norm_std": 1.0,
    }
    torch.save(checkpoint, combo_dir / "best.pth")
    np.savez(combo_dir / "maha_stats.npz", mean=np.zeros(2), inv_cov=np.eye(2), has_pca=False)

    downloads = []
    monkeypatch.setattr(plugin, "ensure_artifacts_downloaded", lambda: downloads.append(True))
    monkeypatch.setattr(plugin, "wav_to_logmel", lambda path: np.full(
        (1, 2, 2), 10.0 if "abnormal" in Path(path).parts else 1.0, dtype=np.float32,
    ))
    monkeypatch.setattr(plugin, "audio_base64_to_logmel", lambda value: np.ones((1, 2, 2)))
    monkeypatch.setattr(plugin, "fine_tune", lambda *args: (0.1, {}))
    monkeypatch.setattr(plugin, "get_cls_embedding_single", lambda model, tensor, device:
                        np.repeat(float(tensor.flatten()[0]), 2))
    monkeypatch.setattr(plugin, "fit_mahalanobis", lambda *args, **kwargs: (np.zeros(2), np.eye(2), None))
    monkeypatch.setattr(plugin, "run_inference_single", lambda *args: 0.2)
    monkeypatch.setattr(plugin.Ml41MeatCuringMachineryAcousticAnomalyPlugin, "_score_spectrogram",
                        lambda *args: (0.2, 12.0))

    remote = tmp_path / "mlflow-model"

    class Tracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def log_params(self, values):
            pass

        def log_metrics(self, values):
            pass

        def upload_artifacts(self, path, artifact_path):
            assert artifact_path == "model"
            shutil.copytree(path, remote, dirs_exist_ok=True)

        def download_artifacts(self, destination, artifact_path):
            assert artifact_path == "model"
            return shutil.copytree(remote, Path(destination) / "model")

    monkeypatch.setattr(plugin, "BaseMLflowTracker", Tracker)
    monkeypatch.setattr(mlflow_utils, "BaseMLflowTracker", Tracker)
    instance = plugin.Ml41MeatCuringMachineryAcousticAnomalyPlugin()
    instance.load()
    downloads.clear()
    batch = tmp_path / "batch.zip"
    with zipfile.ZipFile(batch, "w") as archive:
        archive.writestr("manifest.csv", "filename,machine,machine_id,snr\na.wav,fan,id_00,0_dB\n")
        archive.writestr("a.wav", b"test audio")
    return SimpleNamespace(instance=instance, combo_dir=combo_dir, remote=remote,
                           downloads=downloads, batch=batch, checkpoint=checkpoint)


def training_zip(tmp_path, with_anomalies):
    path = tmp_path / "train.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(5):
            archive.writestr(f"0_dB_fan/id_00/normal/{i}.wav", b"normal")
        if with_anomalies:
            archive.writestr("0_dB_fan/id_00/abnormal/0.wav", b"abnormal")
    return path


@pytest.mark.parametrize("source", ["local", "mlflow"])
def test_train_persists_calibration_for_inline_and_batch_after_reload(runtime, tmp_path, source):
    instance = runtime.instance
    # Prime the cache with the original checkpoint and its static threshold.
    assert instance.predict_inline(features=FEATURES).threshold_used == THRESHOLDS[COMBINATION]
    result = instance.train(data_path=str(training_zip(tmp_path, True)), mlflow_run_id="run-1")
    learned = result.per_combination[0].threshold
    assert learned > 12.0 > THRESHOLDS[COMBINATION]
    assert runtime.downloads == []  # train must not replace its new checkpoint from S3
    assert instance.predict_inline(features=FEATURES).threshold_used == learned

    for checkpoint_path in (runtime.combo_dir / "best.pth", runtime.remote / "fan/id_00/0_dB/best.pth"):
        assert torch.load(checkpoint_path, weights_only=False)["threshold"] == learned

    # A separate plugin instance must recover calibration, not depend on global state.
    fresh = plugin.Ml41MeatCuringMachineryAcousticAnomalyPlugin()
    fresh.load()
    if source == "mlflow":
        # Deliberately diverge local calibration to detect use of the wrong model's threshold.
        checkpoint = torch.load(runtime.combo_dir / "best.pth", weights_only=False)
        checkpoint["threshold"] = 1.0
        torch.save(checkpoint, runtime.combo_dir / "best.pth")
    run_id = "run-1" if source == "mlflow" else ""
    inline = fresh.predict_inline(features=FEATURES, mlflow_run_id=run_id)
    batch = fresh.predict_batch(data_path=str(runtime.batch), mlflow_run_id=run_id).predictions[0]
    assert inline.threshold_used == batch["threshold_used"] == learned
    assert inline.predicted_label == batch["predicted_label"] == 0
    override = fresh.predict_inline(features=FEATURES, threshold=0.0, mlflow_run_id=run_id)
    assert override.threshold_used == 0.0
    assert override.predicted_label == 1
    assert fresh.predict_inline(features=FEATURES, mlflow_run_id=run_id).threshold_used == learned
    routed = PredictModelUseCase(fresh).execute(PredictBatchRequest(
        data_path=str(runtime.batch), mlflow_run_id=run_id, threshold=0.0,
    ))
    assert routed.predictions[0]["threshold_used"] == 0.0
    assert routed.predictions[0]["predicted_label"] == 1


@pytest.mark.parametrize("source", ["local", "mlflow"])
def test_normal_only_training_does_not_reuse_old_calibration(runtime, tmp_path, source):
    result = runtime.instance.train(data_path=str(training_zip(tmp_path, False)), mlflow_run_id="run-1")
    assert result.per_combination[0].threshold is None
    run_id = "run-1" if source == "mlflow" else ""
    fresh = plugin.Ml41MeatCuringMachineryAcousticAnomalyPlugin()
    fresh.load()
    with pytest.raises(UnsupportedMachineConfigurationError, match="no calibrated"):
        fresh.predict_inline(features=FEATURES, mlflow_run_id=run_id)
    batch = fresh.predict_batch(data_path=str(runtime.batch), mlflow_run_id=run_id)
    assert "no calibrated" in batch.predictions[0]["error"]
    assert "predicted_label" not in batch.predictions[0]
    assert fresh.predict_inline(features=FEATURES, threshold=20.0, mlflow_run_id=run_id).predicted_label == 0
    override_batch = fresh.predict_batch(data_path=str(runtime.batch), mlflow_run_id=run_id, threshold=20.0)
    assert override_batch.predictions[0]["threshold_used"] == 20.0
    assert override_batch.predictions[0]["predicted_label"] == 0


@pytest.mark.parametrize("source", ["local", "mlflow"])
@pytest.mark.parametrize("calibration", ["legacy", 0.0, 20.0, None])
def test_checkpoint_loader_compatibility(runtime, source, calibration):
    checkpoint = runtime.checkpoint.copy()
    if calibration != "legacy":
        checkpoint["threshold"] = calibration
    torch.save(checkpoint, runtime.combo_dir / "best.pth")
    expected = THRESHOLDS[COMBINATION] if calibration == "legacy" else calibration
    if source == "local":
        loaded = model_loader.load_checkpoint(*COMBINATION, torch.device("cpu"))
        assert loaded.threshold == expected
    else:
        shutil.copytree(runtime.combo_dir, runtime.remote / "fan/id_00/0_dB")
        combinations, temporary_dir = mlflow_utils.download_user_model_from_mlflow("run-1")
        try:
            assert combinations[COMBINATION]["threshold"] == expected
        finally:
            shutil.rmtree(temporary_dir)
