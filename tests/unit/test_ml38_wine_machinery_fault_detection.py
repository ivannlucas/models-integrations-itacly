"""Tests for ``ml38-wine-machinery-fault-detection`` (a38, DNLS-VINO).

* Endpoint tests run against ``FakePlugin`` (conftest): they check the HTTP wiring — schemas,
  modes, status codes and domain-exception mapping — not the model.
* The pure functions (feature extraction, operating configurations, physics-guided loss) and the
  real ``train()`` (tiny synthetic table, MLflow stubbed) call the plugin's own code.
* The golden test runs the real plugin on the delivered artifacts and is skipped when
  ``artifacts/ml38_wine_machinery_fault_detection/`` is not available locally. The full golden
  verification against the server lives in outputs/a38/verification_report.md.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from app.plugins.ml38_wine_machinery_fault_detection import preprocessing, training
from app.plugins.ml38_wine_machinery_fault_detection.constants import (
    ARTIFACT_FILENAMES,
    ARTIFACT_FOLDER_NAME,
    FEATURE_NAMES,
    TRAIN_ELE_WINDOW_SAMPLES,
    TRAIN_VIB_WINDOW_SAMPLES,
)
from app.plugins.ml38_wine_machinery_fault_detection.exceptions import (
    FeatureContractError,
    UnsupportedOperatingConfigurationError,
)

MODEL_ID = "ml38-wine-machinery-fault-detection"
PREFIX = f"/models/{MODEL_ID}"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "inbox" / "a38" / "manifest.yaml"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / ARTIFACT_FOLDER_NAME

# Features of golden caso_001 (sample_id 609, rounded): only used for the HTTP wiring tests.
INLINE_FEATURES = {
    "vib_mean": -0.0009, "vib_std": 0.2721, "vib_max": 1.37, "vib_min": -1.32, "vib_rms": 0.2721,
    "vib_kurtosis": 0.46, "vib_skewness": 0.02, "vib_spectral_energy": 8.9e9,
    "vib_spectral_centroid": 61000.0, "vib_band_low": 0.02, "vib_band_mid": 0.02,
    "vib_band_high": 0.003, "vib_amp_1x": 1528.7, "vib_amp_2x": 900.0, "vib_ratio_2x_1x": 0.59,
    "ele_mean": 0.1, "ele_std": 14.0, "ele_max": 21.0, "ele_min": -21.0, "ele_rms": 14.0,
    "ele_kurtosis": -1.45, "ele_skewness": 0.001, "ele_spectral_energy": 1.9e13,
    "ele_spectral_centroid": 15000.0,
}
RAW_PAYLOAD = {
    "mode": "raw",
    "vibration_csv": "/tmp/vibration_example.csv",
    "current_csv": "/tmp/current_example.csv",
    "motor": "Motor-4",
    "speed": 70,
    "window_index": 0,
}


# ── HTTP wiring (FakePlugin) ──────────────────────────────────────────────────────────────


def test_health(client):
    """GET /health reports the model as loaded."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    """GET /stats returns the model name."""
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline(client):
    """Inline predict with the 24 features returns family + severity."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", **INLINE_FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["family"] == "alignment"
    assert body["severity"] in (0, 1, None)
    assert isinstance(body["family_probabilities"], dict)


def test_predict_inline_missing_feature_is_422(client):
    """An inline predict without one of the 24 features is rejected by the schema."""
    payload = {"mode": "inline", **INLINE_FEATURES}
    del payload["vib_band_low"]
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


def test_predict_raw(client):
    """A raw predict (CSV pair + motor/speed) is routed to predict_inline with model_key=raw."""
    resp = client.post(f"{PREFIX}/predict", json=RAW_PAYLOAD)
    assert resp.status_code == 200
    assert resp.json()["input_warnings"]


def test_predict_raw_unknown_motor_is_422(client):
    """A motor outside the NLN-EMP configurations is rejected by the schema."""
    resp = client.post(f"{PREFIX}/predict", json={**RAW_PAYLOAD, "motor": "Motor-7"})
    assert resp.status_code == 422


def test_unsupported_configuration_maps_to_422(client, fake_plugins):
    """UnsupportedOperatingConfigurationError (e.g. Motor-4 at 50 %) is a 422, not a 500."""
    fake_plugins[MODEL_ID].raise_on_inline = UnsupportedOperatingConfigurationError("Motor-4 50")
    resp = client.post(f"{PREFIX}/predict", json={**RAW_PAYLOAD, "speed": 50})
    assert resp.status_code == 422
    assert "Motor-4" in resp.text


def test_feature_contract_error_maps_to_422(client, fake_plugins):
    """A batch file without the feature columns is a 422."""
    fake_plugins[MODEL_ID].raise_on_batch = FeatureContractError("Faltan columnas")
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/x.csv"})
    assert resp.status_code == 422


def test_predict_batch(client):
    """Batch predict returns one prediction per row."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/x.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert [p["row"] for p in body["predictions"]] == [0, 1]


def test_train_returns_200_with_metrics(client):
    """POST /train returns the CV metrics of scripts.train."""
    resp = client.post(f"{PREFIX}/train",
                       json={"data_path": "/tmp/train.parquet", "mlflow_run_id": "run-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["family_cv_macro_f1_mean"], float)
    assert "bearing" in body["severity_cv"]
    assert body["mlflow_run_id"] == "run-test-ml38"


def test_train_without_mlflow_run_id_returns_422(client):
    """POST /train without mlflow_run_id (or with an empty one) is rejected with 422."""
    assert client.post(f"{PREFIX}/train", json={"data_path": "/tmp/t.csv"}).status_code == 422
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/t.csv", "mlflow_run_id": " "})
    assert resp.status_code == 422


# ── pure functions (real plugin code) ─────────────────────────────────────────────────────


@pytest.mark.parametrize("motor,speed", [("Motor-2", 50), ("Motor-2", 75), ("Motor-2", 100),
                                         ("Motor-4", 70)])
def test_documented_configurations_are_accepted(motor, speed):
    """The four NLN-EMP configurations of the training data are accepted."""
    preprocessing.check_operating_configuration(motor, speed)


@pytest.mark.parametrize("motor,speed", [("Motor-4", 100), ("Motor-2", 70), ("Motor-1", 50)])
def test_undocumented_configurations_are_rejected(motor, speed):
    """Any other motor/speed pair is a domain error (memoria §8.2)."""
    with pytest.raises(UnsupportedOperatingConfigurationError):
        preprocessing.check_operating_configuration(motor, speed)


def test_get_rpm_matches_delivered_code():
    """get_rpm keeps the delivered nominal values (1470 / 2950 rpm), not the memoria's."""
    assert preprocessing.get_rpm("Motor-2", 50) == pytest.approx(735.0)
    assert preprocessing.get_rpm("Motor-4", "70") == pytest.approx(2065.0)


def test_feature_vector_layout_and_values():
    """A pure 1X sine gives the expected vector: 24 values, energy at 1X, ratio ≈ 0."""
    rpm = preprocessing.get_rpm("Motor-2", 100)
    t = np.arange(80000) / 20000.0   # 4 s: 0.25 Hz bins, 24.5 Hz falls on a bin
    vib = np.sin(2 * np.pi * rpm / 60.0 * t)
    ele = 10 * np.sin(2 * np.pi * 50 * t)
    vec = preprocessing.build_feature_vector(vib, ele, "Motor-2", 100)
    feats = dict(zip(FEATURE_NAMES, vec))
    assert len(vec) == 24
    assert feats["vib_rms"] == pytest.approx(1 / math.sqrt(2), rel=1e-3)
    assert feats["vib_band_low"] == pytest.approx(1.0, abs=1e-6)
    assert feats["vib_ratio_2x_1x"] < 1e-3
    assert feats["ele_rms"] == pytest.approx(10 / math.sqrt(2), rel=1e-3)


def test_signal_length_warnings():
    """Only signals with a length other than the training window produce warnings."""
    assert not preprocessing.signal_length_warnings(TRAIN_VIB_WINDOW_SAMPLES,
                                                    TRAIN_ELE_WINDOW_SAMPLES)
    assert len(preprocessing.signal_length_warnings(4096, 4096)) == 2


def test_read_signal_window_index(tmp_path):
    """window_index selects the signal column after time; out of range is a contract error."""
    path = tmp_path / "s.csv"
    pd.DataFrame({"time": [0.0, 1.0], "0": [1.0, 2.0], "1": [3.0, 4.0]}).to_csv(path, index=False)
    assert preprocessing.read_signal(str(path), 1).tolist() == [3.0, 4.0]
    with pytest.raises(FeatureContractError):
        preprocessing.read_signal(str(path), 2)


def test_feature_table_requires_all_columns():
    """A batch table without the 24 features is a FeatureContractError (422)."""
    with pytest.raises(FeatureContractError):
        preprocessing.feature_frame_from_table(pd.DataFrame({"vib_mean": [0.0]}))


def test_physics_guided_loss_equals_original_formula():
    """mean(per-sample loss) == BCE(y, z) + λ·mean(relu(hb − σ(z))) of the original tape loop."""
    keras = pytest.importorskip("keras")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(7, 1)).astype("float32")
    y = rng.integers(0, 2, size=(7, 1)).astype("float32")
    hb = rng.uniform(size=7).astype("float32")
    per_sample = keras.ops.convert_to_numpy(
        training.physics_guided_loss(0.3)(np.concatenate([y, hb[:, None]], axis=1), logits)
    )
    prob = 1 / (1 + np.exp(-logits[:, 0]))
    bce = -(y[:, 0] * np.log(prob) + (1 - y[:, 0]) * np.log(1 - prob))
    expected = bce.mean() + 0.3 * np.maximum(hb - prob, 0).mean()
    assert per_sample.shape == (7,)
    assert float(per_sample.mean()) == pytest.approx(float(expected), rel=1e-5)


def test_prepare_training_frame_rejects_single_group():
    """GroupKFold needs at least 2 groups: a clear ValueError instead of a sklearn error."""
    df = pd.DataFrame({c: [1.0, 2.0] for c in FEATURE_NAMES})
    df["family"], df["severity"], df["group_id"] = ["healthy", "bearing"], [-1, 1], [0, 0]
    with pytest.raises(ValueError, match="grupos"):
        training.prepare_training_frame(df)


def _synthetic_training_table(n_groups_per_family: int = 3, windows: int = 4) -> pd.DataFrame:
    """Small separable table to exercise train() end to end (not a golden case)."""
    rng = np.random.default_rng(1)
    rows, group = [], 0
    for k, family in enumerate(["healthy", "bearing", "unbalance"]):
        for g in range(n_groups_per_family):
            severity = -1 if family == "healthy" else (1 if g == 0 else g + 1)
            for _ in range(windows):
                feats = rng.normal(loc=k * 3.0, size=len(FEATURE_NAMES))
                rows.append({**dict(zip(FEATURE_NAMES, feats)), "family": family,
                             "severity": severity, "group_id": group})
            group += 1
    return pd.DataFrame(rows)


class _RecordingTracker:
    """BaseMLflowTracker stand-in that keeps the uploaded bundle folder."""

    uploads: list[str] = []

    def __init__(self, run_id=""):
        self.run_id = run_id
        self.params, self.metrics = {}, {}

    def log_params(self, params):
        """Record params."""
        self.params.update(params)

    def log_metrics(self, metrics):
        """Record metrics."""
        self.metrics.update(metrics)

    def upload_artifacts(self, local_dir, artifact_path=""):
        """Copy the folder before the plugin deletes it."""
        import shutil  # pylint: disable=import-outside-toplevel
        dest = pathlib.Path(self.keep_dir) / artifact_path
        shutil.copytree(local_dir, dest)
        _RecordingTracker.uploads.append(str(dest))

    keep_dir = ""


def test_train_real_code_saves_reloadable_bundle_to_mlflow(monkeypatch, tmp_path):
    """train() runs the ported procedure, uploads 4 files and they reload as a usable bundle."""
    pytest.importorskip("keras")
    from app.domain.services import mlflow_artifacts  # pylint: disable=import-outside-toplevel
    from app.plugins.ml38_wine_machinery_fault_detection import (  # pylint: disable=import-outside-toplevel
        inference, mlflow_utils, plugin,
    )

    monkeypatch.setattr(training, "TRAIN_EPOCHS", 2)
    monkeypatch.setattr(training, "TRAIN_RF_N_ESTIMATORS", 10)
    _RecordingTracker.keep_dir = str(tmp_path / "mlflow")
    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", _RecordingTracker)
    data = tmp_path / "train.csv"
    table = _synthetic_training_table()
    table.to_csv(data, index=False)

    resp = plugin.Ml38WineMachineryFaultDetectionPlugin().train(
        data_path=str(data), mlflow_run_id="run-x"
    )
    assert resp.n_samples == len(table)
    assert resp.n_groups == 9
    assert set(resp.severity_cv) == {"bearing", "unbalance"}
    assert resp.mlflow_run_id == "run-x"

    uploaded = pathlib.Path(_RecordingTracker.uploads[-1])
    assert sorted(p.name for p in uploaded.iterdir()) == sorted(ARTIFACT_FILENAMES)
    bundle = mlflow_utils._load(str(uploaded))  # pylint: disable=protected-access
    preds = inference.predict_matrix(bundle, table[FEATURE_NAMES].to_numpy(float))
    assert {p["family"] for p in preds} <= {"healthy", "bearing", "unbalance"}
    assert all(p["severity"] is None for p in preds if p["family"] == "healthy")


def test_train_missing_columns_is_value_error(monkeypatch, tmp_path):
    """A training table without group_id answers 400 (ValueError) before training."""
    from app.domain.services import mlflow_artifacts  # pylint: disable=import-outside-toplevel
    from app.plugins.ml38_wine_machinery_fault_detection import plugin  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", _RecordingTracker)
    data = tmp_path / "train.csv"
    _synthetic_training_table().drop(columns=["group_id"]).to_csv(data, index=False)
    with pytest.raises(ValueError, match="group_id"):
        plugin.Ml38WineMachineryFaultDetectionPlugin().train(data_path=str(data),
                                                             mlflow_run_id="run-x")


# ── golden cases against the delivered artifacts (skipped without them) ─────────────────


@pytest.mark.skipif(
    not all((ARTIFACT_DIR / name).exists() for name in ARTIFACT_FILENAMES),
    reason="artefactos de ml38 no disponibles en local",
)
def test_golden_cases_with_delivered_artifacts():
    """Every golden case of the manifest: same family/severity, |Δ confidence| ≤ 0.05."""
    yaml = pytest.importorskip("yaml")
    from app.plugins.ml38_wine_machinery_fault_detection.plugin import (  # pylint: disable=import-outside-toplevel
        Ml38WineMachineryFaultDetectionPlugin,
    )

    cases = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))["golden_cases"]
    instance = Ml38WineMachineryFaultDetectionPlugin()
    instance.load()
    for case in cases:
        got = instance.predict_inline(features=case["input"])
        exp = case["expected"]
        assert got.family == exp["family"], case["id"]
        assert got.severity == exp["severity"], case["id"]
        assert got.severity_label == exp["severity_label"], case["id"]
        if exp["severity_confidence"] is None:
            assert got.severity_confidence is None, case["id"]
        else:
            assert abs(got.severity_confidence - exp["severity_confidence"]) <= 0.05, case["id"]
