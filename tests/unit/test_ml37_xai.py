"""Real-code tests (no FakePlugin) for ml37's explainability layer — deliverable a39, absorbed
into this plugin (app/plugins/ml37_wine_press_predictive_maintenance/xai.py).

- Integrated Gradients against a finite-difference reference on a tiny Keras model, on whatever
  Keras backend this process has (torch in the server, often tensorflow under pytest).
- SHAP background sampling, loading and per-cycle wiring (xai_error, SHAP budget in batch).
- The background produced by a real (1-epoch) retrain.
- Golden cases of inbox/a39/manifest.yaml through the real plugin, skipped when the real
  artifacts or the delivered tensor are not available locally.
"""
# pylint: disable=protected-access  # exercises the plugin's private per-cycle helpers
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras  # noqa: E402  pylint: disable=wrong-import-position

from app.plugins.ml37_wine_press_predictive_maintenance import (  # noqa: E402  pylint: disable=wrong-import-position
    model_loader,
    plugin,
    training,
    xai,
)
from app.plugins.ml37_wine_press_predictive_maintenance.constants import (  # noqa: E402  pylint: disable=wrong-import-position
    F_SCHEMA,
    PHYSICAL_SENSORS,
    SHAP_BACKGROUND_FILENAME,
    TARGET_NAMES,
    TIMESTEPS,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_N_CLASSES = {"cooler": 3, "valve": 4, "pump": 3, "accumulator": 4}


def _tiny_model(timesteps: int = 2) -> "keras.Model":
    """4 softmax heads named like the real ones on a small dense trunk. Random biases: with the
    default zero biases the baseline (input 0) gives a uniform softmax, every class ties for the
    max and its gradient is undefined there (the real trained model has biases)."""
    keras.utils.set_random_seed(0)
    inp = keras.layers.Input(shape=(timesteps, F_SCHEMA), name="X")
    x = keras.layers.Flatten()(inp)
    x = keras.layers.Dense(6, activation="tanh", bias_initializer="random_normal")(x)
    outputs = [
        keras.layers.Dense(
            _N_CLASSES[t], activation="softmax", bias_initializer="random_normal", name=t,
        )(x)
        for t in TARGET_NAMES
    ]
    return keras.Model(inp, outputs)


def _finite_difference_ig(model, x: np.ndarray, steps: int) -> dict[str, np.ndarray]:
    """Reference IG: same Riemann sum, gradients by central differences of max softmax."""
    flat = x.reshape(-1).astype(np.float64)
    eps = 1e-3
    result = {t: np.zeros_like(flat) for t in TARGET_NAMES}
    for alpha in np.linspace(0.0, 1.0, steps + 1):
        point = alpha * flat
        perturbed = np.repeat(point[None, :], 2 * flat.size, axis=0)
        for i in range(flat.size):
            perturbed[2 * i, i] += eps
            perturbed[2 * i + 1, i] -= eps
        preds = model.predict(perturbed.reshape((-1,) + x.shape[1:]).astype(np.float32), verbose=0)
        for head, target in enumerate(TARGET_NAMES):
            score = np.asarray(preds[head]).max(axis=1)
            result[target] += (score[0::2] - score[1::2]) / (2 * eps)
    return {t: (flat * g / (steps + 1)).reshape(x.shape) for t, g in result.items()}


def test_integrated_gradients_matches_finite_difference_reference():
    """IG (shared forward pass, max over classes per point) equals the textbook Riemann sum."""
    model = _tiny_model()
    x = np.random.default_rng(1).normal(size=(1, 2, F_SCHEMA)).astype(np.float32)
    got = xai.integrated_gradients(model, x, steps=4)
    expected = _finite_difference_ig(model, x, steps=4)
    for target in TARGET_NAMES:
        assert got[target].shape == x.shape
        # atol: float32 central differences resolve the score to ~1e-7 / (2 * eps) ≈ 5e-5 per
        # gradient; the attributions themselves are ~1e-2.
        np.testing.assert_allclose(got[target], expected[target], rtol=2e-2, atol=3e-5)


def test_integrated_gradients_batch_equals_one_cycle_at_a_time():
    """Several cycles in one call give the same attributions as one call per cycle."""
    model = _tiny_model()
    x = np.random.default_rng(2).normal(size=(3, 2, F_SCHEMA)).astype(np.float32)
    together = xai.integrated_gradients(model, x, steps=5)
    for i in range(3):
        alone = xai.integrated_gradients(model, x[i:i + 1], steps=5)
        for target in TARGET_NAMES:
            np.testing.assert_allclose(together[target][i:i + 1], alone[target], rtol=1e-5,
                                       atol=1e-8)


def test_sensor_importance_ignores_virtual_and_reserved_slots():
    """Only the 14 physical sensors are reported: |attr| mean over cycles and timesteps."""
    attrs = np.zeros((2, 3, F_SCHEMA), dtype=np.float32)
    attrs[:, :, 0] = -2.0               # PS1
    attrs[0, :, 13] = 3.0               # VS1, one cycle only
    attrs[:, :, len(PHYSICAL_SENSORS):] = 99.0  # CE/CP/SE + RESERVED_*
    importance = xai.sensor_importance(attrs)
    assert list(importance) == list(PHYSICAL_SENSORS)
    assert importance["PS1"] == pytest.approx(2.0)
    assert importance["VS1"] == pytest.approx(1.5)
    assert importance["PS2"] == 0.0


def test_background_positions_replay_the_original_rng_sequence():
    """xai_analysis.py draws the 150 IG test cycles first, then the 80 background cycles."""
    rng = np.random.default_rng(42)
    rng.choice(331, size=150, replace=False)
    expected = rng.choice(1543, size=80, replace=False)
    np.testing.assert_array_equal(xai.background_positions(1543, 331), expected)


def test_background_positions_small_splits():
    """With fewer train cycles than 80 the whole train split is the background."""
    positions = xai.background_positions(12, 3)
    assert sorted(positions.tolist()) == list(range(12))


def test_build_shap_background_takes_unscaled_physical_sensors_of_train_cycles():
    """(n, 600, 14) float32, raw values of the chosen train cycles, extra slots dropped."""
    x = np.zeros((20, TIMESTEPS, 17), dtype=np.float64)
    x[:, :, :] = np.arange(20)[:, None, None]
    x[:, :, 14:] = -1.0  # virtual sensors of X.npy, never part of the background
    train_idx = np.array([3, 5, 7, 11, 13, 17])
    bg = xai.build_shap_background(x, train_idx, n_test=4)
    assert bg.shape == (6, TIMESTEPS, len(PHYSICAL_SENSORS))
    assert bg.dtype == np.float32
    assert sorted(bg[:, 0, 0].astype(int).tolist()) == train_idx.tolist()


def test_shap_background_is_scaled_with_the_bundle_scaler(tmp_path):
    """The stored background is raw; loading pads it to 40 slots and applies THIS scaler."""
    raw = np.full((2, TIMESTEPS, len(PHYSICAL_SENSORS)), 5.0, dtype=np.float32)
    path = tmp_path / SHAP_BACKGROUND_FILENAME
    np.save(path, raw)
    scaler = {"mean": [1.0] * F_SCHEMA, "scale": [2.0] * F_SCHEMA}
    bg = model_loader._load_shap_background(str(path), scaler)
    assert bg.shape == (2, TIMESTEPS, F_SCHEMA)
    np.testing.assert_allclose(bg[:, :, :14], 2.0)   # (5 - 1) / 2
    np.testing.assert_allclose(bg[:, :, 14:], -0.5)  # padding 0 scaled like the real input
    assert model_loader._load_shap_background(str(tmp_path / "missing.npy"), scaler) is None


def test_shap_background_with_wrong_shape_is_rejected(tmp_path):
    """A scaled (n, 600, 40) file from the old ml39 plugin must not be silently re-scaled."""
    path = tmp_path / SHAP_BACKGROUND_FILENAME
    np.save(path, np.zeros((2, TIMESTEPS, F_SCHEMA), dtype=np.float32))
    with pytest.raises(ValueError, match="sin escalar"):
        model_loader._load_shap_background(str(path), {"mean": [0.0] * 40, "scale": [1.0] * 40})


_IG = {t: {s: 0.1 for s in PHYSICAL_SENSORS} for t in TARGET_NAMES}


def test_explain_cycle_without_background_keeps_ig_and_reports_shap_error(monkeypatch):
    """include_shap on a bundle without shap_background: IG is returned, SHAP goes to xai_error."""
    monkeypatch.setattr(plugin.xai, "compute_ig_importance", lambda model, x: _IG)
    out = plugin._explain_cycle({"model": None, "shap_background": None}, np.zeros(1), True)
    assert out["ig_importance"] == _IG
    assert out["shap_importance"] is None
    assert SHAP_BACKGROUND_FILENAME in str(out["xai_error"])


def test_explain_cycle_failure_goes_to_xai_error(monkeypatch):
    """An XAI failure never discards the prediction: it becomes xai_error."""
    def boom(model, x):
        raise RuntimeError("gradiente no disponible")

    monkeypatch.setattr(plugin.xai, "compute_ig_importance", boom)
    out = plugin._explain_cycle({"model": None, "shap_background": None}, np.zeros(1), False)
    assert out["ig_importance"] is None
    assert "gradiente no disponible" in str(out["xai_error"])


def _batch_csv(tmp_path, n_cycles: int) -> str:
    rows = [
        {"cycle_id": c, "t": t, **{s: 1.0 for s in PHYSICAL_SENSORS}}
        for c in range(n_cycles) for t in (0.0, 0.1)
    ]
    path = tmp_path / "batch.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(name="batch_plugin")
def _batch_plugin(monkeypatch):
    """Plugin whose prediction is stubbed and whose _explain_cycle records its calls."""
    calls = []
    fields = {
        "predictions": {t: 0 for t in TARGET_NAMES}, "risk": {t: 0.0 for t in TARGET_NAMES},
        "confidence_level": {t: "green" for t in TARGET_NAMES},
        "probabilities": {t: {0: 1.0} for t in TARGET_NAMES}, "alerts": [], "rule_evidence": [],
    }
    monkeypatch.setattr(plugin, "_predict_cycle", lambda bundle, readings: (dict(fields), None))

    def explain(bundle, x_scaled, include_shap):
        _ = bundle, x_scaled
        calls.append(include_shap)
        return {"ig_importance": _IG, "shap_importance": _IG if include_shap else None,
                "xai_error": None}

    monkeypatch.setattr(plugin, "_explain_cycle", explain)
    instance = plugin.Ml37WinePressPredictiveMaintenancePlugin()
    instance._bundle = {"model": None}
    return instance, calls


def test_predict_batch_shap_only_for_the_first_n_samples_cycles(batch_plugin, tmp_path):
    """include_shap in batch is capped by n_samples; the other cycles still get IG."""
    instance, calls = batch_plugin
    resp = instance.predict_batch(
        data_path=_batch_csv(tmp_path, 4), include_xai=True, include_shap=True, n_samples=2,
    )
    assert calls == [True, True, False, False]
    assert [p.shap_importance is not None for p in resp.predictions] == [True, True, False, False]
    assert all(p.ig_importance == _IG for p in resp.predictions)


def test_predict_batch_explains_every_cycle_by_default(batch_plugin, tmp_path):
    """Batch default (include_xai=true): IG for every cycle, no SHAP."""
    instance, calls = batch_plugin
    resp = instance.predict_batch(data_path=_batch_csv(tmp_path, 3))
    assert calls == [False, False, False]
    assert all(p.ig_importance == _IG for p in resp.predictions)


def test_predict_batch_without_xai_skips_the_explanation(batch_plugin, tmp_path):
    """include_xai=false: no IG cost per cycle."""
    instance, calls = batch_plugin
    resp = instance.predict_batch(data_path=_batch_csv(tmp_path, 3), include_xai=False)
    assert not calls
    assert all(p.ig_importance is None for p in resp.predictions)


def _training_frame(n_per_class: int = 10) -> pd.DataFrame:
    """Cycles whose sensors are constant in time: sensor j of cycle i reads 100 + i + j/100."""
    rows = []
    for i in range(3 * n_per_class):
        cooler = (3, 20, 100)[i % 3]
        for t in (0.0, 1.0, 2.0):
            rows.append({
                "cycle_id": i, "t": t,
                **{s: 100.0 + i + j / 100 for j, s in enumerate(PHYSICAL_SENSORS)},
                "y_cooler": cooler, "y_valve": 100, "y_pump": 0, "y_accumulator": 130,
            })
    return pd.DataFrame(rows)


def test_retrain_returns_a_raw_background_from_its_own_train_split(monkeypatch):
    """A retrained model carries a SHAP background drawn from ITS train cycles, unscaled."""
    monkeypatch.setattr(training, "TRAIN_EPOCHS_MAX", 1)
    result = training.train_lstm_multitask(_training_frame())
    n_train = result.split_sizes["n_train"]
    bg = result.shap_background
    assert bg.shape == (n_train, TIMESTEPS, len(PHYSICAL_SENSORS))
    cycles = np.round(bg[:, :, 0] - 100.0).astype(int)  # raw values identify the cycle
    assert (cycles == cycles[:, :1]).all()               # constant in time, not scaled
    assert len(set(cycles[:, 0])) == n_train             # n_train distinct cycles
    np.testing.assert_allclose(bg[:, 0, 1] - bg[:, 0, 0], 0.01, atol=1e-4)


def test_train_uploads_the_shap_background(monkeypatch, tmp_path):
    """train() writes shap_background.npy next to the model in the MLflow run."""
    background = np.ones((3, TIMESTEPS, len(PHYSICAL_SENSORS)), dtype=np.float32)
    fake_model = MagicMock()
    fake_model.save.side_effect = lambda path: Path(path).write_bytes(b"keras")
    result = training.LstmTrainResult(
        model=fake_model, scaler={"mean": [0.0], "scale": [1.0]}, class_mappings={},
        metrics={"accuracy": {}, "balanced_accuracy": {}, "f1_macro": {}},
        split_sizes={"n_train": 3, "n_val": 1, "n_test": 1}, epochs_run=1,
        shap_background=background,
    )
    monkeypatch.setattr(plugin.training, "train_lstm_multitask", lambda df: result)
    monkeypatch.setattr(plugin, "require_train_tracker", lambda run_id: MagicMock())
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(plugin, "upload_model_to_mlflow",
                        lambda tracker, write, prefix: write(str(model_dir)))
    data = tmp_path / "train.csv"
    _training_frame(2).to_csv(data, index=False)

    instance = plugin.Ml37WinePressPredictiveMaintenancePlugin()
    instance._bundle = {"model": None}
    instance.train(data_path=str(data), mlflow_run_id="run-1")
    np.testing.assert_array_equal(np.load(model_dir / SHAP_BACKGROUND_FILENAME), background)


# ── golden cases against the real artifact ────────────────────────────────────

_ARTIFACT_DIR = _REPO_ROOT / "artifacts" / "ml37_wine_press_predictive_maintenance"
_RAW_TENSOR = _REPO_ROOT / "inbox" / "a37" / "codigo" / "data" / "processed" / "tensors" / "X.npy"

# inbox/a39/manifest.yaml golden_cases (original TensorFlow _integrated_gradients, steps=50).
_GOLDEN = {
    33: {
        "predictions": {"cooler": 3, "valve": 100, "pump": 0, "accumulator": 130},
        "ig_top3": {
            "cooler": {"PS6": 0.000318, "TS4": 0.000241, "TS1": 0.000224},
            "valve": {"PS6": 0.000153, "PS5": 0.00014, "TS4": 0.000138},
            "pump": {"FS1": 0.000153, "EPS1": 0.000095, "VS1": 0.00009},
            "accumulator": {"PS6": 0.000229, "TS4": 0.000171, "FS1": 0.000168},
        },
    },
    1842: {
        "predictions": {"cooler": 100, "valve": 90, "pump": 1, "accumulator": 100},
        "ig_top3": {
            "cooler": {"PS4": 0.003605, "TS1": 0.003385, "PS6": 0.002427},
            "valve": {"PS4": 0.00044, "TS1": 0.000301, "PS6": 0.000227},
            "pump": {"PS4": 0.000707, "TS1": 0.000516, "TS4": 0.000403},
            "accumulator": {"PS4": 0.000301, "TS1": 0.000189, "TS2": 0.000166},
        },
    },
}


@pytest.fixture(scope="module", name="real_plugin")
def _real_plugin():
    if not ((_ARTIFACT_DIR / "best_model.keras").is_file() and _RAW_TENSOR.is_file()):
        pytest.skip("real ml37 artifacts / a37 raw tensor not available locally")
    instance = plugin.Ml37WinePressPredictiveMaintenancePlugin()
    instance.load()
    return instance, np.load(_RAW_TENSOR, mmap_mode="r")


@pytest.mark.parametrize("cycle_id", sorted(_GOLDEN))
def test_real_plugin_reproduces_a39_golden_case(real_plugin, cycle_id):
    """The real plugin's inline response carries the a39 golden IG top-3 (1% relative)."""
    instance, x_raw = real_plugin
    cycle = x_raw[cycle_id - 1]
    readings = [
        {"t": t / 10.0, **{s: float(cycle[t, j]) for j, s in enumerate(PHYSICAL_SENSORS)}}
        for t in range(cycle.shape[0])
    ]
    resp = instance.predict_inline(features={"readings": readings})
    case = _GOLDEN[cycle_id]
    assert resp.predictions == case["predictions"]
    assert resp.xai_error is None
    for head, expected in case["ig_top3"].items():
        got = dict(sorted(resp.ig_importance[head].items(), key=lambda kv: -kv[1])[:3])
        assert list(got) == list(expected), f"cycle {cycle_id} head {head}: {got}"
        for sensor, value in expected.items():
            assert got[sensor] == pytest.approx(value, rel=0.01)
