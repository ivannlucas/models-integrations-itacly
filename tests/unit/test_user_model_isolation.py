"""Concurrency regression tests: a user's retrained model must never leak to other requests.

There is one plugin instance per model, shared by every request, and FastAPI runs ``/predict``
in a thread pool. ml2, ml4, ml8, ml9, ml17, ml21, ml25, ml30, ml34, ml35 and ml46 used to swap the
user's model into ``self`` for the duration of a request and restore it in ``finally``. Two failure modes:

1. A concurrent request WITHOUT mlflow_run_id is served the other user's model while the swap
   is in place.
2. Two overlapping requests WITH different runs restore in the wrong order, so the base model
   is permanently replaced by a user's model until the pod restarts.

Only the step that *uses* the model is replaced by a probe that records which model each
thread saw and can pause on an event, which makes the interleavings deterministic. The
assertions check behaviour (which model a request without run is served), not internals.
"""
import tempfile
import threading
import zipfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

BASE = "BASE"
WAIT_S = 5


class _Stop(Exception):
    """Raised by the probe once the model has been observed: the rest of predict is irrelevant."""


class Probe:
    """Records which model each thread used and optionally pauses that thread inside the call."""

    def __init__(self):
        self.seen: dict[str, list[str]] = {}
        self.pauses: dict[str, tuple[threading.Event, threading.Event]] = {}

    def pause(self, thread_name):
        inside, release = threading.Event(), threading.Event()
        self.pauses[thread_name] = (inside, release)
        return inside, release

    def touch(self, model_name):
        name = threading.current_thread().name
        self.seen.setdefault(name, []).append(model_name)
        if name in self.pauses:
            inside, release = self.pauses[name]
            inside.set()
            assert release.wait(WAIT_S), f"{name}: never released"


def _start(name, fn):
    errors = []

    def target():
        try:
            fn()
        except _Stop:
            pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    return thread, errors


def _call(fn):
    try:
        fn()
    except _Stop:
        pass


# ── ml9 ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml9(monkeypatch, tmp_path):
    from app.plugins.ml9_cereals_infestation_sequence_classifier import plugin

    probe = Probe()
    monkeypatch.setattr(plugin.model_loader, "load_artifacts",
                        lambda: ({"name": BASE}, f"{BASE}-scaler", {"bundle": BASE}))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: ({"name": f"USER-{run_id}"}, f"scaler-{run_id}",
                                        {"bundle": run_id}, tempfile.mkdtemp()))
    payload = {"X_seq": np.zeros((1, 2, 2)), "y_seq": np.zeros(1), "window_meta": None,
               "feature_columns": []}
    monkeypatch.setattr(plugin.preprocessing, "missing_required_columns", lambda df: [])
    monkeypatch.setattr(plugin.preprocessing, "build_raw_dataframe", lambda rows: pd.DataFrame(rows))
    monkeypatch.setattr(plugin.preprocessing, "build_windows", lambda df, bundle, **kw: (payload, None, False))
    monkeypatch.setattr(plugin.preprocessing, "last_window_index", lambda payload: 0)

    def run_inference(checkpoint, scaler, x_seq):
        probe.touch(checkpoint["name"])
        raise _Stop

    monkeypatch.setattr(plugin.postprocessing, "run_inference", run_inference)
    instance = plugin.Ml9CerealsInfestationSequenceClassifierPlugin()
    instance.load()
    csv = tmp_path / "series.csv"
    pd.DataFrame({"sample_id": [1]}).to_csv(csv, index=False)
    calls = {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"rows": [{"sample_id": 1}]}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=str(csv), mlflow_run_id=run_id),
    }
    return probe, calls


# ── ml2 ───────────────────────────────────────────────────────────────────────

class _FakeModel:
    """Stands in for any served model: calling or predicting with it reports its name."""

    def __init__(self, name, probe):
        self.name, self.probe = name, probe

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self

    def __call__(self, *_args, **_kwargs):
        self.probe.touch(self.name)
        raise _Stop

    def predict(self, *_args, **_kwargs):
        self.probe.touch(self.name)
        raise _Stop


def _zip_with(tmp_path, filename):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(filename, b"not-decoded")
    return str(path)


def _csv_with(tmp_path, columns):
    path = tmp_path / "data.csv"
    pd.DataFrame({column: [1.0] for column in columns}).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def ml2(monkeypatch, tmp_path):
    from app.plugins.ml2_fungal_cnn_disease_detection import plugin

    probe = Probe()

    def bundle(name):
        return {"model": _FakeModel(name, probe), "model_id": name, "image_size": 8,
                "device": "cpu", "classes": ["a", "b"]}

    monkeypatch.setattr(plugin, "load_model_bundle", lambda: bundle(BASE))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (bundle(f"USER-{run_id}"), tempfile.mkdtemp()))
    monkeypatch.setattr(plugin, "image_base64_to_tensor", lambda *a, **k: MagicMock())
    monkeypatch.setattr(plugin, "image_path_to_tensor_and_image", lambda *a, **k: (MagicMock(), None))

    class Tracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def get_params(self):
            probe.touch(f"stats-{self.run_id}")  # pause point while stats() holds the run
            return {}

        def get_metrics(self):
            return {}

    monkeypatch.setattr(plugin, "BaseMLflowTracker", Tracker)
    instance = plugin.Ml2FungalCnnDiseaseDetectionPlugin()
    instance.load()
    images = tmp_path / "images.zip"
    with zipfile.ZipFile(images, "w") as archive:
        archive.writestr("leaf.jpg", b"not-decoded")
    calls = {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"image_base64": "x"}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=str(images), mlflow_run_id=run_id),
        "stats": lambda run_id: instance.stats(mlflow_run_id=run_id),
    }
    return probe, calls


# ── ml4 ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml4(monkeypatch, tmp_path):
    from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_model", lambda: (_FakeModel(BASE, probe), "cpu"))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), "cpu", tempfile.mkdtemp()))
    monkeypatch.setattr(plugin, "preprocess_image", lambda image_bytes: MagicMock())
    instance = plugin.Ml4LacticCnnThermalEarlyDiseaseDetectionPlugin()
    instance.load()
    images = _zip_with(tmp_path, "udder.jpg")
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"image_base64": "eA=="}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=images, mlflow_run_id=run_id),
    }


# ── ml8 ───────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml8(monkeypatch, tmp_path):
    from app.plugins.ml8_cereals_img_anomaly_detector import plugin

    probe = Probe()

    def bundle(name):
        return {"model": _FakeModel(name, probe), "model_id": name, "image_size": 8, "device": "cpu",
                "idx_to_class": {}, "idx_to_cereal": {}}

    monkeypatch.setattr(plugin, "load_model_bundle", lambda: bundle(BASE))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (bundle(f"USER-{run_id}"), tempfile.mkdtemp()))
    monkeypatch.setattr(plugin, "image_base64_to_tensor", lambda *a, **k: MagicMock())
    monkeypatch.setattr(plugin, "image_path_to_tensor_and_image", lambda *a, **k: (MagicMock(), None))
    instance = plugin.Ml8CerealsImgAnomalyDetectorPlugin()
    instance.load()
    images = _zip_with(tmp_path, "grain.jpg")
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"image_base64": "x"}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=images, mlflow_run_id=run_id),
    }


# ── ml17 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml17(monkeypatch, tmp_path):
    from app.plugins.ml17_meat_market_price_analysis import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_model", lambda: _FakeModel(BASE, probe))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), tempfile.mkdtemp()))
    monkeypatch.setattr(plugin.Ml17MeatMarketPriceAnalysisPlugin, "_build_frame",
                        lambda self, features: pd.DataFrame({"x": [1.0]}))
    instance = plugin.Ml17MeatMarketPriceAnalysisPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["date"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"date": "2024-01-01"},
                                                                 mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml21 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml21(monkeypatch, tmp_path):
    from app.plugins.ml21_cereals_price_spatial import plugin

    probe = Probe()
    cls = plugin.Ml21CerealsPriceSpatialPlugin
    monkeypatch.setattr(plugin, "load_model_bundle", lambda: ({"name": BASE}, {}, None))
    monkeypatch.setattr(cls, "_ensure_feature_importance", staticmethod(lambda metadata, models: None))
    monkeypatch.setattr(cls, "_compute_h3_reference_stats", staticmethod(lambda panel, metadata: (None, None)))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: ({"name": f"USER-{run_id}"}, {}, tempfile.mkdtemp()))

    def predict_single(self, raw_row, metadata, models):
        probe.touch(models["name"])
        raise _Stop

    monkeypatch.setattr(cls, "_predict_single", predict_single)
    instance = cls()
    instance.load()
    csv = _csv_with(tmp_path, ["provincia"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"provincia": "Burgos", "cereal_predominante": "trigo", "date": "2024-01"},
            mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml25 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml25(monkeypatch, tmp_path):
    from app.plugins.ml25_wine_sulphites import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_artifacts",
                        lambda: (_FakeModel(BASE, probe), _FakeModel(BASE, probe), {}))
    monkeypatch.setattr(plugin, "download_user_predictor_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), _FakeModel(f"USER-{run_id}", probe),
                                        {}, tempfile.mkdtemp()))
    monkeypatch.setattr(plugin, "map_request_to_wine_dict", lambda req: {})
    monkeypatch.setattr(plugin, "build_simulation_grid",
                        lambda wine, delta_max, **kw: (np.zeros(1), pd.DataFrame(), pd.DataFrame()))
    instance = plugin.Ml25WineSulphitesPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["pH"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(
            features={"delta_max": 40.0, "min_molecular": 0.6, "max_total": 200.0}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml30 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml30(monkeypatch, tmp_path):
    from app.plugins.ml30_meat_traceability_detection import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_artifacts", lambda: (f"{BASE}-pre", _FakeModel(BASE, probe), ["f"]))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (f"pre-{run_id}", _FakeModel(f"USER-{run_id}", probe), ["f"],
                                        tempfile.mkdtemp()))
    monkeypatch.setattr(plugin, "build_dataframe_from_features", lambda features: pd.DataFrame({"f": [1.0]}))
    monkeypatch.setattr(plugin, "build_dataframe_from_csv", lambda path: pd.DataFrame({"f": [1.0]}))
    monkeypatch.setattr(plugin, "enforce_data_contract", lambda *a, **k: [])

    def run_inference(preprocessor, mlp, df):
        mlp.probe.touch(mlp.name)
        raise _Stop

    monkeypatch.setattr(plugin, "run_inference", run_inference)
    instance = plugin.Ml30MeatTraceabilityDetectionPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["f"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"f": 1.0}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml34 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml34(monkeypatch, tmp_path):
    from app.plugins.ml34_dairy_pasteurization_energy_ga import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_artifacts", lambda: (_FakeModel(BASE, probe), "sx", "sy", {}))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), "sx", "sy", {}, tempfile.mkdtemp()))

    def predict_scenario(model, *args, **kwargs):
        model.probe.touch(model.name)
        raise _Stop

    monkeypatch.setattr(plugin, "predict_scenario", predict_scenario)
    instance = plugin.Ml34DairyPasteurizationEnergyGaPlugin()
    instance.load()
    features = {"T_in_leche": 4.0, "F_flow": 1000.0, "T_servicio": 80.0, "t_ciclo": 10.0, "Delta_P": 1.0}
    csv = _csv_with(tmp_path, list(plugin.FEATURES))
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features=features, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml35 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml35(monkeypatch, tmp_path):
    from app.plugins.ml35_dairy_ann_cleaning_cost import plugin

    probe = Probe()
    monkeypatch.setattr(plugin, "load_artifacts", lambda: (_FakeModel(BASE, probe), "sx", "sy"))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), "sx", "sy", tempfile.mkdtemp()))
    keys = ("temp_entrada_leche", "temp_ambiente", "temp_setpoint_leche", "temp_proceso_leche",
            "temp_agua_servicio", "flujo_leche_lh", "horas_desde_limpieza", "presion_diferencial_bar")
    monkeypatch.setattr(plugin, "_resolve_features", lambda features: {k: 1000.0 for k in keys})
    monkeypatch.setattr(plugin, "_compute_pu", lambda temp, flujo: 1000.0)
    monkeypatch.setattr(plugin, "_build_df", lambda *args: None)

    def infer(model, scaler_x, scaler_y, df):
        model.probe.touch(model.name)
        raise _Stop

    monkeypatch.setattr(plugin, "_infer", infer)
    instance = plugin.Ml35DairyAnnCleaningCostPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["temp_entrada_leche"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml46 ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def ml46(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from app.plugins.ml46_dairy_fouling_clog_detection import plugin

    probe = Probe()
    cfg = SimpleNamespace(seq_len=1, stride=1)
    artifacts = SimpleNamespace(predicate_thresholds={})
    monkeypatch.setattr(plugin.model_loader, "load_artifacts",
                        lambda: (_FakeModel(BASE, probe), cfg, artifacts, {}, {}))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), cfg, artifacts, {},
                                        tempfile.mkdtemp()))
    monkeypatch.setattr(plugin.preprocessing, "build_raw_dataframe", lambda rows: pd.DataFrame(rows))
    monkeypatch.setattr(plugin.preprocessing, "prepare_sequences", lambda df, cfg, art: (None, None, None))
    monkeypatch.setattr(plugin.preprocessing, "last_window_only", lambda seq, idx, seq_len: (0, 0))

    def run_inference(model, *args, **kwargs):
        model.probe.touch(model.name)
        raise _Stop

    monkeypatch.setattr(plugin.postprocessing, "run_inference", run_inference)
    monkeypatch.setattr(plugin.postprocessing, "run_inference_single_window", run_inference)
    instance = plugin.Ml46DairyFoulingClogDetectionPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["asset_id"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"rows": [{"asset_id": 1}]},
                                                                 mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


# ── ml13 / ml14 / ml18 / ml26 (integrated from main) ───────────────────────────────

@pytest.fixture
def ml13(monkeypatch, tmp_path):
    from app.plugins.ml13_wine_price_fluctuation_prediction import plugin

    probe = Probe()
    monkeypatch.setattr(plugin.model_loader, "load_artifacts",
                        lambda: {"model": _FakeModel(BASE, probe), "model_type": "logreg"})
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: ({"model": _FakeModel(f"USER-{run_id}", probe), "model_type": "logreg"},
                                        tempfile.mkdtemp()))

    def score(_df, bundle):
        bundle["model"].probe.touch(bundle["model"].name)
        raise _Stop

    cls = plugin.Ml13WinePriceFluctuationPredictionPlugin
    monkeypatch.setattr(cls, "_score", staticmethod(score))
    instance = cls()
    instance.load()
    csv = _csv_with(tmp_path, ["price_red"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"rows": []}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


@pytest.fixture
def ml14(monkeypatch, tmp_path):
    from app.plugins.ml14_wine_phyto_price_forecast import plugin

    probe = Probe()
    monkeypatch.setattr(plugin.model_loader, "load_artifact_bundle", lambda: {"model": _FakeModel(BASE, probe)})
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: ({"model": _FakeModel(f"USER-{run_id}", probe)}, tempfile.mkdtemp()))

    def run_inference(bundle, _rows):
        bundle["model"].probe.touch(bundle["model"].name)
        raise _Stop

    monkeypatch.setattr(plugin.preprocessing, "run_inference", run_inference)
    instance = plugin.Ml14WinePhytoPriceForecastPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["date"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"rows": []}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


@pytest.fixture
def ml18(monkeypatch, tmp_path):
    from app.plugins.ml18_meat_spatial_price_forecast import plugin

    probe = Probe()
    monkeypatch.setattr(plugin.model_loader, "load_artifact_bundle", lambda: {"model": _FakeModel(BASE, probe)})
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: ({"model": _FakeModel(f"USER-{run_id}", probe)}, tempfile.mkdtemp()))

    def run_inference(bundle, _rows):
        bundle["model"].probe.touch(bundle["model"].name)
        raise _Stop

    monkeypatch.setattr(plugin.inference, "run_inference", run_inference)
    instance = plugin.Ml18MeatSpatialPriceForecastPlugin()
    instance.load()
    csv = _csv_with(tmp_path, ["CCAA"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={"rows": []}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


@pytest.fixture
def ml26(monkeypatch, tmp_path):
    from app.plugins.ml26_wine_sulfite_gru_pso_forecast import plugin

    probe = Probe()
    monkeypatch.setattr(plugin.model_loader, "load_artifacts", lambda: (_FakeModel(BASE, probe), {}))
    monkeypatch.setattr(plugin, "download_user_model_from_mlflow",
                        lambda run_id: (_FakeModel(f"USER-{run_id}", probe), tempfile.mkdtemp()))
    monkeypatch.setattr(plugin.preprocessing, "frames_from_inline", lambda features: (None, None))
    monkeypatch.setattr(plugin.preprocessing, "frames_from_batch_csv", lambda df: (None, None))

    def lot_predictions(model, _lots, _readings):
        model.probe.touch(model.name)
        raise _Stop

    cls = plugin.Ml26WineSulfiteGruPsoForecastPlugin
    monkeypatch.setattr(cls, "_lot_predictions", staticmethod(lot_predictions))
    instance = cls()
    instance.load()
    csv = _csv_with(tmp_path, ["lot_id"])
    return probe, {
        "predict_inline": lambda run_id: instance.predict_inline(features={}, mlflow_run_id=run_id),
        "predict_batch": lambda run_id: instance.predict_batch(data_path=csv, mlflow_run_id=run_id),
    }


CASES = [("ml9", "predict_inline"), ("ml9", "predict_batch"),
         ("ml2", "predict_inline"), ("ml2", "predict_batch"), ("ml2", "stats")] + [
    (name, method)
    for name in ("ml4", "ml8", "ml13", "ml14", "ml17", "ml18", "ml21", "ml25", "ml26", "ml30", "ml34", "ml35", "ml46")
    for method in ("predict_inline", "predict_batch")
]


def _plugin(request, name):
    return request.getfixturevalue(name)


@pytest.mark.parametrize("plugin_name,method", CASES)
def test_request_without_run_never_sees_concurrent_user_model(request, plugin_name, method):
    probe, calls = _plugin(request, plugin_name)
    inside, release = probe.pause("A")
    thread_a, errors = _start("A", lambda: calls[method]("run-A"))
    assert inside.wait(WAIT_S), "request A never reached the model"

    _call(lambda: calls["predict_inline"](""))  # request B, no run, while A is mid-request

    release.set()
    thread_a.join(WAIT_S)
    assert not errors, errors
    assert probe.seen["MainThread"] == [BASE]


@pytest.mark.parametrize("plugin_name,method", CASES)
def test_overlapping_user_requests_leave_base_model_in_place(request, plugin_name, method):
    probe, calls = _plugin(request, plugin_name)
    a_inside, a_release = probe.pause("A")
    b_inside, b_release = probe.pause("B")

    thread_a, errors_a = _start("A", lambda: calls[method]("run-A"))
    assert a_inside.wait(WAIT_S)
    thread_b, errors_b = _start("B", lambda: calls[method]("run-B"))  # starts while A holds run-A
    assert b_inside.wait(WAIT_S)
    a_release.set()
    thread_a.join(WAIT_S)                                              # A finishes first...
    b_release.set()
    thread_b.join(WAIT_S)                                              # ...then B
    assert not errors_a and not errors_b, (errors_a, errors_b)

    _call(lambda: calls["predict_inline"](""))  # any later request without run
    assert probe.seen["MainThread"] == [BASE]
