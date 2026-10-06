"""Endpoint + unit tests for ``ml3-wine-disease-pest-forecast``."""
import numpy as np
import pandas as pd
import pytest

PREFIX = "/models/ml3-wine-disease-pest-forecast"

_ROW = {
    "Fecha": "2021-06-11 00:00:00",
    "Temp_Amb_C": 20.0,
    "Hum_Rel_Pct": 76.0,
    "Lluvia_mm": 0.0,
    "Viento_kmh": 5.9,
    "CO2_ppm": 436.16,
    "VOC_ppb": 135.27,
    "Hum_Suelo_Pct": 32.9,
    "pH_Suelo": 6.88,
}

INLINE_PAYLOAD = {"mode": "inline", "rows": [_ROW] * 168}


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "ml3-wine-disease-pest-forecast"
    assert body["loaded"] is True


def test_stats(client):
    body = client.get(f"{PREFIX}/stats").json()
    assert body["model_name"] == "ml3-wine-disease-pest-forecast"
    assert body["task_type"]


def test_predict_inline_rejected(client):
    """Inline prediction removed for this model (product decision) — the /predict
    route only accepts PredictBatchRequest now, so "mode": "inline" fails Pydantic
    validation (422) instead of reaching the plugin. Batch is unaffected."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/vineyard.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == "ml3-wine-disease-pest-forecast"
    assert isinstance(body["predictions"], list)


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/train.csv", "mlflow_run_id": ""})
    assert resp.status_code == 200
    body = resp.json()
    assert body["detail"]
    assert isinstance(body["accuracy"], float)
    assert isinstance(body["f1_macro"], float)
    assert isinstance(body["n_windows_train"], int)


def test_validate_raw_columns_rejects_missing_columns():
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import validate_raw_columns

    df = pd.DataFrame([{"Fecha": "2021-06-11 00:00:00", "ID_Serie": 1}])
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        validate_raw_columns(df)


def test_validate_raw_columns_accepts_full_contract():
    from app.plugins.ml3_wine_disease_pest_forecast.constants import RAW_FIXED_COLUMNS
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import validate_raw_columns

    row = {"Fecha": "2021-06-11 00:00:00", "ID_Serie": 1}
    row.update({c: 0.0 for c in RAW_FIXED_COLUMNS})
    validate_raw_columns(pd.DataFrame([row]))  # must not raise


def test_apply_feature_engineering_generates_model_features():
    from app.plugins.ml3_wine_disease_pest_forecast.constants import MODEL_FEATURES
    from app.plugins.ml3_wine_disease_pest_forecast.feature_engineering import apply_feature_engineering

    rows = pd.DataFrame(
        [{"Fecha": pd.Timestamp("2021-06-11 00:00:00") + pd.Timedelta(hours=h),
          "Temp_Amb_C": 20.0, "Hum_Rel_Pct": 95.0, "Lluvia_mm": 2.0, "Viento_kmh": 5.0,
          "CO2_ppm": 430.0, "VOC_ppb": 130.0, "Hum_Suelo_Pct": 33.0, "pH_Suelo": 6.8,
          "ID_Serie": 1} for h in range(48)]
    )
    out = apply_feature_engineering(rows)
    for col in MODEL_FEATURES:
        assert col in out.columns
    # wetness accumulates while leaf is wet (rain > 0.1)
    assert out["Horas_Humedad_Foliar"].iloc[-1] == pytest.approx(48.0)
    # GDD accumulates over 2 days at base 10 with mean 20 -> 20 accumulated
    assert out["GDD_Acumulado"].iloc[-1] == pytest.approx(20.0)


def test_tail_or_pad_pads_short_series():
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import _tail_or_pad

    df = pd.DataFrame({"a": [1, 2, 3]})
    padded = _tail_or_pad(df, 5)
    assert len(padded) == 5
    assert list(padded["a"]) == [1, 1, 1, 2, 3]


def test_tail_or_pad_truncates_long_series():
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import _tail_or_pad

    df = pd.DataFrame({"a": list(range(10))})
    tail = _tail_or_pad(df, 3)
    assert len(tail) == 3
    assert list(tail["a"]) == [7, 8, 9]


def test_build_window_tensor_shape_and_dtype():
    from app.plugins.ml3_wine_disease_pest_forecast.constants import MODEL_FEATURES, WINDOW_SIZE
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import build_window_tensor

    class _Scaler:
        def transform(self, df):
            return df.values.astype(np.float32)

    df = pd.DataFrame(
        [{"Fecha": pd.Timestamp("2021-06-11 00:00:00") + pd.Timedelta(hours=h),
          **{c: 1.0 for c in MODEL_FEATURES}} for h in range(200)]
    )
    x_tensor, last_fecha, window_df = build_window_tensor(df, _Scaler())
    assert x_tensor.shape == (1, WINDOW_SIZE, len(MODEL_FEATURES))
    assert last_fecha == df["Fecha"].iloc[-1]
    assert len(window_df) == WINDOW_SIZE


def test_build_window_tensor_sorts_before_windowing_when_rows_arrive_out_of_order():
    """Regression: rows must be sorted by Fecha BEFORE taking the last WINDOW_SIZE, not
    after — otherwise an out-of-chronological-order input (e.g. a batch CSV exported
    without an explicit ORDER BY) silently windows on the wrong 168 hours."""
    from app.plugins.ml3_wine_disease_pest_forecast.constants import MODEL_FEATURES, WINDOW_SIZE
    from app.plugins.ml3_wine_disease_pest_forecast.preprocessing import build_window_tensor

    class _Scaler:
        def transform(self, df):
            return df.values.astype(np.float32)

    df = pd.DataFrame(
        [{"Fecha": pd.Timestamp("2021-06-11 00:00:00") + pd.Timedelta(hours=h),
          **{c: 1.0 for c in MODEL_FEATURES}} for h in range(200)]
    )
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    _, last_fecha, window_df = build_window_tensor(shuffled, _Scaler())

    assert last_fecha == df["Fecha"].iloc[-1]
    assert len(window_df) == WINDOW_SIZE
    assert list(window_df["Fecha"]) == list(df["Fecha"].iloc[-WINDOW_SIZE:])


def test_build_treatment_fallback_for_unknown_class():
    from app.plugins.ml3_wine_disease_pest_forecast.postprocessing import build_treatment

    assert build_treatment("CLASE_DESCONOCIDA") == "Aviso: N/A"
    assert "Químico" in build_treatment("OIDIO")


def test_stats_fetches_mlflow_metrics_when_run_id_given():
    """Regression: stats(mlflow_run_id=...) accepted the parameter but never used it —
    every retrain's /stats always reported the served model's static REPORTED_METRICS,
    no matter which trained run was asked about, same class of bug found and fixed in
    ml43_cereals_dnsl_anomaly_fault_detection/ml21_cereals_price_spatial (see plugin.py::stats)."""
    from unittest.mock import patch

    from app.plugins.ml3_wine_disease_pest_forecast.plugin import Ml3WineDiseasePestForecastPlugin

    plugin = Ml3WineDiseasePestForecastPlugin()
    with patch(
        "app.plugins.ml3_wine_disease_pest_forecast.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {"test_f1": 0.87}
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["test_f1"] == 0.87
    assert resp.metrics["mlflow"]["metrics"]["test_f1"] == 0.87


def test_stats_without_run_id_returns_static_metrics():
    from app.plugins.ml3_wine_disease_pest_forecast.constants import REPORTED_METRICS
    from app.plugins.ml3_wine_disease_pest_forecast.plugin import Ml3WineDiseasePestForecastPlugin

    plugin = Ml3WineDiseasePestForecastPlugin()
    resp = plugin.stats()

    assert resp.metrics == dict(REPORTED_METRICS)


def test_stats_overwrites_nested_classification_and_regression_keys():
    """Regression: the flattening loop only adds FLAT top-level keys (train()'s own
    accuracy/precision_macro/.../mae/mse/r2 naming) — it never touches
    REPORTED_METRICS's nested classification.*/regression.* sub-dicts, which is what
    the platform's Estadísticas screen (statsMetricsList, one dot-level of flattening)
    actually renders. Every retrain's /stats kept showing the served model's fixed
    classification.accuracy/regression.mae/etc. regardless of which run was asked
    about. mse->rmse must be an explicit sqrt, not a same-value alias."""
    from unittest.mock import patch

    from app.plugins.ml3_wine_disease_pest_forecast.constants import REPORTED_METRICS
    from app.plugins.ml3_wine_disease_pest_forecast.plugin import Ml3WineDiseasePestForecastPlugin

    plugin = Ml3WineDiseasePestForecastPlugin()
    with patch(
        "app.plugins.ml3_wine_disease_pest_forecast.plugin.BaseMLflowTracker"
    ) as mock_tracker_cls:
        mock_tracker_cls.return_value.get_metrics.return_value = {
            "accuracy": 0.81, "precision_macro": 0.79, "recall_macro": 0.77,
            "f1_macro": 0.78, "f1_weighted": 0.80, "mae": 0.05, "mse": 0.0064, "r2": 0.91,
        }
        mock_tracker_cls.return_value.get_params.return_value = {}
        resp = plugin.stats(mlflow_run_id="some-run-id")

    assert resp.metrics["classification"]["accuracy"] == 0.81
    assert resp.metrics["classification"]["precision_macro"] == 0.79
    assert resp.metrics["classification"]["recall_macro"] == 0.77
    assert resp.metrics["classification"]["f1_macro"] == 0.78
    assert resp.metrics["classification"]["f1_weighted"] == 0.80
    assert resp.metrics["regression"]["mae"] == 0.05
    assert resp.metrics["regression"]["rmse"] == 0.08  # sqrt(0.0064), not the raw mse
    assert resp.metrics["regression"]["r2"] == 0.91

    # The module-level REPORTED_METRICS constant must be untouched by the mutation above.
    assert REPORTED_METRICS["classification"]["accuracy"] == 0.7052
    assert REPORTED_METRICS["regression"]["rmse"] == 0.0936
