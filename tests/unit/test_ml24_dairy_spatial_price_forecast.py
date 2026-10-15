"""Endpoint tests for ``ml24-dairy-spatial-price-forecast`` (CatBoost h=1/3/6, wiring only).

Correctness against the real artifact (golden_cases, parity with inference_smoke.csv) is
validated by the `verification` skill (outputs/a24/verification_report.md), not here — these
tests use FakePlugin and only check request/response schemas and the domain-error -> 422 mapping.
The last block tests the real input-handling functions of preprocessing.py (pure pandas).
"""
import pandas as pd
import pytest

from app.plugins.ml24_dairy_spatial_price_forecast import preprocessing
from app.plugins.ml24_dairy_spatial_price_forecast.exceptions import (
    InsufficientPriceHistoryError,
    SeriesContractError,
    UnknownCcaaError,
)

PREFIX = "/models/ml24-dairy-spatial-price-forecast"
MODEL_ID = "ml24-dairy-spatial-price-forecast"


def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline_default(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"]) == 3
    assert {"y_pred", "period", "feature_period", "ccaa_code"} <= body["predictions"][0].keys()
    assert "main_source_lag_months" in body["data_freshness"]


def test_predict_inline_with_filters(client):
    payload = {"mode": "inline", "ccaa": ["CYL"], "horizons": [3], "as_of_period": "2019-12"}
    resp = client.post(f"{PREFIX}/predict", json=payload)
    assert resp.status_code == 200
    assert [row["horizon_months"] for row in resp.json()["predictions"]] == [3]


@pytest.mark.parametrize("payload", [
    {"mode": "inline", "horizons": [2]},              # only the official horizons 1/3/6
    {"mode": "inline", "as_of_period": "2019-13"},    # YYYY-MM
    {"mode": "inline", "as_of_period": "201912"},
])
def test_predict_inline_invalid_payload_rejected(client, payload):
    assert client.post(f"{PREFIX}/predict", json=payload).status_code == 422


@pytest.mark.parametrize("exc", [
    UnknownCcaaError("CCAA no reconocida(s) o no servida(s) por el modelo: ['ESP']"),
    InsufficientPriceHistoryError("Ninguna CCAA solicitada tiene los 13 meses consecutivos"),
    SeriesContractError("faltan los productos requeridos"),
])
def test_predict_inline_domain_errors_map_to_422(client, fake_plugins, exc):
    fake_plugins[MODEL_ID].raise_on_inline = exc
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline"})
    assert resp.status_code == 422
    assert str(exc) in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/series.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert body["n_predictions"] == len(body["predictions"])


def test_predict_batch_contract_error_maps_to_422(client, fake_plugins):
    fake_plugins[MODEL_ID].raise_on_batch = SeriesContractError("faltan columnas obligatorias ['es_reconstruido']")
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/series.csv"})
    assert resp.status_code == 422


def test_train_returns_200_with_metrics(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/series.csv", "mlflow_run_id": "test-run-id"})
    assert resp.status_code == 200
    body = resp.json()
    assert {"h1", "h3", "h6", "task_type", "training_time_s", "mlflow_run_id"} <= body.keys()
    assert {"train", "validation", "test", "ml_alpha"} <= body["h1"].keys()
    assert {"mae", "smape", "accuracy_pct"} <= body["h6"]["test"].keys()


def test_train_without_mlflow_run_id_returns_422(client):
    assert client.post(f"{PREFIX}/train", json={"data_path": "/tmp/series.csv"}).status_code == 422


def test_train_with_empty_mlflow_run_id_returns_422(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/series.csv", "mlflow_run_id": "  "})
    assert resp.status_code == 422


# ── preprocessing.py (real code, no artifacts) ──────────────────────────────────

def _series(months: int, ccaa=("CASTILLA Y LEON", "GALICIA")) -> pd.DataFrame:
    rows = []
    periods = pd.period_range("2018-01", periods=months, freq="M")
    for name in ccaa:
        for period in periods:
            for product in preprocessing.MODEL_PRODUCTS:
                rows.append({"anio": period.year, "mes": period.month, "ccaa": name, "producto": product,
                             "metrica": preprocessing.TARGET_METRIC, "valor": 0.7, "es_reconstruido": False})
    rows.append({"anio": 2018, "mes": 1, "ccaa": ccaa[0], "producto": "BATIDOS DE LECHE",
                 "metrica": preprocessing.TARGET_METRIC, "valor": 1.0, "es_reconstruido": False})
    return pd.DataFrame(rows)


def test_validate_series_keeps_only_model_products():
    out = preprocessing.validate_series(_series(14))
    assert set(out["producto"]) == set(preprocessing.MODEL_PRODUCTS)


def test_validate_series_requires_es_reconstruido():
    with pytest.raises(SeriesContractError, match="es_reconstruido"):
        preprocessing.validate_series(_series(14).drop(columns=["es_reconstruido"]))


def test_validate_series_requires_context_products():
    df = _series(14)
    with pytest.raises(SeriesContractError, match="LECHE ENTERA"):
        preprocessing.validate_series(df[df["producto"] != "LECHE ENTERA"])


def test_read_series_csv_filters_and_validates(tmp_path):
    path = tmp_path / "series.csv"
    _series(14).to_csv(path, index=False)
    out = preprocessing.read_series_csv(str(path))
    assert "BATIDOS DE LECHE" not in set(out["producto"])
    assert len(out) == 14 * 2 * len(preprocessing.MODEL_PRODUCTS)


def test_truncate_series_is_inclusive():
    out = preprocessing.truncate_series(_series(14), "2018-06")
    assert out[["anio", "mes"]].drop_duplicates().shape[0] == 6
    with pytest.raises(SeriesContractError):
        preprocessing.truncate_series(_series(14), "2010-01")


@pytest.mark.parametrize("requested,expected", [
    (None, None),
    (["cyl", "AND"], ["CASTILLA Y LEON", "ANDALUCIA"]),
    (["GALICIA"], ["GALICIA"]),
])
def test_resolve_ccaa(requested, expected):
    assert preprocessing.resolve_ccaa(requested) == expected


@pytest.mark.parametrize("bad", [["ESP"], ["T.ESPANA"], ["XXX"]])
def test_resolve_ccaa_rejects_unknown_and_spain(bad):
    with pytest.raises(UnknownCcaaError):
        preprocessing.resolve_ccaa(bad)


def test_insufficient_history_detects_short_or_broken_series():
    panel = pd.DataFrame({
        "ccaa": ["A"] * 13 + ["B"] * 12,
        "period_dt": list(pd.date_range("2019-01-01", periods=13, freq="MS"))
        + list(pd.date_range("2019-02-01", periods=12, freq="MS")),
        "price": [0.7] * 25,
    })
    assert list(preprocessing.ccaa_with_insufficient_history(panel, "price")) == ["B"]
    panel.loc[5, "price"] = None   # a gap inside A's last 13 months
    assert set(preprocessing.ccaa_with_insufficient_history(panel, "price")) == {"A", "B"}
