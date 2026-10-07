"""Tests for ``ml13-wine-price-fluctuation-prediction``.

Endpoint tests run against the FakePlugin wiring from conftest.py. The preprocessing/training
unit tests exercise the real plugin modules on small deterministic synthetic series — they check
wiring and invariants, not correctness (golden cases live in inbox/a13/manifest.yaml and are
checked by the verification skill).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import InsufficientDataError
from app.plugins.ml13_wine_price_fluctuation_prediction import preprocessing, training
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import FEATURE_COLUMNS

PREFIX = "/models/ml13-wine-price-fluctuation-prediction"
MODEL_ID = "ml13-wine-price-fluctuation-prediction"


def _rows(n_weeks: int, start: date = date(2023, 1, 2), seed: int = 0) -> list[dict]:
    """Weekly rows (campaign/week/price_red) starting on an ISO Monday."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_weeks):
        d = start + timedelta(weeks=i)
        iso_year, iso_week, _ = d.isocalendar()
        campaign = f"{iso_year}/{iso_year + 1}" if iso_week >= 31 else f"{iso_year - 1}/{iso_year}"
        price = 40 + 3 * np.sin(i / 6) + 0.05 * i + rng.normal(0, 0.6)
        rows.append({"campaign": campaign, "week": iso_week, "price_red": round(float(price), 2)})
    return rows


INLINE_PAYLOAD = {"mode": "inline", "rows": _rows(20)}


# ── endpoint wiring (FakePlugin) ─────────────────────────────────────────────

def test_health(client):
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline(client):
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert 0.0 <= body["pred_proba_up"] <= 1.0
    assert body["alerta_subida"] in (0, 1)
    assert body["horizon_weeks"] == 4


def test_predict_inline_custom_threshold(client):
    resp = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "threshold": 0.9})
    assert resp.status_code == 200
    assert resp.json()["alerta_subida"] == 0


def test_predict_inline_bulletin_schema_accepted(client):
    rows = [{"bulletin": f"SEMANA {w}/2024", "price_red": 40.0 + w / 10} for w in range(1, 21)]
    assert client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows}).status_code == 200


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (20 weeks, Bollinger-20 warm-up) must fail validation."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": _rows(10)})
    assert resp.status_code == 422


def test_predict_inline_missing_price_column_rejected(client):
    rows = [{"campaign": "2023/2024", "week": w} for w in range(1, 21)]
    assert client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows}).status_code == 422


def test_predict_inline_missing_time_key_rejected(client):
    rows = [{"price_red": 40.0} for _ in range(20)]
    assert client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows}).status_code == 422


def test_predict_inline_insufficient_data_maps_to_422(client, fake_plugins):
    """InsufficientDataError (fewer than 20 valid weeks after cleaning) must map to 422."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientDataError("at least 20 are required")
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "20" in resp.json()["detail"]


def test_predict_batch(client):
    resp = client.post(f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/wine_prices.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == len(body["predictions"])
    assert body["n_predictions"] == sum(p["pred_proba_up"] is not None for p in body["predictions"])


def test_train(client):
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine_prices.csv"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_model_type"] in ("logreg", "xgboost")
    assert body["n_test_rows"] == 24
    assert 0.0 <= body["auc"] <= 1.0


# ── preprocessing (real module) ──────────────────────────────────────────────

def test_parse_bulletin_uses_campaign_end_year():
    assert preprocessing.parse_bulletin_to_campaign_week("SEMANA 50/2023") == ("2022/2023", 50)
    assert preprocessing.parse_bulletin_to_campaign_week(" semana 1/2024 ") == ("2023/2024", 1)
    with pytest.raises(ValueError):
        preprocessing.parse_bulletin_to_campaign_week("WEEK 1/2024")
    with pytest.raises(ValueError):
        preprocessing.parse_bulletin_to_campaign_week("SEMANA 54/2024")


def test_campaign_week_to_date_crosses_year_boundary():
    first_half = pd.Series({"campaign": "2023/2024", "week": 50})
    second_half = pd.Series({"campaign": "2023/2024", "week": 1})
    assert preprocessing.parse_campaign_date(first_half).date() == date(2023, 12, 11)
    assert preprocessing.parse_campaign_date(second_half).date() == date(2024, 1, 1)
    assert preprocessing.parse_campaign_date(pd.Series({"campaign": "bad", "week": 1})) is None


def test_price_column_priority():
    df = pd.DataFrame({"price_white": [1.0], "price_red": [2.0]})
    assert preprocessing.detect_price_column(df) == "price_red"
    with pytest.raises(ValueError):
        preprocessing.detect_price_column(pd.DataFrame({"precio": [1.0]}))


def test_inference_features_warmup_and_merge():
    df_raw = pd.DataFrame(_rows(30))
    df_norm, feats, price_col = preprocessing.prepare_inference_frame(df_raw)
    # Bollinger-20 is the dominant window: first valid row is the 20th week
    assert len(feats) == 30 - 19
    assert list(feats.columns[: len(FEATURE_COLUMNS) + 1]) == ["price"] + FEATURE_COLUMNS
    feats["pred_proba_up"] = 0.5
    out = preprocessing.merge_predictions(df_norm, feats, price_col)
    assert len(out) == 30  # every input row is kept (left merge)
    assert out["pred_proba_up"].isna().sum() == 19


def test_fewer_than_20_weeks_raises_domain_error():
    rows = _rows(20)
    rows[3]["price_red"] = None  # dropped by cleaning -> 19 valid weeks
    with pytest.raises(InsufficientDataError):
        preprocessing.prepare_inference_frame(pd.DataFrame(rows))


def test_target_definition():
    idx = pd.date_range("2023-01-02", periods=40, freq="7D")
    up = pd.Series(40 * 1.02 ** np.arange(40), index=idx)  # +2%/week -> next-4 mean ~ +5.1%
    df = preprocessing.generate_technical_features(up.to_frame("price"), target_window=4, return_threshold=0.025)
    assert df["target"].eq(1).all()
    slow = pd.Series(40 * 1.005 ** np.arange(40), index=idx)  # +0.5%/week -> next-4 mean ~ +1.3%
    df_slow = preprocessing.generate_technical_features(slow.to_frame("price"), target_window=4, return_threshold=0.025)
    assert df_slow["target"].eq(0).all()
    assert df.index[-1] == idx[-5]  # the last 4 weeks have no closed future window


# ── training (real module) ───────────────────────────────────────────────────

def test_adaptive_splits_ensure_both_classes():
    y = np.array([0] * 60 + [1, 0] * 20)
    for tr, va in training.adaptive_walk_forward_splits(len(y), y, n_splits=5, gap=4):
        assert tr[-1] + 4 < va[0]
        assert set(np.unique(y[va])) == {0, 1}


def test_smart_score_caps_stability():
    assert training.smart_score({"auc_mean": 0.8, "f1_mean": 0.5, "auc_std": 0.0}) == pytest.approx(0.75)
    assert training.smart_score({"auc_mean": 0.8, "f1_mean": 0.5, "auc_std": 0.30}) == pytest.approx(0.55)


def test_train_models_end_to_end_on_synthetic_series():
    df_price = preprocessing.clean_and_index_data(pd.DataFrame(_rows(160, seed=3)))
    result = training.train_models(df_price)
    assert result["model_type"] in ("logreg", "xgboost")
    assert result["n_test"] == 24
    assert set(result["test_metrics"]) == {"auc", "accuracy", "f1", "precision", "recall"}
    proba = result["model"].predict_proba(result["scaler"].transform(np.zeros((1, len(FEATURE_COLUMNS)))))
    assert proba.shape == (1, 2)
