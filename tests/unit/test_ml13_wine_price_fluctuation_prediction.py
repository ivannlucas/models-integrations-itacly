"""Tests for ``ml13-wine-price-fluctuation-prediction``.

Endpoint tests run against the FakePlugin wiring from conftest.py. The preprocessing/training
unit tests exercise the real plugin modules on small deterministic synthetic series — they check
wiring and invariants, not correctness (golden cases live in inbox/a13/manifest.yaml and are
checked by the verification skill).
"""
from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.domain.services.exceptions import (
    DataContractError,
    InsufficientDataError,
    UserModelUnavailableError,
)
from app.domain.services import mlflow_artifacts
from app.infrastructure.artifact_store import ArtifactStore
from app.plugins.ml13_wine_price_fluctuation_prediction import (
    model_loader,
    preprocessing,
    training,
)
from app.plugins.ml13_wine_price_fluctuation_prediction import plugin as plugin_mod
from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
    ARTIFACT_FOLDER_NAME,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_FILENAME,
    MODEL_CONFIG_FILENAME,
    MODEL_FILENAME,
    SCALER_FILENAME,
)
from app.plugins.ml13_wine_price_fluctuation_prediction.plugin import (
    Ml13WinePriceFluctuationPredictionPlugin,
)

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
    """The health endpoint reports the model as loaded."""
    body = client.get(f"{PREFIX}/health").json()
    assert body["status"] == "ok"
    assert body["model"] == MODEL_ID
    assert body["loaded"] is True


def test_stats(client):
    """The stats endpoint returns the model name."""
    assert client.get(f"{PREFIX}/stats").json()["model_name"] == MODEL_ID


def test_predict_inline(client):
    """An inline prediction returns probability, alert flag and horizon."""
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] == MODEL_ID
    assert 0.0 <= body["pred_proba_up"] <= 1.0
    assert body["alerta_subida"] in (0, 1)
    assert body["horizon_weeks"] == 4


def test_predict_inline_custom_threshold(client):
    """A custom threshold of 0.9 suppresses the rise alert."""
    resp = client.post(f"{PREFIX}/predict", json={**INLINE_PAYLOAD, "threshold": 0.9})
    assert resp.status_code == 200
    assert resp.json()["alerta_subida"] == 0


def test_predict_inline_bulletin_schema_accepted(client):
    """Rows using the bulletin schema ("SEMANA w/yyyy") are accepted."""
    rows = [{"bulletin": f"SEMANA {w}/2024", "price_red": 40.0 + w / 10} for w in range(1, 21)]
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows})
    assert resp.status_code == 200


def test_predict_inline_too_few_rows_rejected(client):
    """rows below the DTO min_length (20 weeks, Bollinger-20 warm-up) must fail validation."""
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": _rows(10)})
    assert resp.status_code == 422


def test_predict_inline_missing_price_column_rejected(client):
    """Rows without a price column are rejected with 422."""
    rows = [{"campaign": "2023/2024", "week": w} for w in range(1, 21)]
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows})
    assert resp.status_code == 422


def test_predict_inline_missing_time_key_rejected(client):
    """Rows without campaign/week or bulletin are rejected with 422."""
    rows = [{"price_red": 40.0} for _ in range(20)]
    resp = client.post(f"{PREFIX}/predict", json={"mode": "inline", "rows": rows})
    assert resp.status_code == 422


def test_predict_inline_insufficient_data_maps_to_422(client, fake_plugins):
    """InsufficientDataError (fewer than 20 valid weeks after cleaning) must map to 422."""
    fake_plugins[MODEL_ID].raise_on_inline = InsufficientDataError("at least 20 are required")
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "20" in resp.json()["detail"]


def test_predict_batch(client):
    """Batch prediction returns one row per input and counts the scored ones."""
    resp = client.post(
        f"{PREFIX}/predict", json={"mode": "batch", "data_path": "/tmp/wine_prices.csv"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == len(body["predictions"])
    assert body["n_predictions"] == sum(
        p["pred_proba_up"] is not None for p in body["predictions"]
    )


def test_train(client):
    """The train endpoint returns the best model type and test metrics."""
    resp = client.post(
        f"{PREFIX}/train",
        json={"data_path": "/tmp/wine_prices.csv", "mlflow_run_id": "run-1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_model_type"] in ("logreg", "xgboost")
    assert body["n_test_rows"] == 24
    assert 0.0 <= body["auc"] <= 1.0


# ── preprocessing (real module) ──────────────────────────────────────────────

def test_parse_bulletin_uses_campaign_end_year():
    """Bulletins map to the campaign ending in the bulletin year; bad formats raise."""
    assert preprocessing.parse_bulletin_to_campaign_week("SEMANA 50/2023") == ("2022/2023", 50)
    assert preprocessing.parse_bulletin_to_campaign_week(" semana 1/2024 ") == ("2023/2024", 1)
    with pytest.raises(ValueError):
        preprocessing.parse_bulletin_to_campaign_week("WEEK 1/2024")
    with pytest.raises(ValueError):
        preprocessing.parse_bulletin_to_campaign_week("SEMANA 54/2024")


def test_campaign_week_to_date_crosses_year_boundary():
    """Campaign weeks map to dates across the year boundary; bad campaigns give None."""
    first_half = pd.Series({"campaign": "2023/2024", "week": 50})
    second_half = pd.Series({"campaign": "2023/2024", "week": 1})
    assert preprocessing.parse_campaign_date(first_half).date() == date(2023, 12, 11)
    assert preprocessing.parse_campaign_date(second_half).date() == date(2024, 1, 1)
    assert preprocessing.parse_campaign_date(pd.Series({"campaign": "bad", "week": 1})) is None


def test_price_column_priority():
    """price_red takes priority over other price columns; no price column raises."""
    df = pd.DataFrame({"price_white": [1.0], "price_red": [2.0]})
    assert preprocessing.detect_price_column(df) == "price_red"
    with pytest.raises(ValueError):
        preprocessing.detect_price_column(pd.DataFrame({"precio": [1.0]}))


def test_inference_features_warmup_and_merge():
    """Inference features drop the 19 warm-up weeks and the merge keeps every input row."""
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
    """Fewer than 20 valid weeks after cleaning raise InsufficientDataError."""
    rows = _rows(20)
    rows[3]["price_red"] = None  # dropped by cleaning -> 19 valid weeks
    with pytest.raises(InsufficientDataError):
        preprocessing.prepare_inference_frame(pd.DataFrame(rows))


@pytest.mark.parametrize("bad_price", [0.0, -5.0, float("inf"), "n/d"])
@pytest.mark.parametrize("position", [3, 15, -1])
def test_invalid_price_is_rejected_not_silently_used(bad_price, position):
    """PR review: a 0 price gave logret=-inf that dropna() does not filter; a negative/inf price
    silently dropped the latest week. Any informed price must be finite and > 0."""
    rows = _rows(30)
    rows[position]["price_red"] = bad_price
    with pytest.raises(DataContractError, match="price_red"):
        preprocessing.prepare_inference_frame(pd.DataFrame(rows))


def test_missing_price_is_still_skipped_like_original_etl():
    """A missing price is dropped by cleaning, as in the original ETL."""
    rows = _rows(30)
    rows[10]["price_red"] = None
    _, feats, _ = preprocessing.prepare_inference_frame(pd.DataFrame(rows))
    assert np.isfinite(feats[FEATURE_COLUMNS].to_numpy()).all()
    assert len(feats) == 29 - 19


def test_invalid_price_maps_to_422(client, fake_plugins):
    """DataContractError on an invalid price maps to 422."""
    fake_plugins[MODEL_ID].raise_on_inline = DataContractError(
        "'price_red' debe ser un número finito y mayor que 0"
    )
    resp = client.post(f"{PREFIX}/predict", json=INLINE_PAYLOAD)
    assert resp.status_code == 422
    assert "price_red" in resp.json()["detail"]


def test_target_definition():
    """The target is 1 when the next-4-week mean return exceeds the threshold."""
    idx = pd.date_range("2023-01-02", periods=40, freq="7D")
    up = pd.Series(40 * 1.02 ** np.arange(40), index=idx)  # +2%/week -> next-4 mean ~ +5.1%
    df = preprocessing.generate_technical_features(
        up.to_frame("price"), target_window=4, return_threshold=0.025
    )
    assert df["target"].eq(1).all()
    slow = pd.Series(40 * 1.005 ** np.arange(40), index=idx)  # +0.5%/week -> next-4 mean ~ +1.3%
    df_slow = preprocessing.generate_technical_features(
        slow.to_frame("price"), target_window=4, return_threshold=0.025
    )
    assert df_slow["target"].eq(0).all()
    assert df.index[-1] == idx[-5]  # the last 4 weeks have no closed future window


# ── training (real module) ───────────────────────────────────────────────────

def test_adaptive_splits_ensure_both_classes():
    """Adaptive walk-forward splits keep the gap and both classes in validation."""
    y = np.array([0] * 60 + [1, 0] * 20)
    for tr, va in training.adaptive_walk_forward_splits(len(y), y, n_splits=5, gap=4):
        assert tr[-1] + 4 < va[0]
        assert set(np.unique(y[va])) == {0, 1}


def test_smart_score_caps_stability():
    """smart_score penalises the AUC std up to a cap."""
    stable = {"auc_mean": 0.8, "f1_mean": 0.5, "auc_std": 0.0}
    unstable = {"auc_mean": 0.8, "f1_mean": 0.5, "auc_std": 0.30}
    assert training.smart_score(stable) == pytest.approx(0.75)
    assert training.smart_score(unstable) == pytest.approx(0.55)


def test_train_models_end_to_end_on_synthetic_series():
    """train_models picks a model and reports test metrics on a synthetic series."""
    df_price = preprocessing.clean_and_index_data(pd.DataFrame(_rows(160, seed=3)))
    result = training.train_models(df_price)
    assert result["model_type"] in ("logreg", "xgboost")
    assert result["n_test"] == 24
    assert set(result["test_metrics"]) == {"auc", "accuracy", "f1", "precision", "recall"}
    proba = result["model"].predict_proba(
        result["scaler"].transform(np.zeros((1, len(FEATURE_COLUMNS))))
    )
    assert proba.shape == (1, 2)


# ── real plugin: retraining contract (MLflow only, no silent fallback) ───────

@pytest.fixture(name="real_plugin")
def fixture_real_plugin(monkeypatch):
    """The real plugin loaded with a bundle trained on the synthetic series.

    Fixed artifacts live in S3, so load() reads the bundle from a stubbed loader.
    """
    result = training.train_models(
        preprocessing.clean_and_index_data(pd.DataFrame(_rows(160, seed=3)))
    )
    bundle = {
        "model": result["model"], "scaler": result["scaler"],
        "schema": {"feature_columns": FEATURE_COLUMNS}, "model_type": result["model_type"],
    }
    plugin = Ml13WinePriceFluctuationPredictionPlugin()
    with monkeypatch.context() as patch:
        patch.setattr(model_loader, "load_artifacts", lambda: bundle)
        plugin.load()
    return plugin


@pytest.fixture(name="train_csv")
def fixture_train_csv(tmp_path):
    """CSV with 160 synthetic weeks for training."""
    path = tmp_path / "wine_prices.csv"
    pd.DataFrame(_rows(160, seed=3)).to_csv(path, index=False)
    return str(path)


def test_real_train_publishes_the_new_base_and_uploads_to_the_run(
    real_plugin, train_csv, monkeypatch, isolated_base_store
):
    """train() publishes v1 with all artifacts and uploads them to the MLflow run."""
    repo_dir = ArtifactStore(ARTIFACT_FOLDER_NAME).local_dir  # the plugin's store dir
    before = set(repo_dir.rglob("*")) if repo_dir.exists() else set()
    tracker = MagicMock()
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="run-1")
    assert resp.n_test_rows == 24 and resp.upload_warning is None
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id == "run-1"
    assert tracker.upload_artifacts.call_args.kwargs["artifact_path"] == "model"
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    for name in (MODEL_FILENAME, SCALER_FILENAME, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME):
        assert (v1 / name).exists()
    after = set(repo_dir.rglob("*")) if repo_dir.exists() else set()
    assert after == before  # nothing written under the repo's artifacts/


def test_real_train_without_run_publishes_the_base(real_plugin, train_csv):
    """train() without a run id still publishes the base."""
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="")
    assert resp.base_version.version == "v1" and resp.base_version.mlflow_run_id is None


def test_real_train_mlflow_failure_is_a_warning(real_plugin, train_csv, monkeypatch):
    """An MLflow upload failure becomes a warning, the base is still published."""
    tracker = MagicMock(upload_artifacts=MagicMock(side_effect=ConnectionError("mlflow caído")))
    monkeypatch.setattr(plugin_mod, "BaseMLflowTracker", MagicMock(return_value=tracker))
    resp = real_plugin.train(data_path=train_csv, mlflow_run_id="run-1")
    assert "mlflow caído" in resp.upload_warning and resp.base_version.version == "v1"


def test_declares_its_base_folder():
    """The plugin declares its artifact folder as BASE_ARTIFACT_FOLDER."""
    assert Ml13WinePriceFluctuationPredictionPlugin.BASE_ARTIFACT_FOLDER == ARTIFACT_FOLDER_NAME


def test_train_without_mlflow_run_id_is_accepted(client):
    """The train endpoint accepts a request without mlflow_run_id."""
    resp = client.post(f"{PREFIX}/train", json={"data_path": "/tmp/wine_prices.csv"})
    assert resp.status_code == 200


def test_real_predict_with_a_run_without_model_never_uses_the_base(real_plugin, monkeypatch):
    """A run without a model raises UserModelUnavailableError instead of using the base."""
    tracker = MagicMock(download_artifacts=MagicMock(return_value=""))
    monkeypatch.setattr(mlflow_artifacts, "BaseMLflowTracker", MagicMock(return_value=tracker))
    with pytest.raises(UserModelUnavailableError, match="run-vacio"):
        real_plugin.predict_inline(features={"rows": _rows(30)}, mlflow_run_id="run-vacio")


def test_real_batch_rows_carry_their_own_alert_label(real_plugin, train_csv):
    """Each batch row gets alerta_subida like predict_inline does; warm-up rows stay null."""
    rows = real_plugin.predict_batch(data_path=train_csv).predictions
    scored = [r for r in rows if r["pred_proba_up"] is not None]
    assert scored and all(r["alerta_subida"] == int(r["pred_proba_up"] >= 0.5) for r in scored)
    assert all(r["alerta_subida"] is None for r in rows if r["pred_proba_up"] is None)
