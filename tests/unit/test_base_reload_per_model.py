"""Reload of a promoted base, per model (spec 2026-10-14 §7).

For every model that promotes its base: a real ModelContainer starts on v0 (a copy of the
delivered artifacts in a temp store), /train runs through the container, and the fresh plugin
instance must read every file from versions/v1/ (the inherited v0 files + the retrained bundle)
and predict with it. The per-plugin publish tests only check the files written; this checks
that each plugin's own loader reads them back. ml14 has its own test in test_ml14_training.py.
"""
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter

from app.infrastructure.artifact_store import ArtifactStore
from app.infrastructure.http.dependencies.container import ModelContainer
from app.registry import REGISTRY

REPO_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


class _RecordingStore(ArtifactStore):
    """ArtifactStore that remembers every path the plugin's loader resolves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read: list[Path] = []

    def path(self, filename):
        resolved = super().path(filename)
        self.read.append(resolved)
        return resolved


def _entry(plugin_cls):
    return next(e for e in REGISTRY if e.plugin_class is plugin_cls)


def _start_on_v0(monkeypatch, root: Path, folder: str, loader_module, plugin_cls, seed_v0=None):
    """Container serving v0 from root/<folder> (the repo's artifacts, or *seed_v0* when absent)."""
    v0 = root / folder
    if seed_v0 is not None:
        seed_v0(v0)
    elif (REPO_ARTIFACTS / folder).exists():
        shutil.copytree(REPO_ARTIFACTS / folder, v0,
                        ignore=shutil.ignore_patterns("versions", "current.json"))
    else:
        pytest.skip(f"artifacts/{folder} not available locally (they live in S3)")
    store = _RecordingStore(folder, root=root, use_s3=False)
    monkeypatch.setattr(loader_module, "_store", store)
    container = ModelContainer(plugin_cls())
    container.init()
    assert container.base_version()["version"] == "v0"
    return container, store


def _train_and_reload(container, store, root: Path, folder: str, data_path: str, **train_fields):
    """POST /train through the container; check the fresh instance reads only versions/v1/."""
    v0_files = {p.name for p in (root / folder).iterdir() if p.is_file()}
    served_before = container._plugin  # pylint: disable=protected-access
    store.read.clear()

    entry = _entry(type(served_before))
    resp = container.train_use_case.execute(
        entry.train_request_type(data_path=data_path, mlflow_run_id="", **train_fields)
    )

    v1 = root / folder / "versions" / "v1"
    assert resp.base_version.version == "v1"
    assert container.base_version()["version"] == "v1"
    assert container._plugin is not served_before  # pylint: disable=protected-access
    assert container.service.is_loaded()
    assert store.read, "the fresh instance did not resolve any artifact"
    assert all(p == v1 or v1 in p.parents for p in store.read), store.read
    assert v0_files <= {p.name for p in v1.iterdir()}, "v1 must inherit every v0 file"
    return resp


def _predict_batch(container, data_path: str):
    entry = _entry(type(container._plugin))  # pylint: disable=protected-access
    body = TypeAdapter(entry.predict_request_type).validate_python({"mode": "batch", "data_path": data_path})
    return container.predict_use_case.execute(body)


# ── ml13 ───────────────────────────────────────────────────────────────────────

def _ml13_rows(n_weeks: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_weeks):
        d = date(2023, 1, 2) + timedelta(weeks=i)
        iso_year, iso_week, _ = d.isocalendar()
        campaign = f"{iso_year}/{iso_year + 1}" if iso_week >= 31 else f"{iso_year - 1}/{iso_year}"
        price = 40 + 3 * np.sin(i / 6) + 0.05 * i + rng.normal(0, 0.6)
        rows.append({"campaign": campaign, "week": iso_week, "price_red": round(float(price), 2)})
    return rows


def test_ml13_reloads_the_promoted_base(monkeypatch, tmp_path, isolated_base_store):
    from app.plugins.ml13_wine_price_fluctuation_prediction import model_loader, preprocessing, training
    from app.plugins.ml13_wine_price_fluctuation_prediction.constants import (
        ARTIFACT_FOLDER_NAME, FEATURE_COLUMNS, FEATURE_SCHEMA_FILENAME, MODEL_CONFIG_FILENAME,
        MODEL_FILENAME, SCALER_FILENAME,
    )
    from app.plugins.ml13_wine_price_fluctuation_prediction.plugin import (
        Ml13WinePriceFluctuationPredictionPlugin,
    )

    def seed_v0(v0: Path):
        # ml13's delivered artifacts are only in S3: a v0 in the same format, trained on synthetic data.
        result = training.train_models(preprocessing.clean_and_index_data(pd.DataFrame(_ml13_rows(160, seed=1))))
        v0.mkdir(parents=True)
        joblib.dump(result["model"], v0 / MODEL_FILENAME)
        joblib.dump(result["scaler"], v0 / SCALER_FILENAME)
        (v0 / FEATURE_SCHEMA_FILENAME).write_text(json.dumps({"feature_columns": FEATURE_COLUMNS}))
        (v0 / MODEL_CONFIG_FILENAME).write_text(json.dumps({"model_type": result["model_type"]}))

    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml13WinePriceFluctuationPredictionPlugin, seed_v0=seed_v0)
    csv = tmp_path / "wine_prices.csv"
    pd.DataFrame(_ml13_rows(160, seed=3)).to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    assert _predict_batch(container, str(csv))


# ── ml15 ───────────────────────────────────────────────────────────────────────

def test_ml15_reloads_the_promoted_base_with_inherited_reference_csvs(monkeypatch, tmp_path, isolated_base_store):
    from app.plugins.ml15_wine_ipi_price_forecast import model_loader
    from app.plugins.ml15_wine_ipi_price_forecast.constants import (
        ANCHOR_COLUMN, ARTIFACT_FOLDER_NAME, FEATURE_COLUMNS, REFERENCE_FINANCIAL_FILENAME,
        REFERENCE_PRICES_FILENAME, TARGET_COLUMN,
    )
    from app.plugins.ml15_wine_ipi_price_forecast.plugin import Ml15WineIpiPriceForecastPlugin

    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml15WineIpiPriceForecastPlugin)
    rng = np.random.default_rng(0)
    n = 60
    ipi = 100 + np.cumsum(rng.normal(0.2, 0.5, n + 6))
    panel = pd.DataFrame({"date": pd.date_range("2018-01-01", periods=n, freq="MS").strftime("%Y-%m-%d")})
    for col in FEATURE_COLUMNS:
        panel[col] = rng.normal(100, 5, n)
    panel[ANCHOR_COLUMN] = ipi[:n]
    target = ipi[6:n + 6].copy()
    target[-6:] = np.nan
    panel[TARGET_COLUMN] = target
    csv = tmp_path / "panel.csv"
    panel.to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    # Without new reference CSVs the v1 serves the inherited ones, read from versions/v1/.
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    store.read.clear()
    assert model_loader.reference_data_paths() == (str(v1 / REFERENCE_PRICES_FILENAME),
                                                   str(v1 / REFERENCE_FINANCIAL_FILENAME))
    assert _predict_batch(container, str(csv))


# ── ml16 ───────────────────────────────────────────────────────────────────────

def test_ml16_reloads_the_promoted_base(monkeypatch, tmp_path, isolated_base_store):
    from app.plugins.ml16_meat_raw_material_price_alert import model_loader
    from app.plugins.ml16_meat_raw_material_price_alert.constants import (
        ARTIFACT_FOLDER_NAME, RAW_REQUIRED_COLUMNS, TARGETS,
    )
    from app.plugins.ml16_meat_raw_material_price_alert.plugin import Ml16MeatRawMaterialPriceAlertPlugin

    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml16MeatRawMaterialPriceAlertPlugin)
    rng = np.random.default_rng(0)
    n = 72
    dates = pd.date_range("2015-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"fecha": dates.strftime("%Y-%m-%d"), "month": dates.month})
    for col in RAW_REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = rng.normal(100, 10, n)
    for target in TARGETS:
        df[target] = np.tile([0, 0, 1], n // 3)
    csv = tmp_path / "ml16.csv"
    df.to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    assert _predict_batch(container, str(csv))


# ── ml17 ───────────────────────────────────────────────────────────────────────

def test_ml17_reloads_the_promoted_base(monkeypatch, tmp_path, isolated_base_store):
    from app.plugins.ml17_meat_market_price_analysis import model_loader
    from app.plugins.ml17_meat_market_price_analysis.constants import ARTIFACT_FOLDER_NAME, FEATURE_COLUMNS
    from app.plugins.ml17_meat_market_price_analysis.plugin import Ml17MeatMarketPriceAnalysisPlugin

    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml17MeatMarketPriceAnalysisPlugin)
    rng = np.random.default_rng(0)
    n = 36
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for col in FEATURE_COLUMNS:
        df[col] = rng.normal(150, 10, n)
    df["month_sin"] = np.sin(2 * np.pi * dates.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dates.month / 12)
    csv = tmp_path / "ml17.csv"
    df.to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    assert _predict_batch(container, str(csv))


# ── ml18 ───────────────────────────────────────────────────────────────────────

def test_ml18_reloads_the_promoted_base(monkeypatch, tmp_path, isolated_base_store):
    pytest.importorskip("tensorflow")
    from app.plugins.ml18_meat_spatial_price_forecast import model_loader, training
    from app.plugins.ml18_meat_spatial_price_forecast.constants import ARTIFACT_FOLDER_NAME
    from app.plugins.ml18_meat_spatial_price_forecast.plugin import Ml18MeatSpatialPriceForecastPlugin

    real_train_gru = training.train_gru
    monkeypatch.setattr(training, "train_gru", lambda path, **kw: real_train_gru(path, **{**kw, "epochs": 2}))
    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml18MeatSpatialPriceForecastPlugin)
    rng = np.random.default_rng(0)
    rows = []
    for ccaa in ("ANDALUCIA", "MURCIA", "BALEARES"):
        for i, fecha in enumerate(pd.date_range("2015-01-01", periods=40, freq="MS")):
            rows.append({
                "Fecha": fecha.strftime("%Y-%m-%d"), "CCAA": ccaa, "Producto": "CARNE POLLO",
                "Poblacion": 1e6, "RentaHogar": 25000 + 10 * i, "CONSUMO X CAPITA": 1.0 + rng.normal(0, 0.05),
                "PENETRACION (%)": 50.0, "PRECIO MEDIO KG": 4 + 0.02 * i + rng.normal(0, 0.05),
            })
    csv = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(csv, sep=";", index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    assert _predict_batch(container, str(csv))


# ── ml21 ───────────────────────────────────────────────────────────────────────

class _FastSearch:
    """GridSearchCV stand-in: fits the estimator once with few trees / iterations (same code
    path, seconds instead of minutes on the full delivered panel)."""

    def __init__(self, estimator, param_grid, **_kwargs):
        self.estimator = estimator
        self.best_params_ = {}
        self.best_score_ = 0.0

    def fit(self, X, y):
        params = self.estimator.get_params()
        if "n_estimators" in params:
            self.estimator.set_params(n_estimators=5)
        if "max_iter" in params:
            self.estimator.set_params(solver="liblinear", max_iter=50)
        self.best_estimator_ = self.estimator.fit(X, y)
        return self


def test_ml21_reloads_the_promoted_base_and_its_new_panel(monkeypatch, tmp_path, isolated_base_store):
    import sklearn.model_selection

    from app.plugins.ml21_cereals_price_spatial import model_loader
    from app.plugins.ml21_cereals_price_spatial.constants import ARTIFACT_FOLDER_NAME, DATASET_FILENAME
    from app.plugins.ml21_cereals_price_spatial.plugin import Ml21CerealsPriceSpatialPlugin

    monkeypatch.setattr(sklearn.model_selection, "GridSearchCV", _FastSearch)
    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml21CerealsPriceSpatialPlugin)
    # The delivered panel plus one new month: the promoted base must serve that month.
    panel = pd.read_csv(isolated_base_store / ARTIFACT_FOLDER_NAME / DATASET_FILENAME)
    last = pd.to_datetime(panel["date"]).max()
    new_month = panel[pd.to_datetime(panel["date"]) == last].copy()
    new_month["date"] = (last + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    csv = tmp_path / "panel.csv"
    pd.concat([panel, new_month], ignore_index=True).to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    served = container._plugin._panel_df  # pylint: disable=protected-access
    assert served["date"].max() == last + pd.DateOffset(months=1)
    batch = tmp_path / "new_month.csv"
    new_month.to_csv(batch, index=False)
    assert _predict_batch(container, str(batch))


# ── ml23 ───────────────────────────────────────────────────────────────────────

def test_ml23_reloads_the_promoted_base(monkeypatch, tmp_path, isolated_base_store):
    pytest.importorskip("torch")
    from app.plugins.ml23_lactic_market_price_forecast import model_loader
    from app.plugins.ml23_lactic_market_price_forecast import plugin as plugin_mod
    from app.plugins.ml23_lactic_market_price_forecast.constants import ARTIFACT_FOLDER_NAME

    monkeypatch.setattr(plugin_mod, "TRAIN_EPOCHS", 2)
    monkeypatch.setattr(plugin_mod, "TRAIN_N_SEEDS", 1)
    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    plugin_mod.Ml23LacticMarketPriceForecastPlugin)
    rng = np.random.default_rng(0)
    rows = []
    for producto, canal in (("LECHE ENTERA", "HIPER"), ("LECHE DESNATADA", "SUPER")):
        for i, fecha in enumerate(pd.date_range("2015-01-01", periods=60, freq="MS")):
            rows.append({"fecha": fecha.strftime("%Y-%m-%d"), "producto": producto, "canal": canal,
                         "target_precio_medio": 0.8 + 0.002 * i + rng.normal(0, 0.01),
                         "ipc_general_lag1": 100 + 0.1 * i, "mes": fecha.month})
    csv = tmp_path / "ml23.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    assert _predict_batch(container, str(csv))


def _predict_inline(container, payload: dict):
    entry = _entry(type(container._plugin))  # pylint: disable=protected-access
    body = TypeAdapter(entry.predict_request_type).validate_python({"mode": "inline", **payload})
    return container.predict_use_case.execute(body)


# ── ml19 ───────────────────────────────────────────────────────────────────────

def test_ml19_reloads_the_promoted_base_and_its_new_dataset(monkeypatch, tmp_path, isolated_base_store):
    from app.plugins.ml19_cereals_cost_forecast import model_loader
    from app.plugins.ml19_cereals_cost_forecast._vendor import training_core
    from app.plugins.ml19_cereals_cost_forecast.constants import ARTIFACT_FOLDER_NAME, DATASET_FE_FILENAME
    from app.plugins.ml19_cereals_cost_forecast.plugin import Ml19CerealsCostForecastPlugin

    for params in (training_core.XGB_PARAMS, training_core.XGBC_PARAMS, training_core.RF_PARAMS):
        monkeypatch.setitem(params, "n_estimators", 10)
    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml19CerealsCostForecastPlugin)
    # The delivered dataset plus one month: the promoted base must serve that month.
    dataset = pd.read_csv(isolated_base_store / ARTIFACT_FOLDER_NAME / DATASET_FE_FILENAME)
    last = pd.to_datetime(dataset["date"]).max()
    new_month = dataset.tail(1).copy()
    new_month["date"] = (last + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    csv = tmp_path / "dataset_v7_fe.csv"
    pd.concat([dataset, new_month], ignore_index=True).to_csv(csv, index=False)

    _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(csv))
    out = _predict_inline(container, {})
    assert out.date_label == (last + pd.DateOffset(months=1)).strftime("%Y-%m")


# ── ml24 ───────────────────────────────────────────────────────────────────────

def test_ml24_reloads_the_promoted_base_with_its_series(monkeypatch, isolated_base_store):
    pytest.importorskip("catboost")
    from app.plugins.ml24_dairy_spatial_price_forecast import model_loader
    from app.plugins.ml24_dairy_spatial_price_forecast.constants import (
        ARTIFACT_FOLDER_NAME, SERIES_FILENAME, TRAINING_CFG,
    )
    from app.plugins.ml24_dairy_spatial_price_forecast.plugin import Ml24DairySpatialPriceForecastPlugin

    monkeypatch.setitem(TRAINING_CFG, "catboost_iterations", 20)
    container, store = _start_on_v0(monkeypatch, isolated_base_store, ARTIFACT_FOLDER_NAME, model_loader,
                                    Ml24DairySpatialPriceForecastPlugin)
    served_v0 = container._plugin._reference["series"]  # pylint: disable=protected-access
    series_csv = isolated_base_store / ARTIFACT_FOLDER_NAME / SERIES_FILENAME

    resp = _train_and_reload(container, store, isolated_base_store, ARTIFACT_FOLDER_NAME, str(series_csv))
    assert resp.test_year > 2020   # derived from the series, not the delivered 2019/2020
    v1 = isolated_base_store / ARTIFACT_FOLDER_NAME / "versions" / "v1"
    assert (v1 / SERIES_FILENAME).stat().st_size < series_csv.stat().st_size   # only the 5 model products
    served_v1 = container._plugin._reference["series"]  # pylint: disable=protected-access
    assert len(served_v1) == len(served_v0)
    out = _predict_inline(container, {"ccaa": ["CYL"], "horizons": [1]})
    assert out.n_predictions == 1
