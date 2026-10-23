"""Direct unit tests for Ml25WineSulphitesPlugin state, load, and stats (no real artifacts)."""
from unittest.mock import MagicMock, patch

from app.plugins.ml25_wine_sulphites.plugin import Ml25WineSulphitesPlugin

_FAKE_METADATA = {
    "metrics": {
        "quality_cv": {"mae_mean": 0.427},
        "bound_cv": {"mae_mean": 14.5},
    }
}
_FAKE_ARTIFACTS = (MagicMock(), MagicMock(), _FAKE_METADATA)
_PATCH_LOAD = "app.plugins.ml25_wine_sulphites.plugin.load_artifacts"


def _loaded_plugin() -> Ml25WineSulphitesPlugin:
    """Return a Ml25WineSulphitesPlugin with load_artifacts mocked and load() already called."""
    plugin = Ml25WineSulphitesPlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS):
        plugin.load()
    return plugin


# ── load / is_loaded ──────────────────────────────────────────────────────────

def test_is_loaded_false_before_load():
    """Plugin reports not loaded before load() is called."""
    assert Ml25WineSulphitesPlugin().is_loaded() is False


def test_is_loaded_true_after_load():
    """Plugin reports loaded after a successful load() call."""
    assert _loaded_plugin().is_loaded() is True


def test_load_calls_load_artifacts_exactly_once():
    """load() invokes load_artifacts() exactly once."""
    plugin = Ml25WineSulphitesPlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS) as mock_la:
        plugin.load()
    mock_la.assert_called_once()


def test_second_load_reloads_artifacts():
    """Calling load() twice invokes load_artifacts() twice (hot-reload supported)."""
    plugin = Ml25WineSulphitesPlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS) as mock_la:
        plugin.load()
        plugin.load()
    assert mock_la.call_count == 2


# ── stats() ───────────────────────────────────────────────────────────────────

def test_stats_returns_correct_model_name():
    """stats() returns the model name 'ml25-wine-sulphites'."""
    assert _loaded_plugin().stats().model_name == "ml25-wine-sulphites"


def test_stats_returns_correct_version():
    """stats() returns version '1.3.0'."""
    assert _loaded_plugin().stats().version == "1.3.0"


def test_stats_initial_prediction_count_is_zero():
    """Total predictions is 0 immediately after loading, before any inference."""
    assert _loaded_plugin().stats().runtime_stats.total_predictions == 0


def test_stats_avg_latency_is_none_without_predictions():
    """avg_latency_ms is None when no predictions have been made."""
    assert _loaded_plugin().stats().runtime_stats.avg_latency_ms is None


def test_stats_has_at_least_one_input_field():
    """stats() returns at least one input field descriptor."""
    assert len(_loaded_plugin().stats().inputs) > 0


def test_stats_has_at_least_one_output_field():
    """stats() returns at least one output field descriptor."""
    assert len(_loaded_plugin().stats().outputs) > 0


def test_stats_metrics_contains_mae_quality():
    """stats() metrics dict contains the 'mae_quality' key."""
    assert "mae_quality" in _loaded_plugin().stats().metrics


def test_stats_metrics_contains_mae_bound_so2():
    """stats() metrics dict contains the 'mae_bound_so2_mg_l' key."""
    assert "mae_bound_so2_mg_l" in _loaded_plugin().stats().metrics


def test_stats_mae_quality_matches_metadata():
    """mae_quality in stats() reflects the value stored in the loaded metadata."""
    stats = _loaded_plugin().stats()
    assert stats.metrics["mae_quality"] == 0.427


def test_stats_mae_bound_matches_metadata():
    """mae_bound_so2_mg_l in stats() reflects the value stored in the loaded metadata."""
    stats = _loaded_plugin().stats()
    assert stats.metrics["mae_bound_so2_mg_l"] == 14.5


# ── predict_batch: simulation_trajectory per row ─────────────────────────────

import numpy as np  # noqa: E402

_WINE_COLUMNS = (
    "fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,"
    "free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol"
)
_WINE_ROW = "7.0,0.27,0.36,20.7,0.045,{free},170,1.001,3.0,0.45,8.8"


class _FakeBound:
    """Bound SO2 model returning a constant 90 mg/L (in the log1p space the plugin decodes)."""

    def predict(self, rows):
        return np.log1p(np.full(len(rows), 90.0))


class _FakeQuality:
    """Quality model rising gently along the simulated doses, so there is a real optimum."""

    def predict(self, rows):
        return np.linspace(5.8, 6.4, len(rows))


def _plugin_with_fake_models() -> Ml25WineSulphitesPlugin:
    plugin = Ml25WineSulphitesPlugin()
    with patch(_PATCH_LOAD, return_value=(_FakeQuality(), _FakeBound(), _FAKE_METADATA)):
        plugin.load()
    return plugin


def _write_csv(tmp_path, n_rows, extra_column=False):
    header = ("deposito," if extra_column else "") + _WINE_COLUMNS
    lines = [header]
    for i in range(n_rows):
        prefix = f"D-{i:02d}," if extra_column else ""
        lines.append(prefix + _WINE_ROW.format(free=20 + i))
    path = tmp_path / ("wine_extra.csv" if extra_column else "wine.csv")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def _without_trajectory(predictions):
    return [{k: v for k, v in p.items() if k != "simulation_trajectory"} for p in predictions]


def test_predict_batch_trajectory_on_every_row_under_cap(tmp_path):
    plugin = _plugin_with_fake_models()
    resp = plugin.predict_batch(data_path=_write_csv(tmp_path, 5))
    assert len(resp.predictions) == 5
    for p in resp.predictions:
        traj = p["simulation_trajectory"]
        assert len(traj) > 1
        assert sum(1 for point in traj if point["is_recommended"]) == 1


def test_predict_batch_trajectory_first_row_only_over_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("app.plugins.ml25_wine_sulphites.plugin.BATCH_TRAJECTORY_MAX_ROWS", 2)
    plugin = _plugin_with_fake_models()
    resp = plugin.predict_batch(data_path=_write_csv(tmp_path, 3))
    assert "simulation_trajectory" in resp.predictions[0]
    assert all("simulation_trajectory" not in p for p in resp.predictions[1:])


def test_predict_batch_values_unchanged_by_trajectory(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path, 4)
    plugin = _plugin_with_fake_models()
    monkeypatch.setattr("app.plugins.ml25_wine_sulphites.plugin.BATCH_TRAJECTORY_MAX_ROWS", 1)
    capped = plugin.predict_batch(data_path=csv).predictions
    monkeypatch.setattr("app.plugins.ml25_wine_sulphites.plugin.BATCH_TRAJECTORY_MAX_ROWS", 100)
    full = plugin.predict_batch(data_path=csv).predictions
    assert _without_trajectory(capped) == _without_trajectory(full)


def test_predict_batch_ignores_extra_tank_column(tmp_path):
    plugin = _plugin_with_fake_models()
    plain = plugin.predict_batch(data_path=_write_csv(tmp_path, 3)).predictions
    with_tank = plugin.predict_batch(data_path=_write_csv(tmp_path, 3, extra_column=True)).predictions
    assert all("error" not in p for p in with_tank)
    assert with_tank == plain
