"""Direct unit tests for WineSulphitePlugin state, load, and stats (no real artifacts)."""
from unittest.mock import MagicMock, patch

from app.plugins.ml25_wine_sulphites.plugin import WineSulphitePlugin

_FAKE_METADATA = {
    "metrics": {
        "quality_cv": {"mae_mean": 0.427},
        "bound_cv": {"mae_mean": 14.5},
    }
}
_FAKE_ARTIFACTS = (MagicMock(), MagicMock(), _FAKE_METADATA)
_PATCH_LOAD = "app.plugins.ml25_wine_sulphites.plugin.load_artifacts"


def _loaded_plugin() -> WineSulphitePlugin:
    """Return a WineSulphitePlugin with load_artifacts mocked and load() already called."""
    plugin = WineSulphitePlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS):
        plugin.load()
    return plugin


# ── load / is_loaded ──────────────────────────────────────────────────────────

def test_is_loaded_false_before_load():
    """Plugin reports not loaded before load() is called."""
    assert WineSulphitePlugin().is_loaded() is False


def test_is_loaded_true_after_load():
    """Plugin reports loaded after a successful load() call."""
    assert _loaded_plugin().is_loaded() is True


def test_load_calls_load_artifacts_exactly_once():
    """load() invokes load_artifacts() exactly once."""
    plugin = WineSulphitePlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS) as mock_la:
        plugin.load()
    mock_la.assert_called_once()


def test_second_load_reloads_artifacts():
    """Calling load() twice invokes load_artifacts() twice (hot-reload supported)."""
    plugin = WineSulphitePlugin()
    with patch(_PATCH_LOAD, return_value=_FAKE_ARTIFACTS) as mock_la:
        plugin.load()
        plugin.load()
    assert mock_la.call_count == 2


# ── stats() ───────────────────────────────────────────────────────────────────

def test_stats_returns_correct_model_name():
    """stats() returns the model name 'wine-sulphite'."""
    assert _loaded_plugin().stats().model_name == "wine-sulphite"


def test_stats_returns_correct_version():
    """stats() returns version '1.2.0'."""
    assert _loaded_plugin().stats().version == "1.2.0"


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
