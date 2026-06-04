"""Direct unit tests for wine-sulphite helper functions.

Tests preprocessing, postprocessing, and the real plugin (mocked models).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.plugins.wine_sulphite.postprocessing import (
    apply_operational_constraints,
    compute_molecular_so2,
    decode_bound_predictions,
    select_recommendation,
)
from app.plugins.wine_sulphite.preprocessing import (
    FEATURES_BOUND,
    FEATURES_QUAL,
    PKA_SO2,
    build_simulation_grid,
    map_request_to_wine_dict,
)
from app.plugins.wine_sulphite.predict_dto import PredictInlineRequest


# ── preprocessing tests ───────────────────────────────────────────────────

class TestMapRequestToWineDict:
    """Tests for map_request_to_wine_dict field mapping."""

    def test_maps_snake_case_to_space_separated(self):
        """Verify snake_case fields map to space-separated wine feature names."""
        req = PredictInlineRequest(
            mode="inline",
            fixed_acidity=7.4,
            volatile_acidity=0.66,
            citric_acid=0.0,
            residual_sugar=1.8,
            chlorides=0.075,
            density=0.9978,
            pH=3.51,
            sulphates=0.56,
            alcohol=9.4,
            free_sulfur_dioxide=11.0,
            total_sulfur_dioxide=34.0,
            min_molecular=0.6,
            max_total=200.0,
            delta_max=40.0,
        )
        result = map_request_to_wine_dict(req)
        assert result["fixed acidity"] == 7.4
        assert result["pH"] == 3.51
        assert result["free sulfur dioxide"] == 11.0
        assert result["total sulfur dioxide"] == 34.0

    def test_all_keys_present(self):
        """Verify all expected feature keys are present in the output dict."""
        req = PredictInlineRequest(
            mode="inline",
            fixed_acidity=1, volatile_acidity=2, citric_acid=3,
            residual_sugar=4, chlorides=5, density=6, pH=7,
            sulphates=8, alcohol=9, free_sulfur_dioxide=10,
            total_sulfur_dioxide=11, min_molecular=0.6,
            max_total=200, delta_max=40,
        )
        result = map_request_to_wine_dict(req)
        expected_keys = {
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "density", "pH",
            "sulphates", "alcohol", "free sulfur dioxide",
            "total sulfur dioxide",
        }
        assert set(result.keys()) == expected_keys


class TestBuildSimulationGrid:
    """Tests for build_simulation_grid with various delta_max values."""

    def test_returns_expected_types(self):
        """Verify build_simulation_grid returns numpy array and DataFrames."""
        base = {
            "fixed acidity": 7.4, "volatile acidity": 0.66,
            "citric acid": 0.0, "residual sugar": 1.8,
            "chlorides": 0.075, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4,
            "free sulfur dioxide": 11.0, "total sulfur dioxide": 34.0,
        }
        free_targets, qual_df, bound_df = build_simulation_grid(base, delta_max=40.0)
        assert isinstance(free_targets, np.ndarray)
        assert len(free_targets) == 41  # 11.0 to 51.0 step 1
        assert list(qual_df.columns) == FEATURES_QUAL
        assert list(bound_df.columns) == FEATURES_BOUND
        assert len(qual_df) == 41
        assert len(bound_df) == 41

    def test_delta_max_zero_returns_single_point(self):
        """Verify zero delta_max returns a single simulation point."""
        base = {
            "fixed acidity": 7.4, "volatile acidity": 0.66,
            "citric acid": 0.0, "residual sugar": 1.8,
            "chlorides": 0.075, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4,
            "free sulfur dioxide": 11.0, "total sulfur dioxide": 34.0,
        }
        free_targets, qual_df, bound_df = build_simulation_grid(base, delta_max=0)
        assert len(free_targets) == 1
        assert free_targets[0] == 11.0

    def test_grid_starts_at_current_free_so2(self):
        """Verify the grid starts at the current free SO2 level."""
        base = {
            "fixed acidity": 7.4, "volatile acidity": 0.66,
            "citric acid": 0.0, "residual sugar": 1.8,
            "chlorides": 0.075, "density": 0.9978, "pH": 3.51,
            "sulphates": 0.56, "alcohol": 9.4,
            "free sulfur dioxide": 5.0, "total sulfur dioxide": 20.0,
        }
        free_targets, _, _ = build_simulation_grid(base, delta_max=10)
        assert free_targets[0] == 5.0
        assert free_targets[-1] == 15.0


# ── postprocessing tests ──────────────────────────────────────────────────

class TestDecodeBoundPredictions:
    """Tests for decode_bound_predictions output properties."""

    def test_decode_and_monotonic(self):
        """Verify decode_bound_predictions returns non-negative, monotonic values."""
        raw = np.log1p(np.array([5.0, 10.0, 15.0, 8.0, 12.0]))
        free = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
        decoded = decode_bound_predictions(raw, free)
        assert len(decoded) == 5
        assert (decoded >= 0).all()
        # sorted by free targets, monotonic
        order = np.argsort(free)
        assert (np.diff(decoded[order]) >= 0).all() or len(decoded) <= 1

    def test_all_zeros(self):
        """Verify decode_bound_predictions handles all-zero input."""
        raw = np.zeros(5)
        free = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        decoded = decode_bound_predictions(raw, free)
        assert len(decoded) == 5
        assert (decoded >= 0).all()


class TestComputeMolecularSO2:
    """Tests for compute_molecular_so2 formula correctness."""

    def test_computes_correctly(self):
        """Verify compute_molecular_so2 returns correct values."""
        # molecular = free / (1 + 10^(pH - pKa))
        free = np.array([10.0, 20.0, 30.0])
        pH = 3.5
        result = compute_molecular_so2(free, pH)
        expected = free / (1.0 + 10.0 ** (pH - PKA_SO2))
        np.testing.assert_array_almost_equal(result, expected)

    def test_zero_free_so2(self):
        """Verify compute_molecular_so2 returns 0 for zero free SO2."""
        result = compute_molecular_so2(np.array([0.0]), 3.5)
        assert result[0] == 0.0


class TestApplyOperationalConstraints:
    """Tests for apply_operational_constraints filtering logic."""

    def test_valid_points_returned(self):
        """Verify apply_operational_constraints filters points correctly."""
        free = np.array([10.0, 20.0, 30.0])
        bounds = np.array([5.0, 8.0, 12.0])
        totals = np.array([15.0, 28.0, 42.0])
        molecular = np.array([0.8, 0.5, 0.3])
        qualities = np.array([6.0, 6.5, 7.0])
        result = apply_operational_constraints(
            free, bounds, totals, molecular, qualities,
            min_molecular=0.5, max_total=40.0,
        )
        vf, vb, vt, vm, vq = result
        assert len(vf) == 2
        np.testing.assert_array_equal(vf, [10.0, 20.0])
        np.testing.assert_array_equal(vq, [6.0, 6.5])

    def test_no_valid_points_raises(self):
        """Verify apply_operational_constraints raises ValueError when nothing is valid."""
        free = np.array([10.0, 20.0])
        bounds = np.array([5.0, 8.0])
        totals = np.array([50.0, 60.0])  # exceeds max_total
        molecular = np.array([0.1, 0.05])  # below min_molecular
        qualities = np.array([6.0, 6.5])
        with pytest.raises(ValueError, match="No simulation point"):
            apply_operational_constraints(
                free, bounds, totals, molecular, qualities,
                min_molecular=0.5, max_total=40.0,
            )


class TestSelectRecommendation:
    """Tests for select_recommendation quality-gain logic."""

    def test_significant_gain_selects_best_idx(self):
        """Verify select_recommendation picks the best quality index when gain is significant."""
        qualities = np.array([5.0, 6.5, 7.0, 6.0])
        free = np.array([10.0, 11.0, 12.0, 13.0])
        bounds = np.array([5.0, 6.0, 7.0, 8.0])
        totals = np.array([15.0, 17.0, 19.0, 21.0])
        moleculars = np.array([0.8, 0.7, 0.6, 0.5])

        idx, reason, intervention = select_recommendation(
            free, bounds, totals, moleculars, qualities,
            baseline_quality=5.0, mae_quality=0.5,
        )
        assert intervention is True
        assert idx == 2  # best quality at index 2 (7.0)
        assert "Significant improvement" in reason

    def test_no_significant_gain_selects_index_0(self):
        """Verify select_recommendation returns index 0 with no intervention when gain is small."""
        qualities = np.array([5.0, 5.1, 5.05])
        free = np.array([10.0, 11.0, 12.0])
        bounds = np.array([5.0, 6.0, 7.0])
        totals = np.array([15.0, 17.0, 19.0])
        moleculars = np.array([0.8, 0.7, 0.6])

        idx, reason, intervention = select_recommendation(
            free, bounds, totals, moleculars, qualities,
            baseline_quality=5.0, mae_quality=0.5,
        )
        assert intervention is False
        assert idx == 0
        assert "not significant" in reason

    def test_edge_case_equal_quality(self):
        """Verify select_recommendation handles equal quality values correctly."""
        qualities = np.array([5.0, 5.0, 5.0])
        free = np.array([10.0, 11.0, 12.0])
        bounds = np.array([5.0, 6.0, 7.0])
        totals = np.array([15.0, 17.0, 19.0])
        moleculars = np.array([0.8, 0.7, 0.6])

        idx, reason, intervention = select_recommendation(
            free, bounds, totals, moleculars, qualities,
            baseline_quality=5.0, mae_quality=0.1,
        )
        # gain is 0, which is ≤ threshold
        assert intervention is False
        assert idx == 0


# ── Real plugin unit tests (mocked model loading) ─────────────────────────

class TestWineSulphitePluginDirect:
    """Tests for the real WineSulphitePlugin class with mocked model loading."""

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_initial_state(self, mock_load):
        """Verify the plugin starts in an unloaded state with zero stats."""
        mock_load.return_value = (MagicMock(), MagicMock(), {})
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
        plugin = WineSulphitePlugin()
        assert plugin.is_loaded() is False
        assert plugin._predict_count == 0
        assert plugin._last_predict_at is None

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_load_sets_loaded(self, mock_load):
        """Verify calling load() makes is_loaded() return True."""
        mock_load.return_value = (MagicMock(), MagicMock(), {})
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
        plugin = WineSulphitePlugin()
        plugin.load()
        assert plugin.is_loaded() is True

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_stats_structure(self, mock_load):
        """Verify stats() returns a StatsResponse with expected fields."""
        mock_load.return_value = (MagicMock(), MagicMock(), {})
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
        plugin = WineSulphitePlugin()
        plugin.load()
        stats = plugin.stats()
        assert stats.model_name == "wine-sulphite"
        assert stats.predict_count == 0
        assert stats.last_predict_at is None

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_train_raises_not_supported(self, mock_load):
        """Verify train() raises TrainingNotSupportedError."""
        mock_load.return_value = (MagicMock(), MagicMock(), {})
        from app.domain.services.exceptions import TrainingNotSupportedError
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin
        plugin = WineSulphitePlugin()
        plugin.load()
        with pytest.raises(TrainingNotSupportedError):
            plugin.train(data_path="/some/path.zip")

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_predict_inline_with_mocked_models(self, mock_load):
        """Verify predict_inline returns expected fields with mocked models."""
        import numpy as np
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin

        # Build a range of free SO2 values matching expected grid
        base_free = 11.0
        delta_max = 40.0
        n_grid = int(delta_max) + 1  # 41 points

        mock_bound = MagicMock()
        mock_bound.predict.return_value = np.log1p(np.full(n_grid, 5.0))
        mock_qual = MagicMock()
        # Quality predictions: start low, go high
        qual_vals = np.linspace(6.0, 7.5, n_grid)
        mock_qual.predict.return_value = qual_vals

        mock_load.return_value = (mock_qual, mock_bound, {
            "metrics": {
                "quality_cv": {"mae_mean": 0.5},
                "bound_cv": {"mae_mean": 15.0},
            }
        })
        plugin = WineSulphitePlugin()
        plugin.load()

        features = {
            "mode": "inline",
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.66,
            "citric_acid": 0.0,
            "residual_sugar": 1.8,
            "chlorides": 0.075,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
            "free_sulfur_dioxide": base_free,
            "total_sulfur_dioxide": 34.0,
            "min_molecular": 0.3,
            "max_total": 200.0,
            "delta_max": delta_max,
        }
        result = plugin.predict_inline(features=features)
        assert result["model_id"] == "wine-sulphite"
        assert "prediction" in result
        assert "confidence" in result
        assert "recommended_free_so2" in result
        assert plugin._predict_count == 1

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_predict_inline_no_valid_point_raises(self, mock_load):
        """Verify predict_inline raises NoValidSimulationPointError when no feasible SO2 dose exists."""
        import numpy as np
        from app.domain.services.exceptions import NoValidSimulationPointError
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin

        base_free = 11.0
        n_grid = 41
        mock_bound = MagicMock()
        mock_bound.predict.return_value = np.log1p(np.full(n_grid, 5.0))
        mock_qual = MagicMock()
        mock_qual.predict.return_value = np.linspace(6.0, 7.5, n_grid)

        mock_load.return_value = (mock_qual, mock_bound, {})
        plugin = WineSulphitePlugin()
        plugin.load()

        features = {
            "mode": "inline",
            "fixed_acidity": 7.4,
            "volatile_acidity": 0.66,
            "citric_acid": 0.0,
            "residual_sugar": 1.8,
            "chlorides": 0.075,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
            "free_sulfur_dioxide": base_free,
            "total_sulfur_dioxide": 34.0,
            "min_molecular": 5.0,
            "max_total": 10.0,
            "delta_max": 40.0,
        }
        with pytest.raises(NoValidSimulationPointError):
            plugin.predict_inline(features=features)

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_predict_batch_with_mocked_models(self, mock_load):
        """Verify predict_batch returns predictions list with mocked models."""
        import numpy as np
        import tempfile
        from pathlib import Path
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin

        n_grid = 41
        mock_bound = MagicMock()
        mock_bound.predict.return_value = np.log1p(np.full(n_grid, 5.0))
        mock_qual = MagicMock()
        mock_qual.predict.return_value = np.linspace(6.0, 7.5, n_grid)

        mock_load.return_value = (mock_qual, mock_bound, {})
        plugin = WineSulphitePlugin()
        plugin.load()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "test.csv"
            csv_path.write_text(
                "fixed acidity,volatile acidity,citric acid,residual sugar,"
                "chlorides,density,pH,sulphates,alcohol,"
                "free sulfur dioxide,total sulfur dioxide\n"
                "7.4,0.66,0.0,1.8,0.075,0.9978,3.51,0.56,9.4,11.0,34.0\n"
            )
            result = plugin.predict_batch(data_path=str(csv_path))
            assert "predictions" in result
            assert len(result["predictions"]) == 1

    @patch("app.plugins.wine_sulphite.plugin.load_artifacts")
    def test_predict_batch_with_error(self, mock_load):
        """Verify predict_batch handles errors gracefully and includes error status in results."""
        import numpy as np
        import tempfile
        from pathlib import Path
        from app.plugins.wine_sulphite.plugin import WineSulphitePlugin

        # Model bound.predict returns non-numeric data, causing error in inference
        mock_bound = MagicMock()
        mock_bound.predict.return_value = "bad_data"
        mock_qual = MagicMock()
        mock_qual.predict.return_value = np.linspace(6.0, 7.5, 41)

        mock_load.return_value = (mock_qual, mock_bound, {})
        plugin = WineSulphitePlugin()
        plugin.load()

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "test.csv"
            csv_path.write_text(
                "fixed acidity,volatile acidity,citric acid,residual sugar,"
                "chlorides,density,pH,sulphates,alcohol,"
                "free sulfur dioxide,total sulfur dioxide\n"
                "7.4,0.66,0.0,1.8,0.075,0.9978,3.51,0.56,9.4,11.0,34.0\n"
            )
            result = plugin.predict_batch(data_path=str(csv_path))
            assert len(result["predictions"]) == 1
            assert "error" in result["predictions"][0]
