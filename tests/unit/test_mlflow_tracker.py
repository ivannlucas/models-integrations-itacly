"""Unit tests for BaseMLflowTracker."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.domain.services.mlflow_tracker import BaseMLflowTracker


class TestBaseMLflowTrackerEmptyRunId:
    """Empty run_id should produce no-ops, never touch mlflow."""

    def setup_method(self) -> None:
        """Create a tracker without run_id."""
        self.tracker = BaseMLflowTracker(run_id="")  # pylint: disable=attribute-defined-outside-init

    def test_is_connected_false(self) -> None:
        """The tracker reports itself as not connected."""
        assert not self.tracker.is_connected()

    def test_log_params_noop(self) -> None:
        """log_params is a no-op that does not raise."""
        self.tracker.log_params({"lr": 0.01})  # should not raise

    def test_log_metrics_noop(self) -> None:
        """log_metrics is a no-op that does not raise."""
        self.tracker.log_metrics({"acc": 0.95})  # should not raise

    def test_set_tags_noop(self) -> None:
        """set_tags is a no-op that does not raise."""
        self.tracker.set_tags({"env": "test"})  # should not raise

    def test_upload_artifacts_noop(self) -> None:
        """upload_artifacts is a no-op that does not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "dummy.txt").write_text("hello", encoding="utf-8")
            self.tracker.upload_artifacts(tmp)

    def test_download_artifacts_returns_empty(self) -> None:
        """download_artifacts returns an empty path."""
        result = self.tracker.download_artifacts("/tmp")
        assert result == ""

    def test_get_metrics_returns_empty(self) -> None:
        """get_metrics returns an empty dict."""
        assert self.tracker.get_metrics() == {}

    def test_get_params_returns_empty(self) -> None:
        """get_params returns an empty dict."""
        params = self.tracker.get_params()
        assert isinstance(params, dict) and not params

    def test_get_tags_returns_empty(self) -> None:
        """get_tags returns an empty dict."""
        tags = self.tracker.get_tags()
        assert isinstance(tags, dict) and not tags


class TestBaseMLflowTrackerConnected:
    """Non-empty run_id triggers real mlflow calls (mocked)."""

    RUN_ID = "test-run-123"

    def setup_method(self) -> None:
        """Create a tracker bound to RUN_ID."""
        self.tracker = BaseMLflowTracker(run_id=self.RUN_ID)  # pylint: disable=attribute-defined-outside-init

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_is_connected_true(self, mock_build: MagicMock) -> None:
        """A non-empty run_id reports a connected tracker."""
        _ = mock_build
        assert self.tracker.is_connected()

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_log_params(self, mock_build: MagicMock) -> None:
        """log_params logs each param as a string on the run."""
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.log_params({"lr": 0.01, "epochs": 10})
        client.log_param.assert_any_call(self.RUN_ID, "lr", "0.01")
        client.log_param.assert_any_call(self.RUN_ID, "epochs", "10")

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_log_metrics(self, mock_build: MagicMock) -> None:
        """log_metrics logs each metric on the run with its step."""
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.log_metrics({"acc": 0.95, "loss": 0.05}, step=1)
        client.log_metric.assert_any_call(self.RUN_ID, "acc", 0.95, step=1)
        client.log_metric.assert_any_call(self.RUN_ID, "loss", 0.05, step=1)

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_set_tags(self, mock_build: MagicMock) -> None:
        """set_tags sets each tag on the run."""
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.set_tags({"env": "prod"})
        client.set_tag.assert_called_once_with(self.RUN_ID, "env", "prod")

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_upload_artifacts(self, mock_build: MagicMock) -> None:
        """upload_artifacts logs the directory's files as artifacts."""
        client = MagicMock()
        mock_build.return_value = client
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.pth").write_text("weights", encoding="utf-8")
            self.tracker.upload_artifacts(tmp)
            client.log_artifact.assert_called_once()

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_download_artifacts(self, mock_build: MagicMock) -> None:
        """download_artifacts downloads into dst_path and returns the local path."""
        client = MagicMock()
        mock_build.return_value = client
        with tempfile.TemporaryDirectory() as tmp:
            result = self.tracker.download_artifacts(tmp, artifact_path="model")
            client.download_artifacts.assert_called_once_with(
                self.RUN_ID, "model", dst_path=tmp
            )
            expected = os.path.normpath(os.path.join(tmp, "model"))
            assert result == expected

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_download_artifacts_failure_returns_empty(self, mock_build: MagicMock) -> None:
        """A download failure returns an empty path instead of raising."""
        client = MagicMock()
        client.download_artifacts.side_effect = RuntimeError("connection error")
        mock_build.return_value = client
        result = self.tracker.download_artifacts("/tmp", artifact_path="model")
        assert result == ""

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_metrics(self, mock_build: MagicMock) -> None:
        """get_metrics returns the run's metrics."""
        client = MagicMock()
        run = MagicMock()
        run.data.metrics = {"acc": 0.95, "loss": 0.05}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_metrics() == {"acc": 0.95, "loss": 0.05}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_metrics_failure_returns_empty(self, mock_build: MagicMock) -> None:
        """A get_run failure makes get_metrics return an empty dict."""
        client = MagicMock()
        client.get_run.side_effect = RuntimeError("not found")
        mock_build.return_value = client
        metrics = self.tracker.get_metrics()
        assert isinstance(metrics, dict) and not metrics

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_params(self, mock_build: MagicMock) -> None:
        """get_params returns the run's params."""
        client = MagicMock()
        run = MagicMock()
        run.data.params = {"lr": "0.01"}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_params() == {"lr": "0.01"}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_tags(self, mock_build: MagicMock) -> None:
        """get_tags returns the run's tags."""
        client = MagicMock()
        run = MagicMock()
        run.data.tags = {"env": "prod"}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_tags() == {"env": "prod"}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_connect_rebuilds_client(self, mock_build: MagicMock) -> None:
        """connect() switches run_id and rebuilds the client."""
        self.tracker.connect("new-run-456")
        assert self.tracker.run_id == "new-run-456"
        mock_build.assert_called_once()
