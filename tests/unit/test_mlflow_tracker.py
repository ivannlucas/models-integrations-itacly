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
        self.tracker = BaseMLflowTracker(run_id="")

    def test_is_connected_false(self) -> None:
        assert not self.tracker.is_connected()

    def test_log_params_noop(self) -> None:
        self.tracker.log_params({"lr": 0.01})  # should not raise

    def test_log_metrics_noop(self) -> None:
        self.tracker.log_metrics({"acc": 0.95})  # should not raise

    def test_set_tags_noop(self) -> None:
        self.tracker.set_tags({"env": "test"})  # should not raise

    def test_upload_artifacts_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "dummy.txt").write_text("hello")
            self.tracker.upload_artifacts(tmp)

    def test_download_artifacts_returns_empty(self) -> None:
        result = self.tracker.download_artifacts("/tmp")
        assert result == ""

    def test_get_metrics_returns_empty(self) -> None:
        assert self.tracker.get_metrics() == {}

    def test_get_params_returns_empty(self) -> None:
        assert self.tracker.get_params() == {}

    def test_get_tags_returns_empty(self) -> None:
        assert self.tracker.get_tags() == {}


class TestBaseMLflowTrackerConnected:
    """Non-empty run_id triggers real mlflow calls (mocked)."""

    RUN_ID = "test-run-123"

    def setup_method(self) -> None:
        self.tracker = BaseMLflowTracker(run_id=self.RUN_ID)

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_is_connected_true(self, mock_build: MagicMock) -> None:
        assert self.tracker.is_connected()

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_log_params(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.log_params({"lr": 0.01, "epochs": 10})
        client.log_param.assert_any_call(self.RUN_ID, "lr", "0.01")
        client.log_param.assert_any_call(self.RUN_ID, "epochs", "10")

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_log_metrics(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.log_metrics({"acc": 0.95, "loss": 0.05}, step=1)
        client.log_metric.assert_any_call(self.RUN_ID, "acc", 0.95, step=1)
        client.log_metric.assert_any_call(self.RUN_ID, "loss", 0.05, step=1)

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_set_tags(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        mock_build.return_value = client
        self.tracker.set_tags({"env": "prod"})
        client.set_tag.assert_called_once_with(self.RUN_ID, "env", "prod")

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_upload_artifacts(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        mock_build.return_value = client
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model.pth").write_text("weights")
            self.tracker.upload_artifacts(tmp)
            client.log_artifact.assert_called_once()

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_download_artifacts(self, mock_build: MagicMock) -> None:
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
        client = MagicMock()
        client.download_artifacts.side_effect = RuntimeError("connection error")
        mock_build.return_value = client
        result = self.tracker.download_artifacts("/tmp", artifact_path="model")
        assert result == ""

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_metrics(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        run = MagicMock()
        run.data.metrics = {"acc": 0.95, "loss": 0.05}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_metrics() == {"acc": 0.95, "loss": 0.05}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_metrics_failure_returns_empty(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        client.get_run.side_effect = RuntimeError("not found")
        mock_build.return_value = client
        assert self.tracker.get_metrics() == {}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_params(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        run = MagicMock()
        run.data.params = {"lr": "0.01"}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_params() == {"lr": "0.01"}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_get_tags(self, mock_build: MagicMock) -> None:
        client = MagicMock()
        run = MagicMock()
        run.data.tags = {"env": "prod"}
        client.get_run.return_value = run
        mock_build.return_value = client
        assert self.tracker.get_tags() == {"env": "prod"}

    @patch("app.domain.services.mlflow_tracker.BaseMLflowTracker._build_client")
    def test_connect_rebuilds_client(self, mock_build: MagicMock) -> None:
        self.tracker.connect("new-run-456")
        assert self.tracker.run_id == "new-run-456"
        mock_build.assert_called_once()
