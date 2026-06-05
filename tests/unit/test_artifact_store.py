"""Unit tests for ArtifactStore — local_dir property and upload()."""
from unittest.mock import ANY, MagicMock, patch

from app.infrastructure.artifact_store import ARTIFACTS_ROOT, ArtifactStore


def test_local_dir_returns_correct_path():
    """local_dir property resolves to ARTIFACTS_ROOT / model_name."""
    store = ArtifactStore("my_model")
    assert store.local_dir == ARTIFACTS_ROOT / "my_model"


def test_upload_no_bucket_is_noop(monkeypatch):
    """upload() is a no-op and does not raise when STORAGE_BUCKET is unset."""
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    store = ArtifactStore("my_model")
    store.upload("weights.pkl")  # must not raise


def test_upload_with_bucket_calls_s3(monkeypatch, tmp_path):
    """upload() calls s3.upload_file with the correct bucket, key, and local path when STORAGE_BUCKET is set."""
    monkeypatch.setenv("STORAGE_BUCKET", "test-bucket")

    (tmp_path / "weights.pkl").write_bytes(b"fake data")

    mock_s3 = MagicMock()
    store = ArtifactStore("my_model")
    store._local_dir = tmp_path

    with patch("app.infrastructure.artifact_store._build_s3_client", return_value=mock_s3):
        store.upload("weights.pkl")

    mock_s3.upload_file.assert_called_once_with(
        str(tmp_path / "weights.pkl"),
        "test-bucket",
        "artifacts/fixed/my_model/weights.pkl",
        Config=ANY,
    )
