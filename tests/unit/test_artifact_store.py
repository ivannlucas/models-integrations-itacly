"""Unit tests for ArtifactStore — local_dir property, path resolution, and download."""
from pathlib import Path

import pytest

from app.infrastructure.artifact_store import ARTIFACTS_ROOT, ArtifactStore, _file_needs_download


def test_local_dir_returns_correct_path():
    """local_dir property resolves to ARTIFACTS_ROOT / model_name."""
    store = ArtifactStore("my_model")
    assert store.local_dir == ARTIFACTS_ROOT / "my_model"


# ── path() ────────────────────────────────────────────────────────────────────

def test_path_returns_local_path_when_file_exists(tmp_path):
    """path() returns the correct Path when the file already exists locally."""
    (tmp_path / "model.pkl").write_bytes(b"data")
    store = ArtifactStore("my_model")
    store._local_dir = tmp_path
    assert store.path("model.pkl") == tmp_path / "model.pkl"


def test_path_raises_file_not_found_when_missing_and_no_s3(monkeypatch, tmp_path):
    """path() raises FileNotFoundError when the file is absent and STORAGE_BUCKET is unset."""
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    store = ArtifactStore("my_model")
    store._local_dir = tmp_path
    with pytest.raises(FileNotFoundError):
        store.path("missing.pkl")


# ── download_all_if_needed() ──────────────────────────────────────────────────

def test_download_all_if_needed_raises_without_bucket(monkeypatch):
    """download_all_if_needed() raises EnvironmentError when STORAGE_BUCKET is not set."""
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    store = ArtifactStore("my_model")
    with pytest.raises(EnvironmentError):
        store.download_all_if_needed()


# ── _file_needs_download() ────────────────────────────────────────────────────

def test_file_needs_download_absent_file_returns_true():
    """_file_needs_download() returns True when the local file does not exist."""
    assert _file_needs_download(Path("/nonexistent/path/file.pkl"), 100) is True


def test_file_needs_download_same_size_returns_false(tmp_path):
    """_file_needs_download() returns False when the local file size matches the remote size."""
    f = tmp_path / "model.pkl"
    f.write_bytes(b"x" * 100)
    assert _file_needs_download(f, 100) is False


def test_file_needs_download_different_size_returns_true(tmp_path):
    """_file_needs_download() returns True when the local file size differs from the remote size."""
    f = tmp_path / "model.pkl"
    f.write_bytes(b"x" * 50)
    assert _file_needs_download(f, 100) is True
