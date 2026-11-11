"""ml4 /train data_path resolution: ZIP by content, nested dataset root, clear errors."""
import os
import tempfile
import zipfile

import pytest

from app.plugins.ml4_lactic_cnn_thermal_early_disease_detection.plugin import (
    _is_zip,
    _load_training_set,
)

_N_PER_CLASS = 6


def _write_tids_zip(zip_path, prefix: str = "", macosx: bool = False) -> None:
    """ZIP with ID_Labels.csv + TIDS_cropped/{healthy,SCM}/<ID>.jpg under ``prefix``."""
    rows = ["ID,label"]
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(_N_PER_CLASS * 2):
            label = i % 2
            folder = "SCM" if label else "healthy"
            rows.append(f"{i},{label}")
            zf.writestr(f"{prefix}TIDS_cropped/{folder}/{i}.jpg", b"jpg")
        zf.writestr(f"{prefix}ID_Labels.csv", "\n".join(rows) + "\n")
        if macosx:
            zf.writestr("__MACOSX/._ID_Labels.csv", b"meta")


def test_zip_without_extension_is_extracted(tmp_path):
    """An s3:// key uploaded by the platform may have no ".zip" suffix: detect it by content."""
    data = tmp_path / "upload-0f3a"
    _write_tids_zip(data)
    paths, labels = _load_training_set(str(data), str(tmp_path / "x"))
    assert len(paths) == _N_PER_CLASS * 2
    assert sorted(set(labels)) == [0, 1]


@pytest.mark.parametrize("prefix", ["", "train_tids/", "a/b/"])
def test_dataset_root_found_at_any_depth(tmp_path, prefix):
    """ID_Labels.csv locates the dataset root, at the ZIP root or inside wrapping folders."""
    data = tmp_path / "ds.zip"
    _write_tids_zip(data, prefix=prefix, macosx=True)
    paths, _ = _load_training_set(str(data), str(tmp_path / "x"))
    assert len(paths) == _N_PER_CLASS * 2


def test_s3_zip_without_suffix(tmp_path, monkeypatch):
    """s3:// download keeps the key's (empty) suffix; the ZIP is still recognised."""
    src = tmp_path / "src.zip"
    _write_tids_zip(src)

    def fake_download(_uri):
        fd, tmp = tempfile.mkstemp()
        with os.fdopen(fd, "wb") as fh:
            fh.write(src.read_bytes())
        return tmp

    monkeypatch.setattr("app.infrastructure.artifact_store.download_s3_file", fake_download)
    paths, _ = _load_training_set("s3://bucket/uploads/abc123", str(tmp_path / "x"))
    assert len(paths) == _N_PER_CLASS * 2


def test_non_zip_file_is_a_clear_error(tmp_path):
    """A file that is not a ZIP is rejected naming the expected layout."""
    bad = tmp_path / "data.csv"
    bad.write_text("ID,label\n1,0\n")
    assert not _is_zip(str(bad))
    with pytest.raises(ValueError, match="no es un ZIP ni una carpeta"):
        _load_training_set(str(bad), str(tmp_path / "x"))


def test_zip_without_labels_is_a_clear_error(tmp_path):
    """A ZIP without ID_Labels.csv keeps the explicit contract error."""
    data = tmp_path / "ds.zip"
    with zipfile.ZipFile(data, "w") as zf:
        zf.writestr("TIDS_cropped/healthy/1.jpg", b"jpg")
    with pytest.raises(ValueError, match="ID_Labels.csv"):
        _load_training_set(str(data), str(tmp_path / "x"))
