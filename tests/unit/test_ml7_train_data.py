"""ml7 /train input: a ZIP (local or s3://) like the other image plugins, or a local dataset.

Before, /train only accepted a folder or dataset.yaml inside the container, so it could not be
used from the platform. The yaml inside a ZIP is not trusted: the delivered
create_dataset_yaml() writes absolute paths of the author's machine.
"""
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest
import yaml

from app.plugins.ml7_cereals_grain_pest_detection import plugin

_NAMES = ["cf", "sz", "rd", "tc", "os"]


def _dataset(root: Path, *, names=None, splits=("train", "val", "test"), yaml_extra=None) -> Path:
    """Minimal YOLO dataset: one empty image/label per split and a dataset.yaml."""
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / "1.jpg").write_bytes(b"jpg")
        (root / "labels" / split / "1.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    cfg = {"train": "images/train", "val": "images/val", "nc": 5,
           "names": _NAMES if names is None else names, **(yaml_extra or {})}
    (root / "dataset.yaml").write_text(yaml.safe_dump(cfg))
    return root


def _zip(src: Path, dest: Path, arc_prefix: str = "") -> Path:
    with zipfile.ZipFile(dest, "w") as zf:
        for f in src.rglob("*"):
            if f.is_file():
                zf.write(f, f"{arc_prefix}{f.relative_to(src)}")
    return dest


@pytest.mark.parametrize("arc_prefix", ["", "train_yolo/"])
def test_zip_at_root_or_in_one_folder(tmp_path, arc_prefix):
    """The dataset may be at the ZIP root or inside a single top-level folder."""
    zip_path = _zip(_dataset(tmp_path / "ds"), tmp_path / "ds.zip", arc_prefix)
    with plugin._local_dataset_yaml(str(zip_path)) as dataset_yaml:  # pylint: disable=protected-access
        cfg = yaml.safe_load(dataset_yaml.read_text())
        assert Path(cfg["path"]) == dataset_yaml.parent
        assert (cfg["train"], cfg["val"], cfg["test"]) == ("images/train", "images/val", "images/test")
        assert cfg["names"] == _NAMES
        assert plugin._count_images(dataset_yaml, "train") == 1  # pylint: disable=protected-access
        extract_dir = dataset_yaml.parent if not arc_prefix else dataset_yaml.parent.parent
    assert not extract_dir.exists()  # removed on exit


def test_zip_split_paths_are_replaced(tmp_path):
    """Absolute paths of another machine (or of the container) in the yaml are ignored."""
    ds = _dataset(tmp_path / "ds", splits=("train", "val"), yaml_extra={
        "path": "/etc", "train": "/home/autor/data/splits/images/train",
        "val": "../../outside", "test": "/home/autor/data/splits/images/test",
    })
    with plugin._local_dataset_yaml(str(_zip(ds, tmp_path / "ds.zip"))) as dataset_yaml:  # pylint: disable=protected-access
        cfg = yaml.safe_load(dataset_yaml.read_text())
    assert Path(cfg["path"]) == dataset_yaml.parent
    assert (cfg["train"], cfg["val"]) == ("images/train", "images/val")
    assert "test" not in cfg  # no images/test in the ZIP


def test_s3_zip_is_downloaded(tmp_path, monkeypatch):
    """An s3:// ZIP goes through local_file_path, like the other image plugins."""
    zip_path = _zip(_dataset(tmp_path / "ds"), tmp_path / "ds.zip")

    def fake_download(_uri):
        fd, tmp = tempfile.mkstemp(suffix=".zip")
        with open(fd, "wb") as fh:
            fh.write(zip_path.read_bytes())
        return tmp

    monkeypatch.setattr("app.infrastructure.artifact_store.download_s3_file", fake_download)
    with plugin._local_dataset_yaml("s3://bucket/train/ds.zip") as dataset_yaml:  # pylint: disable=protected-access
        assert (dataset_yaml.parent / "images" / "val" / "1.jpg").exists()


def test_s3_non_zip_rejected(tmp_path, monkeypatch):
    """A downloaded yaml alone has no images next to it: only ZIPs are valid over s3://."""
    yaml_file = _dataset(tmp_path / "ds") / "dataset.yaml"

    def fake_download(_uri):
        fd, tmp = tempfile.mkstemp(suffix=".yaml")
        with open(fd, "wb") as fh:
            fh.write(yaml_file.read_bytes())
        return tmp

    monkeypatch.setattr("app.infrastructure.artifact_store.download_s3_file", fake_download)
    with pytest.raises(ValueError, match="s3://"):
        with plugin._local_dataset_yaml("s3://bucket/train/dataset.yaml"):  # pylint: disable=protected-access
            pass


def test_zip_with_other_number_of_classes_rejected(tmp_path):
    """The detector head and TRAINING_CLASS_ID_TO_CODE have 5 classes."""
    ds = _dataset(tmp_path / "ds", names=["cf", "sz", "rd", "tc", "os", "unknown"])
    with pytest.raises(ValueError, match="6 clases"):
        with plugin._local_dataset_yaml(str(_zip(ds, tmp_path / "ds.zip"))):  # pylint: disable=protected-access
            pass


def test_zip_without_val_rejected(tmp_path):
    """Never train without a validation split."""
    ds = _dataset(tmp_path / "ds", splits=("train",))
    with pytest.raises(ValueError, match="images/val"):
        with plugin._local_dataset_yaml(str(_zip(ds, tmp_path / "ds.zip"))):  # pylint: disable=protected-access
            pass


def test_zip_without_dataset_yaml_rejected(tmp_path):
    """A ZIP without dataset.yaml is a 400, not an Ultralytics error."""
    ds = _dataset(tmp_path / "ds")
    (ds / "dataset.yaml").unlink()
    with pytest.raises(ValueError, match="dataset.yaml"):
        with plugin._local_dataset_yaml(str(_zip(ds, tmp_path / "ds.zip"))):  # pylint: disable=protected-access
            pass


def test_local_dir_and_yaml_used_in_place(tmp_path):
    """A local folder or yaml keeps working as before, without copying or rewriting."""
    ds = _dataset(tmp_path / "ds")
    original = (ds / "dataset.yaml").read_text()
    for data_path in (ds, ds / "dataset.yaml"):
        with plugin._local_dataset_yaml(str(data_path)) as dataset_yaml:  # pylint: disable=protected-access
            assert dataset_yaml == ds / "dataset.yaml"
    assert (ds / "dataset.yaml").read_text() == original
    shutil.rmtree(ds)


def test_validate_best_writes_inside_the_train_dir(tmp_path, monkeypatch):
    """Without project=, Ultralytics wrote runs/detect/val-N under its global runs_dir."""
    calls = {}

    class _Box:  # pylint: disable=too-few-public-methods
        map50, map, mp, mr = 0.5, 0.3, 0.4, 0.6

    class _FakeYolo:  # pylint: disable=too-few-public-methods
        def val(self, **kwargs):
            calls.update(kwargs)
            return type("Metrics", (), {"box": _Box})()

    monkeypatch.setattr(plugin, "_load_yolo_from_path", lambda _p: _FakeYolo())
    plugin._validate_best(tmp_path / "best.pt", tmp_path / "dataset.yaml", "cpu", tmp_path)  # pylint: disable=protected-access
    assert calls["project"] == str(tmp_path)
    assert calls["name"] == "val"
