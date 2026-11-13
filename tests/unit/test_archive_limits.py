"""safe_extract_zip: bounded extraction of the ZIPs users send to /train and batch /predict."""
import zipfile

import pytest

from app.domain.services.exceptions import ArchiveLimitExceededError
from app.infrastructure.archive import safe_extract_zip
from tests.unit.test_mlflow_persistence_contract import ML10, ML10_PREFIX


def _zip(path, members, compression=zipfile.ZIP_DEFLATED):
    """Write a ZIP at path with the given name->bytes members and return its path."""
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return str(path)


def test_extracts_a_normal_dataset_with_folders(tmp_path):
    """A normal ZIP with nested folders is extracted intact."""
    src = _zip(tmp_path / "ok.zip", {"data/fly/a.jpg": b"\xff\xd8" + bytes(range(256)) * 40,
                                     "data/tick/b.jpg": b"img", "data/labels.json": b"{}"})
    safe_extract_zip(src, str(tmp_path / "out"))
    assert (tmp_path / "out/data/fly/a.jpg").read_bytes().startswith(b"\xff\xd8")
    assert (tmp_path / "out/data/labels.json").read_bytes() == b"{}"


def test_rejects_a_zip_bomb_by_compression_ratio(tmp_path):
    """A highly compressed member is rejected as a zip bomb."""
    src = _zip(tmp_path / "bomb.zip", {"zeros.bin": b"\0" * (8 * 1024 ** 2)})  # ~1000x
    with pytest.raises(ArchiveLimitExceededError, match="zip bomb"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_small_members_are_exempt_from_the_ratio(tmp_path):
    """Small members are not subject to the compression-ratio check."""
    src = _zip(tmp_path / "json.zip", {"annotations.json": b'{"a": 1}' * 50_000})  # 400 KB, >200x
    safe_extract_zip(src, str(tmp_path / "out"))


def test_rejects_too_many_files(tmp_path, monkeypatch):
    """More members than ARCHIVE_MAX_FILES are rejected."""
    monkeypatch.setenv("ARCHIVE_MAX_FILES", "3")
    src = _zip(tmp_path / "many.zip", {f"{i}.jpg": b"x" for i in range(4)})
    with pytest.raises(ArchiveLimitExceededError, match="ARCHIVE_MAX_FILES"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_rejects_total_size_over_the_limit(tmp_path, monkeypatch):
    """An uncompressed total above ARCHIVE_MAX_TOTAL_BYTES is rejected."""
    monkeypatch.setenv("ARCHIVE_MAX_TOTAL_BYTES", "1000")
    src = _zip(tmp_path / "big.zip", {"a.bin": b"x" * 600, "b.bin": b"y" * 600}, zipfile.ZIP_STORED)
    with pytest.raises(ArchiveLimitExceededError, match="ARCHIVE_MAX_TOTAL_BYTES"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_not_a_zip_is_a_value_error(tmp_path):
    """A file that is not a ZIP raises ValueError."""
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="ZIP válido"):
        safe_extract_zip(str(bad), str(tmp_path / "out"))


def test_paths_escaping_the_destination_are_rejected(tmp_path):
    """Members whose path escapes the destination are rejected and not written."""
    src = _zip(tmp_path / "slip.zip", {"../evil.txt": b"x"})
    with pytest.raises(ArchiveLimitExceededError, match="fuera del destino"):
        safe_extract_zip(src, str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


def test_archive_limit_maps_to_413(client, fake_plugins, monkeypatch, lacteo_inline_payload):
    """ArchiveLimitExceededError maps to HTTP 413 on /predict and /train."""
    fake_plugins[ML10].raise_on_inline = ArchiveLimitExceededError("demasiado grande")
    assert client.post(f"{ML10_PREFIX}/predict", json=lacteo_inline_payload).status_code == 413

    def failing_train(**_kwargs):
        raise ArchiveLimitExceededError("demasiado grande")
    monkeypatch.setattr(fake_plugins[ML10], "train", failing_train)
    resp = client.post(
        f"{ML10_PREFIX}/train", json={"data_path": "/tmp/x.zip", "mlflow_run_id": "run-1"}
    )
    assert resp.status_code == 413


def test_invalid_env_value_falls_back_to_the_default(tmp_path, monkeypatch):
    """An unparsable env limit falls back to the default instead of failing."""
    monkeypatch.setenv("ARCHIVE_MAX_FILES", "200000  # comentario de un --env-file")
    safe_extract_zip(_zip(tmp_path / "ok.zip", {"a.jpg": b"x"}), str(tmp_path / "out"))


def test_skips_os_metadata_created_when_zipping_on_macos_or_windows(tmp_path):
    """__MACOSX/, AppleDouble ._* files and .DS_Store/Thumbs.db are never extracted.

    With them, a ZIP made with "Compress" in Finder had two top-level entries (wrapper folder
    + __MACOSX) and the plugins that look for one wrapper folder (ml5, ml8) failed.
    """
    src = _zip(tmp_path / "mac.zip", {
        "dataset/trigo/train/sano/a.jpg": b"img",
        "__MACOSX/dataset/trigo/train/sano/._a.jpg": b"meta",
        "dataset/trigo/train/sano/._b.jpg": b"meta",
        "dataset/.DS_Store": b"ds", "dataset/Thumbs.db": b"tb",
    })
    out = tmp_path / "out"
    safe_extract_zip(src, str(out))
    assert [p.name for p in out.iterdir()] == ["dataset"]
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["dataset/trigo/train/sano/a.jpg"]


def test_ml8_training_zip_built_on_macos_is_read(tmp_path):
    """ml8: wrapper folder + __MACOSX used to give 'No se encontraron imágenes de entrenamiento'."""
    from app.plugins.ml8_cereals_img_anomaly_detector.plugin import _load_records

    members = {f"cereales/trigo/train/sano/{i}.jpg": b"img" for i in range(5)}
    members.update({f"__MACOSX/cereales/trigo/train/sano/._{i}.jpg": b"m" for i in range(5)})
    members["cereales/maiz/train/hongos/x.webp"] = b"img"
    out = tmp_path / "out"
    safe_extract_zip(_zip(tmp_path / "ml8.zip", members), str(out))
    train, val = _load_records(str(out))
    assert len(train) + len(val) == 6


def test_ml8_training_error_explains_the_expected_structure(tmp_path):
    """A wrong layout (accents/capitals) gets the exact folder names and what was found."""
    from app.plugins.ml8_cereals_img_anomaly_detector.plugin import _load_records

    out = tmp_path / "out"
    members = {"Maíz/train/sano/a.jpg": b"img", "Trigo/train/sano/b.jpg": b"img"}
    safe_extract_zip(_zip(tmp_path / "bad.zip", members), str(out))
    with pytest.raises(ValueError, match=r"'arroz', 'maiz', 'sorgo', 'trigo'.*'Maíz', 'Trigo'"):
        _load_records(str(out))


def test_ml5_training_zip_built_on_macos_finds_its_wrapper_folder(tmp_path):
    """ml5: the CVAT layout inside one wrapper folder, with __MACOSX next to it."""
    from app.plugins.ml5_meat_cow_behaviour import plugin as ml5

    members = {
        "dataset/raw_frames/clip1/img_00001.jpg": b"img",
        "dataset/annotations/clip1/annotations/instances_default.json": b"{}",
        "__MACOSX/dataset/raw_frames/clip1/._img_00001.jpg": b"m",
    }
    seen = {}

    def fake_load_all_clips(raw_frames_dir, annotations_dir):
        seen["dirs"] = (raw_frames_dir, annotations_dir)
        return {"images": []}

    src = _zip(tmp_path / "ml5.zip", members)
    import unittest.mock as mock  # pylint: disable=import-outside-toplevel
    with mock.patch.object(ml5, "load_all_clips", fake_load_all_clips):
        with pytest.raises(ValueError):  # empty fake dataset; we only check the layout was found
            ml5._load_training_clips(src, str(tmp_path / "out"))
    assert seen["dirs"][0].endswith("dataset/raw_frames")
