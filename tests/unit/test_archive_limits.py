"""safe_extract_zip: bounded extraction of the ZIPs users send to /train and batch /predict."""
import zipfile

import pytest

from app.domain.services.exceptions import ArchiveLimitExceededError
from app.infrastructure.archive import safe_extract_zip
from tests.unit.test_mlflow_persistence_contract import ML10, ML10_PREFIX


def _zip(path, members, compression=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return str(path)


def test_extracts_a_normal_dataset_with_folders(tmp_path):
    src = _zip(tmp_path / "ok.zip", {"data/fly/a.jpg": b"\xff\xd8" + bytes(range(256)) * 40,
                                     "data/tick/b.jpg": b"img", "data/labels.json": b"{}"})
    safe_extract_zip(src, str(tmp_path / "out"))
    assert (tmp_path / "out/data/fly/a.jpg").read_bytes().startswith(b"\xff\xd8")
    assert (tmp_path / "out/data/labels.json").read_bytes() == b"{}"


def test_rejects_a_zip_bomb_by_compression_ratio(tmp_path):
    src = _zip(tmp_path / "bomb.zip", {"zeros.bin": b"\0" * (8 * 1024 ** 2)})  # ~1000x
    with pytest.raises(ArchiveLimitExceededError, match="zip bomb"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_small_members_are_exempt_from_the_ratio(tmp_path):
    src = _zip(tmp_path / "json.zip", {"annotations.json": b'{"a": 1}' * 50_000})  # 400 KB, >200x
    safe_extract_zip(src, str(tmp_path / "out"))


def test_rejects_too_many_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_MAX_FILES", "3")
    src = _zip(tmp_path / "many.zip", {f"{i}.jpg": b"x" for i in range(4)})
    with pytest.raises(ArchiveLimitExceededError, match="ARCHIVE_MAX_FILES"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_rejects_total_size_over_the_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_MAX_TOTAL_BYTES", "1000")
    src = _zip(tmp_path / "big.zip", {"a.bin": b"x" * 600, "b.bin": b"y" * 600}, zipfile.ZIP_STORED)
    with pytest.raises(ArchiveLimitExceededError, match="ARCHIVE_MAX_TOTAL_BYTES"):
        safe_extract_zip(src, str(tmp_path / "out"))


def test_not_a_zip_is_a_value_error(tmp_path):
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="ZIP válido"):
        safe_extract_zip(str(bad), str(tmp_path / "out"))


def test_paths_escaping_the_destination_are_rejected(tmp_path):
    src = _zip(tmp_path / "slip.zip", {"../evil.txt": b"x"})
    with pytest.raises(ArchiveLimitExceededError, match="fuera del destino"):
        safe_extract_zip(src, str(tmp_path / "out"))
    assert not (tmp_path / "evil.txt").exists()


def test_archive_limit_maps_to_413(client, fake_plugins, monkeypatch, lacteo_inline_payload):
    fake_plugins[ML10].raise_on_inline = ArchiveLimitExceededError("demasiado grande")
    assert client.post(f"{ML10_PREFIX}/predict", json=lacteo_inline_payload).status_code == 413

    def failing_train(**_kwargs):
        raise ArchiveLimitExceededError("demasiado grande")
    monkeypatch.setattr(fake_plugins[ML10], "train", failing_train)
    resp = client.post(f"{ML10_PREFIX}/train", json={"data_path": "/tmp/x.zip", "mlflow_run_id": "run-1"})
    assert resp.status_code == 413


def test_invalid_env_value_falls_back_to_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHIVE_MAX_FILES", "200000  # comentario de un --env-file")
    safe_extract_zip(_zip(tmp_path / "ok.zip", {"a.jpg": b"x"}), str(tmp_path / "out"))
