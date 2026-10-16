"""Versioned base model in ArtifactStore: current.json + versions/<vN>/, never overwritten."""
import json

import pytest

import app.infrastructure.artifact_store as store_mod
from app.domain.services.exceptions import BaseVersionNotFoundError
from app.infrastructure.artifact_store import ArtifactStore
from tests.unit.fake_s3 import FakeS3

ROOT = "artifacts/fixed/m/"


@pytest.fixture(autouse=True)
def _fresh_active_versions(monkeypatch):
    monkeypatch.setattr(store_mod, "_ACTIVE_VERSIONS", {})


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(store_mod, "_build_s3_client", lambda: fake)
    monkeypatch.setenv("STORAGE_BUCKET", "xai")
    return fake


@pytest.fixture
def bundle(tmp_path):
    def _make(**files: bytes):
        d = tmp_path / f"bundle_{len(list(tmp_path.glob('bundle_*')))}"
        d.mkdir()
        for name, data in files.items():
            (d / name).write_bytes(data)
        return d
    return _make


def test_without_pointer_the_base_is_v0_at_the_root(s3, tmp_path):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    assert store.refresh() == "v0"
    assert store.local_dir == tmp_path / "cache" / "m"
    assert store.path("model.pkl").read_bytes() == b"old"
    assert store.version_info() == {"version": "v0", "parent": None, "created_at": None,
                                    "mlflow_run_id": None, "data_path": None, "metrics": {}}


def test_v0_download_ignores_versions_tree_and_pointer(s3, tmp_path):
    s3.put(ROOT + "model.pkl", b"old")
    s3.put(ROOT + "versions/v1/model.pkl", b"new")
    s3.put(ROOT + "versions/v1/version.json", b'{"version": "v1", "parent": "v0"}')
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.download_all_if_needed()   # active version is v0: no current.json was read
    local = tmp_path / "cache" / "m"
    assert (local / "model.pkl").read_bytes() == b"old"
    assert not (local / "versions").exists() and not (local / "current.json").exists()


def test_publish_inherits_the_parent_and_moves_the_pointer_last(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    s3.put(ROOT + "panel.csv", b"ctx")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.refresh()

    info = store.publish_version(bundle(**{"model.pkl": b"new"}),
                                 {"mlflow_run_id": "run-1", "data_path": "s3://d.csv", "metrics": {"mae": 1.5}})

    assert info["version"] == "v1" and info["parent"] == "v0" and info["mlflow_run_id"] == "run-1"
    assert info["metrics"] == {"mae": 1.5} and info["created_at"]
    assert s3.get(ROOT + "versions/v1/model.pkl") == b"new"
    assert s3.get(ROOT + "versions/v1/panel.csv") == b"ctx"
    assert json.loads(s3.get(ROOT + "versions/v1/version.json"))["parent"] == "v0"
    assert json.loads(s3.get(ROOT + "current.json")) == {"version": "v1"}
    assert s3.get(ROOT + "model.pkl") == b"old"          # v0 untouched
    assert store.active_version() == "v1"
    assert store.path("model.pkl").read_bytes() == b"new"

    info2 = store.publish_version(bundle(**{"extra.json": b"{}"}), {"metrics": {}})
    assert info2["version"] == "v2" and info2["parent"] == "v1"
    assert s3.get(ROOT + "versions/v2/model.pkl") == b"new"   # inherited from v1
    assert ROOT + "versions/v2/version.json" in s3.keys()


def test_version_json_of_the_parent_is_not_inherited_as_an_artifact(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.publish_version(bundle(**{"model.pkl": b"a"}), {})
    store.publish_version(bundle(**{"model.pkl": b"b"}), {})
    assert json.loads(s3.get(ROOT + "versions/v2/version.json"))["version"] == "v2"


def test_failed_upload_keeps_the_pointer_and_the_next_publish_skips_the_number(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    s3.fail_upload_on = "version.json"            # every artifact uploads, version.json (last) fails
    with pytest.raises(ConnectionError):
        store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert ROOT + "current.json" not in s3.keys()
    assert store.active_version() == "v0"
    assert ROOT + "versions/v1/model.pkl" in s3.keys()   # the partial folder stays

    s3.fail_upload_on = None
    assert store.publish_version(bundle(**{"model.pkl": b"new"}), {})["version"] == "v2"


def test_refresh_reads_the_pointer_and_stores_of_the_same_folder_share_it(s3, tmp_path):
    s3.put(ROOT + "current.json", b'{"version": "v1"}')
    s3.put(ROOT + "versions/v1/model.pkl", b"new")
    s3.put(ROOT + "versions/v1/version.json", b'{"version": "v1", "parent": "v0"}')
    first, second = ArtifactStore("m", root=tmp_path / "c"), ArtifactStore("m", root=tmp_path / "c")
    assert first.refresh() == "v1"
    assert second.active_version() == "v1"
    assert second.path("model.pkl").read_bytes() == b"new"
    assert second.local_dir == tmp_path / "c" / "m" / "versions" / "v1"
    assert ArtifactStore("m", root=tmp_path / "other").active_version() == "v0"


def test_set_active_version_and_version_info(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.publish_version(bundle(**{"model.pkl": b"a"}), {})
    store.publish_version(bundle(**{"model.pkl": b"b"}), {})

    store.set_active_version("v1")
    assert json.loads(s3.get(ROOT + "current.json")) == {"version": "v1"}
    assert store.version_info()["version"] == "v1"
    assert store.version_info("v2")["parent"] == "v1"
    store.set_active_version("v0")
    assert store.path("model.pkl").read_bytes() == b"old"
    with pytest.raises(BaseVersionNotFoundError, match="v9"):
        store.set_active_version("v9")
    with pytest.raises(BaseVersionNotFoundError):
        store.version_info("v9")
    assert store.version_exists("v0") and store.version_exists("v2") and not store.version_exists("v9")


def test_version_info_is_read_from_s3_when_the_version_is_not_cached(s3, tmp_path):
    s3.put(ROOT + "versions/v4/version.json", b'{"version": "v4", "parent": "v3", "metrics": {"f1": 0.7}}')
    store = ArtifactStore("m", root=tmp_path / "cache")
    assert store.version_info("v4")["metrics"] == {"f1": 0.7}


def test_local_mode_without_bucket(monkeypatch, tmp_path, bundle):
    monkeypatch.delenv("STORAGE_BUCKET", raising=False)
    base = tmp_path / "m"
    base.mkdir()
    (base / "model.pkl").write_bytes(b"old")
    store = ArtifactStore("m", root=tmp_path)
    info = store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert info["version"] == "v1"
    assert (base / "versions" / "v1" / "model.pkl").read_bytes() == b"new"
    assert json.loads((base / "current.json").read_text()) == {"version": "v1"}
    assert (base / "model.pkl").read_bytes() == b"old"

    store_mod._ACTIVE_VERSIONS.clear()  # pylint: disable=protected-access — simulates a restart
    assert ArtifactStore("m", root=tmp_path).refresh() == "v1"


def test_use_s3_false_ignores_the_bucket(s3, tmp_path, bundle):
    store = ArtifactStore("m", root=tmp_path, use_s3=False)
    store.publish_version(bundle(**{"model.pkl": b"new"}), {})
    assert s3.keys() == []
    assert (tmp_path / "m" / "versions" / "v1" / "model.pkl").exists()


def test_invalid_version_strings_are_rejected_and_nothing_changes(s3, tmp_path, bundle):
    s3.put(ROOT + "model.pkl", b"old")
    store = ArtifactStore("m", root=tmp_path / "cache")
    store.publish_version(bundle(**{"model.pkl": b"a"}), {})
    pointer_before = s3.get(ROOT + "current.json")

    with pytest.raises(BaseVersionNotFoundError, match="inválida"):
        store.set_active_version("v1/../../other/versions/v1")
    with pytest.raises(BaseVersionNotFoundError):
        store.version_info("../x")
    assert not store.version_exists("v1\n")
    assert s3.get(ROOT + "current.json") == pointer_before
    assert store.active_version() == "v1"


def test_refresh_fails_visibly_on_a_corrupt_pointer(s3, tmp_path):
    s3.put(ROOT + "current.json", b'{"version": "../evil"}')
    store = ArtifactStore("m", root=tmp_path / "cache")
    with pytest.raises(BaseVersionNotFoundError, match="evil"):
        store.refresh()
    assert store.active_version() == "v0"
